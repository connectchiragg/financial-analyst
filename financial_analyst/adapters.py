"""Read ports and source checks for a manually reviewed PDF fixture.

The bindings below authenticate curated locations. They do not extract new
observations or claim to understand arbitrary report layouts.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
from pathlib import Path
from typing import Protocol, Sequence
from urllib.parse import urlsplit

from .analytics import RevenueObservation


class EvidenceError(ValueError):
    """A fixture assertion lacks authenticated source support."""


@dataclass(frozen=True)
class Citation:
    ref: str
    document_name: str
    page: int
    locator: str
    link: str | None = None


@dataclass(frozen=True)
class Evidence:
    ref: str
    document_name: str
    page: int
    locator: str
    excerpt: str
    link: str | None = None

    @property
    def citation(self) -> Citation:
        return Citation(self.ref, self.document_name, self.page, self.locator, self.link)


class AnalyticsReadPort(Protocol):
    def read_observations(self, company: str, period: str) -> Sequence[RevenueObservation]: ...


class EvidenceReadPort(Protocol):
    def validate_observation(self, observation: RevenueObservation) -> None: ...
    def resolve(self, ref: str) -> Evidence: ...


class PassageReadPort(Protocol):
    def growth_passages(self, company: str, period: str) -> Sequence[Evidence]: ...


def _text(value: str) -> str:
    return " ".join(value.split())


class FileFixtureAdapter:
    """Authenticate an explicit JSON fixture against its original local PDF."""

    mode = "fixture"

    def __init__(self, fixture_path: str | Path, source_path: str | Path | None = None):
        try:
            self._load(fixture_path, source_path)
        except EvidenceError:
            raise
        except (TypeError, ValueError, KeyError, AttributeError, IndexError) as exc:
            raise EvidenceError("Fixture fields or reviewed source bindings are invalid.") from exc

    def _load(self, fixture_path: str | Path, source_path: str | Path | None) -> None:
        try:
            payload = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise EvidenceError("Fixture could not be read as JSON.") from exc
        if not isinstance(payload, dict):
            raise EvidenceError("Fixture JSON must be an object.")
        self._payload = payload
        self._evidence: dict[str, dict] = {}
        self._locations: dict[str, str] = {}
        self._observations: list[RevenueObservation] = []
        self.source_conflicts: tuple[dict, ...] = ()
        source = payload.get("source", {})
        path = Path(source_path or source.get("local_path", ""))
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise EvidenceError("Original source PDF is unavailable.") from exc
        if not source.get("sha256") or digest != source["sha256"]:
            raise EvidenceError("Source PDF identity does not match the reviewed fixture.")
        if not isinstance(source.get("document_name"), str) or not source["document_name"].strip():
            raise EvidenceError("Source document identity is missing.")
        if path.name != source["document_name"]:
            raise EvidenceError("Source document name does not match the original local PDF.")
        self._source = source
        self._link = self._source_link(source)
        try:
            import pdfplumber
        except ImportError as exc:
            raise EvidenceError("PDF fixture validation requires the optional pdfplumber dependency.") from exc
        try:
            with pdfplumber.open(path) as pdf:
                if source.get("page_count") != len(pdf.pages):
                    raise EvidenceError("Source page count does not match the reviewed fixture.")
                self._pages = [
                    {
                        "words": page.extract_words(),
                        "text": _text(" ".join(word["text"] for word in page.extract_words(use_text_flow=True))),
                        "width": page.width,
                        "height": page.height,
                    }
                    for page in pdf.pages
                ]
        except EvidenceError:
            raise
        except Exception as exc:
            raise EvidenceError("Source PDF could not be inspected.") from exc
        self._validate_evidence()
        self.source_conflicts = self._validated_year_end_conflicts()
        fields = RevenueObservation.__dataclass_fields__
        try:
            for raw in payload.get("observations", ()):
                values = {name: raw[name] for name in fields}
                values["value"] = Decimal(values["value"])
                values["evidence_refs"] = tuple(values["evidence_refs"])
                observation = RevenueObservation(**values)
                self.validate_observation(observation)
                self._observations.append(observation)
        except (KeyError, TypeError, InvalidOperation) as exc:
            raise EvidenceError("Fixture observation fields are incomplete or invalid.") from exc
        if not self._observations:
            raise EvidenceError("Fixture has no supported observations.")
        for ref in payload.get("growth_explanation_refs", ()):
            self.resolve(ref)

    def _validated_year_end_conflicts(self) -> tuple[dict, ...]:
        # Preserve the source's printed labels, never fixture-authored handling
        # prose. This scan only authenticates year-end labels, not calendar dates.
        months = "January|February|March|April|May|June|July|August|September|October|November|December"
        labels = {}
        refs = []
        for number, page in enumerate(self._pages, 1):
            matches = re.findall(rf"\bY/E\s+(?:{months})\b", page["text"], re.I)
            if matches:
                labels[f"page_{number}"] = tuple(dict.fromkeys(matches))
        distinct = {label.casefold() for values in labels.values() for label in values}
        if len(distinct) < 2:
            return ()
        for page_key, values in labels.items():
            page = int(page_key.removeprefix("page_"))
            for index, label in enumerate(values):
                ref = f"source-year-end-page-{page}-{index}"
                if ref in self._evidence:
                    raise EvidenceError("Fixture reference conflicts with a verified source-label citation.")
                self._evidence[ref] = {"ref": ref, "document_name": self._source["document_name"],
                                       "page": page, "locator": "Printed fiscal year-end label",
                                       "excerpt": label}
                refs.append(ref)
        return ({"subject": "Fiscal year-end labels", "source_labels": labels,
                 "evidence_refs": tuple(refs),
                 "handling": "Preserve the printed labels; calendar dates are unresolved."},)

    @staticmethod
    def _source_link(source: dict) -> str | None:
        """Only pass through a supplied Drive link whose document ID agrees."""
        link = source.get("drive_url")
        if link is None:
            return None
        parts = urlsplit(link)
        file_id = source.get("drive_file_id")
        if parts.scheme != "https" or parts.netloc != "drive.google.com" or not file_id:
            raise EvidenceError("Source citation link is not an authenticated document link.")
        if parts.path != f"/file/d/{file_id}/view":
            raise EvidenceError("Source citation link does not match its document identity.")
        return link

    def _page(self, number: int) -> dict:
        if type(number) is not int or not 1 <= number <= len(self._pages):
            raise EvidenceError("Evidence refers to an unavailable source page.")
        return self._pages[number - 1]

    def _boxed_text(self, page_number: int, box: Sequence[float]) -> str:
        page = self._page(page_number)
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise EvidenceError("Source location must have four PDF coordinates.")
        try:
            x0, top, x1, bottom = map(float, box)
        except (TypeError, ValueError) as exc:
            raise EvidenceError("Source location coordinates are invalid.") from exc
        if not (0 <= x0 < x1 <= page["width"] and 0 <= top < bottom <= page["height"]):
            raise EvidenceError("Source location is outside its page.")
        words = [word["text"] for word in page["words"]
                 if x0 - .1 <= word["x0"] and word["x1"] <= x1 + .1
                 and top - .1 <= word["top"] and word["bottom"] <= bottom + .1]
        return _text(" ".join(words))

    def _binding(self, page_number: int, binding: dict) -> str:
        if not isinstance(binding, dict) or not isinstance(binding.get("text"), str):
            raise EvidenceError("Reviewed source binding is missing.")
        if self._boxed_text(page_number, binding.get("bbox_pt")) != _text(binding["text"]):
            raise EvidenceError("Reviewed source label does not match its PDF location.")
        return binding["text"]

    def _validate_evidence(self) -> None:
        for raw in self._payload.get("evidence", ()):
            ref = raw.get("ref")
            if not isinstance(ref, str) or not ref or ref in self._evidence:
                raise EvidenceError("Evidence references must be present and unique.")
            if raw.get("document_name") != self._source["document_name"]:
                raise EvidenceError("Evidence document identity conflicts with the source.")
            page = self._page(raw.get("page"))
            excerpt = raw.get("excerpt")
            if not isinstance(excerpt, str) or not excerpt.strip() or not raw.get("locator"):
                raise EvidenceError("Evidence requires a source excerpt and locator.")
            if "bbox_pt" in raw:
                if self._boxed_text(raw["page"], raw["bbox_pt"]) != _text(excerpt):
                    raise EvidenceError("Evidence cell does not match its PDF location.")
            elif _text(excerpt) not in page["text"]:
                raise EvidenceError("Evidence excerpt is absent from its source page.")
            self._evidence[ref] = raw
        for raw in self._evidence.values():
            for ref in raw.get("context_refs", ()):
                self.resolve(ref)

    def resolve(self, ref: str) -> Evidence:
        raw = self._evidence.get(ref)
        if raw is None:
            raise EvidenceError("Evidence reference cannot be resolved.")
        locator = self._locations.get(ref)
        if locator is None:
            if "bbox_pt" in raw:
                locator = "Source cell at PDF points " + ", ".join(str(item) for item in raw["bbox_pt"])
            else:
                locator = "Source passage beginning: " + " ".join(raw["excerpt"].split()[:12])
        return Evidence(ref, raw["document_name"], raw["page"], locator, raw["excerpt"], self._link)

    def read_observations(self, company: str, period: str) -> tuple[RevenueObservation, ...]:
        return tuple(item for item in self._observations if item.company == company and item.period == period)

    def validate_observation(self, observation: RevenueObservation) -> None:
        refs = observation.evidence_refs
        if not refs or len(set(refs)) != len(refs):
            raise EvidenceError("Observation requires unique supporting references.")
        for ref in refs:
            self.resolve(ref)
        cells = [self._evidence[ref] for ref in refs if "bbox_pt" in self._evidence[ref]]
        matching = [cell for cell in cells if cell["excerpt"] == observation.source_value_raw]
        if len(matching) != 1:
            raise EvidenceError("Observation needs exactly one authenticated precise source cell.")
        cell = matching[0]
        try:
            value = Decimal(observation.source_value_raw.replace(",", ""))
        except (AttributeError, InvalidOperation) as exc:
            raise EvidenceError("Source cell is not a numeric revenue amount.") from exc
        if value != observation.value:
            raise EvidenceError("Observation value disagrees with the original source cell.")
        context_refs = cell.get("context_refs", ())
        if len(context_refs) != 1 or context_refs[0] not in refs:
            raise EvidenceError("Observation is missing its cited table context.")
        context = self._evidence[context_refs[0]]
        if context["page"] != cell["page"]:
            raise EvidenceError("Source cell and table context refer to different pages.")
        bindings = context.get("bindings", {})
        page = context["page"]
        expected = {
            "company": observation.company, "metric": observation.source_metric_raw,
            "year_end": observation.year_end_raw, "unit": f"({observation.source_unit_raw})",
            "scope": "(Consol.)" if observation.scope == "consolidated" else None,
        }
        for field, label in expected.items():
            if label is None or self._binding(page, bindings.get(field)) != label:
                raise EvidenceError(f"Observation {field} is unsupported by its table labels.")
        if observation.metric != "net_sales" or observation.source_metric_raw != "Net Sales":
            raise EvidenceError("Fixture supports only the explicit Net Sales to revenue mapping.")
        units = {"INRm": ("INR", "million"), "INRb": ("INR", "billion")}
        if units.get(observation.source_unit_raw) != (observation.currency, observation.source_unit):
            raise EvidenceError("Currency or monetary unit conflicts with the printed table unit.")
        metric_box = bindings["metric"]["bbox_pt"]
        if abs(float(metric_box[1]) - float(cell["bbox_pt"][1])) > 1:
            raise EvidenceError("Source cell is not on the cited metric row.")
        column_matches = []
        for group in bindings.get("column_groups", ()):
            group_text = self._binding(page, group)
            columns = group.get("columns", ())
            if not columns:
                raise EvidenceError("Reviewed table group has no column bindings.")
            for column in columns:
                column_text = self._binding(page, column)
                if cell["ref"] in column.get("cell_refs", ()):
                    column_matches.append((group_text, column_text, column))
            first_x = float(columns[0]["bbox_pt"][0])
            last_x = float(columns[-1]["bbox_pt"][2])
            group_center = (float(group["bbox_pt"][0]) + float(group["bbox_pt"][2])) / 2
            if not first_x - 10 <= group_center <= last_x + 10:
                raise EvidenceError("Fiscal header is outside its reviewed column group.")
        if len(column_matches) != 1:
            raise EvidenceError("Source cell has no unambiguous reviewed column identity.")
        group_text, quarter_text, column = column_matches[0]
        if abs(float(column["bbox_pt"][2]) - float(cell["bbox_pt"][2])) > 1:
            raise EvidenceError("Source cell is not aligned with its cited quarter column.")
        if not float(column["bbox_pt"][3]) < float(cell["bbox_pt"][1]):
            raise EvidenceError("Quarter column header must precede its source cell.")
        if quarter_text == group_text:
            header = f"{group_text} annual total"
            period = group_text.removesuffix("E")
        else:
            header = f"{group_text} / {quarter_text}"
            period = quarter_text.removesuffix("E") + group_text.removesuffix("E")
        if header != observation.source_header_raw or period != observation.period:
            raise EvidenceError("Observation period conflicts with the original fiscal column labels.")
        if quarter_text != group_text and quarter_text.endswith("E"):
            if observation.kind != "broker_estimate":
                raise EvidenceError("Estimate-column source cell cannot be treated as an actual.")
        elif group_text.endswith("E") and observation.kind == "reported_actual":
            self._validate_actual_status(observation)
        elif group_text.endswith("E"):
            if observation.kind != "broker_forecast":
                raise EvidenceError("Forecast-column status is unsupported.")
        elif observation.kind != "reported_actual":
            raise EvidenceError("Historical column does not support the supplied observation kind.")
        self._locations[cell["ref"]] = f"{observation.source_metric_raw} row / {observation.source_header_raw}"
        self._locations[context["ref"]] = "Quarterly table heading and authenticated source labels"

    def _validate_actual_status(self, observation: RevenueObservation) -> None:
        passages = [self.resolve(ref) for ref in observation.evidence_refs]
        period = re.escape(observation.period)
        page_one = [item for item in passages if item.page == 1
                    and re.search(rf"\bdelivered\b[^.]*\brevenue\b[^.]*\bin {period}\b", item.excerpt, re.I)]
        page_three = [item for item in passages if item.page == 3
                      and re.search(rf"\breported {period} revenue\b", item.excerpt, re.I)]
        if not page_one or not page_three:
            raise EvidenceError("Forecast-headed actual needs reported-result corroboration on pages 1 and 3.")
        amount_pattern = rf"\b{re.escape(observation.currency)}(\d+(?:\.\d+)?)([mb])\b"
        matches = re.findall(amount_pattern, page_three[0].excerpt, re.I)
        if not matches:
            raise EvidenceError("Reported-result corroboration is missing its revenue amount.")
        printed, suffix = matches[0]
        sign, digits, exponent = observation.value.as_tuple()
        shift = (3 if observation.source_unit == "billion" else 0) - (3 if suffix.lower() == "b" else 0)
        rounded = Decimal((sign, digits, exponent + shift))
        decimals = len(printed.partition(".")[2])
        with localcontext() as context:
            context.prec = max(40, len(digits) + abs(shift) + decimals + 2)
            if rounded.quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP) != Decimal(printed):
                raise EvidenceError("Reported-result corroboration conflicts with the precise revenue amount.")

    def growth_passages(self, company: str, period: str) -> tuple[Evidence, ...]:
        actuals = [item for item in self.read_observations(company, period) if item.kind == "reported_actual"]
        if len(actuals) != 1:
            raise EvidenceError("Fixture has no unambiguous reported quarter for growth passages.")
        self.validate_observation(actuals[0])
        passages = tuple(self.resolve(ref) for ref in self._payload.get("growth_explanation_refs", ()))
        if not any(re.search(rf"\breported {re.escape(period)} revenue\b", item.excerpt, re.I) for item in passages):
            raise EvidenceError("Growth passages lack the reported-quarter revenue context.")
        for item in passages:
            if (item.page != 3 or "bbox_pt" in self._evidence[item.ref]
                    or not re.search(r"\b(?:revenue|portfolio|medicines|respiratory|vaccine|oncology)", item.excerpt, re.I)
                    or not re.search(r"\b(?:growth|momentum|traction|contribution|execution)\b", item.excerpt, re.I)):
                raise EvidenceError("Curated passage does not support a revenue-growth explanation.")
        return passages
