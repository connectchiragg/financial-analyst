"""Reviewed, source-bound facts for multiple reports, behind read/write ports.

This format is separate from the trusted GSK fixture and pending extraction DB.
Proof packs are curated input: quote checks authenticate their source locations,
not an LLM's interpretation of financial column associations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re
import sqlite3
from urllib.parse import urlsplit

from .adapters import Citation, Evidence, EvidenceError


class CorpusError(EvidenceError):
    """A reviewed corpus, its source proof, or persistent copy is invalid."""


@dataclass(frozen=True)
class FactFilter:
    companies: tuple[str, ...] = ()
    period: str | None = None
    metrics: tuple[str, ...] = ()
    scope: str | None = None
    kind: str | None = None
    currency: str | None = None
    unit: str | None = None
    source_sha256: str | None = None


@dataclass(frozen=True)
class ReviewedFact:
    fact_id: str
    source_sha256: str
    company: str
    period: str | None
    scope: str | None
    metric: str
    kind: str | None
    value_text: str | None
    text_value: str | None
    currency: str | None
    unit: str | None
    raw_labels: dict
    evidence_refs: tuple[str, ...]
    proof: dict
    review: dict

    @property
    def value(self) -> Decimal | None:
        return Decimal(self.value_text) if self.value_text is not None else None

    @property
    def capabilities(self) -> tuple[str, ...]:
        return tuple(self.review.get("capabilities", ())) if self.review.get("status") == "reviewed" else ()


@dataclass(frozen=True)
class CorpusIngestionResult:
    documents_added: int
    evidence_added: int
    facts_added: int


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, RecursionError) as error:
        raise CorpusError("Corpus metadata must contain finite JSON values.") from error


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CorpusError("Duplicate corpus JSON fields are invalid.")
        result[key] = value
    return result


def _load(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(CorpusError("Nonfinite corpus JSON is invalid.")))
    except (OSError, ValueError, TypeError, RecursionError) as error:
        if isinstance(error, CorpusError):
            raise
        raise CorpusError("The reviewed corpus JSON could not be read.") from error


def _text(value, name, nullable=False):
    if value is None and nullable:
        return
    if not isinstance(value, str) or not value.strip():
        raise CorpusError(f"Corpus {name} must be nonblank text.")


def _digest(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise CorpusError("Corpus sources require a canonical SHA-256.")


def _normalized(text):
    return " ".join(text.split())


def _refs(value):
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (tuple, list)) and value and all(isinstance(item, str) and item.strip() for item in value):
        return tuple(value)
    raise CorpusError("Source bindings require one or more evidence references.")


def _number(raw):
    _text(raw, "raw numeric value")
    spelling = raw.strip()
    negative = spelling.startswith("(") and spelling.endswith(")")
    if negative:
        spelling = spelling[1:-1].strip()
    spelling = re.sub(r"(?:%|bps?|x)$", "", spelling, flags=re.I).strip()
    if re.fullmatch(r"[+-]?(?:\d+|\d{1,3}(?:,\d{3})+|\d{1,2}(?:,\d{2})*,\d{3})(?:\.\d+)?|[+-]?\.\d+", spelling) is None:
        raise CorpusError("A reviewed numeric value requires exact numeric source spelling.")
    try:
        value = Decimal(spelling.replace(",", ""))
    except InvalidOperation as error:
        raise CorpusError("A reviewed numeric value is invalid.") from error
    if not value.is_finite():
        raise CorpusError("A reviewed numeric value must be finite.")
    return value.copy_negate() if negative else value


def _fact_copy(fact):
    values = json.loads(_json(asdict(fact)))
    values["evidence_refs"] = tuple(values["evidence_refs"])
    return ReviewedFact(**values)


def _matches(fact, filters):
    return ((not filters.companies or fact.company in filters.companies)
            and (not filters.metrics or fact.metric in filters.metrics)
            and all(getattr(filters, name) is None or getattr(fact, name) == getattr(filters, name)
                for name in ("period", "scope", "kind", "currency", "unit", "source_sha256")))


def _validate_filter(filters):
    if not isinstance(filters, FactFilter):
        raise CorpusError("Corpus reads require a FactFilter.")
    for name in ("companies", "metrics"):
        items = getattr(filters, name)
        if not isinstance(items, tuple):
            raise CorpusError(f"Corpus {name} filters must be unique tuples.")
        for item in items:
            _text(item, name)
        if len(set(items)) != len(items):
            raise CorpusError(f"Corpus {name} filters must be unique tuples.")
    for name in ("period", "scope", "kind", "currency", "unit", "source_sha256"):
        _text(getattr(filters, name), name, nullable=True)
    if filters.source_sha256 is not None:
        _digest(filters.source_sha256)


def _fact_batch(facts):
    if not isinstance(facts, tuple) or any(not isinstance(fact, ReviewedFact) for fact in facts):
        raise CorpusError("Fact validation requires a tuple of reviewed facts.")
    for fact in facts:
        _text(fact.fact_id, "fact identifier")
    if len({fact.fact_id for fact in facts}) != len(facts):
        raise CorpusError("Fact validation requires unique fact identifiers.")


def _ref_batch(refs):
    if not isinstance(refs, tuple):
        raise CorpusError("Evidence resolution requires a tuple of references.")
    for ref in refs:
        _text(ref, "evidence reference")


class ReviewedCorpusAdapter:
    """Validate a private proof pack against all referenced original PDFs."""

    mode = "reviewed_corpus_fixture"
    seed_provenance = ("reviewed_proof_pack",)

    def __init__(self, proof_pack_path, source_catalog_path):
        self._catalog_directory = Path(source_catalog_path).resolve().parent
        pack, catalog = _load(proof_pack_path), _load(source_catalog_path)
        if not isinstance(pack, dict) or pack.get("format_version") != 1:
            raise CorpusError("A reviewed corpus proof pack version 1 is required.")
        documents = catalog.get("documents") if isinstance(catalog, dict) else None
        if not isinstance(documents, list):
            raise CorpusError("The source catalog must contain a documents list.")
        catalog_by_hash = {}
        for document in documents:
            if not isinstance(document, dict):
                raise CorpusError("A catalog document is invalid.")
            _digest(document.get("sha256"))
            if document["sha256"] in catalog_by_hash:
                raise CorpusError("Catalog source hashes must be unique.")
            catalog_by_hash[document["sha256"]] = document
        self._sources, self._paths, self._pages = {}, {}, {}
        self._evidence, self._locations, self._facts = {}, {}, {}
        sources = pack.get("sources")
        if not isinstance(sources, list) or not sources:
            raise CorpusError("The proof pack must declare its source documents.")
        evidence = pack.get("evidence")
        if not isinstance(evidence, list):
            raise CorpusError("The proof pack must contain an evidence list.")
        required_pages = {}
        for item in evidence:
            if not isinstance(item, dict):
                raise CorpusError("Reviewed evidence is invalid.")
            digest, page = item.get("source_sha256"), item.get("page")
            _digest(digest)
            if type(page) is not int or page < 1:
                raise CorpusError("Evidence refers to an unavailable source page.")
            required_pages.setdefault(digest, set()).add(page)
        for source in sources:
            self._source(source, catalog_by_hash, required_pages)
        for item in evidence:
            self._add_evidence(item)
        facts = pack.get("facts")
        if not isinstance(facts, list) or not facts:
            raise CorpusError("The proof pack must contain reviewed facts.")
        for raw in facts:
            fact = self._fact(raw)
            if fact.fact_id in self._facts:
                raise CorpusError("Reviewed fact identifiers must be globally unique.")
            self._facts[fact.fact_id] = fact
        self.proof_pack_sha256 = hashlib.sha256(_json(pack).encode()).hexdigest()

    def _source(self, source, catalog, required_pages):
        if not isinstance(source, dict):
            raise CorpusError("A reviewed source identity is invalid.")
        digest = source.get("sha256")
        _digest(digest)
        if digest in self._sources or digest not in catalog:
            raise CorpusError("A proof source is duplicated or absent from the source catalog.")
        registered = catalog[digest]
        for name in ("document_id", "document_name", "url", "company", "agency"):
            _text(source.get(name), name)
            if registered.get(name) is not None and registered[name] != source[name]:
                raise CorpusError("Proof source metadata conflicts with its catalog identity.")
        try:
            url = urlsplit(source["url"])
            if (url.scheme != "https" or not url.hostname or url.username is not None
                    or url.password is not None or any(character.isspace() for character in source["url"])):
                raise ValueError
            url.port
        except ValueError:
            raise CorpusError("Source citation URLs require HTTPS without credentials.") from None
        aliases = source.get("aliases", [])
        if not isinstance(aliases, list):
            raise CorpusError("Reviewed source aliases must be a list.")
        for alias in aliases:
            _text(alias, "source alias")
        _text(registered.get("local_path"), "source PDF path")
        path = Path(registered["local_path"])
        if not path.is_absolute():
            path = self._catalog_directory / path
        if path.name != source["document_name"] or path.suffix.lower() != ".pdf":
            raise CorpusError("Source filename differs from its reviewed PDF identity.")
        try:
            raw = path.read_bytes()
            if not 0 < len(raw) <= 25 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != digest:
                raise CorpusError("Source PDF hash differs from the reviewed corpus.")
            import pdfplumber
            with pdfplumber.open(BytesIO(raw)) as pdf:
                # Keep physical page indexes and authenticate the entire PDF, but
                # parse text only where the proof pack declares evidence.
                selected = required_pages.get(digest, set())
                pages = tuple({"text": (page.extract_text() or "") if number in selected else "",
                    "words": page.extract_words() if number in selected else (),
                    "width": page.width, "height": page.height}
                    for number, page in enumerate(pdf.pages, start=1))
            if not pages:
                raise CorpusError("A reviewed source PDF must have pages.")
        except CorpusError:
            raise
        except Exception as error:
            raise CorpusError("A reviewed source PDF could not be authenticated.") from error
        self._sources[digest] = json.loads(_json(source))
        self._paths[digest], self._pages[digest] = path, pages

    def reauthenticate(self):
        for digest, path in self._paths.items():
            try:
                if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise CorpusError("A corpus source PDF changed after review.")
            except OSError as error:
                raise CorpusError("A reviewed corpus source PDF is unavailable.") from error

    def _add_evidence(self, raw):
        if not isinstance(raw, dict):
            raise CorpusError("Reviewed evidence is invalid.")
        ref, digest, page = raw.get("ref"), raw.get("source_sha256"), raw.get("page")
        _text(ref, "evidence reference")
        if ref in self._evidence or digest not in self._sources:
            raise CorpusError("Evidence identity is duplicated or has an unknown source.")
        if type(page) is not int or not 1 <= page <= len(self._pages[digest]):
            raise CorpusError("Evidence refers to an unavailable source page.")
        excerpt = raw.get("excerpt")
        _text(excerpt, "evidence excerpt")
        source_page = self._pages[digest][page - 1]
        box = raw.get("bbox_pt")
        if box is not None:
            if (not isinstance(box, list) or len(box) != 4
                    or any(isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) for value in box)):
                raise CorpusError("Evidence PDF coordinates must be finite numeric bounds.")
            x0, top, x1, bottom = box
            if not 0 <= x0 < x1 <= source_page["width"] or not 0 <= top < bottom <= source_page["height"]:
                raise CorpusError("Evidence coordinates exceed the source page.")
            words = [word["text"] for word in source_page["words"] if x0 - .1 <= word["x0"]
                and word["x1"] <= x1 + .1 and top - .1 <= word["top"] and word["bottom"] <= bottom + .1]
            canonical = _normalized(" ".join(words))
            if canonical != _normalized(excerpt):
                raise CorpusError("Evidence differs from its exact source PDF box.")
            locator = "Source PDF box " + ",".join(str(value) for value in box)
        else:
            canonical = excerpt
            if excerpt not in source_page["text"]:
                raise CorpusError("Evidence is not an exact extracted source-page span.")
            locator = "Source passage beginning: " + " ".join(excerpt.split()[:12])
        source = self._sources[digest]
        self._evidence[ref] = Evidence(ref, source["document_name"], page, locator, canonical, source["url"])
        self._locations[ref] = {"source_sha256": digest, "page": page, "bbox_pt": box}

    def _bound_refs(self, refs, digest, available):
        for ref in refs:
            _text(ref, "evidence reference")
            if ref not in available or ref not in self._evidence or self._locations[ref]["source_sha256"] != digest:
                raise CorpusError("Fact proof must use its own cited canonical source references.")

    def _fact(self, raw):
        if not isinstance(raw, dict) or set(raw) != set(ReviewedFact.__dataclass_fields__):
            raise CorpusError("Reviewed fact fields differ from the corpus contract.")
        values = json.loads(_json(raw))
        refs = values["evidence_refs"]
        if (not isinstance(refs, list) or not refs
                or any(not isinstance(ref, str) or not ref.strip() for ref in refs)
                or len(set(refs)) != len(refs)):
            raise CorpusError("Reviewed facts require unique evidence references.")
        values["evidence_refs"] = tuple(refs)
        fact = ReviewedFact(**values)
        for name in ("fact_id", "company", "metric"):
            _text(getattr(fact, name), name)
        for name in ("period", "scope", "kind", "value_text", "text_value", "currency", "unit"):
            _text(getattr(fact, name), name, nullable=True)
        if re.fullmatch(r"[a-z][a-z0-9_]{0,63}", fact.metric) is None:
            raise CorpusError("Reviewed metrics require a canonical snake_case name.")
        _digest(fact.source_sha256)
        if fact.source_sha256 not in self._sources or fact.company != self._sources[fact.source_sha256]["company"]:
            raise CorpusError("Fact company differs from its reviewed source identity.")
        if (fact.value_text is None) == (fact.text_value is None):
            raise CorpusError("A fact requires exactly one numeric value or exact quoted text value.")
        if not isinstance(fact.raw_labels, dict) or not isinstance(fact.proof, dict) or not isinstance(fact.review, dict):
            raise CorpusError("Reviewed labels, proof and review provenance must be objects.")
        status = fact.review.get("status")
        provenance = fact.review.get("provenance")
        if not isinstance(status, str) or status not in {"reviewed", "source_verified", "context_ambiguous"}:
            raise CorpusError("Corpus facts require explicit source-review status.")
        if (not isinstance(provenance, str) or (status == "reviewed" and provenance != "reviewed_proof_pack")
                or (status != "reviewed" and provenance not in {"source_verified_proof_pack", "reviewed_proof_pack"})):
            raise CorpusError("An unapproved proof pack cannot automatically promote facts to reviewed evidence.")
        for name in ("reviewer", "revision"):
            _text(fact.review.get(name), "review " + name)
        capabilities = fact.review.get("capabilities", [])
        if (not isinstance(capabilities, list)
                or any(not isinstance(item, str) or item not in {"source_statement", "analytical_context"} for item in capabilities)
                or len(set(capabilities)) != len(capabilities)):
            raise CorpusError("Corpus review capabilities are invalid.")
        if status == "reviewed" and ("source_statement" not in capabilities or fact.review.get("promotion") is False):
            raise CorpusError("Reviewed facts require an explicit approved source-statement capability.")
        if fact.proof.get("period_role") == "report_context":
            if fact.value_text is not None or "analytical_context" in capabilities:
                raise CorpusError("Report-context periods are qualitative context, not analytical financial periods.")
        if "analytical_context" in capabilities:
            if (status != "reviewed" or fact.value_text is None
                    or any(getattr(fact, name) is None for name in ("period", "scope", "kind", "unit"))):
                raise CorpusError("Analytical review requires complete compatible numeric context.")
            if fact.unit in {"million", "billion", "per_share"} and fact.currency is None:
                raise CorpusError("Analytical monetary values require their source currency.")
            if fact.kind not in {"reported_actual", "broker_estimate", "broker_forecast", "broker_opinion"}:
                raise CorpusError("Analytical review requires a supported financial status.")
            units = {"million", "billion", "percent", "basis_points", "x", "multiple", "per_share",
                "shares", "million_shares", "billion_shares", "count", "ratio", "tonnes"}
            if fact.unit not in units and re.fullmatch(r"[A-Z]{3}/share", fact.unit) is None:
                raise CorpusError("Analytical review requires a supported financial unit category.")
            if fact.unit.endswith("/share") and fact.currency != fact.unit[:3]:
                raise CorpusError("Analytical per-share currency differs from its explicit unit.")
        self._bound_refs(fact.evidence_refs, fact.source_sha256, fact.evidence_refs)
        proof = fact.proof
        value_ref = proof.get("value_ref")
        self._bound_refs((value_ref,), fact.source_sha256, fact.evidence_refs)
        value_evidence = self._evidence[value_ref]
        method = proof.get("method")
        if fact.value_text is not None:
            try:
                value = Decimal(fact.value_text)
            except InvalidOperation as error:
                raise CorpusError("A numeric reviewed fact is invalid.") from error
            if not value.is_finite() or value != _number(fact.raw_labels.get("value")):
                raise CorpusError("Reviewed numeric value differs from its exact printed value.")
            if method == "table_cell":
                if self._locations[value_ref]["bbox_pt"] is None or value_evidence.excerpt != fact.raw_labels["value"]:
                    raise CorpusError("Table facts require one exact numeric source-cell box.")
                self._table_proof(fact)
            elif method == "quoted_number":
                span = proof.get("value_span")
                if (not isinstance(span, list) or len(span) != 2 or any(type(value) is not int for value in span)
                        or not 0 <= span[0] < span[1] <= len(value_evidence.excerpt)
                        or value_evidence.excerpt[span[0]:span[1]] != fact.raw_labels["value"]):
                    raise CorpusError("Quoted numbers require an exact printed numeric span.")
                before = value_evidence.excerpt[span[0] - 1] if span[0] else ""
                after = value_evidence.excerpt[span[1]] if span[1] < len(value_evidence.excerpt) else ""
                if ((before and (before.isdigit() or before in ".,+-"))
                        or (after and after.isdigit())
                        or (after == "," and span[1] + 1 < len(value_evidence.excerpt)
                            and value_evidence.excerpt[span[1] + 1].isdigit())
                        or (after == "." and span[1] + 1 < len(value_evidence.excerpt)
                            and value_evidence.excerpt[span[1] + 1].isdigit())):
                    raise CorpusError("A quoted numeric span cannot omit digits, separators or a printed sign.")
            else:
                raise CorpusError("Numeric facts require a table-cell or quoted-number proof.")
        elif method != "quoted_statement" or fact.text_value not in value_evidence.excerpt:
            raise CorpusError("Qualitative facts require an exact canonical source quotation.")
        self._context_proof(fact)
        return fact

    def _table_proof(self, fact):
        proof, digest, available = fact.proof, fact.source_sha256, fact.evidence_refs
        metric_ref = proof.get("metric_ref")
        columns = _refs(proof.get("column_refs"))
        self._bound_refs((metric_ref, *columns), digest, available)
        bindings = proof.get("bindings", {})
        if not isinstance(bindings, dict):
            raise CorpusError("Table facts require explicit context bindings.")
        if metric_ref not in _refs(bindings.get("metric")) or not set(columns) <= set(_refs(bindings.get("period"))):
            raise CorpusError("Table geometry must use the same reviewed metric and period bindings.")
        value, metric = self._locations[proof["value_ref"]], self._locations[metric_ref]
        if metric["bbox_pt"] is None or value["page"] != metric["page"]:
            raise CorpusError("Numeric cell and metric row must have same-page PDF boxes.")
        cell_box, row_box = value["bbox_pt"], metric["bbox_pt"]
        if min(cell_box[3], row_box[3]) <= max(cell_box[1], row_box[1]):
            raise CorpusError("Numeric cell is outside its reviewed metric row.")
        for ref in columns:
            location = self._locations[ref]
            if location["bbox_pt"] is None or location["page"] != value["page"] or location["bbox_pt"][3] >= cell_box[1]:
                raise CorpusError("Column headers must precede their numeric source cell on the same page.")
        column = self._locations[columns[-1]]["bbox_pt"]
        alignment = proof.get("column_alignment", "right")
        if alignment not in {"right", "center"}:
            raise CorpusError("Source columns require an explicit right or center alignment rule.")
        cell_anchor = cell_box[2] if alignment == "right" else (cell_box[0] + cell_box[2]) / 2
        column_anchor = column[2] if alignment == "right" else (column[0] + column[2]) / 2
        if abs(cell_anchor - column_anchor) > 6:
            raise CorpusError("Numeric cell is outside its reviewed column.")
        selected_header = re.sub(r"\s", "", self._evidence[columns[-1]].excerpt).upper()
        if fact.period is not None:
            fiscal = re.fullmatch(r"([1-4]Q)?(FY(?:\d{2}|\d{4}))", fact.period)
            explicit = re.fullmatch(r"([1-4]Q)?(FY(?:\d{2}|\d{4}))E?", selected_header)
            quarter = re.fullmatch(r"([1-4]Q)E?", selected_header)
            if fiscal is None:
                raise CorpusError("A table fact needs a canonical fiscal period.")
            if explicit is not None:
                if (fiscal[1], fiscal[2]) != (explicit[1], explicit[2]):
                    raise CorpusError("The selected column differs from the reviewed quarter or annual period.")
            elif quarter is not None and fiscal[1] == quarter[1]:
                self._column_group(fact, columns[-1])
            else:
                raise CorpusError("The selected column cannot establish the reviewed quarter or annual period.")
        if re.fullmatch(r"[1-4]QE", selected_header) and fact.kind not in {None, "broker_estimate"}:
            if fact.kind != "broker_forecast":
                raise CorpusError("An explicit estimate column cannot become a reported actual.")
            self._future_forecast(fact)

    def _future_forecast(self, fact):
        """Distinguish forward estimates using a cited report period, not E alone."""
        context = fact.proof.get("forecast_context")
        if not isinstance(context, dict) or set(context) != {"report_period", "report_period_refs"}:
            raise CorpusError("A future-quarter forecast requires source-bound report-period proof.")
        report_period = context["report_period"]
        current = re.fullmatch(r"([1-4])QFY(\d{2}|\d{4})", report_period) if isinstance(report_period, str) else None
        future = re.fullmatch(r"([1-4])QFY(\d{2}|\d{4})", fact.period or "")
        if (current is None or future is None or len(current[2]) != len(future[2])
                or report_period != self._sources[fact.source_sha256].get("report_period")
                or (int(future[2]), int(future[1])) <= (int(current[2]), int(current[1]))):
            raise CorpusError("Forecast quarter must be strictly later than its explicit source report period.")
        refs = _refs(context["report_period_refs"])
        self._bound_refs(refs, fact.source_sha256, fact.evidence_refs)
        quote = " ".join(self._evidence[ref].excerpt for ref in refs)
        pattern = r"(?<!\w)" + current[1] + r"Q\s*FY\s*" + current[2] + r"(?!\d)"
        if re.search(pattern, quote, re.I) is None:
            raise CorpusError("Forecast chronology lacks its exact cited source report period.")

    def _column_group(self, fact, selected_ref):
        group = fact.proof.get("column_group")
        if not isinstance(group, dict):
            raise CorpusError("Grouped fiscal columns require an explicit reviewed year-group band.")
        header_ref = group.get("header_ref")
        members = _refs(group.get("column_refs"))
        self._bound_refs((header_ref, *members), fact.source_sha256, fact.evidence_refs)
        if selected_ref not in members or header_ref not in _refs(fact.proof["bindings"].get("period")):
            raise CorpusError("Selected quarter and fiscal header do not belong to the reviewed year group.")
        band = group.get("x_range")
        page = self._locations[selected_ref]["page"]
        source_page = self._pages[fact.source_sha256][page - 1]
        if (not isinstance(band, list) or len(band) != 2
                or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in band)
                or not 0 <= band[0] < band[1] <= source_page["width"]):
            raise CorpusError("The reviewed fiscal-year group requires finite physical page bounds.")
        header = self._locations[header_ref]
        if header["bbox_pt"] is None or header["page"] != page:
            raise CorpusError("A fiscal-year group needs its exact same-page header box.")
        header_box = header["bbox_pt"]
        if not band[0] <= (header_box[0] + header_box[2]) / 2 <= band[1]:
            raise CorpusError("The fiscal-year header is outside its reviewed column group.")
        for ref in members:
            location = self._locations[ref]
            box = location["bbox_pt"]
            if (box is None or location["page"] != page or box[0] < band[0] - .1
                    or box[2] > band[1] + .1 or box[1] <= header_box[3]):
                raise CorpusError("A quarter column is outside its reviewed fiscal-year group.")
        cell_box = self._locations[fact.proof["value_ref"]]["bbox_pt"]
        # Amounts often have more glyphs than their short quarter header. The
        # aligned column anchor establishes ownership, not the leftmost digit.
        alignment = fact.proof.get("column_alignment", "right")
        anchor = cell_box[2] if alignment == "right" else (cell_box[0] + cell_box[2]) / 2
        if not band[0] - 6 <= anchor <= band[1] + 6:
            raise CorpusError("The numeric cell is outside its reviewed fiscal-year group.")
        expected_year = _normalized(self._evidence[header_ref].excerpt).upper()
        if re.fullmatch(r"FY(?:\d{2}|\d{4})E?", expected_year) is None:
            raise CorpusError("A grouped fiscal-year header must identify one exact fiscal year.")
        fiscal = re.fullmatch(r"([1-4]Q)?(FY(?:\d{2}|\d{4}))", fact.period)
        if fiscal is None or expected_year.removesuffix("E") != fiscal[2]:
            raise CorpusError("The physical fiscal-year group differs from the reviewed period.")
        for word in source_page["words"]:
            label = word["text"].upper()
            if (re.fullmatch(r"FY(?:\d{2}|\d{4})E?", label)
                    and abs(word["top"] - header_box[1]) <= 1
                    and band[0] <= (word["x0"] + word["x1"]) / 2 <= band[1]
                    and label != expected_year):
                raise CorpusError("The reviewed fiscal-year band contains a different year header.")

    def _context_proof(self, fact):
        bindings = fact.proof.get("bindings")
        if not isinstance(bindings, dict):
            raise CorpusError("Facts require explicit reviewed context bindings.")
        quoted = {}
        for name, refs in bindings.items():
            refs = _refs(refs)
            self._bound_refs(refs, fact.source_sha256, fact.evidence_refs)
            quoted[name] = " ".join(self._evidence[ref].excerpt for ref in refs)
        if "company" not in quoted or "metric" not in quoted:
            raise CorpusError("A fact needs source company and metric/topic bindings.")
        aliases = self._sources[fact.source_sha256].get("aliases", [])
        if not any(label in quoted["company"] for label in (fact.company, *aliases)):
            raise CorpusError("The source company binding does not name its reviewed company.")
        for name in ("metric", "period", "scope", "unit", "kind"):
            normalized = getattr(fact, name)
            if normalized is not None:
                if name not in quoted:
                    raise CorpusError(f"Reviewed {name} has no source binding.")
                raw_label = fact.raw_labels.get(name)
                if isinstance(raw_label, str):
                    if not raw_label or raw_label not in quoted[name]:
                        raise CorpusError(f"Reviewed raw {name} label differs from its cited source.")
                elif isinstance(raw_label, list) and raw_label:
                    if any(not isinstance(label, str) or not label or label not in quoted[name] for label in raw_label):
                        raise CorpusError(f"Reviewed raw {name} labels differ from their cited source.")
                else:
                    raise CorpusError(f"Reviewed {name} requires its original raw label.")
        if fact.period is not None:
            compact = re.sub(r"\s", "", quoted["period"]).upper()
            match = re.fullmatch(r"([1-4]Q)?(FY(?:\d{2}|\d{4}))", fact.period)
            if match is None or (fact.period not in compact and
                    not (match[2] in compact and match[1] is not None and match[1] in compact)):
                raise CorpusError("Reviewed fiscal period differs from its original source labels.")
        raw_unit = fact.raw_labels.get("unit")
        if isinstance(raw_unit, str):
            label = re.sub(r"[\s().]", "", raw_unit).upper()
            units = {"INRM": ("INR", "million"), "INRB": ("INR", "billion"),
                "USDM": ("USD", "million"), "USDB": ("USD", "billion"),
                "%": (None, "percent"), "BP": (None, "basis_points"), "BPS": (None, "basis_points")}
            if label in units and (fact.currency, fact.unit) != units[label]:
                raise CorpusError("Reviewed currency or unit differs from its source label.")
        raw_scope = fact.raw_labels.get("scope")
        if fact.scope is not None and isinstance(raw_scope, str):
            label = re.sub(r"[\s().]", "", raw_scope).upper()
            if ((label in {"CONSOL", "CONSOLIDATED", "CONSOLIFRS"} and fact.scope != "consolidated")
                    or (label == "STANDALONE" and fact.scope != "standalone")):
                raise CorpusError("Reviewed scope differs from its explicit source label.")
        raw_kind = fact.raw_labels.get("kind")
        if fact.kind is not None and isinstance(raw_kind, str):
            label = raw_kind.casefold().strip()
            direct_kinds = {"reported_actual": "reported_actual", "actual": "reported_actual",
                "broker_estimate": "broker_estimate", "estimate": "broker_estimate",
                "broker_forecast": "broker_forecast", "forecast": "broker_forecast"}
            if label in direct_kinds and fact.kind != direct_kinds[label]:
                raise CorpusError("Reviewed financial status differs from its explicit source label.")
            if re.fullmatch(r"[1-4]qe", label) and fact.kind != "broker_estimate":
                if fact.kind != "broker_forecast":
                    raise CorpusError("An explicit estimate column cannot become a reported actual.")
                self._future_forecast(fact)
        statuses = fact.proof.get("reported_status_refs", [])
        if not isinstance(statuses, list):
            raise CorpusError("Reported status references must be a list.")
        if statuses:
            self._bound_refs(statuses, fact.source_sha256, fact.evidence_refs)
            status_text = " ".join(self._evidence[ref].excerpt for ref in statuses)
            reported = re.search(r"\b(?:reported|delivered|results|actual)\b", status_text, re.I)
            if not reported or fact.period is None or fact.period not in re.sub(r"\s", "", status_text):
                raise CorpusError("Reported status corroboration lacks the explicit reported period.")
            if fact.kind == "broker_forecast":
                raise CorpusError("Explicit reported-quarter proof cannot be relabeled as forecast.")
        forecast_heading = re.search(r"FY(?:\d{4}|\d{2})E|[1-4]QE", quoted.get("period", ""), re.I)
        if fact.kind == "reported_actual" and forecast_heading and not statuses:
            raise CorpusError("Forecast-headed actuals require explicit reported-period corroboration.")

    def read_facts(self, filters=FactFilter(), *, include_unreviewed=False):
        _validate_filter(filters)
        if type(include_unreviewed) is not bool:
            raise CorpusError("Unreviewed exploration must be explicitly selected.")
        self.reauthenticate()
        return tuple(_fact_copy(fact) for fact in self._facts.values()
            if _matches(fact, filters) and (include_unreviewed or fact.review["status"] == "reviewed"))

    def explore_facts(self, filters=FactFilter()):
        return self.read_facts(filters, include_unreviewed=True)

    def _check_fact(self, fact):
        """Compare canonical proof after the public operation authenticates sources."""
        if not isinstance(fact, ReviewedFact):
            raise CorpusError("A fact differs from its authenticated reviewed proof pack.")
        _text(fact.fact_id, "fact identifier")
        if self._facts.get(fact.fact_id) != fact:
            raise CorpusError("A fact differs from its authenticated reviewed proof pack.")

    def validate_facts(self, facts: tuple[ReviewedFact, ...]):
        _fact_batch(facts)
        self.reauthenticate()
        for fact in facts:
            self._check_fact(fact)

    def validate_fact(self, fact):
        self.validate_facts((fact,))

    def _canonical_ref(self, ref):
        if ref not in self._evidence:
            raise CorpusError("A corpus evidence reference cannot be resolved.")
        return self._evidence[ref]

    def resolve_many(self, refs: tuple[str, ...]) -> tuple[Evidence, ...]:
        _ref_batch(refs)
        self.reauthenticate()
        return tuple(self._canonical_ref(ref) for ref in refs)

    def resolve(self, ref):
        return self.resolve_many((ref,))[0]

    def companies(self, sector=None):
        if sector is not None:
            _text(sector, "sector")
        self.reauthenticate()
        return tuple(sorted({source["company"] for source in self._sources.values()
            if sector is None or source.get("sector") == sector}))

    @property
    def sources(self):
        return tuple(json.loads(_json(source)) for source in self._sources.values())


_DDL = (
    "CREATE TABLE documents (source_sha256 TEXT PRIMARY KEY, metadata_json TEXT NOT NULL)",
    "CREATE TABLE evidence (ref TEXT PRIMARY KEY, source_sha256 TEXT NOT NULL REFERENCES documents(source_sha256), evidence_json TEXT NOT NULL)",
    "CREATE TABLE facts (fact_id TEXT PRIMARY KEY, source_sha256 TEXT NOT NULL REFERENCES documents(source_sha256), company TEXT NOT NULL, period TEXT, scope TEXT, metric TEXT NOT NULL, kind TEXT, currency TEXT, unit TEXT, fact_json TEXT NOT NULL)",
)


def _connect(path, mode):
    try:
        connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=" + mode, uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection
    except sqlite3.Error as error:
        raise CorpusError("The reviewed corpus SQLite file could not be opened.") from error


def _schema(connection):
    if connection.execute("PRAGMA user_version").fetchone()[0] != 1:
        raise CorpusError("Unsupported reviewed corpus SQLite version.")
    actual = {row[0]: " ".join(row[1].split()).casefold()
        for row in connection.execute("SELECT name,sql FROM sqlite_master WHERE type='table'")}
    expected = {re.search(r"CREATE TABLE (\w+)", ddl)[1]: " ".join(ddl.split()).casefold() for ddl in _DDL}
    if actual != expected or connection.execute("SELECT 1 FROM sqlite_master WHERE type NOT IN ('table','index') OR (type='index' AND sql IS NOT NULL)").fetchone():
        raise CorpusError("Reviewed corpus schema differs from its declared fingerprint.")


class SQLiteCorpusAdapter:
    """Read existing reviewed rows, bound to their authenticated proof validator."""

    mode = "sqlite_corpus"
    seed_provenance = ("reviewed_proof_pack",)

    def __init__(self, database_path, validator):
        if not isinstance(validator, ReviewedCorpusAdapter):
            raise CorpusError("SQLite corpus reads require a reviewed source validator.")
        self.database_path, self.validator = Path(database_path), validator
        self._read(lambda connection: None)

    def _read(self, operation):
        self.validator.reauthenticate()
        connection = _connect(self.database_path, "ro")
        try:
            # Every metadata, fact, and evidence comparison belongs to the same
            # read snapshot, including batched validation and citation resolution.
            connection.execute("BEGIN")
            _schema(connection)
            return operation(connection)
        except sqlite3.Error as error:
            raise CorpusError("Reviewed corpus SQLite rows could not be read.") from error
        finally:
            connection.close()

    def read_facts(self, filters=FactFilter(), *, include_unreviewed=False):
        _validate_filter(filters)
        if type(include_unreviewed) is not bool:
            raise CorpusError("Unreviewed exploration must be explicitly selected.")
        clauses, parameters = [], []
        for name, values in (("company", filters.companies), ("metric", filters.metrics)):
            if values:
                clauses.append(name + " IN (" + ",".join("?" for _ in values) + ")")
                parameters.extend(values)
        for name in ("period", "scope", "kind", "currency", "unit", "source_sha256"):
            if getattr(filters, name) is not None:
                clauses.append(name + "=?")
                parameters.append(getattr(filters, name))
        query = "SELECT * FROM facts" + (" WHERE " + " AND ".join(clauses) if clauses else "") + " ORDER BY fact_id"
        def operation(connection):
            facts = []
            for row in connection.execute(query, parameters):
                fact = self._stored_fact(connection, row)
                if fact.review["status"] != "reviewed" and not include_unreviewed:
                    continue
                facts.append(fact)
            return tuple(facts)
        return self._read(operation)

    def explore_facts(self, filters=FactFilter()):
        return self.read_facts(filters, include_unreviewed=True)

    def _stored_fact(self, connection, row):
        if row is None:
            raise CorpusError("The reviewed fact is missing from the bound SQLite corpus.")
        try:
            data = json.loads(row["fact_json"], object_pairs_hook=_pairs)
            data["evidence_refs"] = tuple(data["evidence_refs"])
            fact = ReviewedFact(**data)
        except (ValueError, TypeError, KeyError) as error:
            raise CorpusError("Stored reviewed fact JSON is invalid.") from error
        self.validator._check_fact(fact)
        if any(row[name] != getattr(fact, name) for name in ("fact_id", "source_sha256", "company", "period", "scope", "metric", "kind", "currency", "unit")):
            raise CorpusError("Stored fact context columns differ from their reviewed identity.")
        for ref in fact.evidence_refs:
            self._resolve(connection, ref)
        return fact

    def _resolve(self, connection, ref):
        row = connection.execute("SELECT * FROM evidence WHERE ref=?", (ref,)).fetchone()
        canonical = self.validator._canonical_ref(ref)
        if (row is None or row["source_sha256"] != self.validator._locations[ref]["source_sha256"]
                or row["evidence_json"] != _json(asdict(canonical))):
            raise CorpusError("Stored evidence differs from its canonical reviewed source.")
        document = connection.execute("SELECT metadata_json FROM documents WHERE source_sha256=?", (row["source_sha256"],)).fetchone()
        if document is None or document[0] != _json(self.validator._sources[row["source_sha256"]]):
            raise CorpusError("Stored source metadata differs from its reviewed identity.")
        return canonical

    def resolve_many(self, refs: tuple[str, ...]) -> tuple[Evidence, ...]:
        _ref_batch(refs)
        return self._read(lambda connection: tuple(self._resolve(connection, ref) for ref in refs))

    def resolve(self, ref):
        return self.resolve_many((ref,))[0]

    def validate_facts(self, facts: tuple[ReviewedFact, ...]):
        _fact_batch(facts)
        def operation(connection):
            for fact in facts:
                self.validator._check_fact(fact)
                row = connection.execute("SELECT * FROM facts WHERE fact_id=?", (fact.fact_id,)).fetchone()
                if self._stored_fact(connection, row) != fact:
                    raise CorpusError("The reviewed fact differs from its bound SQLite record.")
        self._read(operation)

    def validate_fact(self, fact):
        self.validate_facts((fact,))

    def companies(self, sector=None):
        if sector is not None:
            _text(sector, "sector")
        def operation(connection):
            present = set()
            for row in connection.execute("SELECT source_sha256,metadata_json FROM documents"):
                canonical = self.validator._sources.get(row["source_sha256"])
                if canonical is None or row["metadata_json"] != _json(canonical):
                    raise CorpusError("Stored corpus membership differs from its reviewed source catalog.")
                present.add(canonical["company"])
            return tuple(sorted({source["company"] for source in self.validator._sources.values()
                if source["company"] in present and (sector is None or source.get("sector") == sector)}))
        return self._read(operation)

    @property
    def sources(self):
        return self.validator.sources


def import_reviewed(proof_pack_path, source_catalog_path, database_path):
    """Validate the whole proof pack before creating or writing a corpus DB."""
    validator = ReviewedCorpusAdapter(proof_pack_path, source_catalog_path)
    validator.reauthenticate()
    approved = validator.read_facts()
    if not approved:
        raise CorpusError("A source-verified pack has no explicitly approved reviewed facts to persist.")
    connection = _connect(database_path, "rwc")
    counts = [0, 0, 0]
    def insert(table, row, key):
        previous = connection.execute(f"SELECT * FROM {table} WHERE {key}=?", (row[key],)).fetchone()
        if previous is not None:
            if dict(previous) != row:
                raise CorpusError(f"Conflicting reviewed {table} record; existing data was preserved.")
            return 0
        connection.execute(f"INSERT INTO {table} (" + ",".join(row) + ") VALUES (" + ",".join("?" for _ in row) + ")", tuple(row.values()))
        return 1
    try:
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("PRAGMA user_version").fetchone()[0] == 0 and not connection.execute("SELECT 1 FROM sqlite_master").fetchone():
                for ddl in _DDL:
                    connection.execute(ddl)
                connection.execute("PRAGMA user_version=1")
            _schema(connection)
            for digest, source in validator._sources.items():
                counts[0] += insert("documents", {"source_sha256": digest, "metadata_json": _json(source)}, "source_sha256")
            for ref, evidence in validator._evidence.items():
                counts[1] += insert("evidence", {"ref": ref, "source_sha256": validator._locations[ref]["source_sha256"], "evidence_json": _json(asdict(evidence))}, "ref")
            for fact in approved:
                row = {name: getattr(fact, name) for name in ("fact_id", "source_sha256", "company", "period", "scope", "metric", "kind", "currency", "unit")}
                row["fact_json"] = _json(asdict(fact))
                counts[2] += insert("facts", row, "fact_id")
    except sqlite3.Error as error:
        raise CorpusError("Reviewed corpus import failed; all rows were rolled back.") from error
    finally:
        connection.close()
    return CorpusIngestionResult(*counts)
