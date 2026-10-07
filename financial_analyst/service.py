"""Assemble supported claims after evidence validation and pure arithmetic."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from .adapters import AnalyticsReadPort, Citation, EvidenceError, EvidenceReadPort, PassageReadPort
from .analytics import ComparisonError, RevenueComparison, RevenueObservation, compare_revenue
from .selection import PassageSelectionPort, SelectionError, validate_selection


@dataclass(frozen=True)
class Claim:
    kind: str
    values: dict[str, Any]
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class Answer:
    status: str
    claims: tuple[Claim, ...] = ()
    comparison: RevenueComparison | None = None
    citations: tuple[Citation, ...] = ()
    execution: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None
    source_conflicts: tuple[dict, ...] = ()


class ApplicationService:
    def __init__(self, analytics: AnalyticsReadPort, evidence: EvidenceReadPort,
                 passages: PassageReadPort | None = None, selector: PassageSelectionPort | None = None):
        self.analytics = analytics
        self.evidence = evidence
        self.passages = passages
        self.selector = selector
        self._begin_request()

    def _begin_request(self) -> None:
        self._retrieval_used = False
        self._retrieval_mode = "not_requested"
        self._database_used = False
        self._llm_execution = {"provider": "none", "mode": "not_called"}

    def _execution(self) -> dict[str, Any]:
        llm = dict(self._llm_execution)
        analytics = getattr(self.analytics, "mode", "fixture")
        return {"mode": "live" if analytics == "sqlite" else "fixture", "analytics": analytics,
                "seed_provenance": tuple(getattr(self.analytics, "seed_provenance", ())),
                "retrieval": self._retrieval_mode if self._retrieval_used else "not_requested",
                "llm": llm, "database": "sqlite" if self._database_used else "none",
                "selection_executed": llm["mode"] in {"live", "test_double"},
                "no_llm": llm["mode"] != "live", "no_database": not self._database_used}

    def _conflicts(self) -> tuple[dict, ...]:
        return tuple(getattr(self.evidence, "source_conflicts", ()))

    def _citations(self, refs: tuple[str, ...]) -> tuple[Citation, ...]:
        refs += tuple(ref for conflict in self._conflicts() for ref in conflict.get("evidence_refs", ()))
        return tuple(self.evidence.resolve(ref).citation for ref in dict.fromkeys(refs))

    def refuse(self, reason: str) -> Answer:
        return Answer("refused", citations=self._citations(()), execution=self._execution(), reason=reason,
                      source_conflicts=self._conflicts())

    def refuse_calendar_dates(self) -> Answer:
        self._begin_request()
        return self.refuse("The source has conflicting fiscal year-end labels. Calendar dates cannot be resolved from this fixture; the original fiscal labels are preserved.")

    def _one(self, company: str, period: str, kind: str) -> RevenueObservation:
        if getattr(self.analytics, "mode", None) == "sqlite":
            bound_source = getattr(self.analytics, "source_sha256", None)
            if not bound_source or bound_source != getattr(self.evidence, "source_sha256", None):
                raise EvidenceError("SQLite source binding does not match the original PDF evidence validator.")
        self._database_used = getattr(self.analytics, "mode", None) == "sqlite"
        observations = self.analytics.read_observations(company, period)
        if any(item.company != company or item.period != period for item in observations):
            raise EvidenceError("Read adapter returned observations for a different company or quarter.")
        matches = [item for item in observations if item.kind == kind]
        if len(matches) != 1:
            raise EvidenceError(f"The read adapter requires exactly one {kind} observation for the requested company and quarter.")
        self.evidence.validate_observation(matches[0])
        return matches[0]

    def _growth_claims(self, company: str, period: str, select: bool = True) -> tuple[Claim, ...]:
        if self.passages is None:
            raise EvidenceError("Curated growth passage retrieval is unavailable.")
        passage_source = getattr(self.passages, "source_sha256", None)
        if passage_source is not None and passage_source != getattr(self.evidence, "source_sha256", None):
            raise EvidenceError("Passage source binding does not match the original PDF evidence validator.")
        self._retrieval_used = True
        self._retrieval_mode = getattr(self.passages, "mode", "fixture")
        if self._retrieval_mode == "fixture":
            self._retrieval_mode = "curated_fixture"
        growth_reader = self.passages.growth_passages
        if not select:
            # Attribution refusal needs canonical source citations, never a
            # relevance model. Semantic retrieval itself invokes that model.
            growth_reader = getattr(self.evidence, "growth_passages", None)
            if not callable(growth_reader):
                raise EvidenceError("No verified allocation of the estimate variance has been supplied.")
            self._retrieval_mode = "curated_fixture"
        try:
            passages = growth_reader(company, period)
        finally:
            retrieval_llm = getattr(self.passages, "execution", None)
            if select and isinstance(retrieval_llm, dict):
                self._llm_execution = dict(retrieval_llm)
        if not passages:
            raise EvidenceError("No supported growth passage was returned for this request. This retrieval outcome does not establish that the source lacks an explanation.")
        verified = []
        for passage in passages:
            # Resolve again through the evidence port. Retrieval cannot supply a
            # different excerpt under an otherwise valid citation identifier.
            supported = self.evidence.resolve(passage.ref)
            if supported != passage:
                raise EvidenceError("Retrieved growth passage conflicts with its source evidence.")
            verified.append(supported)
        if self.selector is not None and select:
            try:
                refs = validate_selection(self.selector.select(company, period, tuple(verified)), verified)
            finally:
                self._llm_execution = dict(self.selector.execution)
            by_ref = {passage.ref: passage for passage in verified}
            verified = [by_ref[ref] for ref in refs]
        return tuple(Claim("growth", {"attribution": "broker", "excerpt": passage.excerpt}, (passage.ref,))
                     for passage in verified)

    def comparison_answer(self, company: str, period: str, include_growth: bool = False,
                          include_yoy: bool = False) -> Answer:
        self._begin_request()
        if not re.fullmatch(r"[1-4]QFY(?:\d{2}|\d{4})", period):
            return self.refuse("This fixture requires an explicit fiscal-quarter label; calendar-period interpretation is unsupported.")
        try:
            actual = self._one(company, period, "reported_actual")
            estimate = self._one(company, period, "broker_estimate")
            prior = None
            if include_yoy:
                year = period.split("FY")[1]
                prior_period = period.split("FY")[0] + "FY" + str(int(year) - 1).zfill(len(year))
                prior = self._one(company, prior_period, "reported_actual")
            result = compare_revenue(actual, estimate, prior)
            direction = "beat" if result.delta_millions > 0 else "missed" if result.delta_millions < 0 else "matched"
            claims = [Claim("comparison", {
                "company": company, "period": period, "currency": actual.currency,
                "unit": "million", "actual": result.actual_millions,
                "estimate": result.estimate_millions, "delta": result.delta_millions,
                "beat_percent": result.beat_percent, "direction": direction,
            }, result.beat_evidence_refs)]
            if prior is not None:
                sign, digits, exponent = prior.value.as_tuple()
                prior_millions = Decimal((sign, digits, exponent + (3 if prior.source_unit == "billion" else 0)))
                claims.append(Claim("yoy", {
                    "company": company, "period": period, "prior_period": prior.period,
                    "currency": actual.currency, "unit": "million",
                    "prior_actual": prior_millions,
                    "yoy_percent": result.yoy_percent,
                }, result.yoy_evidence_refs))
            if include_growth:
                claims.extend(self._growth_claims(company, period))
            refs = tuple(ref for claim in claims for ref in claim.evidence_refs)
            return Answer("answered", tuple(claims), result, self._citations(refs),
                          self._execution(), source_conflicts=self._conflicts())
        except (EvidenceError, ComparisonError, SelectionError) as exc:
            return self.refuse(str(exc))

    def growth_answer(self, company: str, period: str) -> Answer:
        self._begin_request()
        try:
            self._one(company, period, "reported_actual")
            claims = self._growth_claims(company, period)
            refs = tuple(ref for claim in claims for ref in claim.evidence_refs)
            return Answer("answered", claims, citations=self._citations(refs),
                          execution=self._execution(), source_conflicts=self._conflicts())
        except (EvidenceError, ComparisonError, SelectionError) as exc:
            return self.refuse(str(exc))

    def refuse_beat_attribution(self, company: str, period: str, driver: str | None = None) -> Answer:
        self._begin_request()
        try:
            claims = self._growth_claims(company, period, select=False)
            refs = tuple(ref for claim in claims for ref in claim.evidence_refs)
            reason = "This fixture has no supported allocation of the estimate variance to individual drivers. The cited passages describe revenue growth and cannot allocate the comparison amount."
            return Answer("refused", citations=self._citations(refs), execution=self._execution(),
                          reason=reason, source_conflicts=self._conflicts())
        except EvidenceError as exc:
            return self.refuse(str(exc))
