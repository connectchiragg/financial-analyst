"""Assemble supported claims after evidence validation and pure arithmetic."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from .adapters import AnalyticsReadPort, Citation, EvidenceError, EvidenceReadPort, PassageReadPort
from .analytics import ComparisonError, RevenueComparison, RevenueObservation, compare_revenue


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
                 passages: PassageReadPort | None = None):
        self.analytics = analytics
        self.evidence = evidence
        self.passages = passages

    def _execution(self, growth: bool = False) -> dict[str, Any]:
        return {"mode": "fixture", "analytics": "fixture",
                "retrieval": "curated_fixture" if growth else "not_requested",
                "no_llm": True, "no_database": True}

    def _conflicts(self) -> tuple[dict, ...]:
        return tuple(getattr(self.evidence, "source_conflicts", ()))

    def _citations(self, refs: tuple[str, ...]) -> tuple[Citation, ...]:
        refs += tuple(ref for conflict in self._conflicts() for ref in conflict.get("evidence_refs", ()))
        return tuple(self.evidence.resolve(ref).citation for ref in dict.fromkeys(refs))

    def refuse(self, reason: str) -> Answer:
        return Answer("refused", citations=self._citations(()), execution=self._execution(), reason=reason,
                      source_conflicts=self._conflicts())

    def refuse_calendar_dates(self) -> Answer:
        return self.refuse("The source has conflicting fiscal year-end labels. Calendar dates cannot be resolved from this fixture; the original fiscal labels are preserved.")

    def _one(self, company: str, period: str, kind: str) -> RevenueObservation:
        observations = self.analytics.read_observations(company, period)
        if any(item.company != company or item.period != period for item in observations):
            raise EvidenceError("Read adapter returned observations for a different company or quarter.")
        matches = [item for item in observations if item.kind == kind]
        if len(matches) != 1:
            raise EvidenceError(f"Fixture requires exactly one {kind} observation for the requested company and quarter.")
        self.evidence.validate_observation(matches[0])
        return matches[0]

    def _growth_claims(self, company: str, period: str) -> tuple[Claim, ...]:
        if self.passages is None:
            raise EvidenceError("Curated growth passage retrieval is unavailable.")
        passages = self.passages.growth_passages(company, period)
        if not passages:
            raise EvidenceError("Fixture has no cited revenue-growth explanations for this quarter.")
        claims = []
        for passage in passages:
            # Resolve again through the evidence port. Retrieval cannot supply a
            # different excerpt under an otherwise valid citation identifier.
            supported = self.evidence.resolve(passage.ref)
            if supported != passage:
                raise EvidenceError("Retrieved growth passage conflicts with its source evidence.")
            claims.append(Claim("growth", {"attribution": "broker", "excerpt": supported.excerpt}, (supported.ref,)))
        return tuple(claims)

    def comparison_answer(self, company: str, period: str, include_growth: bool = False,
                          include_yoy: bool = False) -> Answer:
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
                          self._execution(include_growth), source_conflicts=self._conflicts())
        except (EvidenceError, ComparisonError) as exc:
            return self.refuse(str(exc))

    def growth_answer(self, company: str, period: str) -> Answer:
        try:
            self._one(company, period, "reported_actual")
            claims = self._growth_claims(company, period)
            refs = tuple(ref for claim in claims for ref in claim.evidence_refs)
            return Answer("answered", claims, citations=self._citations(refs),
                          execution=self._execution(True), source_conflicts=self._conflicts())
        except (EvidenceError, ComparisonError) as exc:
            return self.refuse(str(exc))

    def refuse_beat_attribution(self, company: str, period: str, driver: str | None = None) -> Answer:
        try:
            claims = self._growth_claims(company, period)
            refs = tuple(ref for claim in claims for ref in claim.evidence_refs)
            reason = "This fixture has no supported allocation of the estimate variance to individual drivers. The cited passages describe revenue growth and cannot allocate the comparison amount."
            return Answer("refused", citations=self._citations(refs), execution=self._execution(True),
                          reason=reason, source_conflicts=self._conflicts())
        except EvidenceError as exc:
            return self.refuse(str(exc))
