"""Exact calculations over already source-reviewed financial records.

This module performs no source lookup or inference. Callers must authenticate
facts and resolve the returned evidence references before rendering a claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re
from typing import Protocol

from .analytics import _percent, _shift_decimal, _subtract_exact


class CalculationError(ValueError):
    """The requested records do not support this calculation."""


class NumericFact(Protocol):
    fact_id: str
    company: str
    period: str | None
    scope: str | None
    metric: str
    kind: str
    currency: str | None
    unit: str | None
    value_text: str | None
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class AmountComparison:
    company: str
    period: str
    scope: str
    metric: str
    currency: str
    actual_millions: Decimal
    estimate_millions: Decimal
    delta_millions: Decimal
    variance_percent: Decimal
    fact_ids: tuple[str, ...]
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class RevenueGrowth:
    company: str
    period: str
    prior_period: str
    scope: str
    currency: str
    actual_millions: Decimal
    prior_millions: Decimal
    yoy_percent: Decimal
    actual_source_unit: str
    prior_source_unit: str
    fact_ids: tuple[str, ...]
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class RankedGrowth:
    rank: int
    growth: RevenueGrowth


def _number(fact: NumericFact) -> Decimal:
    for field in ("fact_id", "company", "metric", "kind"):
        value = getattr(fact, field, None)
        if not isinstance(value, str) or not value.strip():
            raise CalculationError(f"A reviewed numeric fact requires {field}.")
    refs = getattr(fact, "evidence_refs", None)
    if (not isinstance(refs, tuple) or not refs
            or any(not isinstance(ref, str) or not ref.strip() for ref in refs)
            or len(set(refs)) != len(refs)):
        raise CalculationError("Each input requires unique source evidence references.")
    value = getattr(fact, "value_text", None)
    if not isinstance(value, str) or not value or len(value) > 128:
        raise CalculationError("Each input requires exact decimal text.")
    try:
        number = Decimal(value)
    except InvalidOperation:
        raise CalculationError("A financial input has invalid decimal text.") from None
    if not number.is_finite():
        raise CalculationError("Financial inputs must be finite.")
    return number


def amount_millions(fact: NumericFact) -> Decimal:
    """Normalize monetary scale exactly; never convert currencies or ratio units."""
    number = _number(fact)
    if (not isinstance(fact.currency, str)
            or re.fullmatch(r"[A-Z]{3}", fact.currency) is None
            or fact.unit not in {"million", "billion"}):
        raise CalculationError("Monetary calculations require an explicit currency and million/billion unit.")
    return _shift_decimal(number, 3) if fact.unit == "billion" else number


def _same_context(left: NumericFact, right: NumericFact, *, same_period: bool) -> None:
    for field in ("company", "scope", "metric", "currency"):
        value = getattr(left, field)
        if value is None or value != getattr(right, field):
            raise CalculationError(f"Comparison requires the same explicit {field}.")
    if same_period and (not left.period or left.period != right.period):
        raise CalculationError("Comparison requires the same explicit fiscal period.")


def _references(*facts: NumericFact) -> tuple[str, ...]:
    return tuple(dict.fromkeys(ref for fact in facts for ref in fact.evidence_refs))


def compare_amounts(actual: NumericFact, estimate: NumericFact) -> AmountComparison:
    """Compute an actual versus broker-estimate variance for one metric/context."""
    _same_context(actual, estimate, same_period=True)
    if actual.kind != "reported_actual" or estimate.kind != "broker_estimate":
        raise CalculationError("An estimate comparison needs reported actual and broker estimate inputs.")
    actual_value, estimate_value = amount_millions(actual), amount_millions(estimate)
    if estimate_value <= 0:
        raise CalculationError("A percentage variance requires a positive estimate denominator.")
    delta = _subtract_exact(actual_value, estimate_value)
    return AmountComparison(actual.company, actual.period, actual.scope, actual.metric,
                            actual.currency, actual_value, estimate_value, delta,
                            _percent(delta, estimate_value), (actual.fact_id, estimate.fact_id),
                            _references(actual, estimate))


def revenue_growth(actual: NumericFact, prior: NumericFact) -> RevenueGrowth:
    """Use the same fiscal quarter in the previous year; retain company scope."""
    _same_context(actual, prior, same_period=False)
    if actual.metric != "net_sales" or actual.kind != "reported_actual" or prior.kind != "reported_actual":
        raise CalculationError("Revenue growth requires two reviewed reported net-sales actuals.")
    current = re.fullmatch(r"([1-4])QFY(\d{2}|\d{4})", actual.period or "")
    previous = re.fullmatch(r"([1-4])QFY(\d{2}|\d{4})", prior.period or "")
    if (current is None or previous is None or current.group(1) != previous.group(1)
            or len(current.group(2)) != len(previous.group(2))
            or int(current.group(2)) != int(previous.group(2)) + 1):
        raise CalculationError("YoY inputs must be the same fiscal quarter in consecutive years.")
    actual_value, prior_value = amount_millions(actual), amount_millions(prior)
    if actual_value < 0 or prior_value <= 0:
        raise CalculationError("Revenue growth requires nonnegative revenue and a positive prior denominator.")
    return RevenueGrowth(actual.company, actual.period, prior.period, actual.scope, actual.currency,
                         actual_value, prior_value, _percent(_subtract_exact(actual_value, prior_value), prior_value),
                         actual.unit, prior.unit, (actual.fact_id, prior.fact_id), _references(actual, prior))


def rank_revenue_growth(pairs: tuple[tuple[NumericFact, NumericFact], ...],
                        requested_companies: tuple[str, ...]) -> tuple[RankedGrowth, ...]:
    """Refuse missing/extra companies; equal growth receives the same rank."""
    if (not isinstance(requested_companies, tuple) or not requested_companies
            or len(set(requested_companies)) != len(requested_companies)
            or any(not isinstance(company, str) or not company.strip() for company in requested_companies)):
        raise CalculationError("Ranking requires a unique explicit company set.")
    rows = tuple(revenue_growth(actual, prior) for actual, prior in pairs)
    companies = tuple(row.company for row in rows)
    if len(companies) != len(set(companies)) or set(companies) != set(requested_companies):
        raise CalculationError("Every requested company needs exactly one compatible revenue pair.")
    if len({row.period for row in rows}) != 1:
        raise CalculationError("Ranking requires the same explicit fiscal quarter for every company.")
    # Scopes may differ across reports; each result preserves its source scope.
    ordered = sorted(rows, key=lambda row: (-row.yoy_percent, row.company))
    ranked, last_value, last_rank = [], None, 0
    for position, row in enumerate(ordered, 1):
        if last_value is None or row.yoy_percent != last_value:
            last_rank = position
        ranked.append(RankedGrowth(last_rank, row))
        last_value = row.yoy_percent
    return tuple(ranked)


@dataclass(frozen=True)
class MetricChange:
    company: str
    metric: str
    period: str
    reference_period: str
    scope: str
    currency: str | None
    unit: str
    actual: Decimal
    reference: Decimal
    delta: Decimal
    relative_percent: Decimal
    delta_unit: str
    basis_points: Decimal | None
    actual_source_unit: str
    reference_source_unit: str
    fact_ids: tuple[str, ...]
    evidence_refs: tuple[str, ...]


def _metric_values(actual: NumericFact, reference: NumericFact):
    """Normalize only explicitly compatible dimensions, never metric semantics."""
    for field in ('company', 'scope', 'metric'):
        if not getattr(actual, field) or getattr(actual, field) != getattr(reference, field):
            raise CalculationError(f'Metric changes require the same explicit {field}.')
    if actual.currency != reference.currency:
        raise CalculationError('Metric changes cannot convert currencies.')
    left, right = _number(actual), _number(reference)
    if actual.currency is not None and actual.unit in {'million', 'billion'}:
        return amount_millions(actual), amount_millions(reference), 'million'
    if actual.unit in {'percent', 'basis_points'} and reference.unit in {'percent', 'basis_points'}:
        if actual.currency is not None:
            raise CalculationError('Percentage metrics cannot carry a monetary currency.')
        return (_shift_decimal(left, -2) if actual.unit == 'basis_points' else left,
                _shift_decimal(right, -2) if reference.unit == 'basis_points' else right, 'percent')
    units = {'per_share', 'x', 'multiple', 'count', 'tonnes', 'days', 'years', 'ratio'}
    if actual.unit != reference.unit or actual.unit not in units:
        raise CalculationError('This metric needs identical reviewed units or a supported scale conversion.')
    if actual.unit == 'per_share' and not actual.currency:
        raise CalculationError('Per-share monetary metrics require their currency.')
    return left, right, actual.unit


def metric_change(actual: NumericFact, reference: NumericFact, *, comparison: str) -> MetricChange:
    """Compare any reviewed metric with compatible units; preserve margin points."""
    if comparison not in {'estimate', 'yoy'}:
        raise CalculationError('Metric changes support estimate variance or fiscal YoY only.')
    if actual.kind != 'reported_actual':
        raise CalculationError('The current metric must be a reported actual.')
    if comparison == 'estimate':
        if reference.kind != 'broker_estimate' or not actual.period or actual.period != reference.period:
            raise CalculationError('Estimate variance requires the same explicit period and a broker estimate.')
    else:
        if reference.kind != 'reported_actual':
            raise CalculationError('YoY requires two reported actuals.')
        current = re.fullmatch(r'([1-4]Q)?FY(\d{2}|\d{4})', actual.period or '')
        previous = re.fullmatch(r'([1-4]Q)?FY(\d{2}|\d{4})', reference.period or '')
        if (current is None or previous is None or current[1] != previous[1]
                or len(current[2]) != len(previous[2]) or int(current[2]) != int(previous[2]) + 1):
            raise CalculationError('YoY requires consecutive years with the same fiscal quarter or annual basis.')
    left, right, unit = _metric_values(actual, reference)
    if right <= 0:
        raise CalculationError('Relative change requires a positive reference; negative or zero bases are unsupported.')
    delta = _subtract_exact(left, right)
    return MetricChange(actual.company, actual.metric, actual.period, reference.period, actual.scope,
                        actual.currency, unit, left, right, delta, _percent(delta, right),
                        'percentage_points' if unit == 'percent' else unit,
                        _shift_decimal(delta, 2) if unit == 'percent' else None,
                        actual.unit, reference.unit, (actual.fact_id, reference.fact_id),
                        _references(actual, reference))
