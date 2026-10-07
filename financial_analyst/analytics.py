"""Pure revenue comparisons; the caller verifies supporting source content."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Context, Decimal, MAX_EMAX, MIN_EMIN, ROUND_HALF_EVEN, localcontext
import re


class ComparisonError(ValueError):
    """The observations do not support the requested comparison."""


@dataclass(frozen=True)
class RevenueObservation:
    company: str
    period: str
    scope: str
    metric: str
    currency: str
    source_unit: str
    value: Decimal
    kind: str
    evidence_refs: tuple[str, ...]
    year_end_raw: str
    source_header_raw: str
    source_value_raw: str
    source_unit_raw: str
    source_metric_raw: str


@dataclass(frozen=True)
class RevenueComparison:
    actual: RevenueObservation
    broker_estimate: RevenueObservation
    prior_year_actual: RevenueObservation | None
    actual_millions: Decimal
    estimate_millions: Decimal
    delta_millions: Decimal
    beat_percent: Decimal
    yoy_percent: Decimal | None
    beat_evidence_refs: tuple[str, ...]
    yoy_evidence_refs: tuple[str, ...]


_PERIOD = re.compile(r"([1-4])QFY([0-9]{2}|[0-9]{4})")
_SOURCE_NUMBER = re.compile(r"[+-]?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?")
_STRING_FIELDS = (
    "company", "period", "scope", "metric", "currency", "source_unit", "kind",
    "year_end_raw", "source_header_raw", "source_value_raw", "source_unit_raw",
    "source_metric_raw",
)
_IDENTITY_FIELDS = ("company", "scope", "metric", "currency", "year_end_raw")


def _validate(observation: RevenueObservation, role: str, expected_kind: str) -> None:
    if not isinstance(observation, RevenueObservation):
        raise ComparisonError(f"{role} must be a RevenueObservation")
    for field in _STRING_FIELDS:
        value = getattr(observation, field)
        if not isinstance(value, str) or not value.strip():
            raise ComparisonError(f"{role}.{field} is required")

    if observation.kind != expected_kind:
        raise ComparisonError(f"{role}.kind must be {expected_kind}")
    if observation.metric != "net_sales":
        raise ComparisonError(f"{role}.metric must be net_sales")
    if _PERIOD.fullmatch(observation.period) is None:
        raise ComparisonError(f"{role}.period must be an explicit fiscal quarter, such as 1QFY42")
    if re.fullmatch(r"[A-Z]{3}", observation.currency) is None:
        raise ComparisonError(f"{role}.currency must be a three-letter uppercase code")

    if not isinstance(observation.evidence_refs, tuple) or not observation.evidence_refs:
        raise ComparisonError(f"{role}.evidence_refs must be a nonempty tuple")
    if any(not isinstance(ref, str) or not ref.strip() for ref in observation.evidence_refs):
        raise ComparisonError(f"{role}.evidence_refs must contain nonempty strings")

    if not isinstance(observation.value, Decimal) or not observation.value.is_finite():
        raise ComparisonError(f"{role}.value must be a finite Decimal")
    if observation.value < 0:
        raise ComparisonError(f"{role}.value must not be negative")

    unit_suffix = {"million": "m", "billion": "b"}.get(observation.source_unit)
    if unit_suffix is None:
        raise ComparisonError(f"{role}.source_unit must be million or billion")
    if observation.source_unit_raw != observation.currency + unit_suffix:
        raise ComparisonError(f"{role}.source_unit_raw conflicts with currency or source_unit")

    raw_value = observation.source_value_raw.strip()
    if _SOURCE_NUMBER.fullmatch(raw_value) is None:
        raise ComparisonError(f"{role}.source_value_raw must be an unambiguous source number")
    if Decimal(raw_value.replace(",", "")) != observation.value:
        raise ComparisonError(f"{role}.source_value_raw does not equal value")


def _require_same_identity(actual: RevenueObservation, other: RevenueObservation, role: str) -> None:
    for field in _IDENTITY_FIELDS:
        if getattr(actual, field) != getattr(other, field):
            raise ComparisonError(f"{role}.{field} must match actual.{field}")


def _shift_decimal(value: Decimal, places: int) -> Decimal:
    """Move the decimal point exactly, without arithmetic-context rounding."""
    sign, digits, exponent = value.as_tuple()
    return Decimal((sign, digits, exponent + places))


def _subtract_exact(left: Decimal, right: Decimal) -> Decimal:
    # Retain every input place, including when amounts exceed 40 digits.
    precision = max(left.adjusted(), right.adjusted()) - min(
        left.as_tuple().exponent, right.as_tuple().exponent
    ) + 2
    context = Context(prec=max(1, precision), rounding=ROUND_HALF_EVEN, Emax=MAX_EMAX, Emin=MIN_EMIN)
    with localcontext(context):
        return left - right


def _percent(delta: Decimal, denominator: Decimal) -> Decimal:
    context = Context(prec=40, rounding=ROUND_HALF_EVEN, Emax=MAX_EMAX, Emin=MIN_EMIN)
    with localcontext(context):
        return _shift_decimal(delta, 2) / denominator


def _evidence_union(*observations: RevenueObservation) -> tuple[str, ...]:
    return tuple(dict.fromkeys(ref for observation in observations for ref in observation.evidence_refs))


def compare_revenue(
    actual: RevenueObservation,
    broker_estimate: RevenueObservation,
    prior_year_actual: RevenueObservation | None = None,
) -> RevenueComparison:
    """Compare already source-verified observations without I/O or input changes.

    Evidence references are checked structurally here. The calling service must
    resolve them and verify the source content before invoking this function.
    Percentages use the estimate or prior-year actual as their denominator;
    they represent relative change, not a change in percentage points.
    """
    _validate(actual, "actual", "reported_actual")
    _validate(broker_estimate, "broker_estimate", "broker_estimate")
    _require_same_identity(actual, broker_estimate, "broker_estimate")
    if actual.period != broker_estimate.period:
        raise ComparisonError("broker_estimate.period must match actual.period")
    if broker_estimate.value == 0:
        raise ComparisonError("broker_estimate.value must be nonzero")

    if prior_year_actual is not None:
        _validate(prior_year_actual, "prior_year_actual", "reported_actual")
        _require_same_identity(actual, prior_year_actual, "prior_year_actual")
        current_period = _PERIOD.fullmatch(actual.period)
        previous_period = _PERIOD.fullmatch(prior_year_actual.period)
        current_quarter, current_year = current_period.groups()
        previous_quarter, previous_year = previous_period.groups()
        if (
            current_quarter != previous_quarter
            or len(current_year) != len(previous_year)
            or int(current_year) - 1 != int(previous_year)
        ):
            raise ComparisonError(
                "prior_year_actual.period must be the same quarter in the previous fiscal year"
            )
        if prior_year_actual.value == 0:
            raise ComparisonError("prior_year_actual.value must be nonzero for YoY")

    actual_millions = _shift_decimal(actual.value, 3) if actual.source_unit == "billion" else actual.value
    estimate_millions = (
        _shift_decimal(broker_estimate.value, 3)
        if broker_estimate.source_unit == "billion" else broker_estimate.value
    )
    delta_millions = _subtract_exact(actual_millions, estimate_millions)
    yoy_percent = None
    yoy_evidence_refs = ()
    if prior_year_actual is not None:
        prior_millions = (
            _shift_decimal(prior_year_actual.value, 3)
            if prior_year_actual.source_unit == "billion" else prior_year_actual.value
        )
        yoy_percent = _percent(_subtract_exact(actual_millions, prior_millions), prior_millions)
        yoy_evidence_refs = _evidence_union(actual, prior_year_actual)

    return RevenueComparison(
        actual=actual,
        broker_estimate=broker_estimate,
        prior_year_actual=prior_year_actual,
        actual_millions=actual_millions,
        estimate_millions=estimate_millions,
        delta_millions=delta_millions,
        beat_percent=_percent(delta_millions, estimate_millions),
        yoy_percent=yoy_percent,
        beat_evidence_refs=_evidence_union(actual, broker_estimate),
        yoy_evidence_refs=yoy_evidence_refs,
    )
