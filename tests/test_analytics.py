"""Synthetic calculator cases; no private research content."""

from dataclasses import FrozenInstanceError, replace
from decimal import Decimal, Inexact, ROUND_UP, getcontext, localcontext
import unittest

from financial_analyst.analytics import ComparisonError, RevenueObservation, compare_revenue


def observation(value="1200", *, kind="reported_actual", period="1QFY42", source_unit="million"):
    return RevenueObservation(
        company="SYNTHETIC_COMPANY",
        period=period,
        scope="consolidated",
        metric="net_sales",
        currency="USD",
        source_unit=source_unit,
        value=Decimal(value.replace(",", "")),
        kind=kind,
        evidence_refs=("synthetic.shared", f"synthetic.{kind}.{period}"),
        year_end_raw="Y/E June",
        source_header_raw=f"SYNTHETIC {period}",
        source_value_raw=value,
        source_unit_raw="USDb" if source_unit == "billion" else "USDm",
        source_metric_raw="Net Sales",
    )


class RevenueComparisonTests(unittest.TestCase):
    def setUp(self):
        self.actual = observation("1,200")
        self.estimate = observation("1,000", kind="broker_estimate")
        self.prior = observation("800", period="1QFY41")

    def test_positive_comparison_and_optional_yoy(self):
        result = compare_revenue(self.actual, self.estimate, self.prior)
        self.assertEqual(result.actual_millions, Decimal("1200"))
        self.assertEqual(result.estimate_millions, Decimal("1000"))
        self.assertEqual(result.delta_millions, Decimal("200"))
        self.assertEqual(result.beat_percent, Decimal("20"))
        self.assertEqual(result.yoy_percent, Decimal("50"))
        self.assertIs(result.actual, self.actual)
        self.assertIs(result.broker_estimate, self.estimate)
        self.assertIs(result.prior_year_actual, self.prior)
        self.assertEqual(result.actual.source_value_raw, "1,200")
        self.assertEqual(result.beat_evidence_refs, (
            "synthetic.shared", "synthetic.reported_actual.1QFY42", "synthetic.broker_estimate.1QFY42",
        ))
        self.assertEqual(result.yoy_evidence_refs, (
            "synthetic.shared", "synthetic.reported_actual.1QFY42", "synthetic.reported_actual.1QFY41",
        ))
        without_prior = compare_revenue(self.actual, self.estimate)
        self.assertIsNone(without_prior.yoy_percent)
        self.assertEqual(without_prior.yoy_evidence_refs, ())

    def test_equal_and_below_estimate_keep_signed_results(self):
        for value, expected_delta, expected_percent in (("1000", "0", "0"), ("900", "-100", "-10")):
            with self.subTest(value=value):
                result = compare_revenue(observation(value), self.estimate)
                self.assertEqual(result.delta_millions, Decimal(expected_delta))
                self.assertEqual(result.beat_percent, Decimal(expected_percent))

    def test_explicit_billion_conversion_is_exact_and_preserves_inputs(self):
        actual = observation("1.2", source_unit="billion")
        prior = observation("0.8", period="1QFY41", source_unit="billion")
        result = compare_revenue(actual, self.estimate, prior)
        self.assertEqual(result.delta_millions, Decimal("200"))
        self.assertEqual(result.beat_percent, Decimal("20"))
        self.assertEqual(result.yoy_percent, Decimal("50"))
        self.assertEqual(result.actual.value, Decimal("1.2"))
        self.assertEqual(result.actual.source_unit_raw, "USDb")
        self.assertEqual(result.actual.source_value_raw, "1.2")

    def test_percentage_has_40_digits_independent_of_caller_context(self):
        with localcontext() as context:
            context.prec = 6
            context.rounding = ROUND_UP
            context.traps[Inexact] = True
            result = compare_revenue(observation("110"), observation("90", kind="broker_estimate"))
            self.assertEqual(result.beat_percent, Decimal("22.22222222222222222222222222222222222222"))
            self.assertEqual(getcontext().prec, 6)
            self.assertEqual(getcontext().rounding, ROUND_UP)
            self.assertTrue(getcontext().traps[Inexact])

    def test_money_delta_does_not_round_long_input_coefficients(self):
        actual = observation("123456789012345678901234567890123456789012345")
        estimate = observation("123456789012345678901234567890123456789012344", kind="broker_estimate")
        self.assertEqual(compare_revenue(actual, estimate).delta_millions, Decimal("1"))

    def test_billion_normalization_does_not_round_long_input_coefficients(self):
        actual = observation("123456789012345678901234567890123456789012.345", source_unit="billion")
        estimate = observation("123456789012345678901234567890123456789012344", kind="broker_estimate")
        with localcontext() as context:
            context.prec = 6
            result = compare_revenue(actual, estimate)
        self.assertEqual(result.actual_millions, Decimal("123456789012345678901234567890123456789012345"))
        self.assertEqual(result.delta_millions, Decimal("1"))
        self.assertEqual(result.actual.source_value_raw, "123456789012345678901234567890123456789012.345")

    def test_contracts_are_frozen_and_result_references_are_immutable(self):
        result = compare_revenue(self.actual, self.estimate)
        with self.assertRaises(FrozenInstanceError):
            self.actual.value = Decimal("1")
        with self.assertRaises(FrozenInstanceError):
            result.delta_millions = Decimal("1")
        self.assertIsInstance(result.beat_evidence_refs, tuple)

    def test_comparison_identity_and_period_must_match(self):
        cases = (
            ("company", "SYNTHETIC_OTHER", "company"),
            ("scope", "standalone", "scope"),
            ("period", "2QFY42", "period"),
            ("year_end_raw", "Y/E September", "year_end_raw"),
        )
        for field, value, reason in cases:
            with self.subTest(field=field):
                with self.assertRaisesRegex(ComparisonError, reason):
                    compare_revenue(self.actual, replace(self.estimate, **{field: value}))
        other_currency = replace(self.estimate, currency="EUR", source_unit_raw="EURm")
        with self.assertRaisesRegex(ComparisonError, "currency must match"):
            compare_revenue(self.actual, other_currency)
        with self.assertRaisesRegex(ComparisonError, "metric must be net_sales"):
            compare_revenue(self.actual, replace(self.estimate, metric="operating_profit"))

    def test_wrong_actual_estimate_or_forecast_kind_is_rejected(self):
        for actual_kind, estimate_kind in (
            ("broker_estimate", "broker_estimate"),
            ("reported_actual", "reported_actual"),
            ("broker_forecast", "broker_estimate"),
            ("reported_actual", "broker_forecast"),
        ):
            with self.subTest(actual_kind=actual_kind, estimate_kind=estimate_kind):
                with self.assertRaisesRegex(ComparisonError, "kind must be"):
                    compare_revenue(replace(self.actual, kind=actual_kind), replace(self.estimate, kind=estimate_kind))

    def test_missing_identity_provenance_and_evidence_are_rejected(self):
        required_strings = (
            "company", "period", "scope", "metric", "currency", "source_unit", "kind",
            "year_end_raw", "source_header_raw", "source_value_raw", "source_unit_raw", "source_metric_raw",
        )
        for field in required_strings:
            with self.subTest(field=field):
                with self.assertRaisesRegex(ComparisonError, field):
                    compare_revenue(replace(self.actual, **{field: " "}), self.estimate)
        for refs in ((), ("",), (" ",), (None,), ["synthetic.reference"]):
            with self.subTest(refs=refs):
                with self.assertRaisesRegex(ComparisonError, "evidence_refs"):
                    compare_revenue(replace(self.actual, evidence_refs=refs), self.estimate)

    def test_nonfinite_negative_and_nondecimal_revenue_are_rejected(self):
        for value in (Decimal("NaN"), Decimal("sNaN"), Decimal("Infinity"), Decimal("-Infinity"), Decimal("-1"), 1200.0):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ComparisonError, "value must"):
                    compare_revenue(replace(self.actual, value=value), self.estimate)

    def test_units_and_original_numeric_value_must_be_consistent(self):
        cases = (
            {"source_unit": "thousand"},
            {"source_unit_raw": "EURm"},
            {"source_unit_raw": "USDb"},
            {"source_value_raw": "1,201"},
            {"source_value_raw": "1,,200"},
            {"source_value_raw": "1e3"},
        )
        for changes in cases:
            with self.subTest(changes=changes):
                with self.assertRaises(ComparisonError):
                    compare_revenue(replace(self.actual, **changes), self.estimate)

    def test_zero_denominators_are_rejected_but_zero_actual_is_valid(self):
        with self.assertRaisesRegex(ComparisonError, "broker_estimate.value must be nonzero"):
            compare_revenue(self.actual, observation("0", kind="broker_estimate"))
        with self.assertRaisesRegex(ComparisonError, "prior_year_actual.value must be nonzero"):
            compare_revenue(self.actual, self.estimate, observation("0", period="1QFY41"))
        self.assertEqual(compare_revenue(observation("0"), self.estimate).beat_percent, Decimal("-100"))

    def test_prior_year_requires_matching_identity_kind_and_fiscal_period(self):
        cases = (
            {"company": "SYNTHETIC_OTHER"},
            {"scope": "standalone"},
            {"metric": "operating_profit"},
            {"currency": "EUR", "source_unit_raw": "EURm"},
            {"kind": "broker_estimate"},
            {"period": "2QFY41"},
            {"period": "1QFY40"},
            {"period": "1QFY2041"},
            {"year_end_raw": "Y/E September"},
        )
        for changes in cases:
            with self.subTest(changes=changes):
                with self.assertRaises(ComparisonError):
                    compare_revenue(self.actual, self.estimate, replace(self.prior, **changes))

    def test_annual_estimated_and_unresolved_period_labels_are_rejected(self):
        for period in ("FY42", "1QFY42E", "Q1 2042", "5QFY42", "1QFY?"):
            with self.subTest(period=period):
                with self.assertRaisesRegex(ComparisonError, "explicit fiscal quarter"):
                    compare_revenue(replace(self.actual, period=period), replace(self.estimate, period=period))


if __name__ == "__main__":
    unittest.main()
