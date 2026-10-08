"""Financial invariants; synthetic records are not reviewed research facts."""
from decimal import Decimal
from types import SimpleNamespace
import unittest

from financial_analyst.corpus_analytics import (
    CalculationError, metric_change, amount_millions, compare_amounts, rank_revenue_growth, revenue_growth,
)


def fact(value='120', *, company='Example A', period='1QFY27', scope='consolidated',
         metric='net_sales', kind='reported_actual', currency='INR', unit='million',
         ref='current'):
    return SimpleNamespace(fact_id=ref, company=company, period=period, scope=scope,
                           metric=metric, kind=kind, currency=currency, unit=unit,
                           value_text=value, evidence_refs=(ref, 'source-header'))


class CorpusAnalyticsTests(unittest.TestCase):
    def test_mixed_monetary_scales_do_not_change_comparison(self):
        actual = fact('1.2', unit='billion')
        estimate = fact('1000', kind='broker_estimate', ref='estimate')
        result = compare_amounts(actual, estimate)
        self.assertEqual(result.actual_millions, Decimal('1200'))
        self.assertEqual(result.delta_millions, Decimal('200'))
        self.assertEqual(result.variance_percent, Decimal('20'))
        self.assertEqual(result.evidence_refs, ('current', 'source-header', 'estimate'))

    def test_scale_change_retains_more_than_default_decimal_precision(self):
        value = '1234567890123456789012345678901234567890.123456789'
        self.assertEqual(amount_millions(fact(value, unit='billion')),
                         Decimal('1234567890123456789012345678901234567890123.456789'))

    def test_currency_and_ratio_units_never_convert_to_amounts(self):
        for changes in ({'unit':'percent'}, {'unit':'basis_points'}, {'unit':'multiple'},
                        {'unit':'per_share'}, {'currency':None}, {'currency':'inr'}):
            with self.subTest(changes=changes), self.assertRaises(CalculationError):
                amount_millions(fact(**changes))

    def test_comparison_refuses_incompatible_context_and_wrong_status(self):
        actual = fact()
        for changes in ({'company':'Example B'}, {'scope':'standalone'}, {'scope':None},
                        {'metric':'ebitda'}, {'currency':'USD'}, {'period':'2QFY27'},
                        {'kind':'broker_forecast'}):
            with self.subTest(changes=changes), self.assertRaises(CalculationError):
                compare_amounts(actual, fact('100', kind='broker_estimate', ref='estimate',
                                             **{k:v for k,v in changes.items() if k != 'kind'})
                                if 'kind' not in changes else fact('100', ref='estimate', **changes))

    def test_miss_and_zero_variance_are_signed_and_source_supported(self):
        estimate = fact('100', kind='broker_estimate', ref='estimate')
        self.assertEqual(compare_amounts(fact('80'), estimate).variance_percent, Decimal('-20'))
        self.assertEqual(compare_amounts(fact('100'), estimate).variance_percent, Decimal('0'))

    def test_zero_and_negative_denominators_refuse(self):
        for value in ('0', '-1'):
            with self.subTest(value=value), self.assertRaises(CalculationError):
                compare_amounts(fact(), fact(value, kind='broker_estimate', ref='estimate'))
            with self.subTest(value=value), self.assertRaises(CalculationError):
                revenue_growth(fact(), fact(value, period='1QFY26', ref='prior'))

    def test_nonfinite_values_and_missing_evidence_refuse(self):
        for value in ('NaN', 'Infinity', '-Infinity', 'not-a-number', None):
            with self.subTest(value=value), self.assertRaises(CalculationError):
                amount_millions(fact(value))
        missing = fact()
        missing.evidence_refs = ()
        with self.assertRaises(CalculationError):
            amount_millions(missing)

    def test_yoy_requires_same_quarter_previous_year_and_reported_actual(self):
        for changes in ({'period':'2QFY26'}, {'period':'1QFY25'}, {'period':'FY26'},
                        {'period':'1QFY2026'}, {'kind':'broker_forecast'}, {'scope':None}):
            prior = fact('100', ref='prior')
            prior.period = '1QFY26'
            for key, value in changes.items():
                setattr(prior, key, value)
            with self.subTest(changes=changes), self.assertRaises(CalculationError):
                revenue_growth(fact(), prior)

    def test_ranking_preserves_company_scopes_and_units(self):
        pairs = (
            (fact('1.5', company='Example B', unit='billion', scope='standalone'),
             fact('1000', company='Example B', period='1QFY26', scope='standalone', ref='prior-b')),
            (fact('120', company='Example A'), fact('100', period='1QFY26', ref='prior-a')),
        )
        result = rank_revenue_growth(pairs, ('Example A', 'Example B'))
        self.assertEqual([row.growth.company for row in result], ['Example B', 'Example A'])
        self.assertEqual([row.growth.yoy_percent for row in result], [Decimal('50'), Decimal('20')])
        self.assertEqual(result[0].growth.scope, 'standalone')
        self.assertEqual(result[0].growth.actual_source_unit, 'billion')
        self.assertEqual(result[0].growth.actual_millions, Decimal('1500'))

    def test_ranking_refuses_partial_or_duplicate_company_coverage(self):
        pair = (fact(), fact('100', period='1QFY26', ref='prior'))
        for pairs, requested in (((pair,), ('Example A', 'Example B')),
                                 ((pair, pair), ('Example A',)),
                                 ((pair,), ('Example A', 'Example A')),
                                 ((pair,), ())):
            with self.subTest(requested=requested), self.assertRaises(CalculationError):
                rank_revenue_growth(pairs, requested)

    def test_equal_growth_is_a_tie_and_does_not_invent_an_ordering(self):
        pairs = tuple((fact('120', company=name), fact('100', company=name,
                       period='1QFY26', ref='prior-'+name)) for name in ('Example B', 'Example A'))
        rows = rank_revenue_growth(pairs, ('Example A', 'Example B'))
        self.assertEqual([row.rank for row in rows], [1, 1])
        self.assertEqual({row.growth.company for row in rows}, {'Example A', 'Example B'})

    def test_ranking_refuses_different_current_quarters(self):
        pairs = (
            (fact(), fact('100', period='1QFY26', ref='prior-a')),
            (fact('140', company='Example B', period='2QFY27'),
             fact('100', company='Example B', period='2QFY26', ref='prior-b')),
        )
        with self.assertRaises(CalculationError):
            rank_revenue_growth(pairs, ('Example A', 'Example B'))


class GenericMetricTests(unittest.TestCase):
    def pair(self,current,reference,**context):
        return (fact(current,**context),fact(reference,kind='broker_estimate',ref='estimate',**context))

    def test_margin_change_is_points_and_basis_points_with_relative_change_separate(self):
        actual,estimate=self.pair('31.5','30',metric='ebitda_margin',currency=None,unit='percent')
        result=metric_change(actual,estimate,comparison='estimate')
        self.assertEqual(result.delta,Decimal('1.5'))
        self.assertEqual(result.delta_unit,'percentage_points')
        self.assertEqual(result.basis_points,Decimal('150'))
        self.assertEqual(result.relative_percent,Decimal('5'))

    def test_basis_point_and_percent_units_normalize_only_same_metric(self):
        actual=fact('3150',metric='ebitda_margin',currency=None,unit='basis_points')
        prior=fact('30',metric='ebitda_margin',currency=None,unit='percent',period='1QFY26',ref='prior')
        result=metric_change(actual,prior,comparison='yoy')
        self.assertEqual(result.actual,Decimal('31.50'))
        self.assertEqual(result.basis_points,Decimal('150'))

    def test_eps_counts_volumes_and_multiples_keep_their_dimensions(self):
        for metric,unit,currency in (('eps','per_share','INR'),('tractor_sales','count',None),
                                      ('volume','tonnes',None),('p_e','x',None)):
            with self.subTest(metric=metric):
                actual,estimate=self.pair('12','10',metric=metric,unit=unit,currency=currency)
                result=metric_change(actual,estimate,comparison='estimate')
                self.assertEqual(result.delta,Decimal('2'))
                self.assertEqual(result.unit,unit)
                self.assertEqual(result.relative_percent,Decimal('20'))

    def test_annual_yoy_is_supported_but_annual_and_quarterly_cannot_mix(self):
        actual=fact('120',period='FY27',metric='adj_pat')
        prior=fact('100',period='FY26',metric='adj_pat',ref='prior')
        self.assertEqual(metric_change(actual,prior,comparison='yoy').relative_percent,Decimal('20'))
        prior.period='1QFY26'
        with self.assertRaises(CalculationError):metric_change(actual,prior,comparison='yoy')

    def test_unknown_scope_unit_currency_and_forecasts_refuse(self):
        for changes in ({'scope':None},{'unit':'mystery'},{'currency':'USD'},
                        {'kind':'broker_forecast'},{'period':'2QFY27'}):
            actual,estimate=self.pair('12','10',metric='eps',unit='per_share')
            for key,value in changes.items():setattr(estimate,key,value)
            with self.subTest(changes=changes),self.assertRaises(CalculationError):
                metric_change(actual,estimate,comparison='estimate')

    def test_nonpositive_reference_does_not_invent_growth_percent(self):
        for value in ('0','-10'):
            actual,estimate=self.pair('12',value,metric='adj_pat')
            with self.assertRaises(CalculationError):metric_change(actual,estimate,comparison='estimate')


if __name__ == '__main__':
    unittest.main()
