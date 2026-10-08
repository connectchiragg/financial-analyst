"""Canonical local tools over an explicitly reviewed multi-source read adapter."""
from __future__ import annotations

from dataclasses import asdict

from .corpus_adapter import CorpusError, FactFilter
from .corpus_analytics import CalculationError, compare_amounts, rank_revenue_growth, revenue_growth, metric_change
from .service import Answer, Claim


class CorpusService:
    def __init__(self, adapter):
        self.adapter = adapter

    def _execution(self, used=True):
        mode = getattr(self.adapter, 'mode', 'unavailable')
        sqlite = mode == 'sqlite_corpus'
        return {'mode': 'live' if sqlite else 'fixture', 'analytics': mode,
                'database': 'sqlite_corpus' if used and sqlite else 'none',
                'seed_provenance': tuple(getattr(self.adapter, 'seed_provenance', ())),
                'retrieval': 'reviewed_context_filter' if used else 'not_requested',
                'llm': {'provider': 'none', 'mode': 'not_called'},
                'no_llm': True, 'no_database': not (used and sqlite)}

    def refuse(self, reason, *, used=False):
        return Answer('refused', execution=self._execution(used), reason=reason)

    def candidates(self, filters):
        facts = self.adapter.read_facts(filters)
        if len({fact.fact_id for fact in facts}) != len(facts):
            raise CorpusError('The read adapter returned duplicate fact identities.')
        batch = getattr(self.adapter, 'validate_facts', None)
        if callable(batch):
            batch(facts)
        for fact in facts:
            if ((filters.companies and fact.company not in filters.companies)
                    or (filters.metrics and fact.metric not in filters.metrics)
                    or any(getattr(filters, field) is not None
                           and getattr(fact, field) != getattr(filters, field)
                           for field in ('period', 'scope', 'kind', 'currency', 'unit', 'source_sha256'))):
                raise CorpusError('The read adapter returned a record outside the requested financial context.')
            if not callable(batch):
                self.adapter.validate_fact(fact)
            if fact.review.get('status') != 'reviewed' or 'source_statement' not in fact.capabilities:
                raise CorpusError('A requested source record lacks approved factual review.')
        return facts

    def _citations(self, claims):
        refs = tuple(dict.fromkeys(ref for claim in claims for ref in claim.evidence_refs))
        resolver = getattr(self.adapter, 'resolve_many', None)
        evidence = resolver(refs) if callable(resolver) else tuple(self.adapter.resolve(ref) for ref in refs)
        if tuple(item.ref for item in evidence) != refs:
            raise CorpusError('Citation resolution returned different references.')
        return tuple(item.citation for item in evidence)

    @staticmethod
    def _fact_claim(fact):
        return Claim('source_fact', {
            'fact_id': fact.fact_id, 'company': fact.company, 'period': fact.period,
            'scope': fact.scope, 'metric': fact.metric, 'kind': fact.kind,
            'currency': fact.currency, 'unit': fact.unit, 'value_text': fact.value_text,
            'quote': fact.text_value, 'raw_labels': fact.raw_labels,
            'review_capabilities': fact.capabilities,
            'period_role': fact.proof.get('period_role', 'financial_period' if fact.value_text is not None else 'source_statement'),
        }, fact.evidence_refs)

    def lookup(self, filters, fact_ids):
        try:
            eligible = {fact.fact_id: fact for fact in self.candidates(filters)}
            if (not isinstance(fact_ids, tuple) or not fact_ids or len(set(fact_ids)) != len(fact_ids)
                    or any(fact_id not in eligible for fact_id in fact_ids)):
                raise CorpusError('Selected facts must be unique reviewed records matching every requested filter.')
            selected = tuple(eligible[fact_id] for fact_id in fact_ids)
            if filters.companies and set(fact.company for fact in selected) != set(filters.companies):
                raise CorpusError('The selected facts do not cover every requested company.')
            claims = tuple(self._fact_claim(fact) for fact in selected)
            return Answer('answered', claims, citations=self._citations(claims), execution=self._execution())
        except (CorpusError, ValueError) as error:
            return self.refuse(str(error), used=True)

    def _one(self, company, period, metric, kind, **context):
        facts = self.candidates(FactFilter(companies=(company,), period=period,
                                          metrics=(metric,), kind=kind, **context))
        analytical = tuple(fact for fact in facts if 'analytical_context' in fact.capabilities)
        if len(analytical) != 1:
            raise CorpusError('Exactly one fully reviewed compatible analytical input is required for each role.')
        return analytical[0]

    @staticmethod
    def _prior_period(period):
        from re import fullmatch
        match = fullmatch(r'([1-4]Q)?FY(\d{2}|\d{4})', period or '')
        if match is None:
            raise CorpusError('YoY requires an explicit fiscal-quarter label.')
        return (match[1] or '') + 'FY' + str(int(match[2]) - 1).zfill(len(match[2]))

    def compare(self, company, period, metric='net_sales', *, include_yoy=False, scope=None, currency=None, unit=None):
        try:
            context=dict(scope=scope,currency=currency,unit=unit)
            actual = self._one(company, period, metric, 'reported_actual', **context)
            estimate = self._one(company, period, metric, 'broker_estimate', **context)
            if actual.currency and actual.unit in {'million', 'billion'}:
                result = compare_amounts(actual, estimate)
                claims = [Claim('metric_comparison', asdict(result), result.evidence_refs)]
            else:
                result = metric_change(actual, estimate, comparison='estimate')
                claims = [Claim('metric_estimate_change', asdict(result), result.evidence_refs)]
            if include_yoy:
                prior = self._one(company, self._prior_period(period), metric, 'reported_actual', **context)
                growth = metric_change(actual, prior, comparison='yoy')
                claims.append(Claim('metric_yoy_change', asdict(growth), growth.evidence_refs))
            return Answer('answered', tuple(claims), citations=self._citations(claims), execution=self._execution())
        except (CorpusError, CalculationError, ValueError) as error:
            return self.refuse(str(error), used=True)

    def yoy(self, company, period, metric='net_sales', *, scope=None, currency=None, unit=None):
        try:
            context=dict(scope=scope,currency=currency,unit=unit)
            actual = self._one(company, period, metric, 'reported_actual', **context)
            prior = self._one(company, self._prior_period(period), metric, 'reported_actual', **context)
            growth = metric_change(actual, prior, comparison='yoy')
            claims = (Claim('metric_yoy_change', asdict(growth), growth.evidence_refs),)
            return Answer('answered', claims, citations=self._citations(claims), execution=self._execution())
        except (CorpusError, CalculationError, ValueError) as error:
            return self.refuse(str(error), used=True)

    def rank(self, companies, period, *, scope=None, currency=None, unit=None):
        try:
            context=dict(scope=scope,currency=currency,unit=unit)
            pairs = tuple((self._one(company, period, 'net_sales', 'reported_actual', **context),
                           self._one(company, self._prior_period(period), 'net_sales', 'reported_actual', **context))
                          for company in companies)
            ranked = rank_revenue_growth(pairs, companies)
            claims = tuple(Claim('ranked_revenue_yoy', {'rank': row.rank, **asdict(row.growth)},
                                 row.growth.evidence_refs) for row in ranked)
            return Answer('answered', claims, citations=self._citations(claims), execution=self._execution())
        except (CorpusError, CalculationError, ValueError) as error:
            return self.refuse(str(error), used=True)

    def valuation(self, company, period):
        """Check arithmetic of one printed valuation table, preserving its conflicts.

        This is not a cross-period/company financial comparison. All operands
        must be independently reviewed source numbers in the same exact table
        quotation. Unknown accounting scope stays unknown in the output.
        """
        from decimal import Decimal, Context, localcontext
        from .analytics import _subtract_exact
        metrics=('valuation_ebitda','valuation_multiple','target_ev','cash_surplus',
                 'equity_value','shares_outstanding','target_price')
        try:
            facts=self.candidates(FactFilter(companies=(company,),period=period,metrics=metrics))
            by_metric={metric:tuple(fact for fact in facts if fact.metric==metric) for metric in metrics}
            if any(len(records)!=1 for records in by_metric.values()):
                raise CorpusError('One reviewed source value is required for every printed valuation role.')
            inputs=tuple(by_metric[metric][0] for metric in metrics)
            bridges=self.candidates(FactFilter(companies=(company,),period=period,metrics=('valuation_bridge',)))
            if len(bridges)!=1 or bridges[0].text_value is None:
                raise CorpusError('An exact source valuation-table quotation is required.')
            bridge=bridges[0]
            bridge_ref=bridge.proof.get('value_ref')
            bridge_evidence=self.adapter.resolve(bridge_ref)
            headers={tuple(fact.proof.get('column_refs',())) for fact in inputs}
            if len(headers)!=1 or not next(iter(headers)):
                raise CorpusError('Valuation operands require the same physical fiscal table column.')
            for fact in inputs:
                binding=fact.proof.get('bindings',{}).get('kind',())
                binding=(binding,) if isinstance(binding,str) else tuple(binding)
                cell=self.adapter.resolve(fact.proof.get('value_ref'))
                if (fact.proof.get('method')!='table_cell' or bridge_ref not in binding
                        or fact.source_sha256!=bridge.source_sha256 or cell.page!=bridge_evidence.page
                        or cell.excerpt!=fact.raw_labels.get('value')
                        or any(not isinstance(fact.raw_labels.get(field),str)
                            or fact.raw_labels[field] not in bridge_evidence.excerpt for field in ('value','metric','unit'))):
                    raise CorpusError('Valuation inputs must belong to the same original printed table and exact rows.')
            ebitda,multiple,ev,cash,equity,shares,target=inputs
            if (any(fact.unit!='million' or fact.currency!='INR' for fact in (ebitda,ev,cash,equity))
                    or multiple.unit!='x' or multiple.currency is not None
                    or shares.unit!='million' or shares.currency is not None
                    or target.unit!='per_share' or target.currency!='INR'):
                raise CorpusError('The printed valuation table has incompatible monetary/share dimensions.')
            numbers=tuple(Decimal(fact.value_text) for fact in inputs)
            if any(not value.is_finite() or value<=0 for value in numbers):
                raise CorpusError('Printed valuation operands must be positive finite source numbers.')
            earnings,ratio,printed_ev,printed_cash,printed_equity,printed_shares,printed_target=numbers
            with localcontext(Context(prec=max(50,sum(len(number.as_tuple().digits) for number in numbers)+16))):
                computed_ev=earnings*ratio
                computed_per_share=printed_equity/printed_shares
                values={'company':company,'period':period,'scope':None,'currency':'INR','unit':'million',
                    'valuation_ebitda':earnings,'valuation_multiple':ratio,'computed_ev':computed_ev,
                    'printed_ev':printed_ev,'ev_difference':_subtract_exact(computed_ev,printed_ev),
                    'printed_cash':printed_cash,'printed_equity':printed_equity,'printed_shares_millions':printed_shares,
                    'computed_per_share':computed_per_share,'printed_target':printed_target,
                    'per_share_difference':_subtract_exact(computed_per_share,printed_target),
                    'source_consistency':'unreconciled_printed_values' if computed_ev!=printed_ev or computed_per_share!=printed_target else 'arithmetic_matches',
                    'calculation_basis':'same printed valuation table; accounting scope unspecified',
                    'fact_ids':tuple(fact.fact_id for fact in (*inputs,bridge)),
                    'evidence_refs':tuple(dict.fromkeys(ref for fact in (*inputs,bridge) for ref in fact.evidence_refs))}
            claim=Claim('source_table_reconciliation',values,values['evidence_refs'])
            return Answer('answered',(claim,),citations=self._citations((claim,)),execution=self._execution())
        except (CorpusError,ValueError,TypeError,ArithmeticError) as error:
            return self.refuse(str(error),used=True)
