"""Whole-question, source and tool boundaries with synthetic inference."""
from dataclasses import replace
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from financial_analyst.corpus_adapter import CorpusError, ReviewedCorpusAdapter
from financial_analyst.corpus_agent import CorpusToolAgent, _question_context
from financial_analyst.corpus_service import CorpusService
from tests.test_agent import FakeInference
from tests.test_corpus_adapter import create_corpus, create_grouped_corpus
from tests.test_corpus_service import ReviewedMemoryAdapter
from tests.support import _write_pdf


def request(tool='lookup',**changes):
    return dict(tool=tool,companies=['Example Pharma'],period='1QFY27',metrics=['net_sales'],
                sector=None,include_yoy=False,scope=None,kind=None,currency=None,unit=None,**changes)


def plan(item=None):
    return {'requests':[item or request()], 'unsupported_parts':[]}


def decision(ids=('example-revenue',)):
    return {'complete':True,'unsupported_parts':[],
            'selections':[{'call_index':0,'fact_ids':list(ids)}]}


class CorpusAgentTests(unittest.TestCase):
    def setUp(self):
        self.directory=TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        pack,catalog,_,self.pdf=create_corpus(Path(self.directory.name))
        self.adapter=ReviewedCorpusAdapter(pack,catalog)
        self.service=CorpusService(self.adapter)

    def run_agent(self,*responses,question='What was Example Pharma revenue in 1QFY27?'):
        inference=FakeInference(*responses)
        answer=CorpusToolAgent(inference,self.service).answer(question)
        return answer,inference

    def assert_refused(self,answer):
        self.assertEqual(answer.status,'refused')
        self.assertEqual(answer.claims,())

    def test_graph_selects_known_source_fact_and_authenticates_citations(self):
        answer,inference=self.run_agent(plan(),decision())
        self.assertEqual(answer.status,'answered')
        self.assertEqual(answer.claims[0].values['value_text'],self.adapter.read_facts()[0].value_text)
        self.assertEqual({ref for claim in answer.claims for ref in claim.evidence_refs},
                         {citation.ref for citation in answer.citations})
        self.assertEqual(len(inference.calls),2)
        self.assertEqual(answer.execution['llm']['mode'],'test_double')
        self.assertEqual(answer.execution['agent']['local_tool_calls'],1)

    def test_runtime_rejects_unknown_metric_and_filter_despite_provider_schema(self):
        for changes in ({'metrics':['invented']},{'period':'2QFY99'},{'scope':'wrong'},
                        {'kind':'forecast'},{'currency':'USD'},{'unit':'billion'},{'companies':['Missing']},
                        {'include_yoy':'yes'},{'metrics':['net_sales','net_sales']}):
            item=request();item.update(changes)
            with self.subTest(changes=changes):
                answer,inference=self.run_agent(plan(item))
                self.assert_refused(answer)
                self.assertEqual(len(inference.calls),1)
                self.assertEqual(answer.execution['agent']['local_tool_calls'],0)

    def test_explicit_period_cannot_be_ignored_by_null_filter(self):
        item=request();item['period']=None
        answer,_=self.run_agent(plan(item),decision(),question='What was Example Pharma revenue in Q2 FY27?')
        self.assert_refused(answer)

    def test_calendar_year_ranges_never_call_provider_or_infer_fiscal_mapping(self):
        for time in ('2024-2025','2024–25','2024/2025','2024 to 2025','2024 through 2025','2024'):
            with self.subTest(time=time):
                answer,inference=self.run_agent(plan(),decision(),
                    question=f'What was the revenue for GSK in {time}?')
                self.assert_refused(answer)
                self.assertEqual(inference.calls,[])
                self.assertIn('calendar-to-fiscal mappings',answer.reason)
        context=_question_context('Compare 1QFY2027 versus Q1 FY2026, and FY2028 forecasts')
        self.assertFalse(context['calendar'])
        self.assertEqual(context['periods'],{'1QFY2027','1QFY2026','FY2028'})

    def test_approving_models_cannot_override_explicit_annual_scope_currency_or_status(self):
        questions=('What was Example Pharma standalone revenue in 1QFY27?',
            'What is Example Pharma revenue forecast in 1QFY27?',
            'What was Example Pharma annual revenue in FY28?',
            'What was Example Pharma annual revenue?',
            'What was Example Pharma revenue in USD for 1QFY27?')
        for question in questions:
            with self.subTest(question=question):
                answer,inference=self.run_agent(plan(),decision(),question=question)
                self.assert_refused(answer)
                self.assertLessEqual(len(inference.calls),2)

    def test_explicit_supported_scope_and_currency_are_filtered_before_lookup(self):
        with patch.object(self.adapter,'read_facts',wraps=self.adapter.read_facts) as reads:
            answer,_=self.run_agent(plan(),decision(),
                question='What was Example Pharma consolidated revenue in INR for 1QFY27?')
        self.assertEqual(answer.status,'answered')
        filtered=[call.args[0] for call in reads.call_args_list if call.args and call.args[0].companies]
        self.assertTrue(filtered)
        self.assertTrue(all(item.scope=='consolidated' and item.currency=='INR' for item in filtered))

    def test_explicit_actual_status_is_filtered_before_lookup(self):
        with patch.object(self.adapter,'read_facts',wraps=self.adapter.read_facts) as reads:
            answer,_=self.run_agent(plan(),decision(),
                question='What was Example Pharma reported revenue in 1QFY27?')
        self.assertEqual(answer.status,'answered')
        filtered=[call.args[0] for call in reads.call_args_list if call.args and call.args[0].companies]
        self.assertTrue(filtered)
        self.assertTrue(all(item.kind=='reported_actual' for item in filtered))
        self.assertEqual(answer.claims[0].values['kind'],'reported_actual')

    def test_explicit_estimate_status_cannot_be_overridden_by_actual_plan(self):
        item=request();item['kind']='reported_actual'
        answer,inference=self.run_agent(plan(item),decision(),
            question='What was Example Pharma estimated revenue in 1QFY27?')
        self.assert_refused(answer)
        self.assertEqual(len(inference.calls),1)
        self.assertEqual(answer.execution['agent']['local_tool_calls'],0)

    def test_forecast_lookup_preserves_authenticated_future_quarter_and_status(self):
        pack_path,catalog_path,pack,self.pdf=create_grouped_corpus(Path(self.directory.name),selected_quarter='2QE')
        fact=pack['facts'][0]
        fact.update(period='2QFY27',kind='broker_forecast')
        fact['raw_labels'].update(period=['FY27E','2QE'],kind='2QE')
        fact['proof']['bindings']['kind']='column';fact['proof']['reported_status_refs']=[]
        fact['proof']['forecast_context']={'report_period':'1QFY27','report_period_refs':['status']}
        pack_path.write_text(json.dumps(pack))
        self.adapter=ReviewedCorpusAdapter(pack_path,catalog_path);self.service=CorpusService(self.adapter)
        item=request();item['period']='2QFY27'
        answer,_=self.run_agent(plan(item),decision(),question='Give Example Pharma revenue forecast for 2QFY27.')
        self.assertEqual(answer.status,'answered')
        self.assertEqual(answer.claims[0].values['kind'],'broker_forecast')
        self.assertEqual(answer.claims[0].values['period'],'2QFY27')

    def test_currency_conversion_and_ambiguous_dollars_refuse_before_provider(self):
        for question in ('Convert Example Pharma INR revenue into USD in 1QFY27.',
                'Give Example Pharma revenue in dollars for 1QFY27.'):
            answer,inference=self.run_agent(plan(),decision(),question=question)
            self.assert_refused(answer)
            self.assertEqual(inference.calls,[])

    def test_company_specific_metric_period_catalog_keeps_original_metric_names(self):
        actual,quote=self.adapter.read_facts()
        memory=ReviewedMemoryAdapter(self.adapter,(replace(actual,metric='total_income'),quote))
        memory.sources=self.adapter.sources;memory.companies=self.adapter.companies
        self.service=CorpusService(memory)
        item=request();item['metrics']=['total_income']
        answer,inference=self.run_agent(plan(item),decision(),question='What was Example Pharma total income in 1QFY27?')
        self.assertEqual(answer.status,'answered')
        payload=json.loads(inference.calls[0].messages[1].content)
        available=payload['company_metric_periods']['Example Pharma']
        self.assertEqual(available['total_income'],['1QFY27'])
        self.assertNotIn('net_sales',available)

    def planning_payload_for_synthetic_companies(self,question):
        actual,_=self.adapter.read_facts()
        catalog=(('GSK Pharma','Healthcare'),("Divi's Laboratories",'Healthcare'),
            ('Glenmark Pharma','Healthcare'),('Escorts Kubota','Automobiles'),
            ('Persistent Systems','Technology'),('APL Apollo Tubes','Industrials'))
        facts=tuple(replace(actual,fact_id=f'synthetic-{index}',company=company,
            metric='total_income' if company=='APL Apollo Tubes' else 'net_sales')
            for index,(company,_) in enumerate(catalog))
        memory=ReviewedMemoryAdapter(self.adapter,facts)
        memory.sources=tuple({'company':company,'aliases':[], 'sector':sector,'report_period':'1QFY27','agency':'Synthetic'}
                             for company,sector in catalog)
        memory.companies=lambda sector=None:tuple(company for company,value in catalog if sector is None or value==sector)
        self.service=CorpusService(memory)
        answer,inference=self.run_agent({'requests':[],'unsupported_parts':['metadata inspection only']},question=question)
        self.assert_refused(answer)
        self.assertEqual(answer.execution['agent']['local_tool_calls'],0)
        return json.loads(inference.calls[0].messages[1].content),inference.calls[0].schema

    def test_explicit_unique_short_company_narrows_planning_details_only(self):
        for name,canonical in (('GSK','GSK Pharma'),('Persistent','Persistent Systems'),('Escorts','Escorts Kubota')):
            with self.subTest(name=name):
                payload,schema=self.planning_payload_for_synthetic_companies(f'What was {name} revenue in 1QFY27?')
                self.assertEqual(len(payload['company_inventory']),6)
                self.assertEqual({item['company'] for item in payload['sources']},{canonical})
                self.assertEqual(set(payload['company_metric_periods']),{canonical})
                self.assertIn(name,payload['sources'][0]['aliases'])
                metrics=schema['properties']['requests']['items']['properties']['metrics']['items']['enum']
                self.assertNotIn('total_income',metrics)
                self.assertNotIn('value_text',json.dumps(payload))
                self.assertNotIn('1,200',json.dumps(payload))
        payload,_=self.planning_payload_for_synthetic_companies('What was revenue across the corpus?')
        self.assertEqual(len(payload['sources']),6)
        self.assertEqual(len(payload['company_metric_periods']),6)

    def test_rank_question_retains_every_explicit_company_in_planning_catalog(self):
        requested={'GSK Pharma',"Divi's Laboratories",'Glenmark Pharma','Escorts Kubota'}
        payload,_=self.planning_payload_for_synthetic_companies(
            "Rank GSK, Divi's, Glenmark and Escorts by revenue YoY in 1QFY27.")
        self.assertEqual({item['company'] for item in payload['sources']},requested)
        self.assertEqual(set(payload['company_metric_periods']),requested)
        ranking=next(item['request'] for item in payload['canonical_plan_examples'] if item['request']['tool']=='rank')
        self.assertEqual(set(ranking['companies']),requested)
        self.assertFalse(ranking['include_yoy']);self.assertIsNone(ranking['sector']);self.assertIsNone(ranking['kind'])

    def test_explicit_sector_keeps_all_source_members_in_planning_catalog(self):
        payload,_=self.planning_payload_for_synthetic_companies('Summarize Healthcare performance in 1QFY27.')
        healthcare={'GSK Pharma',"Divi's Laboratories",'Glenmark Pharma'}
        self.assertEqual({item['company'] for item in payload['sources']},healthcare)
        self.assertEqual(set(payload['company_metric_periods']),healthcare)
        self.assertEqual(len(payload['company_inventory']),6)

    def test_unknown_or_missing_entity_cannot_substitute_a_reviewed_company(self):
        for question in ('What was Unknown Enterprise revenue in 1QFY27?',
                         'What was revenue in 1QFY27?'):
            answer,inference=self.run_agent(plan(),decision(),question=question)
            self.assert_refused(answer)
            self.assertEqual(inference.calls,[])
            self.assertIn('unknown entity cannot be replaced',answer.reason)

    def test_unsupported_debt_or_ebitda_cannot_return_revenue_only(self):
        for question in ('What were Example Pharma revenue and debt in 1QFY27?',
                         'What were Example Pharma revenue and EBITDA in 1QFY27?'):
            answer,inference=self.run_agent(plan(),decision(),question=question)
            self.assert_refused(answer)
            self.assertEqual(inference.calls,[])

    def test_approving_models_cannot_drop_supported_debt_or_ebitda_family(self):
        actual,quote=self.adapter.read_facts()
        for metric in ('debt','ebitda'):
            other=replace(actual,fact_id='synthetic-'+metric,metric=metric)
            memory=ReviewedMemoryAdapter(self.adapter,(actual,other,quote))
            memory.sources=self.adapter.sources;memory.companies=self.adapter.companies
            self.service=CorpusService(memory)
            answer,inference=self.run_agent(plan(),decision(),
                question=f'What were Example Pharma revenue and {metric} in 1QFY27?')
            self.assert_refused(answer)
            self.assertEqual(len(inference.calls),2)

    def test_supported_revenue_and_ebitda_both_survive_coverage(self):
        actual,quote=self.adapter.read_facts()
        ebitda=replace(actual,fact_id='synthetic-ebitda',metric='ebitda')
        memory=ReviewedMemoryAdapter(self.adapter,(actual,ebitda,quote))
        memory.sources=self.adapter.sources;memory.companies=self.adapter.companies
        self.service=CorpusService(memory)
        item=request();item['metrics']=['net_sales','ebitda']
        answer,_=self.run_agent(plan(item),decision(('example-revenue','synthetic-ebitda')),
            question='What were Example Pharma revenue and EBITDA in 1QFY27?')
        self.assertEqual(answer.status,'answered')
        self.assertEqual({claim.values['metric'] for claim in answer.claims},{'net_sales','ebitda'})

    def test_another_company_debt_cannot_supply_missing_named_company_metric(self):
        actual,quote=self.adapter.read_facts()
        other=replace(actual,fact_id='other-debt',company='Second Company',metric='debt')
        memory=ReviewedMemoryAdapter(self.adapter,(actual,other,quote))
        memory.sources=(*self.adapter.sources,{'company':'Second Company','aliases':[],'sector':'Synthetic','report_period':'1QFY27'})
        memory.companies=lambda sector=None:('Example Pharma','Second Company')
        self.service=CorpusService(memory)
        answer,inference=self.run_agent(plan(),decision(),
            question='Give Example Pharma and Second Company revenue and debt in 1QFY27.')
        self.assert_refused(answer)
        self.assertEqual(inference.calls,[])

    def test_planning_schema_uses_unique_metrics_and_canonical_tool_examples(self):
        actual,quote=self.adapter.read_facts()
        bridge=replace(quote,metric='valuation_bridge')
        memory=ReviewedMemoryAdapter(self.adapter,(actual,bridge))
        memory.sources=self.adapter.sources;memory.companies=self.adapter.companies
        self.service=CorpusService(memory)
        answer,inference=self.run_agent(plan(),decision())
        self.assertEqual(answer.status,'answered')
        properties=inference.calls[0].schema['properties']['requests']['items']['properties']
        metrics=properties['metrics']['items']['enum']
        self.assertEqual(metrics.count('valuation_bridge'),1)
        self.assertEqual(len(metrics),len(set(metrics)))
        self.assertIn('underlying metric',properties['metrics']['description'])
        self.assertIn('MUST be null',properties['sector']['description'])
        self.assertIn('MUST be null',properties['kind']['description'])
        self.assertIn('True only for compare',properties['include_yoy']['description'])
        payload=json.loads(inference.calls[0].messages[1].content)
        example=payload['canonical_plan_examples'][0]['request']
        self.assertEqual(example['tool'],'compare')
        self.assertEqual(example['companies'],['Example Pharma'])
        self.assertEqual(example['metrics'],['net_sales'])
        self.assertIsNone(example['sector']);self.assertIsNone(example['kind'])
        self.assertTrue(example['include_yoy'])
        self.assertNotIn('value_text',json.dumps(payload))
        self.assertTrue(all(call.max_output_tokens==1024 for call in inference.calls))

    def test_provider_errors_keep_only_safe_http_failure_categories(self):
        for stage in ('plan','coverage'):
            for status in (400,429,503):
                with self.subTest(stage=stage,status=status):
                    failure=ValueError(f'HTTP {status}: private provider payload and secret')
                    responses=(failure,) if stage=='plan' else (plan(),failure)
                    answer,inference=self.run_agent(*responses)
                    self.assert_refused(answer)
                    record=answer.execution['agent']['inference_calls'][-1]
                    self.assertEqual(record['stage'],stage)
                    self.assertEqual(record['failure_category'],f'http_{status}')
                    self.assertEqual(record['outcome'],'provider_error')
                    self.assertNotIn('private provider payload',str(answer))
                    self.assertNotIn('secret',str(answer))
                    if status==429:self.assertIn('Retry this question later',answer.reason)
                    if status==400:self.assertIn('structured response (HTTP 400)',answer.reason)

    def test_valuation_without_complete_reviewed_roles_refuses_before_coverage(self):
        item=request();item.update(tool='valuation',metrics=[])
        answer,inference=self.run_agent(plan(item),question='Reconcile the Example Pharma target valuation in 1QFY27.')
        self.assert_refused(answer)
        self.assertEqual(len(inference.calls),1)
        self.assertEqual(answer.execution['agent']['local_tool_calls'],1)

    def test_unknown_empty_or_duplicate_selections_refuse_whole_question(self):
        for ids in ((),('unknown',),('example-revenue','example-revenue')):
            with self.subTest(ids=ids):
                answer,_=self.run_agent(plan(),decision(ids))
                self.assert_refused(answer)

    def test_uncovered_part_cannot_render_partial_supported_claims(self):
        coverage=decision();coverage['unsupported_parts']=['unsupported quarter']
        answer,_=self.run_agent(plan(),coverage)
        self.assert_refused(answer)
        bad=plan();bad['unsupported_parts']=['other company']
        answer,inference=self.run_agent(bad)
        self.assert_refused(answer)
        self.assertEqual(len(inference.calls),1)

    def test_semantic_selection_cannot_drop_a_requested_metric(self):
        actual,quote=self.adapter.read_facts()
        item=request();item['metrics']=[actual.metric,quote.metric]
        answer,_=self.run_agent(plan(item),decision())
        self.assert_refused(answer)

    def test_final_render_reauthenticates_source_after_model_call(self):
        class MutatingInference(FakeInference):
            def infer(inner,req):
                response=super().infer(req)
                if len(inner.calls)==2:
                    self.pdf.write_bytes(self.pdf.read_bytes()+b'tampered')
                return response
        inference=MutatingInference(plan(),decision())
        answer=CorpusToolAgent(inference,self.service).answer('What was Example Pharma revenue?')
        self.assert_refused(answer)

    def test_second_call_failure_and_unfinished_plan_do_not_leak_payload(self):
        answer,_=self.run_agent(plan(),ValueError('private exception'))
        self.assert_refused(answer)
        self.assertNotIn('private exception',str(answer))
        answer,_=self.run_agent('not json')
        self.assert_refused(answer)

    def test_graph_keeps_fresh_state_when_reused(self):
        inference=FakeInference(plan(),decision(),{'requests':[],'unsupported_parts':['missing']})
        agent=CorpusToolAgent(inference,self.service)
        self.assertEqual(agent.answer('What was Example Pharma revenue?').status,'answered')
        second=agent.answer('Unsupported Example Pharma question')
        self.assert_refused(second)
        self.assertEqual(second.execution['agent']['local_tool_calls'],0)
        self.assertEqual(len(second.execution['agent']['inference_calls']),1)

    def test_calculation_retains_canonical_values_not_model_supplied_values(self):
        actual,quote=self.adapter.read_facts()
        actual=replace(actual,value_text='120')
        estimate=replace(actual,fact_id='estimate',value_text='100',kind='broker_estimate')
        prior=replace(actual,fact_id='prior',value_text='80',period='1QFY26')
        memory=ReviewedMemoryAdapter(self.adapter,(actual,estimate,prior,quote))
        memory.sources=self.adapter.sources
        memory.companies=self.adapter.companies
        self.service=CorpusService(memory)
        item=request();item.update(tool='compare',include_yoy=True)
        answer,_=self.run_agent(plan(item),decision(()))
        self.assertEqual(answer.status,'answered')
        self.assertEqual(str(answer.claims[0].values['delta_millions']),'20')
        self.assertEqual(answer.claims[1].values['relative_percent'],Decimal('50'))

    def test_calculation_passes_explicit_source_context_to_canonical_service(self):
        actual,quote=self.adapter.read_facts()
        actual=replace(actual,value_text='120')
        estimate=replace(actual,fact_id='estimate',value_text='100',kind='broker_estimate')
        memory=ReviewedMemoryAdapter(self.adapter,(actual,estimate,quote))
        memory.sources=self.adapter.sources;memory.companies=self.adapter.companies
        self.service=CorpusService(memory)
        item=request();item.update(tool='compare',scope='consolidated',currency='INR',unit='million')
        with patch.object(self.service,'compare',wraps=self.service.compare) as compare:
            answer,_=self.run_agent(plan(item),decision(()),
                question='Compare Example Pharma consolidated actual revenue with the estimate in INR for 1QFY27.')
        self.assertEqual(answer.status,'answered')
        self.assertEqual(compare.call_args.kwargs,{'include_yoy':False,'scope':'consolidated','currency':'INR','unit':'million'})

    def test_coverage_reads_full_authentic_quotation_including_late_condition(self):
        import textwrap
        root=Path(self.directory.name)
        pack_path=root/'proof-pack.json';catalog_path=root/'source-catalog.json'
        pack=json.loads(pack_path.read_text());catalog=json.loads(catalog_path.read_text())
        quote=('Portfolio execution supported revenue growth. '+
            'Supply execution supported existing medicines. '*9+
            'However, the forecast depends on easing geopolitical disruption in 4Q and is not assured.')
        lines=textwrap.wrap(quote,width=70)
        _write_pdf(self.pdf,[[(40,760,'Example Pharma'),(40,720,'1QFY27 consolidated reported_actual'),
            (40,700,'Net Sales'),(312.22,700,'1,200'),(300,730,'1QFY27'),(450,730,'(INRm)'),
            *[(40,670-index*12,line) for index,line in enumerate(lines)],
            (40,400,'Target price INR380; CMP INR369; rating Neutral.'),(40,370,'Gross margin 27.2% in 1QFY27.')]])
        canonical='\n'.join(lines)
        digest=hashlib.sha256(self.pdf.read_bytes()).hexdigest()
        pack['sources'][0]['sha256']=digest;catalog['documents'][0]['sha256']=digest
        for evidence in pack['evidence']:
            evidence['source_sha256']=digest
            if evidence['ref']=='growth':evidence['excerpt']=canonical
        for fact in pack['facts']:
            fact['source_sha256']=digest
            if fact['fact_id']=='example-growth':fact['text_value']=canonical
        pack_path.write_text(json.dumps(pack));catalog_path.write_text(json.dumps(catalog))
        self.adapter=ReviewedCorpusAdapter(pack_path,catalog_path);self.service=CorpusService(self.adapter)
        item=request();item['metrics']=['revenue_growth_reason']
        answer,inference=self.run_agent(plan(item),decision(('example-growth',)),
            question='Explain Example Pharma revenue growth and retain every stated condition.')
        self.assertEqual(answer.status,'answered')
        payload=json.loads(inference.calls[1].messages[1].content)
        values=payload['canonical_tools'][0]['claims'][0]['values']
        self.assertEqual(values['quote'],canonical)
        self.assertNotIn('quote_preview',values)
        self.assertIn('is not assured.',values['quote'][320:])
        self.assertEqual(answer.claims[0].values['quote'],canonical)

    def test_empty_or_long_questions_never_call_inference(self):
        for question in ('',' '*3,'x'*2001):
            answer,inference=self.run_agent(question=question)
            self.assert_refused(answer)
            self.assertEqual(inference.calls,[])


if __name__=='__main__':unittest.main()
