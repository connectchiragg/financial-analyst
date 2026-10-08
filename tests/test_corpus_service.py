"""Canonical tool behavior with synthetic reviewed records and actual PDF lookup."""
from dataclasses import replace
import copy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from financial_analyst.corpus_adapter import CorpusError, FactFilter, ReviewedCorpusAdapter
from financial_analyst.corpus_service import CorpusService
from tests.test_corpus_adapter import create_corpus
from tests.support import _write_pdf


class ReviewedMemoryAdapter:
    mode = 'test_double'
    seed_provenance = ('synthetic_source_records',)
    def __init__(self, validator, facts):
        self.validator, self.facts = validator, facts
    def read_facts(self, filters):
        return tuple(fact for fact in self.facts if
            (not filters.companies or fact.company in filters.companies)
            and (not filters.metrics or fact.metric in filters.metrics)
            and all(getattr(filters,key) is None or getattr(fact,key)==getattr(filters,key)
                    for key in ('period','scope','kind','currency','unit','source_sha256')))
    def validate_fact(self, fact):
        if fact not in self.facts:
            raise CorpusError('Unknown synthetic record.')
    def resolve(self, ref):
        return self.validator.resolve(ref)


class CorpusServiceTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        pack,catalog,_,_ = create_corpus(Path(self.directory.name))
        self.validator = ReviewedCorpusAdapter(pack,catalog)
        actual,quote = self.validator.read_facts()
        self.actual = replace(actual,value_text='120')
        self.estimate = replace(actual,fact_id='estimate',value_text='100',kind='broker_estimate')
        self.prior = replace(actual,fact_id='prior',value_text='80',period='1QFY26')
        self.quote = quote
        self.adapter = ReviewedMemoryAdapter(self.validator,(self.actual,self.estimate,self.prior,self.quote))
        self.service = CorpusService(self.adapter)

    def test_actual_source_lookup_returns_unchanged_quote_with_citation_closure(self):
        service = CorpusService(self.validator)
        answer = service.lookup(FactFilter(companies=('Example Pharma',),period='1QFY27'),('example-growth',))
        self.assertEqual(answer.status,'answered')
        self.assertEqual(answer.claims[0].values['quote'],self.validator.resolve('growth').excerpt)
        self.assertEqual(set(answer.claims[0].evidence_refs),{citation.ref for citation in answer.citations})
        self.assertIsNone(answer.claims[0].values['scope'])
        self.assertEqual(answer.execution['analytics'],'reviewed_corpus_fixture')

    def test_lookup_cannot_select_unknown_filtered_or_duplicate_ids(self):
        for ids in (('unknown',),('example-growth','unknown'),('example-growth','example-growth')):
            answer = self.service.lookup(FactFilter(companies=('Example Pharma',)),ids)
            self.assertEqual(answer.status,'refused')
            self.assertEqual(answer.claims,())
        answer = self.service.lookup(FactFilter(period='1QFY26'),('example-growth',))
        self.assertEqual(answer.status,'refused')

    def test_compare_and_yoy_compute_values_without_model_output(self):
        answer = self.service.compare('Example Pharma','1QFY27',include_yoy=True)
        self.assertEqual(answer.status,'answered')
        self.assertEqual(answer.claims[0].values['delta_millions'],Decimal('20'))
        self.assertEqual(answer.claims[0].values['variance_percent'],Decimal('20'))
        self.assertEqual(answer.claims[1].values['relative_percent'],Decimal('50'))
        refs={ref for claim in answer.claims for ref in claim.evidence_refs}
        self.assertEqual(refs,{citation.ref for citation in answer.citations})
        self.assertTrue(answer.execution['no_llm'])
        self.assertEqual(answer.execution['analytics'],'test_double')

    def test_complete_question_refuses_if_yoy_input_is_missing(self):
        self.adapter.facts = (self.actual,self.estimate)
        answer = self.service.compare('Example Pharma','1QFY27',include_yoy=True)
        self.assertEqual(answer.status,'refused')
        self.assertEqual(answer.claims,())

    def test_source_statement_only_numeric_record_cannot_enter_calculation(self):
        statement = replace(self.actual,review={**self.actual.review,'capabilities':['source_statement']})
        self.adapter.facts=(statement,self.estimate,self.prior)
        self.assertEqual(self.service.compare('Example Pharma','1QFY27').status,'refused')
        lookup=self.service.lookup(FactFilter(companies=('Example Pharma',)),(statement.fact_id,))
        self.assertEqual(lookup.status,'answered')

    def test_extra_source_quote_does_not_replace_analytical_authority(self):
        statement = replace(self.actual,fact_id='rounded-source',value_text='119.9',
            review={**self.actual.review,'capabilities':['source_statement']})
        self.adapter.facts=(*self.adapter.facts,statement)
        answer=self.service.compare('Example Pharma','1QFY27')
        self.assertEqual(answer.status,'answered')
        self.assertEqual(answer.claims[0].values['actual_millions'],Decimal('120'))

    def test_conflicting_analytical_inputs_refuse_instead_of_choosing_one(self):
        other=replace(self.actual,fact_id='other-primary',value_text='121')
        self.adapter.facts=(*self.adapter.facts,other)
        self.assertEqual(self.service.compare('Example Pharma','1QFY27').status,'refused')

    def test_partial_company_lookup_or_ranking_cannot_render_subset(self):
        answer=self.service.lookup(FactFilter(companies=('Example Pharma','Missing Co')),('example-growth',))
        self.assertEqual(answer.status,'refused')
        answer=self.service.rank(('Example Pharma','Missing Co'),'1QFY27')
        self.assertEqual(answer.status,'refused')
        self.assertEqual(answer.claims,())

    def test_adapter_returning_wrong_context_or_unapproved_fact_is_refused(self):
        self.adapter.read_facts=lambda filters:(replace(self.actual,company='Wrong Co'),)
        self.assertEqual(self.service.compare('Example Pharma','1QFY27').status,'refused')
        self.adapter.read_facts=lambda filters:(replace(self.actual,review={**self.actual.review,'status':'source_verified'}),)
        self.assertEqual(self.service.lookup(FactFilter(),('example-revenue',)).status,'refused')


def create_valuation_corpus(directory):
    """An actual synthetic PDF with one fiscal column and no accounting scope."""
    import pdfplumber
    directory=Path(directory)
    pdf_path=directory/'synthetic-valuation.pdf'
    roles=(
        ('valuation_ebitda','Valuation EBITDA (INRm)','100','INR','million','INRm'),
        ('valuation_multiple','Valuation multiple (x)','6.5',None,'x','x'),
        ('target_ev','Target EV (INRm)','651','INR','million','INRm'),
        ('cash_surplus','Cash surplus (INRm)','149','INR','million','INRm'),
        ('equity_value','Equity value (INRm)','800','INR','million','INRm'),
        ('shares_outstanding','Shares outstanding (m)','2',None,'million','m'),
        ('target_price','Target price (INR/share)','390','INR','per_share','INR/share'),
    )
    lines=[(40,760,'Example Minerals'),(40,740,'Example Minerals valuation bridge'),(376.10,720,'FY28')]
    for index,(_,label,value,_,_,_) in enumerate(roles):
        y=690-index*20
        # Helvetica's numeric glyph width establishes a right-aligned column.
        width=len(value)*5.56-value.count('.')*2.78
        lines.extend(((40,y,label),(400-width,y,value)))
    _write_pdf(pdf_path,[lines])
    digest=hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    with pdfplumber.open(pdf_path) as pdf:
        words=pdf.pages[0].extract_words()
        table_quote=pdf.pages[0].extract_text()
    evidence=[]
    def box(ref,selected):
        evidence.append({'ref':ref,'source_sha256':digest,'page':1,
            'excerpt':' '.join(word['text']for word in selected),
            'bbox_pt':[min(word['x0']for word in selected),min(word['top']for word in selected),
                       max(word['x1']for word in selected),max(word['bottom']for word in selected)]})
    box('valuation-company',[word for word in words if abs(word['top']-(800-760-7.93))<.1])
    box('valuation-column',[word for word in words if word['text']=='FY28'])
    evidence.append({'ref':'valuation-table','source_sha256':digest,'page':1,'excerpt':table_quote})
    review={'provenance':'reviewed_proof_pack','status':'reviewed','reviewer':'synthetic-source-review',
            'revision':'valuation-test-1','capabilities':['source_statement'],'promotion':True}
    facts=[]
    for index,(metric,label,value,currency,unit,raw_unit) in enumerate(roles):
        row=[word for word in words if abs(word['top']-(800-(690-index*20)-7.93))<.1]
        metric_ref,value_ref=f'valuation-metric-{index}',f'valuation-value-{index}'
        box(metric_ref,[word for word in row if word['x0']<250])
        box(value_ref,[word for word in row if word['x0']>250])
        refs=['valuation-company','valuation-column','valuation-table',metric_ref,value_ref]
        facts.append({'fact_id':'valuation-'+metric,'source_sha256':digest,'company':'Example Minerals',
            'period':'FY28','scope':None,'metric':metric,'kind':None,'value_text':value,'text_value':None,
            'currency':currency,'unit':unit,'raw_labels':{'value':value,'metric':label,'period':'FY28',
                'scope':None,'kind':None,'unit':raw_unit},'evidence_refs':refs,
            'proof':{'method':'table_cell','value_ref':value_ref,'metric_ref':metric_ref,
                'column_refs':['valuation-column'],'bindings':{'company':'valuation-company',
                    'period':'valuation-column','metric':metric_ref,'unit':metric_ref,'kind':'valuation-table'}},
            'review':copy.deepcopy(review)})
    facts.append({'fact_id':'valuation-bridge','source_sha256':digest,'company':'Example Minerals',
        'period':'FY28','scope':None,'metric':'valuation_bridge','kind':None,'value_text':None,
        'text_value':table_quote,'currency':None,'unit':None,'raw_labels':{'metric':'valuation bridge','period':'FY28'},
        'evidence_refs':['valuation-company','valuation-column','valuation-table'],
        'proof':{'method':'quoted_statement','value_ref':'valuation-table','bindings':{
            'company':'valuation-company','period':'valuation-column','metric':'valuation-table'}},
        'review':copy.deepcopy(review)})
    source={'sha256':digest,'document_id':'synthetic-valuation','document_name':pdf_path.name,
        'url':'https://example.test/synthetic-valuation.pdf','company':'Example Minerals',
        'agency':'Synthetic Research','aliases':['Example Minerals']}
    pack_path,catalog_path=directory/'valuation-proof.json',directory/'valuation-catalog.json'
    pack_path.write_text(json.dumps({'format_version':1,'sources':[source],'evidence':evidence,'facts':facts}))
    catalog_path.write_text(json.dumps({'documents':[{**source,'local_path':str(pdf_path)}]}))
    return ReviewedCorpusAdapter(pack_path,catalog_path)


class ValuationServiceTests(unittest.TestCase):
    def setUp(self):
        directory=TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.validator=create_valuation_corpus(directory.name)
        self.facts=self.validator.read_facts()
        self.adapter=ReviewedMemoryAdapter(self.validator,self.facts)
        self.service=CorpusService(self.adapter)

    def refuse(self):
        answer=self.service.valuation('Example Minerals','FY28')
        self.assertEqual(answer.status,'refused')
        self.assertEqual(answer.claims,())
        return answer

    def change_role(self,metric,**changes):
        self.adapter.facts=tuple(replace(fact,**changes)if fact.metric==metric else fact for fact in self.facts)

    def test_original_pdf_cells_preserve_printed_conflicts_and_unknown_scope(self):
        answer=CorpusService(self.validator).valuation('Example Minerals','FY28')
        self.assertEqual(answer.status,'answered')
        values=answer.claims[0].values
        self.assertEqual(values['computed_ev'],Decimal('100')*Decimal('6.5'))
        self.assertEqual(values['printed_ev'],Decimal('651'))
        self.assertEqual(values['ev_difference'],Decimal('-1'))
        self.assertEqual(values['printed_equity'],Decimal('800'))
        self.assertEqual(values['printed_shares_millions'],Decimal('2'))
        self.assertEqual(values['computed_per_share'],Decimal('800')/Decimal('2'))
        self.assertEqual(values['printed_target'],Decimal('390'))
        self.assertEqual(values['per_share_difference'],Decimal('10'))
        self.assertEqual(values['source_consistency'],'unreconciled_printed_values')
        self.assertIsNone(values['scope'])
        self.assertIn('accounting scope unspecified',values['calculation_basis'])
        self.assertEqual(set(answer.claims[0].evidence_refs),{citation.ref for citation in answer.citations})
        self.assertTrue(answer.execution['no_llm'])
        self.assertEqual(self.validator.read_facts(),self.facts)
        self.assertTrue(all(fact.capabilities==('source_statement',)for fact in self.facts))

    def test_missing_duplicate_or_wrong_period_role_cannot_enter_arithmetic(self):
        for case in ('missing','duplicate','period'):
            with self.subTest(case=case):
                role=next(fact for fact in self.facts if fact.metric=='equity_value')
                if case=='missing':self.adapter.facts=tuple(fact for fact in self.facts if fact!=role)
                elif case=='duplicate':self.adapter.facts=(*self.facts,replace(role,fact_id='duplicate-equity'))
                else:self.change_role('equity_value',period='FY29')
                self.refuse()

    def test_wrong_table_column_or_raw_row_refuses_even_with_reviewed_metadata(self):
        role=next(fact for fact in self.facts if fact.metric=='equity_value')
        for case in ('table','column','value_ref','raw_metric','raw_unit'):
            with self.subTest(case=case):
                proof=copy.deepcopy(role.proof);raw=copy.deepcopy(role.raw_labels)
                if case=='table':proof['bindings']['kind']='valuation-column'
                elif case=='column':proof['column_refs']=['valuation-company']
                elif case=='value_ref':proof['value_ref']=proof['metric_ref']
                elif case=='raw_metric':raw['metric']='Different table equity'
                else:raw['unit']='USDm'
                self.change_role('equity_value',proof=proof,raw_labels=raw)
                self.refuse()

    def test_wrong_units_zero_denominator_or_nonfinite_numbers_refuse_without_claims(self):
        for changes in ({'unit':'billion'},{'currency':'INR'},{'value_text':'0'},
                        {'value_text':'NaN'},{'value_text':'Infinity'},{'value_text':'-1'}):
            with self.subTest(changes=changes):
                self.change_role('shares_outstanding',**changes)
                self.refuse()

    def test_unapproved_operands_or_bridge_never_gain_math_authority(self):
        for metric in ('valuation_ebitda','valuation_bridge'):
            with self.subTest(metric=metric):
                role=next(fact for fact in self.facts if fact.metric==metric)
                review={**role.review,'status':'source_verified','provenance':'source_verified_proof_pack',
                        'promotion':False,'capabilities':[]}
                self.change_role(metric,review=review)
                self.refuse()
                self.assertEqual(review['capabilities'],[])


if __name__=='__main__':
    unittest.main()
