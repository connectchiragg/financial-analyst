"""Synthetic source proof packs and real temporary reviewed-corpus SQLite."""

from contextlib import closing
from dataclasses import asdict, replace
import copy
from decimal import Decimal, Inexact, localcontext
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import financial_analyst.corpus_adapter as corpus_module
from financial_analyst.corpus_adapter import (CorpusError, FactFilter, ReviewedCorpusAdapter,
    SQLiteCorpusAdapter, import_reviewed)
from tests.support import _write_pdf


def create_corpus(directory,margin_text='Gross margin 27.2% in 1QFY27.',extra_pages=None):
    import pdfplumber
    directory = Path(directory)
    path = directory / 'synthetic-corpus.pdf'
    _write_pdf(path, [[(40,760,'Example Pharma'), (40,720,'1QFY27 consolidated reported_actual'),
        (40,700,'Net Sales'), (312.22,700,'1,200'), (300,730,'1QFY27'), (450,730,'(INRm)'),
        (40,670,'Portfolio execution supported revenue growth.'),
        (40,640,'Target price INR380; CMP INR369; rating Neutral.'),
        (40,610,margin_text)]] + (extra_pages or []))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    source = {'sha256':digest,'document_id':'synthetic-corpus','document_name':path.name,
        'url':'https://example.test/report.pdf','company':'Example Pharma','agency':'Synthetic Research',
        'sector':'Healthcare','report_period':'1QFY27','aliases':['Example Pharma']}
    catalog = {'documents':[{**source,'local_path':str(path)}]}
    with pdfplumber.open(path) as pdf:
        words = pdf.pages[0].extract_words()
        page_text = pdf.pages[0].extract_text()
    def boxed(ref,text,x,y):
        row = [word for word in words if abs(word['top']-(800-y-7.93))<.1]
        candidates = [row[index:index+len(text.split())] for index in range(len(row))
            if ' '.join(word['text'] for word in row[index:index+len(text.split())])==text]
        selected = min(candidates,key=lambda span:abs(span[0]['x0']-x))
        assert ' '.join(word['text'] for word in selected)==text
        return {'ref':ref,'source_sha256':digest,'page':1,'excerpt':text,
            'bbox_pt':[selected[0]['x0'],selected[0]['top'],selected[-1]['x1'],selected[-1]['bottom']]}
    evidence = [boxed('company','Example Pharma',40,760),boxed('period','1QFY27',40,720),
        boxed('scope','consolidated',85.01,720),boxed('kind','reported_actual',145.6,720),
        boxed('metric','Net Sales',40,700),boxed('amount','1,200',300,700),
        boxed('column','1QFY27',300,730),boxed('unit','(INRm)',450,730)]
    for ref, quote in (('growth','Portfolio execution supported revenue growth.'),
            ('valuation','Target price INR380; CMP INR369; rating Neutral.'),
            ('margin',margin_text)):
        assert quote in page_text
        evidence.append({'ref':ref,'source_sha256':digest,'page':1,'excerpt':quote})
    review = {'provenance':'reviewed_proof_pack','status':'reviewed','reviewer':'synthetic-source-review',
        'revision':'1','capabilities':['source_statement','analytical_context']}
    numeric = {'fact_id':'example-revenue','source_sha256':digest,'company':'Example Pharma',
        'period':'1QFY27','scope':'consolidated','metric':'net_sales','kind':'reported_actual',
        'value_text':'1200.00000000000000000000','text_value':None,'currency':'INR','unit':'million',
        'raw_labels':{'value':'1,200','metric':'Net Sales','period':'1QFY27','scope':'consolidated',
            'unit':'(INRm)','kind':'reported_actual'},'evidence_refs':['company','period','scope','kind','metric','amount','column','unit'],
        'proof':{'method':'table_cell','value_ref':'amount','metric_ref':'metric','column_refs':['column'],
            'bindings':{'company':'company','period':['period','column'],'scope':'scope','kind':'kind','metric':'metric','unit':'unit'}},'review':review}
    qualitative = {'fact_id':'example-growth','source_sha256':digest,'company':'Example Pharma',
        'period':'1QFY27','scope':None,'metric':'revenue_growth_reason','kind':None,'value_text':None,
        'text_value':'Portfolio execution supported revenue growth.','currency':None,'unit':None,
        'raw_labels':{'metric':'revenue growth','period':'1QFY27'},'evidence_refs':['company','period','growth'],
        'proof':{'method':'quoted_statement','value_ref':'growth','bindings':{'company':'company','period':'period','metric':'growth'}},
        'review':{**review,'capabilities':['source_statement']}}
    pack={'format_version':1,'sources':[source],'evidence':evidence,'facts':[numeric,qualitative]}
    pack_path, catalog_path = directory/'proof-pack.json', directory/'source-catalog.json'
    pack_path.write_text(json.dumps(pack),encoding='utf-8')
    catalog_path.write_text(json.dumps(catalog),encoding='utf-8')
    return pack_path, catalog_path, pack, path


def create_grouped_corpus(directory,selected_quarter='1Q'):
    import pdfplumber
    pack_path,catalog_path,pack,path=create_corpus(directory)
    _write_pdf(path,[[(40,760,'Example Pharma'),(200,750,'FY26'),(300,750,'FY27E'),
        (200,730,'1Q'),(300,730,selected_quarter),(450,730,'(INRm)'),(40,720,'consolidated'),
        (40,700,'Net Sales'),(200,700,'80'),(288.32 if selected_quarter=='1Q' else 300,700,'1,200'),
        (40,670,'Example Pharma reported 1QFY27 revenue.')]])
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    pack['sources'][0]['sha256']=digest
    with pdfplumber.open(path) as pdf:words=pdf.pages[0].extract_words()
    def box(ref,text,x,y):
        row=[word for word in words if abs(word['top']-(800-y-7.93))<.1]
        matches=[row[index:index+len(text.split())] for index in range(len(row))
            if ' '.join(word['text'] for word in row[index:index+len(text.split())])==text]
        selected=min(matches,key=lambda span:abs(span[0]['x0']-x))
        return {'ref':ref,'source_sha256':digest,'page':1,'excerpt':text,
            'bbox_pt':[selected[0]['x0'],selected[0]['top'],selected[-1]['x1'],selected[-1]['bottom']]}
    pack['evidence']=[box('company','Example Pharma',40,760),box('prior_year','FY26',200,750),
        box('year','FY27E',300,750),box('prior_column','1Q',200,730),box('column',selected_quarter,300,730),
        box('scope','consolidated',40,720),box('metric','Net Sales',40,700),box('amount','1,200',300,700),
        box('prior_amount','80',200,700),box('unit','(INRm)',450,730),
        {'ref':'status','source_sha256':digest,'page':1,'excerpt':'Example Pharma reported 1QFY27 revenue.'}]
    numeric=pack['facts'][0]
    numeric['source_sha256']=digest
    numeric['raw_labels'].update(period=['FY27E','1Q'],kind='reported')
    numeric['evidence_refs']=[item['ref'] for item in pack['evidence']]
    numeric['proof'].update(column_refs=['year','column'],reported_status_refs=['status'],
        column_group={'header_ref':'year','column_refs':['column'],'x_range':[285,350]})
    numeric['proof']['bindings'].update(period=['year','column'],kind='status')
    pack['facts']=[numeric]
    pack_path.write_text(json.dumps(pack),encoding='utf-8')
    catalog_path.write_text(json.dumps({'documents':[{**pack['sources'][0],'local_path':str(path)}]}),encoding='utf-8')
    return pack_path,catalog_path,pack,path


class CorpusAdapterTests(unittest.TestCase):
    def setUp(self):
        self.directory=TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name)
        self.pack_path,self.catalog_path,self.pack,self.pdf=create_corpus(self.root)
        self.database=self.root/'corpus.sqlite'

    def write(self,pack):
        self.pack_path.write_text(json.dumps(pack),encoding='utf-8')

    def validator(self):
        return ReviewedCorpusAdapter(self.pack_path,self.catalog_path)

    def ingest(self):
        return import_reviewed(self.pack_path,self.catalog_path,self.database)

    def test_numeric_and_qualitative_facts_preserve_exact_source_and_context(self):
        adapter=self.validator()
        numeric=adapter.read_facts(FactFilter(metrics=('net_sales',)))[0]
        self.assertEqual(numeric.value,Decimal('1200'))
        self.assertEqual(numeric.value_text,'1200.00000000000000000000')
        self.assertIn('analytical_context',numeric.capabilities)
        quoted=adapter.read_facts(FactFilter(metrics=('revenue_growth_reason',)))[0]
        self.assertIsNone(quoted.scope)
        self.assertEqual(quoted.text_value,adapter.resolve('growth').excerpt)
        self.assertEqual(quoted.capabilities,('source_statement',))

    def test_exact_tuple_filters_do_not_cross_metric_scope_status_or_period(self):
        adapter=self.validator()
        self.assertEqual(len(adapter.read_facts(FactFilter(companies=('Example Pharma',),period='1QFY27',metrics=('net_sales',),scope='consolidated',kind='reported_actual',currency='INR',unit='million'))),1)
        for filters in (FactFilter(period='1QFY26'),FactFilter(scope='standalone'),FactFilter(kind='broker_forecast'),FactFilter(currency='USD'),FactFilter(companies=("Example Pharma' OR 1=1 --",))):
            self.assertEqual(adapter.read_facts(filters),())

    def test_source_verified_pack_remains_exploration_only_and_cannot_create_database(self):
        for fact in self.pack['facts']:
            fact['review']={'provenance':'source_verified_proof_pack','status':'source_verified',
                'reviewer':'synthetic-QA','revision':'1','promotion':False,'capabilities':[]}
        self.write(self.pack)
        adapter=self.validator()
        self.assertEqual(adapter.read_facts(),())
        self.assertEqual(len(adapter.explore_facts()),2)
        with self.assertRaises(CorpusError):self.ingest()
        self.assertFalse(self.database.exists())

    def test_ambiguous_optional_context_preserves_null_without_analytical_capability(self):
        numeric=self.pack['facts'][0]
        numeric.update(scope=None,kind=None)
        numeric['review']['capabilities']=['source_statement']
        self.write(self.pack)
        fact=self.validator().read_facts(FactFilter(metrics=('net_sales',)))[0]
        self.assertIsNone(fact.scope)
        self.assertIsNone(fact.kind)
        self.assertNotIn('analytical_context',fact.capabilities)

    def test_analytical_context_cannot_be_granted_to_ambiguous_facts(self):
        self.pack['facts'][0]['scope']=None
        self.write(self.pack)
        with self.assertRaises(CorpusError):self.validator()

    def test_source_hash_mutation_before_and_after_validation_is_rejected(self):
        adapter=self.validator()
        self.pdf.write_bytes(b'changed source bytes')
        for operation in (adapter.read_facts,lambda:adapter.resolve('amount'),self.validator,self.ingest):
            with self.assertRaises(CorpusError):operation()
        self.assertFalse(self.database.exists())

    def test_relative_source_paths_resolve_from_catalog_parent_not_working_directory(self):
        catalog=json.loads(self.catalog_path.read_text())
        catalog['documents'][0]['local_path']=self.pdf.name
        self.catalog_path.write_text(json.dumps(catalog))
        previous=Path.cwd()
        unrelated=self.root/'unrelated-working-directory'
        unrelated.mkdir()
        try:
            os.chdir(unrelated)
            adapter=self.validator()
            self.assertEqual(len(adapter.read_facts()),2)
            self.ingest()
            self.assertEqual(len(SQLiteCorpusAdapter(self.database,adapter).read_facts()),2)
        finally:
            os.chdir(previous)

    def test_false_value_quote_company_currency_and_unit_fail_before_database_creation(self):
        changes=[lambda p:p['facts'][0].update(value_text='1201'),lambda p:p['facts'][0].update(value_text=1200.0),
            lambda p:p['facts'][0].update(value_text='NaN'),lambda p:p['facts'][0].update(company='Other Pharma'),
            lambda p:p['facts'][0].update(currency='USD'),lambda p:p['facts'][0].update(unit='billion'),
            lambda p:p['facts'][0].update(scope='standalone'),lambda p:p['facts'][0].update(kind='broker_forecast'),
            lambda p:p['facts'][1].update(text_value='Invented causal explanation')]
        for change in changes:
            with self.subTest(change=change):
                candidate=copy.deepcopy(self.pack);change(candidate);self.write(candidate)
                with self.assertRaises(CorpusError):self.ingest()
                self.assertFalse(self.database.exists())

    def test_wrong_table_row_column_and_missing_cell_proofs_are_rejected(self):
        for field,value in (('metric_ref','company'),('column_refs',['metric']),('value_ref','growth')):
            candidate=copy.deepcopy(self.pack);candidate['facts'][0]['proof'][field]=value;self.write(candidate)
            with self.subTest(field=field),self.assertRaises(CorpusError):self.validator()

    def test_source_proof_review_and_returned_dictionaries_cannot_be_tampered(self):
        adapter=self.validator()
        fact=adapter.read_facts()[0]
        fact.review['capabilities']=[]
        with self.assertRaises(CorpusError):adapter.validate_fact(fact)
        fresh=adapter.read_facts()[0]
        self.assertIn('analytical_context',fresh.capabilities)
        with self.assertRaises(CorpusError):adapter.validate_fact(replace(fresh,value_text='999'))

    def test_batches_authenticate_sources_once_and_preserve_reference_order(self):
        self.ingest()
        validator=self.validator()
        adapters=(validator,SQLiteCorpusAdapter(self.database,validator))
        facts=validator.read_facts()
        for adapter in adapters:
            with self.subTest(mode=adapter.mode):
                with patch.object(validator,'reauthenticate',wraps=validator.reauthenticate) as authenticate:
                    adapter.validate_facts(facts)
                    self.assertEqual(authenticate.call_count,1)
                    authenticate.reset_mock()
                    evidence=adapter.resolve_many(('growth','amount','growth'))
                    self.assertEqual(tuple(item.ref for item in evidence),('growth','amount','growth'))
                    self.assertEqual(authenticate.call_count,1)
                    authenticate.reset_mock()
                    self.assertEqual(len(adapter.read_facts()),2)
                    self.assertEqual(authenticate.call_count,1)

    def test_batch_tampered_member_unknown_reference_and_invalid_inputs_fail_closed(self):
        self.ingest()
        validator=self.validator()
        facts=validator.read_facts()
        tampered=(facts[0],replace(facts[1],text_value='Invented source claim'))
        for adapter in (validator,SQLiteCorpusAdapter(self.database,validator)):
            with self.subTest(mode=adapter.mode):
                with self.assertRaises(CorpusError):adapter.validate_facts(tampered)
                with self.assertRaises(CorpusError):adapter.resolve_many(('amount','unknown-ref'))
                for invalid in ([facts[0]],(facts[0],facts[0]),('not-a-fact',)):
                    with self.assertRaises(CorpusError):adapter.validate_facts(invalid)
                for invalid in (['amount'],('amount',None),('amount','')):
                    with self.assertRaises(CorpusError):adapter.resolve_many(invalid)

    def test_batches_reauthenticate_changed_source_even_when_empty(self):
        self.ingest()
        validator=self.validator()
        adapters=(validator,SQLiteCorpusAdapter(self.database,validator))
        facts=validator.read_facts()
        self.pdf.write_bytes(b'changed after adapter initialization')
        for adapter in adapters:
            for operation in (lambda:adapter.validate_facts(facts),lambda:adapter.validate_facts(()),
                    lambda:adapter.resolve_many(('growth','amount')),lambda:adapter.resolve_many(())):
                with self.subTest(mode=adapter.mode),self.assertRaises(CorpusError):operation()

    def test_sqlite_batch_requires_every_fact_and_evidence_to_remain_canonical(self):
        self.ingest()
        validator=self.validator()
        adapter=SQLiteCorpusAdapter(self.database,validator)
        facts=validator.read_facts()
        original=self.database.read_bytes()
        for query in ("DELETE FROM facts WHERE fact_id='example-growth'",
                "UPDATE facts SET fact_json='{}' WHERE fact_id='example-growth'",
                "UPDATE evidence SET evidence_json='{}' WHERE ref='growth'"):
            with self.subTest(query=query):
                self.database.write_bytes(original)
                with closing(sqlite3.connect(self.database)) as db:db.execute(query);db.commit()
                with self.assertRaises(CorpusError):adapter.validate_facts(facts)
        with self.assertRaises(CorpusError):adapter.resolve_many(('amount','growth'))

    def test_sqlite_batch_uses_one_explicit_read_snapshot(self):
        self.ingest()
        adapter=SQLiteCorpusAdapter(self.database,self.validator())
        statements=[]
        connect=corpus_module._connect
        def traced_connect(path,mode):
            connection=connect(path,mode)
            connection.set_trace_callback(lambda sql:statements.append((sql,connection.in_transaction)))
            return connection
        with patch.object(corpus_module,'_connect',side_effect=traced_connect) as opened:
            adapter.resolve_many(('amount','growth'))
        self.assertEqual(opened.call_count,1)
        self.assertEqual(statements[0][0],'BEGIN')
        self.assertTrue(all(in_transaction for sql,in_transaction in statements if sql.startswith('SELECT')))

    def test_sqlite_batch_keeps_snapshot_during_concurrent_change_then_refuses_next_read(self):
        self.ingest()
        with closing(sqlite3.connect(self.database)) as db:db.execute('PRAGMA journal_mode=WAL')
        adapter=SQLiteCorpusAdapter(self.database,self.validator())
        resolve=adapter._resolve
        def concurrently_changed(connection,ref):
            evidence=resolve(connection,ref)
            if ref=='amount':
                with closing(sqlite3.connect(self.database)) as db:
                    db.execute("UPDATE evidence SET evidence_json='{}' WHERE ref='growth'")
                    db.commit()
            return evidence
        with patch.object(adapter,'_resolve',side_effect=concurrently_changed):
            evidence=adapter.resolve_many(('amount','growth'))
        self.assertEqual(evidence[1].excerpt,'Portfolio execution supported revenue growth.')
        with self.assertRaises(CorpusError):adapter.resolve_many(('amount','growth'))

    def test_idempotent_real_sqlite_roundtrip_keeps_precision_and_schema(self):
        with localcontext() as context:
            context.prec=4;context.traps[Inexact]=True
            result=self.ingest()
            adapter=SQLiteCorpusAdapter(self.database,self.validator())
            self.assertEqual(result.facts_added,2)
            self.assertEqual(adapter.read_facts(FactFilter(metrics=('net_sales',)))[0].value_text,self.pack['facts'][0]['value_text'])
        before=self.database.read_bytes()
        self.assertEqual(asdict(self.ingest()),{'documents_added':0,'evidence_added':0,'facts_added':0})
        self.assertEqual(self.database.read_bytes(),before)
        self.assertEqual(adapter.companies('Healthcare'),('Example Pharma',))

    def test_generic_metric_addition_is_a_row_without_schema_updates(self):
        self.ingest()
        with closing(sqlite3.connect(self.database)) as db:
            schema=db.execute('SELECT name,sql FROM sqlite_master ORDER BY name').fetchall()
        new=copy.deepcopy(self.pack['facts'][1]);new.update(fact_id='new-commentary',metric='arbitrary_topic')
        self.pack['facts'].append(new);self.write(self.pack)
        self.assertEqual(self.ingest().facts_added,1)
        with closing(sqlite3.connect(self.database)) as db:
            self.assertEqual(db.execute('SELECT name,sql FROM sqlite_master ORDER BY name').fetchall(),schema)

    def test_conflicting_fact_rolls_back_earlier_new_fact_and_preserves_bytes(self):
        self.ingest()
        candidate=copy.deepcopy(self.pack)
        new=copy.deepcopy(candidate['facts'][1]);new.update(fact_id='new-topic',metric='new_topic')
        candidate['facts'].insert(0,new)
        candidate['facts'][1]['review']['revision']='2'
        self.write(candidate)
        before=self.database.read_bytes()
        with self.assertRaisesRegex(CorpusError,'Conflicting'):self.ingest()
        self.assertEqual(self.database.read_bytes(),before)

    def test_missing_database_read_and_wrong_existing_schema_never_create_or_overwrite(self):
        with self.assertRaises(CorpusError):SQLiteCorpusAdapter(self.database,self.validator())
        self.assertFalse(self.database.exists())
        with closing(sqlite3.connect(self.database)) as db:
            db.execute('CREATE TABLE unrelated (value TEXT)');db.commit()
        before=self.database.read_bytes()
        with self.assertRaises(CorpusError):self.ingest()
        self.assertEqual(self.database.read_bytes(),before)

    def test_sqlite_value_context_review_and_evidence_tampering_fail_closed(self):
        self.ingest();adapter=SQLiteCorpusAdapter(self.database,self.validator())
        original=self.database.read_bytes()
        changes=("UPDATE facts SET company='Other Pharma'", "UPDATE facts SET fact_id='forged' WHERE fact_id='example-revenue'",
            "UPDATE facts SET fact_json='{}'", "UPDATE evidence SET evidence_json='{}'",
            "UPDATE documents SET metadata_json='{}'")
        for query in changes:
            with self.subTest(query=query):
                self.database.write_bytes(original)
                with closing(sqlite3.connect(self.database)) as db:db.execute(query);db.commit()
                with self.assertRaises(CorpusError):adapter.read_facts()

    def test_sqlite_exact_filters_and_no_arbitrary_sql_are_exposed(self):
        self.ingest();adapter=SQLiteCorpusAdapter(self.database,self.validator())
        self.assertEqual(adapter.read_facts(FactFilter(companies=("Example Pharma' OR 1=1 --",))),())
        self.assertEqual(adapter.read_facts(FactFilter(unit='billion')),())
        self.assertFalse(hasattr(adapter,'execute'))
        with self.assertRaises(CorpusError):adapter.read_facts({'company':'Example Pharma'})

    def test_grouped_fiscal_year_actual_requires_source_reviewed_group_membership(self):
        self.pack_path,self.catalog_path,self.pack,self.pdf=create_grouped_corpus(self.root)
        adapter=self.validator()
        self.assertEqual(adapter.read_facts()[0].period,'1QFY27')
        candidate=copy.deepcopy(self.pack)
        candidate['facts'][0]['proof'].pop('column_group')
        self.write(candidate)
        with self.assertRaisesRegex(CorpusError,'year-group'):self.validator()

    def test_historical_same_quarter_cell_cannot_inherit_current_year_and_status(self):
        self.pack_path,self.catalog_path,self.pack,self.pdf=create_grouped_corpus(self.root)
        candidate=copy.deepcopy(self.pack)
        fact=candidate['facts'][0]
        fact['value_text']='80';fact['raw_labels']['value']='80'
        fact['proof'].update(value_ref='prior_amount',column_refs=['year','prior_column'])
        fact['proof']['bindings']['period']=['year','prior_column']
        self.write(candidate)
        with self.assertRaisesRegex(CorpusError,'year group'):self.validator()
        # Even widening the reviewed band to include the historical column
        # cannot hide a genuine FY26 header at the same physical baseline.
        fact['proof']['column_group'].update(column_refs=['prior_column'],x_range=[180,350])
        self.write(candidate)
        with self.assertRaisesRegex(CorpusError,'different year'):self.validator()

    def test_grouped_current_cell_cannot_inherit_unrelated_historical_year_binding(self):
        self.pack_path,self.catalog_path,self.pack,self.pdf=create_grouped_corpus(self.root)
        fact=self.pack['facts'][0]
        fact.update(period='1QFY26',kind=None)
        fact['raw_labels']['period']=['FY26','1Q']
        fact['proof']['bindings']['period']=['prior_year','year','column']
        fact['proof']['reported_status_refs']=[]
        fact['review']['capabilities']=['source_statement']
        self.write(self.pack)
        with self.assertRaisesRegex(CorpusError,'physical fiscal-year group'):self.validator()

    def test_explicit_quarter_header_cannot_be_relabelled_as_annual_period(self):
        fact=self.pack['facts'][0]
        fact['period']='FY27'
        fact['raw_labels']['period']='FY27'
        self.write(self.pack)
        with self.assertRaisesRegex(CorpusError,'quarter or annual'):self.validator()

    def test_actual_corroboration_cannot_turn_explicit_forecast_into_actual(self):
        self.pack_path,self.catalog_path,self.pack,self.pdf=create_grouped_corpus(self.root)
        self.pack['facts'][0]['kind']='broker_forecast'
        self.write(self.pack)
        with self.assertRaisesRegex(CorpusError,'reported-quarter'):self.validator()

    def test_unknown_kind_and_unit_cannot_gain_analytical_capability(self):
        for field,value in (('kind','invented_status'),('unit','invented_unit')):
            candidate=copy.deepcopy(self.pack)
            candidate['facts'][0][field]=value
            self.write(candidate)
            with self.subTest(field=field),self.assertRaisesRegex(CorpusError,'supported financial'):self.validator()

    def test_report_context_is_preserved_for_qualitative_statements_only(self):
        self.pack['facts'][1]['proof']['period_role']='report_context'
        self.write(self.pack)
        self.assertEqual(self.validator().read_facts(FactFilter(metrics=('revenue_growth_reason',)))[0].proof['period_role'],'report_context')
        for capabilities in (['source_statement'],['source_statement','analytical_context']):
            self.pack['facts'][0]['proof']['period_role']='report_context'
            self.pack['facts'][0]['review']['capabilities']=capabilities
            self.write(self.pack)
            with self.assertRaisesRegex(CorpusError,'Report-context'):self.validator()

    def test_forward_qe_forecast_requires_cited_report_period_and_strict_chronology(self):
        self.pack_path,self.catalog_path,self.pack,self.pdf=create_grouped_corpus(self.root,selected_quarter='2QE')
        fact=self.pack['facts'][0]
        fact.update(period='2QFY27',kind='broker_forecast')
        fact['raw_labels'].update(period=['FY27E','2QE'],kind='2QE')
        fact['proof']['bindings']['kind']='column'
        fact['proof']['reported_status_refs']=[]
        fact['proof']['forecast_context']={'report_period':'1QFY27','report_period_refs':['status']}
        self.write(self.pack)
        self.assertEqual(self.validator().read_facts()[0].kind,'broker_forecast')
        for context in (None,{'report_period':'2QFY27','report_period_refs':['status']},
                {'report_period':'1QFY27','report_period_refs':['prior_year']}):
            candidate=copy.deepcopy(self.pack)
            candidate['facts'][0]['proof']['forecast_context']=context
            self.write(candidate)
            with self.subTest(context=context),self.assertRaises(CorpusError):self.validator()

    def test_current_qe_cannot_be_relabelled_as_forward_forecast(self):
        self.pack_path,self.catalog_path,self.pack,self.pdf=create_grouped_corpus(self.root,selected_quarter='1QE')
        fact=self.pack['facts'][0]
        fact.update(kind='broker_forecast')
        fact['raw_labels'].update(period=['FY27E','1QE'],kind='1QE')
        fact['proof']['bindings']['kind']='column'
        fact['proof']['reported_status_refs']=[]
        fact['proof']['forecast_context']={'report_period':'1QFY27','report_period_refs':['status']}
        self.write(self.pack)
        with self.assertRaisesRegex(CorpusError,'strictly later'):self.validator()
        fact['kind']='broker_estimate'
        self.write(self.pack)
        self.assertEqual(self.validator().read_facts()[0].kind,'broker_estimate')

    def test_malformed_source_reference_review_and_catalog_types_refuse_before_creation(self):
        changes=(lambda p:p['facts'][0].update(source_sha256=[]),
            lambda p:p['facts'][0].update(evidence_refs=[{}]),
            lambda p:p['facts'][0]['review'].update(status=[]),
            lambda p:p['facts'][0]['review'].update(capabilities=[{}]))
        for change in changes:
            candidate=copy.deepcopy(self.pack);change(candidate);self.write(candidate)
            with self.subTest(change=change),self.assertRaises(CorpusError):self.ingest()
            self.assertFalse(self.database.exists())
        self.write(self.pack)
        catalog=json.loads(self.catalog_path.read_text());catalog['documents'][0]['local_path']=None
        self.catalog_path.write_text(json.dumps(catalog))
        with self.assertRaises(CorpusError):self.ingest()
        self.assertFalse(self.database.exists())

    def test_quoted_percentage_numeric_span_is_exact_and_not_a_monetary_amount(self):
        self.pack_path,self.catalog_path,self.pack,self.pdf=create_corpus(self.root,
            margin_text='Gross margin 27.2%, in 1QFY27.')
        numeric=copy.deepcopy(self.pack['facts'][0])
        quote=next(item['excerpt'] for item in self.pack['evidence'] if item['ref']=='margin')
        start=quote.index('27.2%')
        numeric.update(fact_id='margin',metric='gross_margin',scope=None,kind=None,value_text='27.2',
            currency=None,unit='percent',raw_labels={'value':'27.2%','metric':'Gross margin',
                'period':'1QFY27','unit':'%'},evidence_refs=['company','period','margin'],
            proof={'method':'quoted_number','value_ref':'margin','value_span':[start,start+len('27.2%')],
                'bindings':{'company':'company','period':'period','metric':'margin','unit':'margin'}},
            review={**numeric['review'],'capabilities':['source_statement']})
        self.pack['facts'].append(numeric);self.write(self.pack)
        fact=self.validator().read_facts(FactFilter(metrics=('gross_margin',)))[0]
        self.assertIsNone(fact.currency)
        self.assertEqual(fact.unit,'percent')
        self.assertEqual(fact.value,Decimal('27.2'))
        numeric['proof']['value_span'][0]+=1
        self.write(self.pack)
        with self.assertRaises(CorpusError):self.validator()
        # Matching a smaller exact substring still cannot turn 27.2 into 7.2.
        numeric['value_text']='7.2'
        numeric['raw_labels']['value']='7.2%'
        self.write(self.pack)
        with self.assertRaisesRegex(CorpusError,'cannot omit'):self.validator()

    def test_quoted_number_cannot_omit_numeric_comma_continuation(self):
        fact=self.pack['facts'][0]
        fact.update(value_text='1')
        fact['raw_labels']['value']='1'
        fact['proof'].update(method='quoted_number',value_span=[0,1])
        self.write(self.pack)
        with self.assertRaisesRegex(CorpusError,'cannot omit'):self.validator()

    def test_only_evidence_pages_are_extracted_but_all_source_bytes_are_authenticated(self):
        import pdfplumber
        self.pack_path,self.catalog_path,self.pack,self.pdf=create_corpus(self.root,
            extra_pages=[[(40,760,'Unreferenced source page')]])
        digest=self.pack['sources'][0]['sha256']
        calls={'text':[],'words':[]}
        original_text=pdfplumber.page.Page.extract_text
        original_words=pdfplumber.page.Page.extract_words

        def extract_text(page,*args,**kwargs):
            calls['text'].append(page.page_number)
            return original_text(page,*args,**kwargs)

        def extract_words(page,*args,**kwargs):
            calls['words'].append(page.page_number)
            return original_words(page,*args,**kwargs)

        with patch.object(pdfplumber.page.Page,'extract_text',extract_text), \
                patch.object(pdfplumber.page.Page,'extract_words',extract_words):
            adapter=self.validator()
        self.assertEqual(calls,{'text':[1],'words':[1]})
        self.assertEqual(len(adapter._pages[digest]),2)
        unused=adapter._pages[digest][1]
        self.assertEqual((unused['width'],unused['height']),(600,800))
        self.assertEqual(unused['text'],'')
        self.assertEqual(tuple(unused['words']),())
        self.assertEqual(len(adapter.read_facts()),2)

        # A mutation solely in an unparsed page still changes the full-file hash.
        raw=self.pdf.read_bytes()
        self.assertIn(b'Unreferenced source page',raw)
        self.pdf.write_bytes(raw.replace(b'Unreferenced source page',b'Changed unused page data'))
        with self.assertRaisesRegex(CorpusError,'changed after review'):
            adapter.read_facts()
        with self.assertRaisesRegex(CorpusError,'hash differs'):
            self.validator()

    def test_evidence_page_types_are_checked_before_pdf_text_extraction(self):
        import pdfplumber
        for page in (True,False,0,-1,'1',[],None):
            candidate=copy.deepcopy(self.pack)
            candidate['evidence'][0]['page']=page
            self.write(candidate)
            with self.subTest(page=page), \
                    patch.object(pdfplumber.page.Page,'extract_text') as extract, \
                    self.assertRaisesRegex(CorpusError,'unavailable source page'):
                self.validator()
            extract.assert_not_called()


if __name__=='__main__':unittest.main()
