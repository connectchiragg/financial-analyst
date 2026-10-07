from contextlib import closing
from dataclasses import asdict, replace
from decimal import Decimal, Inexact, localcontext
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from financial_analyst.document_store import DocumentStore, DocumentStorageError
from financial_analyst.extraction import ExtractionError, extract_document, prepare_document
from financial_analyst.inference import InferenceResponse
from financial_analyst.sqlite_adapter import SQLiteStore
from tests.support import create_fixture


class FactInference:
    provider, model, mode = 'synthetic', 'test-model', 'test_double'
    capabilities = frozenset({'json_schema'})

    def __init__(self, value='120.00000000000000000000', metric='net_sales'):
        self.value, self.metric = value, metric

    def infer(self, request):
        payload = json.loads(request.messages[1].content)
        facts = []
        for passage in payload['passages']:
            if passage['page'] == 2:
                facts.append({'company':'Example Pharma','period':'1QFY27','scope':'consolidated',
                    'metric':self.metric,'kind':None,'currency':'INR','unit':'million',
                    'value_text':self.value,'source_value_raw':'120','source_metric_raw':'Net Sales',
                    'source_period_raw':'FY27E','source_scope_raw':'(Consol.)','source_unit_raw':'(INRm)',
                    'source_kind_raw':'FY27E','evidence_refs':[passage['ref']]})
        return InferenceResponse(json.dumps({'facts':facts}),'stop')


class DocumentStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        fixture = json.loads(create_fixture(self.root/'source').read_text())
        self.pdf = Path(fixture['source']['local_path'])
        self.options = {'document_id':'synthetic-source','document_name':self.pdf.name,
                        'url':'https://example.test/source.pdf','company':'Example Pharma','agency':'Example Broker'}
        self.bundle = extract_document(self.pdf,**self.options,inference=FactInference())
        self.database = self.root/'knowledge.sqlite'
        self.store = DocumentStore(self.database)
        self.store.initialize()

    def ingest(self, bundle=None):
        return self.store.ingest(bundle or self.bundle,source_path=self.pdf)

    def counts(self):
        with closing(sqlite3.connect(self.database)) as db:
            return tuple(db.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
                         for table in ('documents','passages','facts','fact_evidence','extraction_runs'))

    def test_roundtrip_preserves_exact_precision_raw_labels_and_pending_status(self):
        with localcontext() as context:
            context.prec = 4
            context.traps[Inexact] = True
            added = self.ingest()
            fact = self.store.facts()[0]
        self.assertEqual(added.facts_added,1)
        self.assertEqual(fact['value_text'],self.bundle.facts[0].value_text)
        self.assertEqual(Decimal(fact['value_text']).as_tuple(),Decimal(self.bundle.facts[0].value_text).as_tuple())
        self.assertEqual(fact['raw_labels']['source_value_raw'],'120')
        self.assertEqual(fact['raw_labels']['source_period_raw'],'FY27E')
        self.assertIsNone(fact['kind'])
        self.assertEqual(fact['review_state'],'quote_validated_pending_review')
        self.assertEqual(self.store.query_reviewed('Example Pharma','1QFY27'),())

    def test_all_pages_and_generated_prefix_are_separate_from_source_quote(self):
        self.ingest()
        passages = self.store.explore_passages()
        self.assertEqual({item['page'] for item in passages},{1,2,3})
        self.assertEqual([item['quote'] for item in passages],[item.excerpt for item in self.bundle.passages])
        self.assertTrue(all(item['review_state']=='unreviewed' for item in passages))
        self.assertTrue(all('company=Example Pharma' in item['context_prefix'] for item in passages))
        self.assertTrue(all('review_state=unreviewed' not in item['quote'] for item in passages))

    def test_idempotent_import_does_not_change_database_bytes(self):
        self.ingest()
        before = self.database.read_bytes()
        repeated = self.ingest()
        self.assertTrue(all(value==0 for value in asdict(repeated).values()))
        self.assertEqual(self.database.read_bytes(),before)

    def test_source_only_import_then_partial_extraction_appends_without_overwrite(self):
        source_only = prepare_document(self.pdf,**self.options)
        self.ingest(source_only)
        partial = replace(self.bundle,coverage=replace(self.bundle.coverage,attempted_pages=(2,),extracted_pages=(2,),uncovered_pages=(1,3),inference_calls=1))
        result = self.ingest(partial)
        self.assertEqual((result.documents_added,result.passages_added,result.facts_added),(0,0,1))
        self.assertEqual(result.extraction_runs_added,1)
        coverage = self.store.coverage()
        self.assertEqual(len(coverage),2)
        self.assertTrue(all(item['extraction_complete'] is False for item in coverage))
        self.assertTrue(any(item['coverage']['uncovered_pages']==[1,3] for item in coverage))

    def test_generic_metrics_are_rows_without_schema_changes(self):
        self.ingest()
        with closing(sqlite3.connect(self.database)) as db:
            before = db.execute("SELECT name,sql FROM sqlite_master ORDER BY name").fetchall()
        second = extract_document(self.pdf,**self.options,inference=FactInference(metric='custom_metric'))
        self.assertEqual(self.ingest(second).facts_added,1)
        with closing(sqlite3.connect(self.database)) as db:
            self.assertEqual(db.execute("SELECT name,sql FROM sqlite_master ORDER BY name").fetchall(),before)
        self.assertEqual({item['metric'] for item in self.store.facts()},{'net_sales','custom_metric'})

    def test_automatic_review_promotion_is_rejected_before_writes(self):
        changed = replace(self.bundle,facts=(replace(self.bundle.facts[0],review_state='reviewed'),))
        with self.assertRaises((ExtractionError,DocumentStorageError)):
            self.ingest(changed)
        self.assertEqual(self.counts(),(0,0,0,0,0))

    def test_forged_source_passage_identity_or_quote_leaves_no_partial_rows(self):
        candidates = [replace(self.bundle,source=replace(self.bundle.source,sha256='0'*64)),
                      replace(self.bundle,passages=(replace(self.bundle.passages[0],excerpt='Invented quote'),*self.bundle.passages[1:])),
                      replace(self.bundle,facts=(replace(self.bundle.facts[0],source_sha256='0'*64),)),
                      replace(self.bundle,facts=(replace(self.bundle.facts[0],fact_id='invented-id'),))]
        for candidate in candidates:
            with self.subTest(candidate_type=type(candidate).__name__),self.assertRaises((ExtractionError,DocumentStorageError)):
                self.ingest(candidate)
            self.assertEqual(self.counts(),(0,0,0,0,0))

    def test_float_nonfinite_and_mismatched_raw_values_are_rejected(self):
        for value in (120.0,'NaN','Infinity','121'):
            changed = replace(self.bundle,facts=(replace(self.bundle.facts[0],value_text=value),))
            with self.subTest(value=value),self.assertRaises((ExtractionError,DocumentStorageError)):
                self.ingest(changed)
            self.assertEqual(self.counts(),(0,0,0,0,0))

    def test_conflicting_metadata_rolls_back_and_preserves_existing_document(self):
        self.ingest()
        before = self.database.read_bytes()
        changed = prepare_document(self.pdf,**{**self.options,'agency':'Other Broker'})
        with self.assertRaisesRegex(DocumentStorageError,'Conflicting documents'):
            self.ingest(changed)
        self.assertEqual(self.database.read_bytes(),before)

    def test_conflicting_existing_fact_rolls_back_earlier_new_fact_insert(self):
        self.ingest()
        new = extract_document(self.pdf,**self.options,inference=FactInference(metric='custom_metric'))
        with closing(sqlite3.connect(self.database)) as db:
            db.execute("UPDATE facts SET value_text='121'")
            db.commit()
        before = self.database.read_bytes()
        combined = replace(self.bundle,facts=(*new.facts,*self.bundle.facts))
        with self.assertRaisesRegex(DocumentStorageError,'Conflicting facts'):
            self.ingest(combined)
        self.assertEqual(self.database.read_bytes(),before)
        self.assertEqual(self.counts()[2],1)

    def test_missing_database_reads_and_ingestion_never_create_files(self):
        missing = self.root/'missing.sqlite'
        store = DocumentStore(missing)
        for operation in (store.explore_passages,store.facts,store.coverage,
                          lambda:store.query_reviewed('Example Pharma','1QFY27'),
                          lambda:store.ingest(self.bundle,source_path=self.pdf)):
            with self.subTest(operation=operation),self.assertRaises(DocumentStorageError):
                operation()
            self.assertFalse(missing.exists())

    def test_wrong_version_and_existing_analytics_database_are_preserved(self):
        with closing(sqlite3.connect(self.database)) as db:
            db.execute('PRAGMA user_version=99');db.commit()
        before = self.database.read_bytes()
        for operation in (self.store.initialize,self.store.facts,lambda:self.ingest()):
            with self.assertRaises(DocumentStorageError):operation()
            self.assertEqual(self.database.read_bytes(),before)
        old = self.root/'analyst.sqlite'
        SQLiteStore(old).initialize()
        before = old.read_bytes()
        with self.assertRaisesRegex(DocumentStorageError,'schema fingerprint'):
            DocumentStore(old).initialize()
        self.assertEqual(old.read_bytes(),before)

    def test_undeclared_trigger_cannot_promote_model_facts_during_ingestion(self):
        with closing(sqlite3.connect(self.database)) as db:
            db.execute("""CREATE TRIGGER promote_model_fact AFTER INSERT ON facts
                BEGIN UPDATE facts SET review_state='reviewed'; END""")
            db.commit()
        before = self.database.read_bytes()
        with self.assertRaisesRegex(DocumentStorageError, 'undeclared'):
            self.ingest()
        self.assertEqual(self.database.read_bytes(), before)
        self.assertEqual(self.counts(), (0,0,0,0,0))

    def test_reviewed_filters_bind_one_complete_fact_tuple(self):
        self.ingest()
        second = extract_document(self.pdf,**self.options,inference=FactInference(metric='custom_metric'))
        self.ingest(second)
        # Simulate an explicit future review, not model-driven promotion.
        pending = self.store.facts()
        with closing(sqlite3.connect(self.database)) as db:
            from financial_analyst.extraction import _fact_identity
            for fact in pending:
                period = '1QFY26' if fact['metric']=='custom_metric' else fact['period']
                payload = {name:fact[name] for name in ('company','period','scope','metric','kind','currency','unit','value_text')}
                payload.update(fact['raw_labels'])
                payload.update(period=period,kind='reported_actual',evidence=[
                    {'ref':item['passage_ref'],'quote':item['quote']} for item in fact['evidence']])
                reviewed_id = _fact_identity(payload)
                db.execute("UPDATE facts SET fact_id=?,review_state='reviewed',kind='reported_actual',period=? WHERE fact_id=?",
                    (reviewed_id,period,fact['fact_id']))
                db.execute('UPDATE fact_evidence SET fact_id=? WHERE fact_id=?',(reviewed_id,fact['fact_id']))
            db.commit()
        self.assertEqual(len(self.store.query_reviewed('Example Pharma','1QFY27',scope='consolidated',metric='net_sales',kind='reported_actual',currency='INR',unit='million')),1)
        self.assertEqual(self.store.query_reviewed('Example Pharma','1QFY27',metric='custom_metric'),())
        self.assertEqual(self.store.query_reviewed('Example Pharma','1QFY27',scope='standalone'),())
        self.assertEqual(self.store.query_reviewed('Example Pharma','1QFY27',kind='broker_forecast'),())
        self.assertEqual(self.store.query_reviewed('Example Pharma','1QFY27',currency='USD'),())
        self.assertEqual(self.store.query_reviewed('Other Pharma','1QFY27'),())

    def test_null_context_cannot_become_answerable_from_review_flag_alone(self):
        self.ingest()
        with closing(sqlite3.connect(self.database)) as db:
            db.execute("UPDATE facts SET review_state='reviewed',scope=NULL")
            db.commit()
        self.assertEqual(self.store.query_reviewed('Example Pharma','1QFY27'),())

    def test_exploratory_filters_are_parameterized_and_bounded(self):
        self.ingest()
        self.assertEqual(self.store.explore_passages(company="Example Pharma' OR 1=1 --"),())
        self.assertEqual(len(self.store.explore_passages(page=2)),1)
        for limit in (0,True,10001):
            with self.assertRaises(DocumentStorageError):self.store.facts(limit=limit)

    def test_tampered_fact_fields_fail_closed_on_read(self):
        self.ingest()
        original = self.database.read_bytes()
        mutations = (
            ("UPDATE facts SET raw_labels_json=?", ('{',)),
            ("UPDATE facts SET raw_labels_json=?", ('{}',)),
            ("UPDATE facts SET value_text=?", ('NaN',)),
            ("UPDATE facts SET period=?", ('FY99',)),
            ("UPDATE facts SET company=?", ('Other Pharma',)),
            ("UPDATE facts SET review_state=?", ('automatically_reviewed',)),
        )
        for query, parameters in mutations:
            with self.subTest(query=query, parameters=parameters):
                self.database.write_bytes(original)
                with closing(sqlite3.connect(self.database)) as db:
                    db.execute(query, parameters)
                    db.commit()
                with self.assertRaises(DocumentStorageError):
                    self.store.facts()

    def test_missing_or_altered_quote_evidence_fails_closed_on_read(self):
        self.ingest()
        original = self.database.read_bytes()
        for query in ("DELETE FROM fact_evidence", "UPDATE fact_evidence SET quote='fabricated quote'",
                      "UPDATE fact_evidence SET position=9"):
            with self.subTest(query=query):
                self.database.write_bytes(original)
                with closing(sqlite3.connect(self.database)) as db:
                    db.execute(query)
                    db.commit()
                with self.assertRaises(DocumentStorageError):
                    self.store.facts()

    def test_passage_context_offsets_and_review_state_cannot_be_silently_changed(self):
        self.ingest()
        original = self.database.read_bytes()
        for query in ("UPDATE passages SET context_prefix='incorrect company'",
                      "UPDATE passages SET end_offset=end_offset+1",
                      "UPDATE passages SET review_state='reviewed'",
                      "UPDATE passages SET page=99"):
            with self.subTest(query=query):
                self.database.write_bytes(original)
                with closing(sqlite3.connect(self.database)) as db:
                    db.execute(query)
                    db.commit()
                with self.assertRaises(DocumentStorageError):
                    self.store.explore_passages()

    def test_coverage_tampering_is_refused_instead_of_claiming_completion(self):
        partial = replace(self.bundle, coverage=replace(self.bundle.coverage,
            attempted_pages=(2,), extracted_pages=(2,), uncovered_pages=(1,3), inference_calls=1))
        self.ingest(partial)
        original = self.database.read_bytes()
        for query in ("UPDATE extraction_runs SET coverage_json='{}'",
                      "UPDATE extraction_runs SET extraction_complete=1"):
            with self.subTest(query=query):
                self.database.write_bytes(original)
                with closing(sqlite3.connect(self.database)) as db:
                    db.execute(query)
                    db.commit()
                with self.assertRaises(DocumentStorageError):
                    self.store.coverage()


if __name__=='__main__':
    unittest.main()
