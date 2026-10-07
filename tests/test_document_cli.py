"""Document workflow composition uses synthetic providers and real temporary SQLite."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from financial_analyst.cli import main
from financial_analyst.document_store import DocumentStore
from financial_analyst.inference import InferenceResponse, ProviderError
from tests.support import create_fixture


class ProposalInference:
    provider, model, mode = 'synthetic', 'test-model', 'test_double'
    capabilities = frozenset({'json_schema'})
    def __init__(self, fail=False, on_call=None):
        self.fail, self.calls, self.closed = fail, [], False
        self.on_call = on_call
        self.execution = {'provider':self.provider,'model':self.model,'mode':'not_called'}
    def infer(self, request):
        self.calls.append(request)
        self.execution = {'provider':self.provider,'model':self.model,'mode':self.mode}
        if self.on_call is not None:
            self.on_call()
        if self.fail:
            raise ProviderError('private-error-must-not-be-rendered')
        return InferenceResponse('{"facts": []}', 'stop')
    def close(self):
        self.closed = True


class DocumentCliTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.fixture = create_fixture(Path(self.directory.name))
        self.pdf = Path(json.loads(self.fixture.read_text())['source']['local_path'])
        self.database = Path(self.directory.name) / 'knowledge.sqlite'
    def invoke(self, command='extract', extra=(), fail=False, on_call=None):
        inference = ProposalInference(fail, on_call)
        output, errors = io.StringIO(), io.StringIO()
        provider = ['--llm','groq'] if command=='extract' else []
        with patch('financial_analyst.inference.create_inference',return_value=inference) as factory, \
             redirect_stdout(output), redirect_stderr(errors):
            code = main(['--mode','live','--source-pdf',str(self.pdf),'--database',str(self.database),
                         '--format','json',*provider,command,'--document-id','synthetic-id',
                         '--source-url','https://example.test/report','--company','Example Pharma',
                         '--agency','Example Research',*extra])
        return code, json.loads(output.getvalue()) if output.getvalue() else None, errors.getvalue(), inference, factory
    def test_source_only_persists_canonical_passages_without_creating_provider(self):
        code, result, error, _, factory = self.invoke('store-document')
        self.assertEqual(code,0,error)
        factory.assert_not_called()
        self.assertEqual(result['claims'],[])
        self.assertEqual(result['status'],'source_stored_unreviewed')
        self.assertGreater(result['counts']['passages_added'],0)
        self.assertEqual(result['coverage']['stored_pages'],[1,2,3])
        self.assertEqual(DocumentStore(self.database).query_reviewed('Example Pharma','1QFY27'),())
    def test_llm_window_budget_and_pending_review_provenance(self):
        code, result, error, inference, factory = self.invoke(extra=['--pages','2','--max-windows','1'])
        self.assertEqual(code,0,error)
        self.assertEqual(result['status'],'extracted_pending_review')
        self.assertEqual(len(inference.calls),1)
        self.assertEqual(factory.call_args.kwargs['timeout'],60)
        self.assertEqual(result['claims'],[])
        self.assertEqual(result['coverage']['attempted_pages'],[2])
        self.assertTrue(inference.closed)
        self.assertEqual(result['execution']['llm']['mode'],'test_double')
        self.assertEqual(DocumentStore(self.database).query_reviewed('Example Pharma','1QFY27'),())
    def test_provider_failure_stores_only_source_and_reports_failure(self):
        code, result, error, inference, _ = self.invoke(extra=['--pages','2','--max-windows','1'],fail=True)
        self.assertEqual(code,2,error)
        self.assertEqual(result['status'],'source_stored_extraction_failed')
        self.assertEqual(result['counts']['facts_added'],0)
        self.assertGreater(result['counts']['passages_added'],0)
        self.assertEqual(result['coverage']['inference_calls'],1)
        self.assertEqual(result['claims'],[])
        self.assertNotIn('private-error',json.dumps(result))
        self.assertTrue(inference.closed)
    def test_all_pages_config_and_repeat_source_import_are_explicit(self):
        code, result, error, inference, _ = self.invoke(extra=['--pages','all','--max-windows','1'])
        self.assertEqual(code,0,error)
        self.assertEqual(result['coverage']['stored_pages'],[1,2,3])
        self.assertEqual(len(inference.calls),1)
        code, repeated, error, _, factory = self.invoke('store-document')
        self.assertEqual(code,0,error)
        self.assertEqual(repeated['counts']['documents_added'],0)
        self.assertEqual(repeated['counts']['passages_added'],0)
        factory.assert_not_called()

    def test_source_mutation_during_inference_cannot_create_a_database(self):
        code, result, error, inference, _ = self.invoke(
            extra=['--max-windows','1'],
            on_call=lambda:self.pdf.write_bytes(b'source changed during inference'))
        self.assertEqual(code,2)
        self.assertIsNone(result)
        self.assertIn('Document ingestion failed',error)
        self.assertEqual(len(inference.calls),1)
        self.assertTrue(inference.closed)
        self.assertFalse(self.database.exists())

    def test_invalid_page_selection_has_no_provider_or_database_side_effects(self):
        for pages in ('99','1,1',''):
            with self.subTest(pages=pages):
                code, result, error, inference, factory = self.invoke(extra=['--pages',pages])
                self.assertEqual(code,2)
                self.assertIsNone(result)
                self.assertIn('Document ingestion failed',error)
                factory.assert_not_called()
                self.assertEqual(inference.calls,[])
                self.assertFalse(self.database.exists())

    def test_invalid_window_budget_has_no_provider_or_database_side_effects(self):
        code, result, error, inference, factory = self.invoke(extra=['--max-windows','0'])
        self.assertEqual(code,2)
        self.assertIsNone(result)
        self.assertIn('Document ingestion failed',error)
        factory.assert_not_called()
        self.assertEqual(inference.calls,[])
        self.assertFalse(self.database.exists())
