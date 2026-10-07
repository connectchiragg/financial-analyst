"""Local graph CLI integration with synthetic inference, never provider calls."""

from contextlib import redirect_stderr, redirect_stdout
from decimal import Decimal
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from financial_analyst.cli import main
from financial_analyst.inference import InferenceResponse
from tests.support import create_fixture


class PlanningInference:
    capabilities = frozenset({'json_schema'})
    mode = 'test_double'
    provider = 'example-provider'
    model = 'example-model'

    def __init__(self, complete=True):
        self.complete = complete
        self.calls = []
        self.closed = False
        self.execution = {'provider': self.provider, 'model': self.model, 'mode': 'not_called'}

    def infer(self, request):
        self.calls.append(request)
        self.execution = {'provider': self.provider, 'model': self.model, 'mode': 'test_double'}
        result = ({'tool': 'combined_revenue', 'company': 'Example Pharma', 'period': '1QFY27',
                   'include_yoy': True, 'unsupported_parts': []} if len(self.calls) == 1 else
                  {'complete': self.complete, 'unsupported_parts': [] if self.complete else ['Profit is unsupported.']})
        return InferenceResponse(json.dumps(result), 'stop')

    def close(self):
        self.closed = True


class AgentCliTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.fixture = create_fixture(Path(self.directory.name))

    def invoke(self, extra=(), complete=True, mode='fixture', database=None):
        inference = PlanningInference(complete)
        stdout, stderr = io.StringIO(), io.StringIO()
        flags = ['--mode', mode, '--fixture', str(self.fixture), '--format', 'json']
        if database is not None:
            flags += ['--database', str(database)]
        with patch('financial_analyst.inference.create_inference', return_value=inference) as factory, \
             redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                code = main([*flags, *extra, 'agent-ask', 'How did turnover perform and what helped?',
                             '--company', 'Example Pharma', '--period', '1QFY27'])
            except SystemExit as error:
                code = error.code
        return code, stdout.getvalue(), stderr.getvalue(), factory, inference

    def test_neutral_factory_canonical_values_and_cleanup(self):
        for provider in ('groq', 'mistral', 'openai-compatible'):
            with self.subTest(provider=provider):
                extra = ['--llm', provider, '--model', 'example-model']
                if provider == 'openai-compatible':
                    extra += ['--base-url', 'https://example.test/v1']
                code, output, error, factory, inference = self.invoke(extra)
                self.assertEqual(code, 0, error)
                answer = json.loads(output)
                self.assertEqual(answer['status'], 'answered')
                self.assertEqual(Decimal(answer['claims'][0]['values']['delta']), Decimal(20))
                self.assertEqual({claim['kind'] for claim in answer['claims']}, {'comparison', 'yoy', 'growth'})
                refs = {citation['ref'] for citation in answer['citations']}
                self.assertTrue(all(set(claim['evidence_refs']) <= refs for claim in answer['claims']))
                self.assertEqual(len(inference.calls), 2)
                self.assertTrue(inference.closed)
                self.assertEqual(factory.call_args.args, (provider,))
                self.assertTrue(answer['execution']['no_llm'])

    def test_partial_coverage_refuses_with_no_claims(self):
        code, output, error, _, inference = self.invoke(['--llm', 'groq'], complete=False)
        self.assertEqual(code, 1, error)
        self.assertEqual(json.loads(output)['claims'], [])
        self.assertTrue(inference.closed)

    def test_modes_and_unsupported_retrieval_do_not_construct_provider(self):
        for extra, mode, database in [([], 'fixture', None),
                                      (['--llm', 'groq', '--retrieval', 'semantic'], 'fixture', None),
                                      (['--llm', 'groq'], 'live', None),
                                      (['--llm', 'groq'], 'fixture', Path('ignored.sqlite')),
                                      (['--llm', 'groq'], 'local', None)]:
            with self.subTest(extra=extra, mode=mode, database=database):
                code, _, _, factory, _ = self.invoke(extra, mode=mode, database=database)
                self.assertEqual(code, 2)
                factory.assert_not_called()

    def test_sqlite_execution_remains_real_and_source_bound(self):
        from financial_analyst.adapters import FileFixtureAdapter
        from financial_analyst.sqlite_adapter import DocumentRecord, SQLiteStore
        fixture = FileFixtureAdapter(self.fixture)
        database = Path(self.directory.name) / 'analyst.sqlite'
        store = SQLiteStore(database)
        store.initialize()
        store.ingest(DocumentRecord(fixture.document_name, fixture.source_sha256, fixture.source_url),
                     fixture.reviewed_observations(), fixture.reviewed_evidence(), fixture)
        code, output, error, _, inference = self.invoke(['--llm', 'groq'], mode='live', database=database)
        self.assertEqual(code, 0, error)
        execution = json.loads(output)['execution']
        self.assertEqual(execution['database'], 'sqlite')
        self.assertFalse(execution['no_database'])
        self.assertTrue(inference.closed)
