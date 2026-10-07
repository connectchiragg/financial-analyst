from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from financial_analyst.adapters import FileFixtureAdapter
from financial_analyst.cli import main
from tests.support import create_fixture


class KnowledgePreparationCliTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.fixture = create_fixture(Path(self.directory.name))
        self.destination = Path(self.directory.name) / 'prepared'

    def invoke(self, extra=()):
        stdout, stderr = io.StringIO(), io.StringIO()
        flags = ['--mode', 'fixture', '--fixture', str(self.fixture), '--format', 'json',
                 'prepare-knowledge', '--company', 'Example Pharma', '--period', '1QFY27',
                 '--destination', str(self.destination)]
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main([*flags, *extra])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_prepares_source_bound_single_contexts_without_provider_or_database(self):
        with patch('financial_analyst.inference.create_inference') as provider:
            code, output, error = self.invoke(['--kind', 'broker_commentary', '--unit', 'INRm'])
        self.assertEqual(code, 0, error)
        provider.assert_not_called()
        result = json.loads(output)
        self.assertEqual(result['status'], 'prepared')
        self.assertEqual(result['claims'], [])
        self.assertTrue(result['execution']['no_llm'])
        self.assertTrue(result['execution']['no_database'])
        self.assertFalse(result['execution']['remote_upload'])
        self.assertEqual(result['execution']['sanitization'], 'not_performed')
        manifest = json.loads(Path(result['manifest']).read_text())
        fixture = FileFixtureAdapter(self.fixture)
        self.assertEqual(len(manifest['groups']), 1)
        group = manifest['groups'][0]
        self.assertEqual(group['context']['kind'], 'broker_commentary')
        self.assertEqual(group['context']['period'], '1QFY27')
        for doc in group['documents']:
            payload = json.loads((self.destination / doc['file_name']).read_text())
            self.assertEqual(payload['quote'], fixture.resolve(doc['record_ref']).excerpt)
            self.assertEqual(payload['context'], group['context'])
            self.assertEqual(payload['source']['sha256'], fixture.source_sha256)

    def test_empty_eligibility_creates_no_export(self):
        code, output, error = self.invoke(['--kind', 'broker_estimate', '--period', '1QFY26'])
        self.assertEqual(code, 0, error)
        result = json.loads(output)
        self.assertEqual(result['status'], 'no_matches')
        self.assertIsNone(result['manifest'])
        self.assertFalse(self.destination.exists())

    def test_repeat_is_unchanged_and_foreign_manifest_is_not_overwritten(self):
        code, output, error = self.invoke(['--kind', 'reported_actual'])
        self.assertEqual(code, 0, error)
        path = Path(json.loads(output)['manifest'])
        before, modified = path.read_bytes(), path.stat().st_mtime_ns
        code, _, error = self.invoke(['--kind', 'reported_actual'])
        self.assertEqual(code, 0, error)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(path.stat().st_mtime_ns, modified)
        path.write_text('foreign manifest')
        code, output, error = self.invoke(['--kind', 'reported_actual'])
        self.assertEqual(code, 2)
        self.assertEqual(output, '')
        self.assertIn('conflicting', error)
        self.assertEqual(path.read_text(), 'foreign manifest')

    def test_source_authentication_failure_creates_no_export(self):
        payload = json.loads(self.fixture.read_text())
        payload['source']['sha256'] = '0' * 64
        self.fixture.write_text(json.dumps(payload))
        code, output, error = self.invoke()
        self.assertEqual(code, 2)
        self.assertEqual(output, '')
        self.assertIn('source', error.lower())
        self.assertFalse(self.destination.exists())

    def test_inference_or_database_configuration_is_rejected_without_construction(self):
        for extra in (['--llm', 'groq'], ['--database', str(self.destination / 'db.sqlite')]):
            with self.subTest(extra=extra), patch('financial_analyst.inference.create_inference') as provider, \
                 redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                main(['--mode', 'fixture', '--fixture', str(self.fixture), *extra,
                      'prepare-knowledge', '--company', 'Example Pharma', '--period', '1QFY27',
                      '--destination', str(self.destination)])
            self.assertEqual(raised.exception.code, 2)
            provider.assert_not_called()
        self.assertFalse(self.destination.exists())


if __name__ == '__main__':
    unittest.main()
