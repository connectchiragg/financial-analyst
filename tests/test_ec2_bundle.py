"""Offline EC2 packaging with synthetic PDFs, credentials and real SQLite."""
from contextlib import closing, redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import shutil
import sqlite3
import stat
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from financial_analyst.corpus_adapter import import_reviewed
from financial_analyst.corpus_cli import load_config
from financial_analyst.inference import load_api_key
from scripts.prepare_ec2_bundle import BundleError, PROJECT_ROOT, _copy, _validate, main, prepare_bundle
from tests.test_corpus_adapter import create_corpus


KEY = 'synthetic-selected-provider-key-DO-NOT-LOG'
CODE = 'synthetic-browser-access-code-0123456789'


class EC2BundleTests(unittest.TestCase):
    def setUp(self):
        private_root = PROJECT_ROOT/'.local'
        private_root.mkdir(mode=0o700, exist_ok=True)
        self.temporary = TemporaryDirectory(prefix='synthetic-ec2-tests-', dir=private_root)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root/'source'
        self.source.mkdir()
        self.pack, self.catalog, self.pack_data, self.pdf = create_corpus(self.source)
        self.database = self.source/'corpus.sqlite'
        import_reviewed(self.pack, self.catalog, self.database)
        self.env = self.source/'input.env'
        self.env.write_text('TEST_PROVIDER_KEY='+KEY+'\nUNRELATED_SECRET=synthetic-unrelated-key\n')
        self.code = self.source/'access-code'
        self.code.write_text(CODE+'\n')
        self.config_data = {
            'engine':'corpus', 'mode':'live', 'provider':'openai-compatible',
            'model':'synthetic/test-model', 'base_url':'https://example.test/api/v1',
            'api_key_env':'TEST_PROVIDER_KEY', 'env_file':self.env.name,
            'proof_pack':self.pack.name, 'catalog':self.catalog.name,
            'database':self.database.name,
        }
        self.config = self.source/'session.json'
        self.config.write_text(json.dumps(self.config_data))
        self.output = self.root/'bundle'

    def prepare(self):
        return prepare_bundle(self.config, self.output, self.code)

    def test_portable_bundle_keeps_source_bytes_proof_status_provider_and_real_sqlite(self):
        with patch('financial_analyst.inference.create_inference', side_effect=AssertionError('No inference')):
            result = self.prepare()
        self.assertEqual((result.documents, result.facts), (1, 2))
        self.assertEqual((result.provider, result.model), ('openai-compatible','synthetic/test-model'))
        private = self.output/'private'
        self.assertEqual((private/'sources'/self.pdf.name).read_bytes(), self.pdf.read_bytes())
        self.assertEqual((private/'proof-pack.json').read_bytes(), self.pack.read_bytes())
        original_catalog = json.loads(self.catalog.read_text())
        copied_catalog = json.loads((private/'catalog.json').read_text())
        original_catalog['documents'][0]['local_path'] = 'sources/'+self.pdf.name
        self.assertEqual(copied_catalog, original_catalog)
        portable = json.loads((private/'config.json').read_text())
        self.assertEqual(portable['model'], self.config_data['model'])
        self.assertEqual(portable['provider'], self.config_data['provider'])
        self.assertEqual(portable['base_url'], self.config_data['base_url'])
        for field in ('proof_pack','catalog','database','env_file'):
            self.assertFalse(Path(portable[field]).is_absolute())
        self.assertNotIn('UNRELATED_SECRET', (private/'provider.env').read_text())
        self.assertEqual(load_api_key('TEST_PROVIDER_KEY', private/'provider.env'), KEY)
        self.assertEqual((private/'browser.env').read_text(), 'FINANCIAL_ANALYST_ACCESS_CODE='+CODE+'\n')
        verification = json.loads((self.output/'verification.json').read_text())
        self.assertEqual(verification['inference'], 'not_called')
        self.assertFalse(verification['deployed'])
        relocated = self.root/'relocated-release'
        self.output.rename(relocated)
        shutil.rmtree(self.source)
        self.assertEqual(_validate(load_config(relocated/'private'/'config.json')), 2)

    def test_every_bundle_file_and_directory_is_owner_only_even_with_open_umask(self):
        previous = os.umask(0)
        try:
            self.prepare()
        finally:
            os.umask(previous)
        for path in (self.output, *self.output.rglob('*')):
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700 if path.is_dir() else 0o600)

    def test_missing_private_artifact_fails_without_output_or_secret_diagnostic(self):
        for path in (self.pack,self.catalog,self.database,self.env,self.code,self.pdf):
            with self.subTest(path=path.name):
                preserved = path.read_bytes()
                path.unlink()
                try:
                    with self.assertRaises(BundleError) as caught:
                        self.prepare()
                    self.assertNotIn(KEY,str(caught.exception))
                    self.assertNotIn(CODE,str(caught.exception))
                    self.assertFalse(self.output.exists())
                    self.assertEqual(list(self.root.glob('.ec2-prepare-*')), [])
                finally:
                    path.write_bytes(preserved)

    def test_altered_pdf_or_database_fact_is_rejected_before_success(self):
        self.pdf.write_bytes(self.pdf.read_bytes()+b'altered')
        with self.assertRaises(BundleError):
            self.prepare()
        self.assertFalse(self.output.exists())
        self.pdf.write_bytes(self.pdf.read_bytes()[:-7])
        with closing(sqlite3.connect(self.database)) as database:
            database.execute("UPDATE facts SET period='1QFY26' WHERE fact_id='example-revenue'")
            database.commit()
        with self.assertRaises(BundleError):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_incomplete_sqlite_coverage_is_rejected(self):
        with closing(sqlite3.connect(self.database)) as database:
            database.execute("DELETE FROM facts WHERE fact_id='example-revenue'")
            database.commit()
        with self.assertRaises(BundleError):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_copied_source_is_authenticated_again_before_publication(self):
        def corrupt_copied_pdf(source, destination):
            _copy(source, destination)
            if destination.suffix == '.pdf':
                destination.write_bytes(destination.read_bytes()+b'changed copy')
        with patch('scripts.prepare_ec2_bundle._copy',side_effect=corrupt_copied_pdf):
            with self.assertRaises(BundleError):
                self.prepare()
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob('.ec2-prepare-*')), [])

    def test_wal_source_is_copied_as_one_read_only_portable_database(self):
        source = sqlite3.connect(self.database)
        try:
            self.assertEqual(source.execute('PRAGMA journal_mode=WAL').fetchone()[0], 'wal')
            source.execute("UPDATE facts SET fact_json=fact_json WHERE fact_id='example-revenue'")
            source.commit()
            self.assertTrue(Path(str(self.database)+'-wal').exists())
            self.prepare()
            copied = self.output/'private'/'corpus.sqlite'
            self.assertEqual(copied.read_bytes()[18:20], b'\x01\x01')
            self.assertEqual(_validate(load_config(self.output/'private'/'config.json')), 2)
            self.assertFalse(Path(str(copied)+'-wal').exists())
            self.assertFalse(Path(str(copied)+'-shm').exists())
            self.assertEqual(source.execute('PRAGMA journal_mode').fetchone()[0], 'wal')
        finally:
            source.close()

    def test_source_catalog_path_is_resolved_from_config_not_current_directory(self):
        catalog = json.loads(self.catalog.read_text())
        catalog['documents'][0]['local_path'] = self.pdf.name
        self.catalog.write_text(json.dumps(catalog))
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            self.prepare()
        finally:
            os.chdir(previous)
        self.assertEqual(_validate(load_config(self.output/'private'/'config.json')), 2)

    def test_refuses_existing_destination_and_paths_outside_ignored_directory(self):
        self.output.mkdir()
        marker = self.output/'preserve'
        marker.write_text('existing release')
        with self.assertRaises(BundleError):
            self.prepare()
        self.assertEqual(marker.read_text(), 'existing release')
        with self.assertRaises(BundleError):
            prepare_bundle(self.config, PROJECT_ROOT/'public-bundle', self.code)
        with TemporaryDirectory() as unrelated:
            symlink = self.root/'escape'
            symlink.symlink_to(unrelated, target_is_directory=True)
            with self.assertRaises(BundleError):
                prepare_bundle(self.config, symlink/'bundle', self.code)

    def test_unsafe_pdf_filename_mapping_is_rejected(self):
        catalog = json.loads(self.catalog.read_text())
        catalog['documents'][0]['document_name'] = '../synthetic-corpus.pdf'
        self.catalog.write_text(json.dumps(catalog))
        with self.assertRaises(BundleError):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_destination_created_during_preparation_is_preserved(self):
        real_validate = _validate
        calls = 0
        def create_collision(config):
            nonlocal calls
            result = real_validate(config)
            calls += 1
            if calls == 2:
                self.output.mkdir()
                (self.output/'preserve').write_text('concurrent release')
            return result
        with patch('scripts.prepare_ec2_bundle._validate',side_effect=create_collision):
            with self.assertRaises(BundleError):
                self.prepare()
        self.assertEqual((self.output/'preserve').read_text(),'concurrent release')
        self.assertEqual([path.name for path in self.output.iterdir()],['preserve'])

    def test_refuses_fixture_mode_missing_model_and_invalid_access_code(self):
        for changes in ({'mode':'fixture'}, {'model':None}):
            with self.subTest(changes=changes):
                self.config.write_text(json.dumps({**self.config_data,**changes}))
                with self.assertRaises(BundleError):
                    self.prepare()
                self.assertFalse(self.output.exists())
        self.config.write_text(json.dumps(self.config_data))
        self.code.write_text('short')
        with self.assertRaises(BundleError):
            self.prepare()

    def test_cli_prints_only_safe_summary_and_does_not_echo_secrets_on_failure(self):
        stdout,stderr = StringIO(),StringIO()
        arguments = ['--config',str(self.config),'--output',str(self.output),'--access-code-file',str(self.code)]
        with redirect_stdout(stdout),redirect_stderr(stderr):
            self.assertEqual(main(arguments),0)
            self.assertEqual(main(arguments),2)
        output = stdout.getvalue()+stderr.getvalue()
        self.assertIn('inference not called; not deployed',output)
        self.assertNotIn(KEY,output)
        self.assertNotIn(CODE,output)
        self.assertNotIn('synthetic-unrelated-key',output)


if __name__ == '__main__':
    unittest.main()
