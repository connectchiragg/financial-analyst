"""Prepare a private, portable deployment directory without invoking inference."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
from tempfile import TemporaryDirectory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from financial_analyst.corpus_adapter import ReviewedCorpusAdapter, SQLiteCorpusAdapter
from financial_analyst.corpus_cli import load_config
from financial_analyst.inference import load_api_key
from financial_analyst.web import _access_code


class BundleError(ValueError):
    """Preparation failed; diagnostics must not include private file contents."""


@dataclass(frozen=True)
class BundleResult:
    directory: Path
    documents: int
    facts: int
    provider: str
    model: str


def _private_directory(path):
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)


def _write(path, content):
    with path.open('x', encoding='utf-8') as output:
        os.chmod(path, 0o600)
        output.write(content)


def _copy(source, destination):
    # Destination directories are owner-only before source bytes are copied.
    with source.open('rb') as input_file, destination.open('xb') as output:
        os.chmod(destination, 0o600)
        shutil.copyfileobj(input_file, output)


def _validate(config):
    validator = ReviewedCorpusAdapter(config['proof_pack'], config['catalog'])
    approved = validator.read_facts()
    if not approved:
        raise BundleError('A deployment requires approved reviewed facts.')
    adapter = SQLiteCorpusAdapter(config['database'], validator)
    stored = adapter.read_facts(include_unreviewed=True)
    if {fact.fact_id for fact in stored} != {fact.fact_id for fact in approved}:
        raise BundleError('SQLite must contain exactly the selected approved facts.')
    adapter.companies()
    return len(approved)


def _snapshot_database(source, destination):
    # SQLite backup includes committed WAL data and produces one portable file.
    original = sqlite3.connect(source.resolve().as_uri() + '?mode=ro', uri=True)
    copied = None
    try:
        destination.touch(mode=0o600, exist_ok=False)
        copied = sqlite3.connect(destination)
        original.backup(copied)
        copied.commit()
        # Backup can inherit WAL mode. The deployed adapter is read-only and
        # must not need writable WAL/SHM sidecars under systemd hardening.
        if copied.execute('PRAGMA journal_mode=DELETE').fetchone()[0] != 'delete':
            raise BundleError('The copied SQLite snapshot requires a portable journal mode.')
        if copied.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise BundleError('The copied SQLite snapshot is invalid.')
    finally:
        if copied is not None:
            copied.close()
        original.close()


def _publish(bundle, output_path):
    # mkdir is the no-clobber reservation: unlike rename, it never replaces an
    # existing directory created by another preparation while validation runs.
    output_path.mkdir(mode=0o700)
    try:
        for child in bundle.iterdir():
            child.rename(output_path/child.name)
    except BaseException:
        shutil.rmtree(output_path)
        raise


def prepare_bundle(config_path, output_path, access_code_path):
    """Copy one session, validate it offline, then publish its private directory."""
    output_path = Path(output_path).expanduser().absolute()
    private_root = (PROJECT_ROOT / '.local').resolve()
    # Keep bundles beneath the ignored directory, including through symlinks.
    if (not output_path.resolve().is_relative_to(private_root)
            or output_path.resolve() == private_root or output_path.exists()
            or output_path.is_symlink()):
        raise BundleError('Choose a new bundle directory beneath project .local/.')
    try:
        config = load_config(config_path)
        if config['mode'] != 'live' or not config.get('model'):
            raise BundleError('An explicit live provider and model are required.')
        fact_count = _validate(config)
        code = _access_code(Path(access_code_path).read_text(encoding='utf-8').strip())
        key_env = config.get('api_key_env') or {
            'groq': 'GROQ_API_KEY', 'mistral': 'MISTRAL_API_KEY',
            'openai-compatible': 'INFERENCE_API_KEY',
        }[config['provider']]
        key = load_api_key(key_env, config.get('env_file'))
        catalog = json.loads(config['catalog'].read_text(encoding='utf-8'))
        documents = catalog['documents']
        filenames = [document['document_name'] for document in documents]
        if (len(set(filenames)) != len(filenames)
                or any(Path(name).name != name or name in {'.', '..'}
                       or '\\' in name for name in filenames)):
            raise BundleError('Source filenames must be unique portable basenames.')
        _private_directory(output_path.parent)
        with TemporaryDirectory(prefix='.ec2-prepare-', dir=output_path.parent) as temporary:
            bundle = Path(temporary) / 'bundle'
            for directory in (bundle, bundle/'app', bundle/'app'/'financial_analyst',
                              bundle/'private', bundle/'private'/'sources', bundle/'deploy'):
                _private_directory(directory)
            for source in sorted((PROJECT_ROOT/'financial_analyst').glob('*.py')):
                _copy(source, bundle/'app'/'financial_analyst'/source.name)
            _copy(PROJECT_ROOT/'requirements.txt', bundle/'app'/'requirements.txt')
            for source in sorted((PROJECT_ROOT/'deploy'/'ec2').iterdir()):
                if source.is_file():
                    _copy(source, bundle/'deploy'/source.name)
            private = bundle/'private'
            _copy(config['proof_pack'], private/'proof-pack.json')
            for document in documents:
                source = Path(document['local_path'])
                if not source.is_absolute():
                    source = config['catalog'].parent/source
                _copy(source, private/'sources'/document['document_name'])
                document['local_path'] = 'sources/' + document['document_name']
            _write(private/'catalog.json', json.dumps(catalog, ensure_ascii=False, indent=2)+'\n')
            _snapshot_database(config['database'], private/'corpus.sqlite')
            # Copy only the selected provider credential, never unrelated env keys.
            from dotenv import set_key
            _write(private/'provider.env', '')
            set_key(private/'provider.env', key_env, key, quote_mode='always')
            (private/'provider.env').chmod(0o600)
            if load_api_key(key_env, private/'provider.env') != key:
                raise BundleError('The selected credential could not be copied safely.')
            _write(private/'browser.env', 'FINANCIAL_ANALYST_ACCESS_CODE='+code+'\n')
            portable = {field: value for field, value in config.items()
                        if field not in {'proof_pack', 'catalog', 'database', 'env_file'}}
            portable.update(proof_pack='proof-pack.json', catalog='catalog.json',
                            database='corpus.sqlite', env_file='provider.env', api_key_env=key_env)
            _write(private/'config.json', json.dumps(portable, ensure_ascii=False, indent=2)+'\n')
            copied_fact_count = _validate(load_config(private/'config.json'))
            if copied_fact_count != fact_count:
                raise BundleError('The deployment copy changed approved fact coverage.')
            _write(bundle/'verification.json', json.dumps({
                'documents': len(documents), 'facts': fact_count,
                'provider': config['provider'], 'model': config['model'],
                'validation': 'authenticated_sources_and_real_sqlite',
                'inference': 'not_called', 'deployed': False,
            }, indent=2)+'\n')
            # All private bytes are validated before the requested path appears.
            _publish(bundle, output_path)
        return BundleResult(output_path, len(documents), fact_count, config['provider'], config['model'])
    except BundleError:
        raise
    except Exception:
        raise BundleError('Bundle preparation failed; check selected private artifacts and source integrity.') from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT/'.local'/'ec2-bundle')
    parser.add_argument('--access-code-file', type=Path, default=PROJECT_ROOT/'.local'/'browser-access-code')
    args = parser.parse_args(argv)
    try:
        result = prepare_bundle(args.config, args.output, args.access_code_file)
    except BundleError as error:
        print(str(error), file=sys.stderr)
        return 2
    print(f'Private EC2 bundle ready: {result.documents} documents, {result.facts} reviewed facts. '
          'Sources and SQLite validated; inference not called; not deployed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
