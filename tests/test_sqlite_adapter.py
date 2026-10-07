from dataclasses import replace
from contextlib import closing, contextmanager
from decimal import Decimal, Inexact, localcontext
import importlib.util
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from financial_analyst.adapters import EvidenceError, FileFixtureAdapter
from financial_analyst.service import ApplicationService
from financial_analyst.sqlite_adapter import (
    DocumentRecord, IngestionResult, SCHEMA_VERSION,
    SQLiteAnalyticsAdapter, SQLiteStore, StorageError,
)
from tests.support import create_fixture


@contextmanager
def database_connection(path):
    with closing(sqlite3.connect(path)) as connection:
        with connection:
            yield connection


@unittest.skipUnless(importlib.util.find_spec("pdfplumber"), "optional PDF fixture dependency unavailable")
class SQLiteAdapterTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.fixture = FileFixtureAdapter(create_fixture(self.root / "source"))
        self.observations = self.fixture.reviewed_observations()
        self.evidence = self.fixture.reviewed_evidence()
        self.document = DocumentRecord(self.fixture.document_name, self.fixture.source_sha256,
                                       self.fixture.source_url)
        self.database = self.root / "analyst.sqlite3"
        self.store = SQLiteStore(self.database)
        self.store.initialize()

    def ingest(self, observations=None, evidence=None, document=None, validator=None):
        return self.store.ingest(document or self.document,
                                 self.observations if observations is None else observations,
                                 self.evidence if evidence is None else evidence,
                                 validator or self.fixture)

    def reader(self):
        return SQLiteAnalyticsAdapter(self.database, self.fixture.source_sha256)

    def counts(self):
        with database_connection(self.database) as connection:
            return tuple(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                         for table in ("documents", "evidence", "observations", "observation_evidence"))

    def test_authenticated_roundtrip_preserves_every_field_and_evidence(self):
        result = self.ingest()
        self.assertEqual(result, IngestionResult(1, len(self.evidence), 3))
        adapter = self.reader()
        self.assertEqual(adapter.mode, "sqlite")
        self.assertEqual(adapter.seed_provenance, ("reviewed_fixture",))
        for period in ("1QFY27", "1QFY26"):
            expected = self.fixture.read_observations("Example Pharma", period)
            actual = adapter.read_observations("Example Pharma", period)
            self.assertCountEqual(actual, expected)
        self.assertEqual(adapter.read_observations("Other Pharma", "1QFY27"), ())
        self.assertEqual(adapter.read_observations("Example Pharma", "2QFY27"), ())

    def test_database_backed_comparison_is_revalidated_against_the_pdf(self):
        self.ingest()
        service = ApplicationService(self.reader(), self.fixture, self.fixture)
        answer = service.comparison_answer("Example Pharma", "1QFY27", True, True)
        self.assertEqual(answer.status, "answered")
        self.assertEqual(answer.comparison.delta_millions, Decimal("20"))
        self.assertEqual(answer.comparison.yoy_percent, Decimal("50"))
        cited = {item.ref for item in answer.citations}
        self.assertTrue(all(set(claim.evidence_refs) <= cited for claim in answer.claims))

    def test_decimal_exponent_and_digits_survive_without_context_rounding(self):
        actual = replace(self.observations[0], value=Decimal("120." + "0" * 60))
        records = (actual, *self.observations[1:])
        with localcontext() as context:
            context.prec = 6
            context.traps[Inexact] = True
            self.ingest(records)
            loaded = next(item for item in self.reader().read_observations("Example Pharma", "1QFY27")
                          if item.kind == "reported_actual")
        self.assertEqual(loaded.value.as_tuple(), actual.value.as_tuple())
        with database_connection(self.database) as connection:
            stored = connection.execute("SELECT value_text, typeof(value_text) FROM observations WHERE kind='reported_actual' AND period='1QFY27'").fetchone()
        self.assertEqual(stored, (str(actual.value), "text"))

    def test_repeat_import_and_initialize_are_idempotent(self):
        first = self.ingest()
        counts = self.counts()
        self.store.initialize()
        self.assertEqual(self.ingest(), IngestionResult(0, 0, 0))
        self.assertEqual(first.observations_added, 3)
        self.assertEqual(self.counts(), counts)

    def test_conflicting_amount_payload_rolls_back_new_evidence_and_observations(self):
        actual = self.observations[0]
        initial_evidence = tuple(item for item in self.evidence if item.ref in actual.evidence_refs)
        self.ingest((actual,), initial_evidence)
        before = self.counts()
        # Equal financial value still differs in preserved decimal precision.
        # The source validator accepts it; persistence rejects conflicting payloads.
        changed = replace(actual, value=Decimal("120.00"))
        with self.assertRaisesRegex(StorageError, "Conflicting observations"):
            self.ingest((*self.observations[1:], changed))
        self.assertEqual(self.counts(), before)
        self.assertEqual(self.reader().read_observations("Example Pharma", "1QFY26"), ())

    def test_conflicting_evidence_order_does_not_overwrite_a_record(self):
        self.ingest()
        changed = replace(self.observations[0], evidence_refs=self.observations[0].evidence_refs[::-1])
        with self.assertRaisesRegex(StorageError, "Conflicting observation evidence"):
            self.ingest((changed,))
        actual = next(item for item in self.reader().read_observations("Example Pharma", "1QFY27")
                      if item.kind == "reported_actual")
        self.assertEqual(actual.evidence_refs, self.observations[0].evidence_refs)

    def test_conflicting_document_metadata_does_not_overwrite_seed_provenance(self):
        self.ingest()
        with self.assertRaisesRegex(StorageError, "Conflicting documents"):
            self.ingest(document=replace(self.document, provenance="unreviewed"))
        self.assertEqual(self.reader().seed_provenance, ("reviewed_fixture",))

    def test_source_validation_failure_leaves_no_partial_import(self):
        changed = replace(self.observations[0], value=Decimal("121"), source_value_raw="121")
        with self.assertRaises(EvidenceError):
            self.ingest((*self.observations[1:], changed))
        self.assertEqual(self.counts(), (0, 0, 0, 0))

    def test_missing_foreign_or_forged_evidence_is_rejected_before_writes(self):
        for evidence in (tuple(item for item in self.evidence if item.ref != "actual"),
                         (replace(self.evidence[0], document_name="foreign.pdf"),),
                         (replace(self.evidence[0], excerpt="Invented evidence"),)):
            with self.subTest(evidence=evidence):
                with self.assertRaises(StorageError):
                    self.ingest(evidence=evidence)
                self.assertEqual(self.counts(), (0, 0, 0, 0))

    def test_invalid_decimal_types_and_bad_identities_are_rejected(self):
        for value in (120.0, "120", Decimal("NaN"), Decimal("Infinity")):
            with self.subTest(value=value), self.assertRaisesRegex(StorageError, "finite Decimal"):
                self.ingest((replace(self.observations[0], value=value),))
        for digest in ("", "0" * 63, "G" * 64):
            with self.subTest(digest=digest), self.assertRaises(StorageError):
                self.ingest(document=replace(self.document, sha256=digest))
        with self.assertRaisesRegex(StorageError, "source validator"):
            self.ingest(document=replace(self.document, sha256="0" * 64))
        self.assertEqual(self.counts(), (0, 0, 0, 0))

    def test_missing_database_reads_and_ingestion_never_create_a_file(self):
        missing = self.root / "missing.sqlite3"
        with self.assertRaises(StorageError):
            SQLiteAnalyticsAdapter(missing)
        with self.assertRaises(StorageError):
            SQLiteStore(missing).ingest(self.document, self.observations, self.evidence, self.fixture)
        self.assertFalse(missing.exists())

    def test_read_only_connections_cannot_modify_database(self):
        self.ingest()
        before = self.database.read_bytes()
        reader = self.reader()
        statements = []
        original_connect = sqlite3.connect

        # Capture the actual URI used by the read adapter, then attempt a write
        # through precisely the same connection configuration.
        def capture(location, **kwargs):
            statements.append((location, kwargs))
            return original_connect(location, **kwargs)

        from unittest.mock import patch
        with patch("financial_analyst.sqlite_adapter.sqlite3.connect", side_effect=capture):
            reader.read_observations("Example Pharma", "1QFY27")
        uri, kwargs = statements[0]
        self.assertTrue(uri.endswith("?mode=ro"))
        with closing(original_connect(uri, **kwargs)) as connection:
            with self.assertRaises(sqlite3.OperationalError):
                connection.execute("DELETE FROM observations")
        self.assertEqual(self.database.read_bytes(), before)

    def test_unknown_schema_version_is_rejected_without_mutation(self):
        with database_connection(self.database) as connection:
            connection.execute("PRAGMA user_version = 99")
        before = self.database.read_bytes()
        for operation in (self.store.initialize, self.ingest,
                          lambda: SQLiteAnalyticsAdapter(self.database)):
            with self.subTest(operation=operation), self.assertRaisesRegex(StorageError, "schema version"):
                operation()
        self.assertEqual(self.database.read_bytes(), before)

    def test_existing_unversioned_foreign_database_is_preserved(self):
        foreign = self.root / "foreign.sqlite3"
        with database_connection(foreign) as connection:
            connection.execute("CREATE TABLE user_records (value TEXT)")
        before = foreign.read_bytes()
        with self.assertRaises(StorageError):
            SQLiteStore(foreign).initialize()
        self.assertEqual(foreign.read_bytes(), before)

    def test_corrupt_shape_and_tampered_values_cannot_support_answers(self):
        self.ingest()
        with database_connection(self.database) as connection:
            connection.execute("UPDATE observations SET value_text='121', raw_labels_json=? WHERE period='1QFY27' AND kind='reported_actual'",
                               (json.dumps({**{field: getattr(self.observations[0], field) for field in
                                              ("year_end_raw", "source_header_raw", "source_value_raw", "source_unit_raw", "source_metric_raw")},
                                            "source_value_raw": "121"}),))
        answer = ApplicationService(self.reader(), self.fixture).comparison_answer("Example Pharma", "1QFY27")
        self.assertEqual(answer.status, "refused")
        self.assertEqual(answer.claims, ())
        with database_connection(self.database) as connection:
            connection.execute("UPDATE observations SET raw_labels_json='[]'")
        with self.assertRaisesRegex(StorageError, "raw labels"):
            self.reader().read_observations("Example Pharma", "1QFY27")

    def test_new_metric_period_and_kind_are_rows_without_schema_changes(self):
        self.ingest()
        original = self.observations[0]
        candidate = replace(original, metric="revenue", kind="company_reported", period="1QFY2027")
        fixture = self.fixture

        class AlternateMapping:
            # A synthetic future source adapter maps the same printed Net Sales
            # cell to a revenue metric and a four-digit fiscal-year identity.
            source_sha256 = fixture.source_sha256

            def validate_observation(self, observation):
                if observation != candidate:
                    raise EvidenceError("Alternate mapping record is not reviewed.")
                fixture.validate_observation(original)

            def resolve(self, ref):
                return fixture.resolve(ref)

        with database_connection(self.database) as connection:
            before = connection.execute("SELECT sql FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
        self.ingest((candidate,), validator=AlternateMapping())
        self.assertEqual(self.reader().read_observations("Example Pharma", "1QFY2027"), (candidate,))
        with database_connection(self.database) as connection:
            after = connection.execute("SELECT sql FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION)
        self.assertEqual(before, after)

    def test_document_revisions_scope_identical_reference_names_by_hash(self):
        self.ingest()
        second_path = create_fixture(self.root / "revision")
        payload = json.loads(second_path.read_text())
        pdf = Path(payload["source"]["local_path"])
        pdf.write_bytes(pdf.read_bytes() + b"\n% reviewed synthetic revision\n")
        import hashlib
        payload["source"]["sha256"] = hashlib.sha256(pdf.read_bytes()).hexdigest()
        second_path.write_text(json.dumps(payload))
        second = FileFixtureAdapter(second_path)
        second_document = DocumentRecord(second.document_name, second.source_sha256, second.source_url)
        result = self.ingest(second.reviewed_observations(), second.reviewed_evidence(), second_document, second)
        self.assertEqual(result.documents_added, 1)
        self.assertEqual(self.counts()[:3], (2, 2 * len(self.evidence), 6))
        qualified = SQLiteAnalyticsAdapter(self.database).read_observations("Example Pharma", "1QFY27")
        self.assertEqual(len(qualified), 4)
        self.assertTrue(all(":" in ref for item in qualified for ref in item.evidence_refs))
        actual_refs = {item.evidence_refs[1] for item in qualified if item.kind == "reported_actual"}
        self.assertEqual(len(actual_refs), 2)
        bound = SQLiteAnalyticsAdapter(self.database, second.source_sha256)
        self.assertCountEqual(bound.read_observations("Example Pharma", "1QFY27"),
                              second.read_observations("Example Pharma", "1QFY27"))
        # Identical local references and amounts cannot substitute another PDF
        # revision for the database's explicitly bound source identity.
        wrong_source = ApplicationService(bound, self.fixture, self.fixture).comparison_answer(
            "Example Pharma", "1QFY27", True, True)
        self.assertEqual(wrong_source.status, "refused")
        self.assertEqual(wrong_source.claims, ())
        matching_source = ApplicationService(bound, second, second).comparison_answer(
            "Example Pharma", "1QFY27", True, True)
        self.assertEqual(matching_source.status, "answered")
        self.assertEqual(matching_source.comparison.delta_millions, Decimal("20"))
        # An unbound port cannot accidentally reuse these local refs against the
        # first PDF; it needs a source-aware evidence resolver.
        answer = ApplicationService(SQLiteAnalyticsAdapter(self.database), self.fixture).comparison_answer("Example Pharma", "1QFY27")
        self.assertEqual(answer.status, "refused")

    def test_sql_text_in_filters_is_treated_as_data(self):
        self.ingest()
        reader = self.reader()
        self.assertEqual(reader.read_observations("' OR 1=1 --", "1QFY27"), ())
        self.assertEqual(reader.read_observations("Example Pharma", "' OR 1=1 --"), ())
        self.assertEqual(self.counts()[2], 3)

    def test_missing_requested_document_is_explicit(self):
        self.ingest()
        with self.assertRaisesRegex(StorageError, "absent"):
            SQLiteAnalyticsAdapter(self.database, "0" * 64)

    def test_scope_metric_and_kind_filters_never_broaden_selection(self):
        self.ingest()
        reader = self.reader()
        selected = reader.read_observations("Example Pharma", "1QFY27", scope="consolidated",
                                            metric="net_sales", kind="reported_actual")
        self.assertEqual(selected, (self.observations[0],))
        for filters in ({"scope": "standalone"}, {"metric": "operating_profit"},
                        {"kind": "broker_forecast"}, {"metric": "' OR 1=1 --"}):
            with self.subTest(filters=filters):
                self.assertEqual(reader.read_observations("Example Pharma", "1QFY27", **filters), ())
        with self.assertRaises(StorageError):
            reader.read_observations("Example Pharma", "1QFY27", metric="")

    def test_missing_evidence_positions_fail_before_returning_records(self):
        self.ingest()
        with database_connection(self.database) as connection:
            connection.execute("DELETE FROM observation_evidence WHERE position = 0")
        with self.assertRaisesRegex(StorageError, "document bindings or order"):
            self.reader().read_observations("Example Pharma", "1QFY27")


if __name__ == "__main__":
    unittest.main()
