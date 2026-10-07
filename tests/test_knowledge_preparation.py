from dataclasses import FrozenInstanceError, replace
import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from financial_analyst.adapters import FileFixtureAdapter
from financial_analyst.knowledge import KnowledgeError
from financial_analyst.knowledge_preparation import (
    KnowledgePreparationError,
    PreparedKnowledgeDocument,
    PreparedKnowledgeGroup,
    prepare_knowledge_groups,
    write_prepared_knowledge,
)
from tests.support import create_fixture


@unittest.skipUnless(importlib.util.find_spec("pdfplumber"), "PDF dependency unavailable")
class PreparedKnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.fixture_path = create_fixture(self.root / "source")
        self.fixture = FileFixtureAdapter(self.fixture_path)

    def prepare(self, **filters):
        return prepare_knowledge_groups(
            self.fixture, company="Example Pharma", period="1QFY27", **filters,
        )

    def test_exact_filters_are_applied_before_preparation(self):
        groups = self.prepare(scope="consolidated", metric="net_sales", kind="broker_commentary",
                              currency="INR", unit="INRm")
        self.assertEqual(len(groups), 1)
        self.assertEqual({item.record_ref for item in groups[0].documents}, {"growth", "portfolio"})
        for filters in [{"scope": "standalone"}, {"metric": "profit"}, {"kind": "broker_forecast"},
                        {"currency": "USD"}, {"unit": "million"}, {"unit": "INRb"}]:
            with self.subTest(filters=filters):
                self.assertEqual(self.prepare(**filters), ())
        for company, period in [("Other Company", "1QFY27"), ("example pharma", "1QFY27"),
                                ("Example Pharma", "2QFY27"), ("Example Pharma", "1qfy27")]:
            with self.subTest(company=company, period=period):
                self.assertEqual(prepare_knowledge_groups(self.fixture, company=company, period=period), ())

    def test_multi_context_evidence_becomes_distinct_single_context_documents(self):
        groups = self.prepare()
        self.assertEqual({group.context.kind for group in groups},
                         {"reported_actual", "broker_estimate", "broker_commentary"})
        table_documents = [item for group in groups for item in group.documents if item.record_ref == "context"]
        self.assertEqual({item.context.kind for item in table_documents}, {"reported_actual", "broker_estimate"})
        self.assertEqual(len({item.file_name for item in table_documents}), 2)
        for document in table_documents:
            payload = json.loads(document.content)
            self.assertEqual(payload["context"]["kind"], document.context.kind)
            self.assertNotIn("1QFY26", payload["context_prefix"])
            other_kind = "broker_estimate" if document.context.kind == "reported_actual" else "reported_actual"
            self.assertNotIn(other_kind, payload["context_prefix"])
            supports = set(payload["source"]["supporting_evidence_refs"])
            self.assertNotIn("prior", supports)
            self.assertNotIn("estimate" if document.context.kind == "reported_actual" else "actual", supports)

    def test_quotes_citations_support_closure_and_raw_financial_labels_are_preserved(self):
        groups = self.prepare(kind="reported_actual")
        for document in groups[0].documents:
            payload = json.loads(document.content)
            evidence = self.fixture.resolve(document.record_ref)
            self.assertEqual(payload["quote"], evidence.excerpt)
            self.assertEqual(document.citation, evidence.citation)
            self.assertEqual(document.source_sha256, self.fixture.source_sha256)
            self.assertEqual(payload["source"]["sha256"], self.fixture.source_sha256)
            self.assertEqual(payload["source"]["citation"]["page"], evidence.page)
            self.assertEqual(payload["source"]["citation"]["locator"], evidence.locator)
            self.assertEqual(payload["source"]["citation"]["link"], evidence.link)
            self.assertEqual(payload["source"]["supporting_evidence_refs"],
                             [item.ref for item in document.supporting_citations])
            self.assertIn(document.record_ref, payload["source"]["supporting_evidence_refs"])
            for citation in document.supporting_citations:
                self.assertEqual(citation, self.fixture.resolve(citation.ref).citation)
            self.assertTrue(payload["supporting_observations"])
            for raw in payload["supporting_observations"]:
                self.assertEqual(raw["period"], "1QFY27")
                self.assertEqual(raw["kind"], "reported_actual")
                self.assertEqual(raw["value"], "120")
                self.assertEqual(raw["source_value_raw"], "120")
                self.assertEqual(raw["source_unit"], "million")
                self.assertEqual(raw["source_unit_raw"], "INRm")
                self.assertEqual(raw["source_header_raw"], "FY27E / 1Q")
                self.assertEqual(raw["year_end_raw"], "Y/E March")
        actual = next(item for item in groups[0].documents if item.record_ref == "actual")
        self.assertEqual(json.loads(actual.content)["quote"], "120")
        self.assertNotEqual(json.loads(actual.content)["quote"], json.loads(actual.content)["context_prefix"])

    def test_preparation_is_deterministic_across_observation_order_and_local_path(self):
        original = self.prepare()
        payload = json.loads(self.fixture_path.read_text())
        payload["observations"].reverse()
        self.fixture_path.write_text(json.dumps(payload), encoding="utf-8")
        reversed_fixture = FileFixtureAdapter(self.fixture_path)
        self.assertEqual(original, prepare_knowledge_groups(
            reversed_fixture, company="Example Pharma", period="1QFY27",
        ))
        second_path = create_fixture(self.root / "second-source")
        self.assertEqual(original, prepare_knowledge_groups(
            FileFixtureAdapter(second_path), company="Example Pharma", period="1QFY27",
        ))

    def test_export_is_idempotent_and_manifest_binds_content_and_context(self):
        groups = self.prepare()
        destination = self.root / "private-output"
        manifest_path = write_prepared_knowledge(groups, destination)
        snapshot = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in destination.iterdir()}
        self.assertEqual(write_prepared_knowledge(self.prepare(), destination), manifest_path)
        self.assertEqual(snapshot,
                         {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in destination.iterdir()})
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["provenance"], "reviewed_fixture")
        self.assertEqual(manifest["execution"], {
            "operation": "local_preparation_only", "remote_upload": False,
            "llm": "none", "database": "none", "sanitization": "not_performed",
        })
        for group in manifest["groups"]:
            for record in group["documents"]:
                path = destination / record["file_name"]
                content = path.read_bytes()
                self.assertEqual(hashlib.sha256(content).hexdigest(), record["content_sha256"])
                self.assertEqual(json.loads(content)["context"], group["context"])
                self.assertEqual(json.loads(content)["source"]["sha256"], record["source_sha256"])
                self.assertNotIn(b"%PDF-", content)
        self.assertEqual({path.suffix for path in destination.iterdir()}, {".json"})

    def test_conflicting_document_or_manifest_is_refused_before_other_files_are_written(self):
        groups = self.prepare()
        for target in [groups[0].documents[0].file_name, "manifest.json"]:
            with self.subTest(target=target):
                destination = self.root / target.removesuffix(".json")
                destination.mkdir()
                conflict = destination / target
                conflict.write_text("Conflicting content", encoding="utf-8")
                with self.assertRaises(KnowledgePreparationError):
                    write_prepared_knowledge(groups, destination)
                self.assertEqual(list(destination.iterdir()), [conflict])
                self.assertEqual(conflict.read_text(), "Conflicting content")

    def test_different_preparation_requires_a_new_destination(self):
        destination = self.root / "export"
        write_prepared_knowledge(self.prepare(kind="reported_actual"), destination)
        before = {path.name: path.read_bytes() for path in destination.iterdir()}
        with self.assertRaises(KnowledgePreparationError):
            write_prepared_knowledge(self.prepare(kind="broker_estimate"), destination)
        self.assertEqual(before, {path.name: path.read_bytes() for path in destination.iterdir()})

    def test_manifest_injected_after_preflight_is_refused(self):
        groups = self.prepare()
        destination = self.root / "concurrent-export"
        foreign_manifest = destination / "manifest.json"
        original_mkdir = Path.mkdir

        def inject_after_creation(path, *args, **kwargs):
            result = original_mkdir(path, *args, **kwargs)
            if path == destination:
                foreign_manifest.write_text("Foreign manifest created after preflight", encoding="utf-8")
            return result

        with patch.object(Path, "mkdir", inject_after_creation):
            with self.assertRaises(KnowledgePreparationError):
                write_prepared_knowledge(groups, destination)
        self.assertEqual(foreign_manifest.read_text(), "Foreign manifest created after preflight")

    def test_exclusive_create_collision_is_revalidated_before_success(self):
        groups = self.prepare()
        for identical in [False, True]:
            with self.subTest(identical=identical):
                destination = self.root / f"collision-{identical}"
                target = destination / groups[0].documents[0].file_name
                expected = groups[0].documents[0].content
                original_open = Path.open
                injected = False

                def collide(path, mode="r", *args, **kwargs):
                    nonlocal injected
                    if path == target and mode == "xb" and not injected:
                        injected = True
                        with original_open(path, "w", encoding="utf-8") as foreign:
                            foreign.write(expected if identical else "Foreign resource")
                    return original_open(path, mode, *args, **kwargs)

                with patch.object(Path, "open", collide):
                    if identical:
                        manifest_path = write_prepared_knowledge(groups, destination)
                        self.assertTrue(manifest_path.is_file())
                    else:
                        with self.assertRaises(KnowledgePreparationError):
                            write_prepared_knowledge(groups, destination)
                        self.assertFalse((destination / "manifest.json").exists())
                self.assertTrue(injected)
                self.assertEqual(target.read_text(), expected if identical else "Foreign resource")

    def test_filenames_are_bounded_content_ids_and_cannot_traverse_paths(self):
        groups = self.prepare()
        for group in groups:
            self.assertRegex(group.context_id, r"^[a-f0-9]{64}$")
            for document in group.documents:
                self.assertRegex(document.file_name, r"^reviewed-[a-f0-9]{64}\.json$")
                self.assertLess(len(document.file_name), 100)
                self.assertNotIn(self.root.name, document.content)
        document = groups[0].documents[0]
        object.__setattr__(document, "file_name", "../escaped.json")
        destination = self.root / "export"
        with self.assertRaises(KnowledgePreparationError):
            write_prepared_knowledge(groups, destination)
        self.assertFalse(destination.exists())
        self.assertFalse((self.root / "escaped.json").exists())

    def test_symlink_destination_or_resource_is_refused(self):
        groups = self.prepare()
        real = self.root / "real"
        real.mkdir()
        link = self.root / "link"
        link.symlink_to(real, target_is_directory=True)
        with self.assertRaises(KnowledgePreparationError):
            write_prepared_knowledge(groups, link)
        resource = real / groups[0].documents[0].file_name
        outside = self.root / "outside.json"
        outside.write_text(groups[0].documents[0].content, encoding="utf-8")
        resource.symlink_to(outside)
        with self.assertRaises(KnowledgePreparationError):
            write_prepared_knowledge(groups, real)
        self.assertEqual(list(real.iterdir()), [resource])

    def test_foreign_constructed_or_replaced_objects_are_not_labeled_reviewed(self):
        groups = self.prepare()
        group = groups[0]
        document = group.documents[0]
        foreign_group = PreparedKnowledgeGroup(group.context, group.context_id, group.documents)
        with self.assertRaises(KnowledgePreparationError):
            write_prepared_knowledge((foreign_group,), self.root / "foreign")
        foreign_document = PreparedKnowledgeDocument(
            document.record_ref, document.source_sha256, document.context, document.content,
            document.content_sha256, document.file_name, document.citation, document.supporting_citations,
        )
        object.__setattr__(group, "documents", (foreign_document,))
        with self.assertRaises(KnowledgePreparationError):
            write_prepared_knowledge((group,), self.root / "foreign-document")
        with self.assertRaises(KnowledgePreparationError):
            write_prepared_knowledge((replace(group),), self.root / "replaced")

    def test_hash_and_metadata_tampering_are_detected_before_export(self):
        for field, value in [("content", "{}\n"), ("content", None), ("content_sha256", "0" * 64),
                             ("source_sha256", "0" * 64), ("record_ref", "invented"),
                             ("citation", None), ("supporting_citations", [None])]:
            with self.subTest(field=field):
                groups = self.prepare()
                document = groups[0].documents[0]
                object.__setattr__(document, field, value)
                with self.assertRaises(KnowledgePreparationError):
                    write_prepared_knowledge(groups, self.root / field)
                self.assertFalse((self.root / field).exists())

    def test_empty_result_is_explicit_local_preparation_with_no_documents(self):
        groups = prepare_knowledge_groups(self.fixture, company="Other Company", period="1QFY27")
        path = write_prepared_knowledge(groups, self.root / "empty")
        self.assertEqual(json.loads(path.read_text())["groups"], [])
        self.assertEqual(list(path.parent.iterdir()), [path])

    def test_immutable_documents_and_input_validation(self):
        groups = self.prepare()
        with self.assertRaises(FrozenInstanceError):
            groups[0].documents[0].content = "Invented quote"
        with self.assertRaises(FrozenInstanceError):
            groups[0].context_id = "invented"
        with self.assertRaises(KnowledgeError):
            prepare_knowledge_groups(object(), company="Example Pharma", period="1QFY27")
        with self.assertRaises(KnowledgeError):
            self.prepare(scope="")
        with self.assertRaises(KnowledgePreparationError):
            write_prepared_knowledge(list(groups), self.root / "invalid")


if __name__ == "__main__":
    unittest.main()
