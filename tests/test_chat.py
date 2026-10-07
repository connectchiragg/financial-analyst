"""Interactive session checks with synthetic sources and inference only."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from financial_analyst.chat import main, run_chat
from financial_analyst.inference import InferenceResponse
from tests.support import create_fixture


class FakeInference:
    provider = "example-provider"
    model = "example-model"
    mode = "test_double"
    capabilities = frozenset({"json_schema"})

    def __init__(self, foreign=False, incomplete=False, fail=False):
        self.calls = []
        self.closed = 0
        self.foreign, self.incomplete, self.fail = foreign, incomplete, fail

    def infer(self, request):
        self.calls.append(request)
        if self.fail:
            raise RuntimeError("Sensitive provider detail SECRET_KEY_VALUE")
        if request.schema_name.endswith("plan"):
            payload = {"tool": "combined_revenue", "company": "Another Company" if self.foreign else "Example Pharma",
                       "period": "1QFY27", "include_yoy": True, "unsupported_parts": []}
        else:
            payload = {"complete": not self.incomplete, "unsupported_parts": ["unsupported request"] if self.incomplete else []}
        return InferenceResponse(json.dumps(payload), "stop")

    def close(self):
        self.closed += 1


def inputs(*values):
    sequence = iter(values)
    def read(_prompt):
        value = next(sequence)
        if isinstance(value, BaseException):
            raise value
        return value
    return read


class ChatTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.fixture = create_fixture(self.root)
        self.config_path = self.root / "session.json"
        self.config = {"provider": "groq", "model": "example-model", "fixture": self.fixture.name,
                       "company": "Example Pharma", "period": "1QFY27", "retrieval": "fixture"}
        self.write_config()

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config))

    def invoke(self, questions, inference=None):
        inference = inference or FakeInference()
        output = []
        with patch("financial_analyst.chat.create_inference", return_value=inference) as factory:
            code = run_chat(self.config_path, inputs(*questions), output.append)
        return code, "\n".join(output), inference, factory

    def test_repeated_questions_cited_values_and_one_cleanup(self):
        code, output, inference, factory = self.invoke(["How did revenue perform?", "Explain growth and compare revenue.", "/quit"])
        self.assertEqual(code, 0)
        self.assertEqual(len(inference.calls), 4)
        self.assertEqual(inference.closed, 1)
        self.assertEqual(factory.call_count, 1)
        self.assertIn("Experimental", output)
        self.assertIn("Example Pharma, 1QFY27", output)
        self.assertIn("beat by 20 INR million", output)
        self.assertIn("synthetic-report.pdf", output)
        self.assertIn("Broader corpus analytics is unavailable", output)

    def test_quit_eof_interrupt_cleanup_without_inference(self):
        for ending in ["/quit", EOFError(), KeyboardInterrupt()]:
            with self.subTest(ending=type(ending).__name__):
                code, _, inference, _ = self.invoke([ending])
                self.assertEqual(code, 0)
                self.assertFalse(inference.calls)
                self.assertEqual(inference.closed, 1)

    def test_blank_questions_are_skipped(self):
        code, _, inference, _ = self.invoke(["", "   ", "/quit"])
        self.assertEqual(code, 0)
        self.assertFalse(inference.calls)

    def test_foreign_planned_company_is_refused(self):
        code, output, inference, _ = self.invoke(["Compare another company.", "/quit"], FakeInference(foreign=True))
        self.assertEqual(code, 0)
        self.assertIn("Refused", output)
        self.assertNotIn("beat by", output)
        self.assertEqual(len(inference.calls), 1)

    def test_different_period_is_refused_before_inference(self):
        code, output, inference, _ = self.invoke(["What was 2QFY27 revenue?", "/quit"])
        self.assertEqual(code, 0)
        self.assertIn("Refused", output)
        self.assertFalse(inference.calls)

    def test_incomplete_whole_question_refuses_without_partial_claims(self):
        code, output, _, _ = self.invoke(["Compare revenue and something unsupported.", "/quit"], FakeInference(incomplete=True))
        self.assertEqual(code, 0)
        self.assertIn("Refused", output)
        self.assertNotIn("beat by", output)

    def test_provider_error_is_sanitized_and_session_continues(self):
        code, output, inference, _ = self.invoke(["What was revenue?", "Another question?", "/quit"], FakeInference(fail=True))
        self.assertEqual(code, 0)
        self.assertIn("Refused", output)
        self.assertNotIn("SECRET_KEY_VALUE", output)
        self.assertEqual(len(inference.calls), 2)
        self.assertEqual(inference.closed, 1)

    def test_config_rejects_plaintext_credentials_and_semantic_before_provider(self):
        for field, value in [("api_key", "SECRET_KEY_VALUE"), ("retrieval", "semantic"), ("period", "FY27")]:
            with self.subTest(field=field):
                self.config[field] = value
                self.write_config()
                code, output, inference, factory = self.invoke(["/quit"])
                self.assertEqual(code, 2)
                factory.assert_not_called()
                self.assertNotIn("SECRET_KEY_VALUE", output)
                self.config.pop(field)
                self.config.update({"retrieval": "fixture", "period": "1QFY27"})

    def test_unreviewed_company_config_rejected_before_provider(self):
        self.config["company"] = "Another Company"
        self.write_config()
        code, _, _, factory = self.invoke(["/quit"])
        self.assertEqual(code, 2)
        factory.assert_not_called()

    def test_startup_exception_is_sanitized(self):
        output = []
        with patch("financial_analyst.chat.create_inference", side_effect=RuntimeError("SECRET_KEY_VALUE")):
            self.assertEqual(run_chat(self.config_path, inputs("/quit"), output.append), 2)
        self.assertNotIn("SECRET_KEY_VALUE", "\n".join(output))

    def test_failed_agent_construction_closes_provider(self):
        inference = FakeInference()
        output = []
        with patch("financial_analyst.chat.create_inference", return_value=inference), \
             patch("financial_analyst.chat.ToolPlanningAgent", side_effect=RuntimeError("SECRET_KEY_VALUE")):
            self.assertEqual(run_chat(self.config_path, inputs("/quit"), output.append), 2)
        self.assertEqual(inference.closed, 1)
        self.assertNotIn("SECRET_KEY_VALUE", "\n".join(output))

    def test_module_entry_accepts_config(self):
        with patch("financial_analyst.chat.run_chat", return_value=0) as runner:
            self.assertEqual(main(["--config", str(self.config_path)]), 0)
        runner.assert_called_once_with(self.config_path)

    def test_duplicate_configuration_keys_rejected_without_provider(self):
        self.config_path.write_text('{"provider":"groq","provider":"mistral"}')
        code, _, _, factory = self.invoke(["/quit"])
        self.assertEqual(code, 2)
        factory.assert_not_called()

    def test_input_error_is_sanitized_and_provider_closed(self):
        code, output, inference, _ = self.invoke([RuntimeError("SECRET_KEY_VALUE")])
        self.assertEqual(code, 2)
        self.assertNotIn("SECRET_KEY_VALUE", output)
        self.assertEqual(inference.closed, 1)

    def test_sqlite_session_reads_actual_database(self):
        from financial_analyst.adapters import FileFixtureAdapter
        from financial_analyst.sqlite_adapter import DocumentRecord, SQLiteStore
        fixture = FileFixtureAdapter(self.fixture)
        database = self.root / "test.sqlite"
        store = SQLiteStore(database)
        store.initialize()
        store.ingest(DocumentRecord(fixture.document_name, fixture.source_sha256, fixture.source_url),
                     fixture.reviewed_observations(), fixture.reviewed_evidence(), fixture)
        self.config["database"] = database.name
        self.write_config()
        code, output, inference, _ = self.invoke(["How did revenue perform?", "/quit"])
        self.assertEqual(code, 0)
        self.assertIn("database: sqlite", output)
        self.assertIn("Database seed provenance: reviewed_fixture", output)
        self.assertIn("beat by 20 INR million", output)
        self.assertEqual(inference.closed, 1)


if __name__ == "__main__":
    unittest.main()
