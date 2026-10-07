from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from financial_analyst.cli import main
from financial_analyst.adapters import FileFixtureAdapter
from financial_analyst.semantic import SemanticGrowthAdapter
from financial_analyst.selection import GroqPassageSelector
from financial_analyst.service import ApplicationService
from tests.support import create_fixture
from tests.test_selection import FakeClient


class SemanticCliTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.fixture = create_fixture(Path(self.directory.name))

    def invoke(self, arguments, response):
        client = FakeClient(response)
        stdout, stderr = io.StringIO(), io.StringIO()
        selector = GroqPassageSelector(client=client)
        with patch("financial_analyst.selection.load_groq_key", return_value="synthetic-credential"), \
             patch("financial_analyst.selection.GroqPassageSelector", return_value=selector), \
             redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(arguments)
        return code, stdout.getvalue(), stderr.getvalue(), client

    def flags(self):
        return ["--mode", "fixture", "--fixture", str(self.fixture), "--retrieval", "semantic",
                "--llm", "groq", "--format", "json"]

    def test_semantic_search_exposes_ordinal_rank_and_truthful_test_execution(self):
        code, output, error, client = self.invoke(
            [*self.flags(), "kb-search", "What helped turnover expand?", "--company", "Example Pharma", "--period", "1QFY27",
             "--scope", "consolidated", "--metric", "net_sales", "--kind", "broker_commentary"],
            {"abstain": False, "selected_refs": ["growth"]},
        )
        self.assertEqual(code, 0, error)
        result = json.loads(output)
        self.assertEqual(result["status"], "retrieved")
        self.assertEqual(result["execution"]["retrieval"], "groq_semantic_selection")
        self.assertEqual(result["execution"]["llm"]["mode"], "test_double")
        self.assertTrue(result["execution"]["no_llm"])
        self.assertEqual(result["hits"][0]["rank"], 1)
        self.assertNotIn("score", result["hits"][0])
        self.assertEqual(result["claims"], [])
        self.assertEqual(len(client.calls), 1)

    def test_combined_semantic_answer_calls_model_once_and_preserves_canonical_claims(self):
        code, output, error, client = self.invoke(
            [*self.flags(), "combined", "--company", "Example Pharma", "--period", "1QFY27", "--yoy"],
            {"abstain": False, "selected_refs": ["growth"]},
        )
        self.assertEqual(code, 0, error)
        answer = json.loads(output)
        self.assertEqual(answer["status"], "answered")
        self.assertEqual(answer["claims"][0]["values"]["delta"], "20")
        self.assertEqual(answer["execution"]["retrieval"], "groq_semantic_selection")
        self.assertTrue(answer["execution"]["selection_executed"])
        refs = {citation["ref"] for citation in answer["citations"]}
        self.assertTrue(all(set(claim["evidence_refs"]) <= refs for claim in answer["claims"]))
        self.assertEqual(len(client.calls), 1)

    def test_semantic_abstention_refuses_without_claiming_source_absence(self):
        code, output, error, client = self.invoke(
            [*self.flags(), "combined", "--company", "Example Pharma", "--period", "1QFY27"],
            {"abstain": True, "selected_refs": []},
        )
        self.assertEqual(code, 1, error)
        answer = json.loads(output)
        self.assertEqual(answer["claims"], [])
        self.assertEqual(answer["status"], "refused")
        self.assertIn("does not establish", answer["reason"])
        self.assertEqual(answer["execution"]["llm"]["outcome"], "abstained")
        self.assertEqual(len(client.calls), 1)

    def test_semantic_port_attribution_refusal_uses_no_model_even_after_reuse(self):
        fixture = FileFixtureAdapter(self.fixture)
        client = FakeClient({"abstain": False, "selected_refs": ["growth"]})
        selector = GroqPassageSelector(client=client)
        service = ApplicationService(fixture, fixture, SemanticGrowthAdapter(fixture, selector))
        first = service.refuse_beat_attribution("Example Pharma", "1QFY27")
        self.assertEqual(first.status, "refused")
        self.assertEqual(client.calls, [])
        self.assertEqual(first.execution["retrieval"], "curated_fixture")
        service.growth_answer("Example Pharma", "1QFY27")
        self.assertEqual(len(client.calls), 1)
        again = service.refuse_beat_attribution("Example Pharma", "1QFY27")
        self.assertEqual(len(client.calls), 1)
        self.assertTrue(again.execution["no_llm"])
        self.assertFalse(again.execution["selection_executed"])


if __name__ == "__main__":
    unittest.main()
