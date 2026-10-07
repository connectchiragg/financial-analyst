from decimal import Decimal
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from financial_analyst.adapters import FileFixtureAdapter
from financial_analyst.agent import ToolPlanningAgent
from financial_analyst.inference import InferenceResponse, ProviderError
from financial_analyst.service import ApplicationService
from tests.support import create_fixture


def plan(tool="combined_revenue", **changes):
    return {"tool": tool, "company": "Example Pharma", "period": "1QFY27", "include_yoy": False,
            "unsupported_parts": [], **changes}


class FakeInference:
    provider, model, mode = "synthetic", "local-test", "test_double"
    capabilities = frozenset({"json_schema"})
    execution = {}

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def infer(self, request):
        self.calls.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        if isinstance(response, InferenceResponse):
            return response
        return InferenceResponse(response if isinstance(response, str) else json.dumps(response), "stop")


class LocalAgentTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        fixture = FileFixtureAdapter(create_fixture(Path(self.directory.name)))
        self.service = ApplicationService(fixture, fixture, fixture)

    def run_agent(self, *responses, question="How much was revenue above estimates and why did it grow?"):
        inference = FakeInference(*responses)
        agent = ToolPlanningAgent(inference, self.service)
        return agent.answer(question, "Example Pharma", "1QFY27"), inference

    def assert_refused(self, answer, expected_calls, expected_tools):
        self.assertEqual(answer.status, "refused")
        self.assertEqual(answer.claims, ())
        self.assertIsNone(answer.comparison)
        self.assertEqual(len(answer.execution["agent"]["inference_calls"]), expected_calls)
        self.assertEqual(answer.execution["agent"]["local_tool_calls"], expected_tools)
        self.assertTrue(answer.execution["agent"]["experimental"])

    def test_real_graph_routes_to_canonical_exact_math_and_citations(self):
        answer, inference = self.run_agent(plan(include_yoy=True), {"complete": True, "unsupported_parts": []})
        self.assertEqual(answer.status, "answered")
        self.assertEqual(answer.comparison.delta_millions, Decimal("20"))
        self.assertEqual(answer.comparison.yoy_percent, Decimal("50"))
        self.assertEqual({c.kind for c in answer.claims}, {"comparison", "yoy", "growth"})
        refs = {c.ref for c in answer.citations}
        self.assertTrue(all(set(c.evidence_refs) <= refs for c in answer.claims))
        self.assertEqual(len(inference.calls), 2)
        self.assertEqual(answer.execution["agent"]["local_tool_calls"], 1)
        self.assertEqual(answer.execution["agent"]["nodes"], ["plan", "validate_arguments", "execute_local_tool", "check_question_coverage", "render_or_refuse"])
        self.assertEqual(answer.execution["llm"]["mode"], "test_double")
        self.assertTrue(answer.execution["no_llm"])
        self.assertTrue(answer.execution["no_database"])
        payload = json.loads(inference.calls[1].messages[1].content)
        self.assertEqual(payload["claims"][0]["values"]["delta"], "20")

    def test_each_allowlisted_tool_runs_once(self):
        for tool, expected in [("compare_revenue", {"comparison"}), ("growth_commentary", {"growth"}), ("combined_revenue", {"comparison", "growth"})]:
            with self.subTest(tool=tool):
                answer, inference = self.run_agent(plan(tool), {"complete": True, "unsupported_parts": []})
                self.assertEqual(answer.status, "answered")
                self.assertEqual({c.kind for c in answer.claims}, expected)
                self.assertEqual(len(inference.calls), 2)
                self.assertEqual(answer.execution["agent"]["local_tool_calls"], 1)

    def test_bad_plan_fields_types_and_context_never_execute_local_service(self):
        invalid = [plan("run_sql"), plan(company="Other Pharma"), plan(period="2QFY27"),
                   plan(include_yoy="true"), plan(sql="SELECT * FROM secret"), plan(actual="999"),
                   plan("growth_commentary", include_yoy=True), plan(unsupported_parts="PAT"),
                   plan(unsupported_parts=[1]), plan(tool=[]), "not json", "[]"]
        with patch.object(self.service, "comparison_answer", wraps=self.service.comparison_answer) as comparison, \
             patch.object(self.service, "growth_answer", wraps=self.service.growth_answer) as growth:
            for response in invalid:
                with self.subTest(response=response):
                    answer, inference = self.run_agent(response)
                    self.assert_refused(answer, 1, 0)
                    self.assertEqual(len(inference.calls), 1)
            comparison.assert_not_called()
            growth.assert_not_called()

    def test_duplicate_json_keys_and_unfinished_output_refuse(self):
        raw = json.dumps(plan())[:-1] + ',"tool":"growth_commentary"}'
        for response in [raw, InferenceResponse(json.dumps(plan()), "length"), "x" * 4097]:
            with self.subTest(response=response):
                answer, _ = self.run_agent(response)
                self.assert_refused(answer, 1, 0)

    def test_unsupported_parts_and_model_abstention_refuse_before_tool(self):
        for response in [plan("refuse"), plan(unsupported_parts=["PAT forecast"]), plan(unsupported_parts=["driver beat allocation"])]:
            with self.subTest(response=response):
                answer, _ = self.run_agent(response, question="Explain the revenue results.")
                self.assert_refused(answer, 1, 0)

    def test_coverage_rejection_or_extra_financial_data_strips_all_claims(self):
        decisions = [{"complete": False, "unsupported_parts": ["PAT"]},
                     {"complete": True, "unsupported_parts": ["standalone forecast"]},
                     {"complete": "yes", "unsupported_parts": []},
                     {"complete": True, "unsupported_parts": [], "answer": "invented"},
                     InferenceResponse('{"complete": true, "unsupported_parts": []}', "length")]
        for decision in decisions:
            with self.subTest(decision=decision):
                answer, _ = self.run_agent(plan(), decision)
                self.assert_refused(answer, 2, 1)
                self.assertTrue(answer.citations)

    def test_provider_failures_refuse_without_leaking_exception_payload(self):
        for responses, calls, tools in [((ProviderError("private-payload"),), 1, 0),
                                        ((plan(), TimeoutError("private-payload")), 2, 1)]:
            with self.subTest(calls=calls):
                answer, _ = self.run_agent(*responses)
                self.assert_refused(answer, calls, tools)
                self.assertNotIn("private-payload", answer.reason)

    def test_reused_agent_starts_with_fresh_request_state(self):
        inference = FakeInference(plan(), {"complete": True, "unsupported_parts": []}, plan("refuse"))
        agent = ToolPlanningAgent(inference, self.service)
        self.assertEqual(agent.answer("Compare and explain revenue", "Example Pharma", "1QFY27").status, "answered")
        refusal = agent.answer("Predict PAT", "Example Pharma", "1QFY27")
        self.assert_refused(refusal, 0, 0)
        self.assertEqual(refusal.citations, ())
        invalid = agent.answer("", "Example Pharma", "1QFY27")
        self.assert_refused(invalid, 0, 0)
        self.assertEqual(invalid.execution["llm"]["mode"], "not_called")
        self.assertEqual(len(inference.calls), 2)

    def test_known_explicit_unsupported_requests_cannot_be_overridden_by_model(self):
        questions = ["Compare revenue and PAT", "Explain EBITDA and revenue", "What was the profit margin?",
                     "What is the EPS and target price?", "Standalone revenue versus estimate", "Forecast next quarter sales",
                     "Convert revenue into USD", "Give revenue in INR billion", "What calendar dates bound this quarter?",
                     "Compare the 2QFY27 actual revenue", "Compare Q2 FY27 revenue", "What were FY27 annual sales?",
                     "How much of the beat came from oncology?", "Allocate the variance to individual drivers",
                     "What was revenue for Other Pharma?"]
        for question in questions:
            with self.subTest(question=question):
                answer, inference = self.run_agent(plan(), {"complete": True, "unsupported_parts": []}, question=question)
                self.assert_refused(answer, 0, 0)
                self.assertEqual(inference.calls, [])

    def test_ordinary_sales_turnover_and_explicit_prior_yoy_remain_supported(self):
        for question in ["How much did sales exceed estimates and why did turnover grow?",
                         "What may explain the revenue growth?",
                         "What was revenue for Example Pharma?",
                         "Compare revenue in Q1 FY27 with the broker estimate",
                         "Compare year-on-year sales in 1QFY27 versus 1QFY26"]:
            with self.subTest(question=question):
                answer, inference = self.run_agent(plan(include_yoy=True), {"complete": True, "unsupported_parts": []}, question=question)
                self.assertEqual(answer.status, "answered")
                self.assertEqual(len(inference.calls), 2)

    def test_safe_provider_http_category_never_copies_error_body(self):
        answer, _ = self.run_agent(ProviderError("HTTP 429 private-payload"))
        self.assert_refused(answer, 1, 0)
        record = answer.execution["agent"]["inference_calls"][0]
        self.assertEqual(record["failure_category"], "http_429")
        self.assertNotIn("private-payload", str(answer))

    def test_no_hidden_inference_in_local_service(self):
        self.service.selector = object()
        with self.assertRaises(ValueError):
            ToolPlanningAgent(FakeInference(), self.service)

    def test_tracing_is_disabled_even_when_environment_requests_it(self):
        with patch.dict(os.environ, {"LANGSMITH_TRACING": "true"}), \
             patch("langsmith.run_trees.RunTree.post") as remote_post:
            answer, _ = self.run_agent(plan(), {"complete": True, "unsupported_parts": []})
            self.assertEqual(answer.status, "answered")
            remote_post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
