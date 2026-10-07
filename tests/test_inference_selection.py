"""Provider swaps must preserve business policy and authenticated answers."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from financial_analyst.adapters import Evidence, FileFixtureAdapter
from financial_analyst.inference import InferenceResponse, ProviderError
from financial_analyst.selection import PassageSelector, SelectionError
from financial_analyst.semantic import SemanticGrowthAdapter
from financial_analyst.service import ApplicationService
from tests.support import create_fixture


class StubInference:
    mode = "test_double"
    capabilities = frozenset({"json_schema"})

    def __init__(self, provider, response=None, finish_reason="stop"):
        self.provider, self.model = provider, "synthetic-model"
        self.response = response if response is not None else {"abstain": False, "selected_refs": ["growth"]}
        self.finish_reason = finish_reason
        self.execution = {"provider": provider, "model": self.model, "mode": "not_called"}
        self.calls = []
        self.closed = False

    def infer(self, request):
        self.calls.append(request)
        self.execution = {"provider": self.provider, "model": self.model, "mode": self.mode, "outcome": "returned"}
        if isinstance(self.response, Exception):
            self.execution["outcome"] = "provider_error"
            raise self.response
        content = self.response if isinstance(self.response, str) else json.dumps(self.response)
        return InferenceResponse(content, self.finish_reason)

    def close(self):
        self.closed = True


class InferenceSelectionTests(unittest.TestCase):
    def setUp(self):
        self.passages = (Evidence("growth", "synthetic.pdf", 3, "paragraph", "Growth reflected portfolio execution."),)

    def test_provider_swap_keeps_selection_request_policy_unchanged(self):
        ports = [StubInference("provider-a"), StubInference("provider-b")]
        for port in ports:
            selector = PassageSelector(port, query="What helped turnover expand?", contexts={"growth": "period=1QFY27"})
            self.assertEqual(selector.select("Example Pharma", "1QFY27", self.passages), ("growth",))
            self.assertEqual(selector.execution["provider"], port.provider)
            self.assertEqual(selector.execution["mode"], "test_double")
            self.assertEqual(selector.execution["outcome"], "validated")
        self.assertEqual(ports[0].calls, ports[1].calls)
        request = ports[0].calls[0]
        self.assertEqual(request.max_output_tokens, 1024)
        self.assertEqual(request.schema["properties"]["selected_refs"]["items"]["enum"], ["growth"])
        self.assertNotIn("model", request.__dict__)
        self.assertEqual(json.loads(request.messages[1].content)["passages"][0]["excerpt"], self.passages[0].excerpt)

    def test_swap_preserves_calculations_quotes_citation_closure_and_refusal(self):
        with TemporaryDirectory() as directory:
            fixture = FileFixtureAdapter(create_fixture(Path(directory)))
            answers = []
            for provider in ("provider-a", "provider-b"):
                port = StubInference(provider)
                selector = PassageSelector(port)
                service = ApplicationService(fixture, fixture, SemanticGrowthAdapter(fixture, selector))
                answer = service.comparison_answer("Example Pharma", "1QFY27", include_growth=True, include_yoy=True)
                self.assertEqual(answer.status, "answered")
                self.assertEqual(answer.execution["retrieval"], "llm_semantic_selection")
                self.assertEqual(answer.execution["llm"]["provider"], provider)
                self.assertTrue(answer.execution["no_llm"])
                self.assertTrue(answer.execution["no_database"])
                self.assertEqual(answer.claims[-1].values["excerpt"], fixture.resolve("growth").excerpt)
                citation_refs = {citation.ref for citation in answer.citations}
                self.assertTrue(all(set(claim.evidence_refs) <= citation_refs for claim in answer.claims))
                answers.append(answer)
                refused = service.refuse_beat_attribution("Example Pharma", "1QFY27")
                self.assertEqual(refused.status, "refused")
                self.assertEqual(refused.claims, ())
                self.assertEqual(len(port.calls), 1)
                self.assertEqual(refused.execution["llm"]["mode"], "not_called")
            self.assertEqual(answers[0].comparison, answers[1].comparison)
            self.assertEqual(answers[0].claims, answers[1].claims)
            self.assertEqual(answers[0].citations, answers[1].citations)

    def test_missing_capability_does_not_weaken_schema_or_call_inference(self):
        port = StubInference("no-schema-provider")
        port.capabilities = frozenset()
        selector = PassageSelector(port)
        with self.assertRaisesRegex(SelectionError, "JSON-schema contract"):
            selector.select("Example Pharma", "1QFY27", self.passages)
        self.assertEqual(port.calls, [])
        self.assertEqual(selector.execution["mode"], "not_called")

    def test_port_cannot_add_claims_refs_or_return_an_incomplete_selection(self):
        cases = [
            ({"abstain": False, "selected_refs": ["invented"]}, "stop"),
            ({"abstain": False, "selected_refs": ["growth"], "amount": "999"}, "stop"),
            ({"abstain": False, "selected_refs": ["growth"]}, "length"),
            ("not json", "stop"),
            ("[" * 2000, "stop"),
        ]
        for response, reason in cases:
            with self.subTest(reason=reason):
                port = StubInference("alternate-provider", response, reason)
                selector = PassageSelector(port)
                with self.assertRaises(SelectionError):
                    selector.select("Example Pharma", "1QFY27", self.passages)
                self.assertEqual(selector.execution["outcome"], "rejected")
                self.assertEqual(len(port.calls), 1)

    def test_provider_failure_and_pre_call_failure_do_not_reuse_validated_outcome(self):
        port = StubInference("alternate-provider")
        selector = PassageSelector(port)
        selector.select("Example Pharma", "1QFY27", self.passages)
        port.response = ProviderError("Alternate provider unavailable; no fallback was performed.")
        with self.assertRaises(ProviderError):
            selector.select("Example Pharma", "1QFY27", self.passages)
        self.assertEqual(selector.execution["outcome"], "provider_error")
        with self.assertRaises(SelectionError):
            selector.select("", "1QFY27", self.passages)
        self.assertEqual(selector.execution, {"provider": port.provider, "model": port.model, "mode": "not_called"})
        self.assertEqual(len(port.calls), 2)

    def test_cleanup_failure_does_not_replace_a_selection(self):
        port = StubInference("alternate-provider")
        selector = PassageSelector(port)
        selected = selector.select("Example Pharma", "1QFY27", self.passages)
        port.close = lambda: (_ for _ in ()).throw(OSError("synthetic cleanup failure"))
        selector.close()
        self.assertEqual(selected, ("growth",))
        self.assertEqual(selector.execution["outcome"], "validated")


if __name__ == "__main__":
    unittest.main()
