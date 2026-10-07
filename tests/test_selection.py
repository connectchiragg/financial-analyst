import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from financial_analyst.adapters import Evidence, FileFixtureAdapter
from financial_analyst.selection import GroqPassageSelector, ProviderError, SelectionError, load_groq_key
from financial_analyst.service import ApplicationService
from tests.support import create_fixture


class FakeClient:
    def __init__(self, response, finish_reason="stop", error=None):
        self.response, self.finish_reason, self.error = response, finish_reason, error
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **payload):
        self.calls.append(payload)
        if self.error:
            raise self.error
        raw = self.response if isinstance(self.response, str) else json.dumps(self.response)
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason=self.finish_reason,
                                                       message=SimpleNamespace(content=raw))])


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.passages = (Evidence("growth", "synthetic.pdf", 3, "growth paragraph",
                                  "Reported revenue growth reflected portfolio execution."),)

    def selector(self, response, **kwargs):
        client = FakeClient(response, **kwargs)
        return GroqPassageSelector(client=client), client

    def test_selects_only_refs_and_labels_injected_client(self):
        selector, client = self.selector({"abstain": False, "selected_refs": ["growth"]})
        self.assertEqual(selector.execution["mode"], "not_called")
        self.assertEqual(selector.select("Example Pharma", "1QFY27", self.passages), ("growth",))
        self.assertEqual(selector.execution["mode"], "test_double")
        self.assertEqual(selector.execution["outcome"], "validated")
        payload = client.calls[0]
        self.assertNotIn("tools", payload)
        self.assertTrue(payload["response_format"]["json_schema"]["strict"])
        self.assertEqual(json.loads(payload["messages"][1]["content"])["passages"][0]["excerpt"], self.passages[0].excerpt)

    def test_refuses_forged_or_extra_claims_and_inconsistent_selection(self):
        cases = [
            {"abstain": False, "selected_refs": ["invented"]},
            {"abstain": False, "selected_refs": ["growth", "growth"]},
            {"abstain": False, "selected_refs": [1]},
            {"abstain": True, "selected_refs": ["growth"]},
            {"abstain": False, "selected_refs": []},
            {"abstain": "false", "selected_refs": ["growth"]},
            {"abstain": False, "selected_refs": ["growth"], "claim": "Invented cause."},
            {"selected_refs": ["growth"]}, [], "not json",
        ]
        for response in cases:
            with self.subTest(response=response):
                selector, _ = self.selector(response)
                with self.assertRaises(SelectionError):
                    selector.select("Example Pharma", "1QFY27", self.passages)
                self.assertEqual(selector.execution["outcome"], "rejected")

    def test_abstention_and_truncation_cannot_supply_an_answer(self):
        for response, reason in [({"abstain": True, "selected_refs": []}, "stop"),
                                 ({"abstain": False, "selected_refs": ["growth"]}, "length")]:
            selector, _ = self.selector(response, finish_reason=reason)
            with self.assertRaises(SelectionError):
                selector.select("Example Pharma", "1QFY27", self.passages)

    def test_provider_errors_are_bounded_without_request_details(self):
        selector, client = self.selector(None, error=TimeoutError("private payload detail"))
        with self.assertRaises(ProviderError) as result:
            selector.select("Example Pharma", "1QFY27", self.passages)
        self.assertIn("no fallback", str(result.exception))
        self.assertNotIn("private", str(result.exception))
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(selector.execution["outcome"], "provider_error")

    def test_transport_cleanup_preserves_a_completed_answer(self):
        selector, client = self.selector({"abstain": False, "selected_refs": ["growth"]})
        client.close = lambda: (_ for _ in ()).throw(OSError("private transport detail"))
        selector._owns_client = True
        selector.select("Example Pharma", "1QFY27", self.passages)
        selector.close()
        self.assertEqual(selector.execution["outcome"], "validated")

    def test_input_bounds_fail_before_a_call(self):
        selector, client = self.selector({"abstain": False, "selected_refs": ["growth"]})
        for company, passages in [("", self.passages), ("Example Pharma", ()),
                                   ("Example Pharma", self.passages * 2),
                                   ("Example Pharma", (Evidence("large", "synthetic.pdf", 1, "text", "x" * 16001),))]:
            with self.subTest(company=company, count=len(passages)):
                with self.assertRaises(SelectionError):
                    selector.select(company, "1QFY27", passages)
        self.assertEqual(client.calls, [])
        self.assertEqual(selector.execution["mode"], "not_called")
        for timeout in [0, 61, True, float("nan")]:
            with self.assertRaises(SelectionError):
                GroqPassageSelector(client=client, timeout=timeout)

    def test_selected_env_file_is_read_without_shell_or_variable_expansion(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("GROQ_API_KEY='synthetic-${UNRELATED}-$(do-not-execute)'\n")
            self.assertEqual(load_groq_key(path), "synthetic-${UNRELATED}-$(do-not-execute)")
            path.write_text("OTHER=synthetic\n")
            with self.assertRaises(SelectionError):
                load_groq_key(path)
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(SelectionError):
                load_groq_key()

    def test_service_keeps_canonical_text_and_validates_any_selection_port(self):
        with TemporaryDirectory() as directory:
            adapter = FileFixtureAdapter(create_fixture(Path(directory)))
            selector, _ = self.selector({"abstain": False, "selected_refs": ["growth"]})
            answer = ApplicationService(adapter, adapter, adapter, selector).comparison_answer("Example Pharma", "1QFY27", True)
            self.assertEqual(answer.status, "answered")
            self.assertEqual(answer.claims[-1].values["excerpt"], adapter.resolve("growth").excerpt)
            self.assertEqual(answer.execution["analytics"], "fixture")
            self.assertEqual(answer.execution["llm"]["mode"], "test_double")
            self.assertTrue(answer.execution["no_llm"])
            self.assertTrue(answer.execution["selection_executed"])
            self.assertTrue(answer.execution["no_database"])

            class ForgedSelector:
                execution = {"provider": "synthetic", "mode": "test_double"}

                def select(self, company, period, passages):
                    return ("invented",)

            answer = ApplicationService(adapter, adapter, adapter, ForgedSelector()).growth_answer("Example Pharma", "1QFY27")
            self.assertEqual(answer.status, "refused")
            self.assertEqual(answer.claims, ())

    def test_attribution_refusal_never_calls_the_model(self):
        with TemporaryDirectory() as directory:
            adapter = FileFixtureAdapter(create_fixture(Path(directory)))
            selector, client = self.selector({"abstain": False, "selected_refs": ["growth"]})
            answer = ApplicationService(adapter, adapter, adapter, selector).refuse_beat_attribution("Example Pharma", "1QFY27")
            self.assertEqual(answer.status, "refused")
            self.assertEqual(client.calls, [])
            self.assertTrue(answer.execution["no_llm"])

    def test_execution_is_per_answer_after_reusing_service(self):
        with TemporaryDirectory() as directory:
            adapter = FileFixtureAdapter(create_fixture(Path(directory)))
            selector, client = self.selector({"abstain": False, "selected_refs": ["growth"]})
            service = ApplicationService(adapter, adapter, adapter, selector)
            self.assertTrue(service.growth_answer("Example Pharma", "1QFY27").execution["selection_executed"])
            comparison = service.comparison_answer("Example Pharma", "1QFY27")
            self.assertTrue(comparison.execution["no_llm"])
            self.assertEqual(comparison.execution["retrieval"], "not_requested")
            unsupported = service.growth_answer("Other Company", "1QFY27")
            self.assertTrue(unsupported.execution["no_llm"])
            self.assertEqual(len(client.calls), 1)

    def test_selection_refusal_reports_retrieval_that_was_attempted(self):
        with TemporaryDirectory() as directory:
            adapter = FileFixtureAdapter(create_fixture(Path(directory)))
            selector, _ = self.selector({"abstain": True, "selected_refs": []})
            answer = ApplicationService(adapter, adapter, adapter, selector).growth_answer("Example Pharma", "1QFY27")
            self.assertEqual(answer.status, "refused")
            self.assertEqual(answer.execution["retrieval"], "curated_fixture")
            self.assertEqual(answer.execution["llm"]["mode"], "test_double")


if __name__ == "__main__":
    unittest.main()
