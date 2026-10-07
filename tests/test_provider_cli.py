"""CLI composition checks use a neutral inference port, never a live provider."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from financial_analyst.cli import main
from financial_analyst.inference import InferenceResponse
from tests.support import create_fixture


class FakeInference:
    capabilities = frozenset({"json_schema"})
    mode = "test_double"

    def __init__(self, provider, model):
        self.provider, self.model = provider, model
        self.calls = []
        self.closed = False
        self.execution = {"provider": provider, "model": model, "mode": "not_called", "outcome": "not_requested"}

    def infer(self, request):
        self.calls.append(request)
        self.execution = {"provider": self.provider, "model": self.model, "mode": self.mode, "outcome": "completed"}
        return InferenceResponse(json.dumps({"abstain": False, "selected_refs": ["growth"]}), "stop")

    def close(self):
        self.closed = True


class ProviderCliTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.fixture = create_fixture(Path(self.directory.name))

    def flags(self):
        return ["--mode", "fixture", "--fixture", str(self.fixture), "--format", "json"]

    def invoke(self, arguments, provider="groq", model="example-model"):
        inference = FakeInference(provider, model)
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("financial_analyst.inference.create_inference", return_value=inference) as factory, \
             redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                code = main(arguments)
            except SystemExit as error:
                code = error.code
        return code, stdout.getvalue(), stderr.getvalue(), factory, inference

    def combined(self):
        return ["combined", "--company", "Example Pharma", "--period", "1QFY27", "--yoy"]

    def test_provider_swap_preserves_calculations_quotes_and_citation_closure(self):
        results = []
        configurations = [
            ("groq", ["--llm", "groq", "--model", "example-model"]),
            ("mistral", ["--llm", "mistral", "--model", "example-model"]),
            ("openai-compatible", ["--llm", "openai-compatible", "--model", "example-model",
                                   "--base-url", "https://inference.example.test/v1", "--api-key-env", "CUSTOM_INFERENCE_KEY"]),
        ]
        for provider, flags in configurations:
            with self.subTest(provider=provider):
                code, output, error, factory, inference = self.invoke([*self.flags(), *flags, *self.combined()], provider)
                self.assertEqual(code, 0, error)
                answer = json.loads(output)
                results.append(answer)
                self.assertEqual(answer["status"], "answered")
                self.assertEqual(answer["claims"][0]["values"]["delta"], "20")
                self.assertEqual(answer["execution"]["llm"]["provider"], provider)
                self.assertEqual(answer["execution"]["llm"]["mode"], "test_double")
                self.assertTrue(answer["execution"]["no_llm"])
                refs = {citation["ref"] for citation in answer["citations"]}
                self.assertTrue(all(set(claim["evidence_refs"]) <= refs for claim in answer["claims"]))
                self.assertEqual(len(inference.calls), 1)
                self.assertTrue(inference.closed)
                self.assertEqual(factory.call_args.args, (provider,))
        self.assertEqual(results[0]["claims"], results[1]["claims"])
        self.assertEqual(results[0]["comparison"], results[1]["comparison"])
        self.assertEqual(results[0]["citations"], results[1]["citations"])
        self.assertEqual(results[0]["claims"], results[2]["claims"])
        self.assertEqual(results[0]["comparison"], results[2]["comparison"])
        self.assertEqual(results[0]["citations"], results[2]["citations"])

    def test_factory_receives_neutral_configuration_without_secret_contents(self):
        credential_file = Path(self.directory.name) / "credentials.env"
        code, _, error, factory, _ = self.invoke(
            [*self.flags(), "--llm", "openai-compatible", "--model", "other-model",
             "--base-url", "https://inference.example.test/v1", "--api-key-env", "OTHER_KEY",
             "--env-file", str(credential_file), *self.combined()], "openai-compatible", "other-model")
        self.assertEqual(code, 0, error)
        factory.assert_called_once_with("openai-compatible", model="other-model", env_file=credential_file,
                                        api_key_env="OTHER_KEY", base_url="https://inference.example.test/v1", timeout=20)

    def test_groq_model_alias_and_factory_default_remain_supported(self):
        for flags, expected in [(["--llm", "groq"], None),
                                (["--llm", "groq", "--groq-model", "old-model"], "old-model"),
                                (["--llm", "groq", "--model", "same", "--groq-model", "same"], "same")]:
            with self.subTest(flags=flags):
                code, _, error, factory, _ = self.invoke([*self.flags(), *flags, *self.combined()])
                self.assertEqual(code, 0, error)
                self.assertEqual(factory.call_args.kwargs["model"], expected)

    def test_mistral_default_and_explicit_model_use_neutral_factory(self):
        for flags, expected in [(["--llm", "mistral"], None),
                                (["--llm", "mistral", "--model", "mistral-model"], "mistral-model")]:
            with self.subTest(flags=flags):
                code, _, error, factory, inference = self.invoke([*self.flags(), *flags, *self.combined()], "mistral")
                self.assertEqual(code, 0, error)
                self.assertEqual(factory.call_args.args, ("mistral",))
                self.assertEqual(factory.call_args.kwargs["model"], expected)
                self.assertTrue(inference.closed)

    def test_semantic_search_changes_provider_without_changing_reviewed_records(self):
        results = []
        for provider in ("groq", "mistral", "openai-compatible"):
            with self.subTest(provider=provider):
                endpoint = ["--base-url", "https://inference.example.test/v1"] if provider == "openai-compatible" else []
                code, output, error, factory, inference = self.invoke(
                    [*self.flags(), "--retrieval", "semantic", "--llm", provider, "--model", "other-model", *endpoint,
                     "kb-search", "What helped turnover expand?", "--company", "Example Pharma", "--period", "1QFY27",
                     "--kind", "broker_commentary"], provider, "other-model")
                self.assertEqual(code, 0, error)
                result = json.loads(output)
                results.append(result)
                self.assertEqual(result["execution"]["retrieval"], "llm_semantic_selection")
                self.assertEqual(result["execution"]["llm"]["provider"], provider)
                self.assertEqual(result["hits"][0]["rank"], 1)
                self.assertNotIn("score", result["hits"][0])
                data = json.loads(inference.calls[0].messages[1].content)
                self.assertEqual(data["query"], "What helped turnover expand?")
                self.assertEqual(result["hits"][0]["record"]["quote"], data["passages"][0]["excerpt"])
                self.assertEqual(len(inference.calls), 1)
                self.assertTrue(inference.closed)
                factory.assert_called_once()
        self.assertEqual(results[0]["hits"], results[1]["hits"])
        self.assertEqual(results[0]["hits"], results[2]["hits"])

    def test_missing_compatible_configuration_and_conflicts_fail_before_factory(self):
        invalid = [
            ["--llm", "openai-compatible"],
            ["--llm", "openai-compatible", "--model", "other-model"],
            ["--llm", "openai-compatible", "--base-url", "https://inference.example.test/v1"],
            ["--llm", "groq", "--model", "one", "--groq-model", "two"],
            ["--llm", "openai-compatible", "--groq-model", "old-model", "--base-url", "https://inference.example.test/v1"],
            ["--llm", "groq", "--base-url", "https://inference.example.test/v1"],
            ["--llm", "mistral", "--groq-model", "old-model"],
            ["--llm", "mistral", "--base-url", "https://inference.example.test/v1"],
            ["--model", "unused"],
        ]
        for flags in invalid:
            with self.subTest(flags=flags):
                code, output, error, factory, inference = self.invoke([*self.flags(), *flags, *self.combined()])
                self.assertEqual(code, 2)
                self.assertEqual(output, "")
                self.assertTrue(error)
                factory.assert_not_called()
                self.assertEqual(inference.calls, [])

    def test_unsupported_question_and_allocation_refusal_do_not_construct_provider(self):
        flags = [*self.flags(), "--llm", "openai-compatible", "--model", "other-model",
                 "--base-url", "https://inference.example.test/v1"]
        requests = [
            ["ask", "--company", "Example Pharma", "--period", "1QFY27", "What was Example Pharma revenue and PAT?"],
            ["beat-attribution", "--company", "Example Pharma", "--period", "1QFY27"],
        ]
        for command in requests:
            with self.subTest(command=command[0]):
                code, output, error, factory, inference = self.invoke([*flags, *command], "openai-compatible")
                self.assertEqual(code, 1, error)
                answer = json.loads(output)
                self.assertEqual(answer["status"], "refused")
                self.assertEqual(answer["claims"], [])
                self.assertTrue(answer["execution"]["no_llm"])
                factory.assert_not_called()
                self.assertEqual(inference.calls, [])

    def test_comparison_import_and_exploratory_search_reject_inference_without_construction(self):
        requests = [
            [*self.flags(), "--llm", "groq", "compare", "--company", "Example Pharma", "--period", "1QFY27"],
            ["--mode", "live", "--fixture", str(self.fixture), "--database", str(Path(self.directory.name) / "not-created.sqlite"),
             "--llm", "groq", "ingest"],
            ["--mode", "local", "--retrieval", "keyword", "--manifest", "not-opened.json", "--llm", "groq", "search", "revenue"],
        ]
        for request in requests:
            with self.subTest(command=request[-1]):
                code, output, error, factory, inference = self.invoke(request)
                self.assertEqual(code, 2)
                self.assertEqual(output, "")
                self.assertTrue(error)
                factory.assert_not_called()
                self.assertEqual(inference.calls, [])
        self.assertFalse((Path(self.directory.name) / "not-created.sqlite").exists())


if __name__ == "__main__":
    unittest.main()
