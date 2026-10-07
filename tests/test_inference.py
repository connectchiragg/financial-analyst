import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx

from financial_analyst.inference import (
    GroqInference, InferenceError, InferenceMessage, InferenceRequest,
    InferenceResponse, MistralInference, OpenAICompatibleInference, ProviderError,
    create_inference, load_api_key,
)


SCHEMA = {"type": "object", "properties": {"selected_refs": {"type": "array", "items": {"type": "string"}}},
          "required": ["selected_refs"], "additionalProperties": False}


def request(**changes):
    fields = {"messages": (InferenceMessage("user", "Select only supported evidence references."),),
              "schema": SCHEMA}
    fields.update(changes)
    return InferenceRequest(**fields)


def completion(content='{"selected_refs":["synthetic"]}', finish_reason="stop"):
    return {"choices": [{"finish_reason": finish_reason, "message": {"content": content}}]}


class GroqClient:
    def __init__(self, result=None, error=None):
        self.calls = []
        self.closed = 0
        self.result = result if result is not None else SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content='{"selected_refs":["synthetic"]}')
        )])
        self.error = error
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **payload):
        self.calls.append(payload)
        if self.error is not None:
            raise self.error
        return self.result

    def close(self):
        self.closed += 1


class InferenceTests(unittest.TestCase):
    def test_groq_uses_standard_request_and_normalizes_sdk_response(self):
        client = GroqClient()
        adapter = GroqInference(client=client)
        self.assertEqual(adapter.execution["mode"], "not_called")
        self.assertEqual(adapter.capabilities, frozenset({"json_schema"}))
        self.assertEqual(adapter.infer(request()), InferenceResponse('{"selected_refs":["synthetic"]}', "stop"))
        self.assertEqual(adapter.execution, {"provider": "groq", "model": "openai/gpt-oss-20b",
                                             "mode": "test_double", "outcome": "returned"})
        payload = client.calls[0]
        self.assertEqual(payload["max_completion_tokens"], 1024)
        self.assertNotIn("max_tokens", payload)
        self.assertTrue(payload["response_format"]["json_schema"]["strict"])
        self.assertEqual(payload["response_format"]["json_schema"]["schema"], SCHEMA)

    def test_compatible_endpoint_uses_same_contract_and_only_selected_destination(self):
        calls = []

        def transport(req):
            calls.append(req)
            return httpx.Response(200, json=completion())

        with httpx.Client(transport=httpx.MockTransport(transport)) as client:
            adapter = OpenAICompatibleInference("synthetic-key", "model-one", "https://example.test/v1/", client=client)
            self.assertEqual(adapter.infer(request()), InferenceResponse('{"selected_refs":["synthetic"]}', "stop"))
            self.assertEqual(adapter.execution["mode"], "test_double")
            self.assertEqual(str(calls[0].url), "https://example.test/v1/chat/completions")
            self.assertEqual(calls[0].headers["authorization"], "Bearer synthetic-key")
            payload = json.loads(calls[0].content)
            self.assertEqual(payload["max_tokens"], 1024)
            self.assertNotIn("max_completion_tokens", payload)
            self.assertEqual(payload["response_format"]["json_schema"]["schema"], SCHEMA)
            self.assertTrue(payload["response_format"]["json_schema"]["strict"])

    def test_incomplete_finish_is_returned_for_business_rejection(self):
        adapter = GroqInference(client=GroqClient(result=completion(finish_reason="length")))
        self.assertEqual(adapter.infer(request()).finish_reason, "length")
        self.assertEqual(adapter.execution["outcome"], "returned")

    def test_mistral_uses_native_labels_and_fixed_chat_endpoint(self):
        calls = []

        def transport(req):
            calls.append(req)
            return httpx.Response(200, json=completion())

        with httpx.Client(transport=httpx.MockTransport(transport)) as client:
            adapter = MistralInference("synthetic-mistral-key", client=client)
            self.assertEqual(adapter.infer(request()), InferenceResponse('{"selected_refs":["synthetic"]}', "stop"))
            self.assertEqual(str(calls[0].url), "https://api.mistral.ai/v1/chat/completions")
            self.assertEqual(adapter.execution, {"provider": "mistral", "model": "mistral-small-latest",
                                                 "mode": "test_double", "outcome": "returned"})
            self.assertEqual(json.loads(calls[0].content)["model"], "mistral-small-latest")

    def test_malformed_provider_responses_are_safe_errors(self):
        for result in [{}, {"choices": []}, {"choices": completion()["choices"] * 2},
                       completion(content=None), completion(content=" "), completion(content="x" * 16001),
                       completion(finish_reason=None)]:
            with self.subTest(result_type=type(result).__name__):
                adapter = GroqInference(client=GroqClient(result=result))
                with self.assertRaises(ProviderError) as error:
                    adapter.infer(request())
                self.assertIn("no fallback", str(error.exception))
                self.assertNotIn("selected_refs", str(error.exception))
                self.assertEqual(adapter.execution["outcome"], "provider_error")

    def test_groq_transport_error_does_not_reveal_details_or_retry(self):
        client = GroqClient(error=TimeoutError("private document and credential detail"))
        adapter = GroqInference(client=client)
        with self.assertRaises(ProviderError) as error:
            adapter.infer(request())
        self.assertNotIn("private", str(error.exception))
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(adapter.execution["outcome"], "provider_error")

    def test_compatible_http_errors_and_invalid_json_are_not_fallbacks(self):
        for status, body in [(400, "private unsupported schema"), (401, "private credential"),
                             (500, "private source quote"), (200, "not json"), (200, "x" * 65537)]:
            calls = []

            def transport(req):
                calls.append(req)
                return httpx.Response(status, text=body)

            with self.subTest(status=status), httpx.Client(transport=httpx.MockTransport(transport)) as client:
                adapter = OpenAICompatibleInference("synthetic-key", "model-one", "https://example.test/v1", client=client)
                with self.assertRaises(ProviderError) as error:
                    adapter.infer(request())
                self.assertNotIn("private", str(error.exception))
                self.assertNotIn("example.test", str(error.exception))
                if status != 200:
                    self.assertIn(f"HTTP {status}", str(error.exception))
                self.assertEqual(len(calls), 1)
                self.assertEqual(adapter.execution["outcome"], "provider_error")

    def test_owned_groq_sdk_does_not_follow_same_origin_redirects(self):
        calls = []

        def transport(req):
            calls.append(req)
            return httpx.Response(307, headers={"Location": "https://api.groq.com/openai/v1/redirected"})

        # Exercise the actual SDK with only an in-memory HTTP transport.
        with patch("financial_analyst.inference.httpx.HTTPTransport", return_value=httpx.MockTransport(transport)) as http_transport:
            adapter = GroqInference("synthetic-key")
            try:
                with self.assertRaises(ProviderError) as error:
                    adapter.infer(request())
                self.assertIn("HTTP 307", str(error.exception))
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0].url.path, "/openai/v1/chat/completions")
                self.assertEqual(calls[0].headers["authorization"], "Bearer synthetic-key")
                self.assertEqual(adapter.execution["outcome"], "provider_error")
                http_transport.assert_called_once_with(retries=0)
            finally:
                adapter.close()
            self.assertTrue(adapter._http_client.is_closed)

    def test_owned_groq_constructor_failure_closes_transport_without_exposing_details(self):
        closed = []
        fake_http = SimpleNamespace(close=lambda: closed.append(True))
        with patch("financial_analyst.inference.httpx.Client", return_value=fake_http) as http_client, \
             patch("financial_analyst.inference.httpx.HTTPTransport") as transport, \
             patch("groq.Groq", side_effect=RuntimeError("private initialization detail")) as sdk:
            with self.assertRaises(ProviderError) as error:
                GroqInference("synthetic", timeout=11)
            self.assertNotIn("private", str(error.exception))
            transport.assert_called_once_with(retries=0)
            http_client.assert_called_once_with(timeout=11, follow_redirects=False, trust_env=False,
                                                transport=transport.return_value)
            sdk.assert_called_once_with(api_key="synthetic", timeout=11, max_retries=0, http_client=fake_http)
            self.assertEqual(closed, [True])

    def test_redirects_never_forward_credentials_even_for_injected_redirecting_client(self):
        calls = []

        def transport(req):
            calls.append(req)
            return httpx.Response(307, headers={"Location": "https://other.test/steal"})

        with httpx.Client(transport=httpx.MockTransport(transport), follow_redirects=True) as client:
            adapter = OpenAICompatibleInference("synthetic-key", "model-one", "https://example.test/v1", client=client)
            with self.assertRaises(ProviderError):
                adapter.infer(request())
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].url.host, "example.test")

    def test_endpoint_policy_rejects_unsafe_configuration_before_any_request(self):
        for endpoint in ["http://example.test/v1", "ftp://localhost/v1", "https://user:password@example.test/v1",
                         "https://@example.test/v1", "https://example.test/v1?q=private",
                         "https://example.test/v1?", "https://example.test/v1#secret", "https://example.test/v1#",
                         "https://example.test:99999/v1", "https://", "https://example.test/\nsecret", None]:
            with self.subTest(endpoint=endpoint), self.assertRaises(InferenceError):
                OpenAICompatibleInference("synthetic", "model-one", endpoint, client=object())
        for endpoint in ["https://example.test/v1", "http://localhost:9000/v1", "http://127.0.0.1/v1", "http://[::1]:9000/v1"]:
            with self.subTest(endpoint=endpoint):
                self.assertEqual(OpenAICompatibleInference("synthetic", "model-one", endpoint, client=object()).base_url, endpoint)

    def test_request_validation_precedes_transport_and_resets_stale_execution(self):
        client = GroqClient()
        adapter = GroqInference(client=client)
        adapter.infer(request())
        invalid = [request(messages=()), request(messages=(InferenceMessage("tool", "value"),)),
                   request(messages=(InferenceMessage("user", "x" * 16001),)), request(messages=[]),
                   request(messages=(InferenceMessage([], "text"),)),
                   request(max_output_tokens=True), request(max_output_tokens=4097), request(temperature=float("nan")),
                   request(temperature=True), request(schema_name="invalid name"), request(schema={"type": "array"}),
                   request(schema={"type": "object", "bad": float("nan")}),
                   request(schema={"type": "object", "description": "x" * 32769})]
        for value in invalid:
            with self.subTest(request_type=type(value).__name__), self.assertRaises(InferenceError):
                adapter.infer(value)
            self.assertEqual(adapter.execution["mode"], "not_called")
        self.assertEqual(len(client.calls), 1)

    def test_configuration_is_checked_before_owned_client_creation(self):
        with patch("groq.Groq") as sdk:
            for timeout in [0, 61, True, float("nan")]:
                with self.assertRaises(InferenceError):
                    GroqInference("synthetic", timeout=timeout)
            for model in [None, "", " ", "x" * 129, "has spaces"]:
                with self.assertRaises(InferenceError):
                    GroqInference("synthetic", model=model)
            for key in [None, "", "synthetic\nheader", "nonascii-\u0100"]:
                with self.assertRaises(InferenceError):
                    GroqInference(key)
            sdk.assert_not_called()

    def test_only_owned_clients_are_closed_and_cleanup_does_not_replace_results(self):
        client = GroqClient()
        injected = GroqInference(client=client)
        injected.close()
        self.assertEqual(client.closed, 0)
        with patch("groq.Groq", return_value=client) as sdk:
            owned = GroqInference("synthetic", timeout=12)
            sdk.assert_called_once_with(api_key="synthetic", timeout=12, max_retries=0,
                                        http_client=owned._http_client)
            self.assertFalse(owned._http_client.follow_redirects)
            self.assertEqual(owned.mode, "live")
            owned.infer(request())
            client.close = lambda: (_ for _ in ()).throw(OSError("private cleanup detail"))
            owned.close()
            self.assertTrue(owned._http_client.is_closed)
            self.assertEqual(owned.execution["outcome"], "returned")

    def test_compatible_owned_client_disables_environment_proxies_and_retries(self):
        fake = SimpleNamespace(close=lambda: None)
        with patch("financial_analyst.inference.httpx.Client", return_value=fake) as client, \
             patch("financial_analyst.inference.httpx.HTTPTransport") as transport:
            adapter = OpenAICompatibleInference("synthetic", "model-one", "https://example.test/v1", timeout=11)
            transport.assert_called_once_with(retries=0)
            client.assert_called_once_with(timeout=11, follow_redirects=False, trust_env=False,
                                           transport=transport.return_value)
            self.assertEqual(adapter.mode, "live")
            adapter.close()

    def test_credential_file_is_explicit_and_never_interpolated(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("CUSTOM_KEY='synthetic-${UNRELATED}-$(do-not-execute)'\n")
            self.assertEqual(load_api_key("CUSTOM_KEY", path), "synthetic-${UNRELATED}-$(do-not-execute)")
            with self.assertRaises(InferenceError):
                load_api_key("MISSING_KEY", path)
            with self.assertRaises(InferenceError):
                load_api_key("CUSTOM_KEY", Path(directory) / "missing")
        with patch.dict("os.environ", {"CUSTOM_KEY": "synthetic-environment"}, clear=True):
            self.assertEqual(load_api_key("CUSTOM_KEY"), "synthetic-environment")
            with self.assertRaises(InferenceError):
                load_api_key("MISSING_KEY")
        for name in [None, "", "CUSTOM KEY", "$(unsafe)", "X" * 129]:
            with self.assertRaises(InferenceError):
                load_api_key(name)

    def test_factory_rejects_incomplete_or_conflicting_configuration_before_loading_keys(self):
        with patch("financial_analyst.inference.load_api_key") as keys:
            invalid = [("unknown", {}), (None, {}), ([], {}),
                       ("groq", {"base_url": "https://example.test/v1"}),
                       ("mistral", {"base_url": "https://example.test/v1"}),
                       ("openai-compatible", {}), ("openai-compatible", {"model": "model-one"}),
                       ("openai-compatible", {"model": "model-one", "base_url": "http://remote.test/v1"}),
                       ("groq", {"timeout": 61})]
            for provider, kwargs in invalid:
                with self.subTest(provider=provider), self.assertRaises(InferenceError):
                    create_inference(provider, **kwargs)
            keys.assert_not_called()

    def test_factory_forwards_only_selected_provider_configuration(self):
        with patch("financial_analyst.inference.load_api_key", return_value="synthetic") as keys, \
             patch("financial_analyst.inference.GroqInference") as groq, \
             patch("financial_analyst.inference.MistralInference") as mistral, \
             patch("financial_analyst.inference.OpenAICompatibleInference") as compatible:
            self.assertIs(create_inference("groq"), groq.return_value)
            keys.assert_called_once_with("GROQ_API_KEY", None)
            groq.assert_called_once_with("synthetic", model="openai/gpt-oss-20b", timeout=20)
            compatible.assert_not_called()
            keys.reset_mock()
            self.assertIs(create_inference("mistral"), mistral.return_value)
            keys.assert_called_once_with("MISTRAL_API_KEY", None)
            mistral.assert_called_once_with("synthetic", model="mistral-small-latest", timeout=20)
            keys.reset_mock()
            create_inference("openai-compatible", model="model-one", base_url="https://example.test/v1",
                             api_key_env="CUSTOM_KEY", timeout=10)
            keys.assert_called_once_with("CUSTOM_KEY", None)
            compatible.assert_called_once_with("synthetic", "model-one", "https://example.test/v1", timeout=10)


if __name__ == "__main__":
    unittest.main()
