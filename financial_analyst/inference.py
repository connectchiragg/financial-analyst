"""Bounded text inference behind a provider-neutral contract.

Adapters return text and completion state only. The application must validate
the returned JSON and its evidence references before using a provider result.
"""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx


class InferenceError(ValueError):
    """Inference configuration or a request violates the local contract."""


class ProviderError(RuntimeError):
    """The configured provider failed; no fallback is permitted."""


@dataclass(frozen=True)
class InferenceMessage:
    role: str
    content: str


@dataclass(frozen=True)
class InferenceRequest:
    messages: tuple[InferenceMessage, ...]
    schema: dict
    schema_name: str = "evidence_selection"
    max_output_tokens: int = 1024
    temperature: float = 0


@dataclass(frozen=True)
class InferenceResponse:
    content: str
    finish_reason: str


class InferencePort(Protocol):
    provider: str
    model: str
    mode: str
    capabilities: frozenset[str]
    execution: dict[str, Any]

    def infer(self, request: InferenceRequest) -> InferenceResponse: ...

    def close(self) -> None: ...


def _model(model: str) -> str:
    if (not isinstance(model, str) or not model.strip() or len(model) > 128
            or any(character.isspace() for character in model)):
        raise InferenceError("A bounded, explicit inference model ID is required.")
    return model


def _timeout(timeout: float) -> float:
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or not 0 < timeout <= 60):
        raise InferenceError("Inference timeout must be between zero and 60 seconds.")
    return timeout


def _key(api_key: str | None) -> str:
    if (not isinstance(api_key, str) or not api_key.strip() or len(api_key) > 4096
            or not api_key.isascii() or any(character in api_key for character in "\r\n\x00")):
        raise InferenceError("A valid API key is required for live inference.")
    return api_key.strip()


def load_api_key(key_env: str, env_file: Path | None = None) -> str:
    """Read one credential without interpolation, shell evaluation, or copying."""
    if (not isinstance(key_env, str) or len(key_env) > 128
            or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key_env) is None):
        raise InferenceError("A valid API-key environment variable name is required.")
    if env_file is None:
        key = os.environ.get(key_env)
    else:
        try:
            path = Path(env_file)
            if not path.is_file():
                raise InferenceError("The selected credential file is unavailable.")
            from dotenv import dotenv_values
            key = dotenv_values(path, interpolate=False).get(key_env)
        except (OSError, UnicodeError, TypeError, ValueError) as exc:
            if isinstance(exc, InferenceError):
                raise
            raise InferenceError("The selected credential file could not be read.") from None
    if not isinstance(key, str) or not key.strip():
        raise InferenceError(f"{key_env} is not configured in the selected credential source.")
    return _key(key)


def _base_url(base_url: str) -> str:
    if (not isinstance(base_url, str) or not base_url or len(base_url) > 2048
            or any(character.isspace() or ord(character) < 32 for character in base_url)
            or "?" in base_url or "#" in base_url):
        raise InferenceError("A bounded inference base URL without query or fragment is required.")
    try:
        parts = urlsplit(base_url)
        host = parts.hostname
        # Accessing port also validates malformed/out-of-range authority values.
        parts.port
        if not host or parts.username is not None or parts.password is not None:
            raise ValueError
        loopback = host.lower() == "localhost"
        if not loopback:
            try:
                loopback = ipaddress.ip_address(host).is_loopback
            except ValueError:
                pass
        if parts.scheme != "https" and not (parts.scheme == "http" and loopback):
            raise ValueError
    except ValueError:
        raise InferenceError("Inference endpoints require HTTPS or loopback HTTP, without URL credentials.") from None
    return base_url.rstrip("/")


def _payload(request: InferenceRequest, model: str, token_field: str) -> dict:
    """Validate before a transport call; copy the schema into a JSON snapshot."""
    if not isinstance(request, InferenceRequest):
        raise InferenceError("A standard inference request is required.")
    if not isinstance(request.messages, tuple) or not 1 <= len(request.messages) <= 16:
        raise InferenceError("Inference requires between one and 16 messages.")
    for message in request.messages:
        if (not isinstance(message, InferenceMessage) or not isinstance(message.role, str)
                or message.role not in {"system", "user", "assistant"}
                or not isinstance(message.content, str) or not message.content.strip()
                or len(message.content) > 16000):
            raise InferenceError("Inference messages require a supported role and bounded nonblank text.")
    if (not isinstance(request.schema, dict) or request.schema.get("type") != "object"
            or not isinstance(request.schema_name, str)
            or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", request.schema_name) is None):
        raise InferenceError("Inference requires a named JSON object schema.")
    if type(request.max_output_tokens) is not int or not 1 <= request.max_output_tokens <= 4096:
        raise InferenceError("Inference output must be bounded between one and 4,096 tokens.")
    if (isinstance(request.temperature, bool) or not isinstance(request.temperature, (int, float))
            or not math.isfinite(request.temperature) or not 0 <= request.temperature <= 2):
        raise InferenceError("Inference temperature must be a finite value between zero and two.")
    payload = {
        "model": model, "temperature": request.temperature, token_field: request.max_output_tokens,
        "messages": [{"role": message.role, "content": message.content} for message in request.messages],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": request.schema_name, "strict": True, "schema": request.schema,
        }},
    }
    try:
        serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        if len(serialized) > 32768:
            raise InferenceError("Inference request exceeds the bounded serialized request size.")
        return json.loads(serialized)
    except (ValueError, TypeError, RecursionError) as exc:
        if isinstance(exc, InferenceError):
            raise
        raise InferenceError("Inference schema must be bounded, finite JSON data.") from None


def _normalize(result: Any) -> InferenceResponse:
    try:
        choices = result["choices"] if isinstance(result, dict) else result.choices
        if not isinstance(choices, (list, tuple)) or len(choices) != 1:
            raise ValueError
        choice = choices[0]
        reason = choice["finish_reason"] if isinstance(choice, dict) else choice.finish_reason
        message = choice["message"] if isinstance(choice, dict) else choice.message
        content = message["content"] if isinstance(message, dict) else message.content
        if (not isinstance(content, str) or not content.strip() or len(content) > 16000
                or not isinstance(reason, str) or not reason or len(reason) > 64):
            raise ValueError
        # Incomplete completion states are returned for business-layer rejection.
        return InferenceResponse(content, reason)
    except (AttributeError, KeyError, IndexError, TypeError, ValueError, RecursionError):
        raise ProviderError("Inference provider returned an invalid response; no fallback was performed.") from None


def _failure(provider: str, exc: Exception) -> ProviderError:
    status = getattr(exc, "status_code", None)
    if status is None and isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
    suffix = f" (HTTP {status})" if type(status) is int and 100 <= status <= 599 else ""
    return ProviderError(f"{provider} inference request failed{suffix}; no fallback was performed.")


class GroqInference:
    """Official Groq SDK transport; SDK objects remain inside this adapter."""

    provider = "groq"
    capabilities = frozenset({"json_schema"})

    def __init__(self, api_key: str | None = None, model: str = "openai/gpt-oss-20b",
                 timeout: float = 20, client=None):
        self.model = _model(model)
        timeout = _timeout(timeout)
        self._owns_client = client is None
        self.mode = "live" if self._owns_client else "test_double"
        self._http_client = None
        if client is None:
            api_key = _key(api_key)
            from groq import Groq
            self._http_client = httpx.Client(
                timeout=timeout, follow_redirects=False, trust_env=False,
                transport=httpx.HTTPTransport(retries=0),
            )
            try:
                client = Groq(api_key=api_key, timeout=timeout, max_retries=0,
                              http_client=self._http_client)
            except Exception:
                try:
                    self._http_client.close()
                except Exception:
                    pass
                raise ProviderError("Groq inference client could not be initialized.") from None
        self._client = client
        self.execution = {"provider": self.provider, "model": self.model, "mode": "not_called"}

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        self.execution = {"provider": self.provider, "model": self.model, "mode": "not_called"}
        payload = _payload(request, self.model, "max_completion_tokens")
        # These exact Groq models default to medium reasoning, which can use
        # the bounded completion budget before strict-schema metadata appears.
        # Keep this provider wire policy out of the standard inference request.
        if self.model in {"openai/gpt-oss-20b", "openai/gpt-oss-120b"}:
            payload["reasoning_effort"] = "low"
        self.execution.update(mode=self.mode, outcome="attempted")
        try:
            result = self._client.chat.completions.create(**payload)
        except Exception as exc:
            self.execution["outcome"] = "provider_error"
            raise _failure(self.provider, exc) from None
        try:
            response = _normalize(result)
        except ProviderError:
            self.execution["outcome"] = "provider_error"
            raise
        self.execution["outcome"] = "returned"
        return response

    def close(self) -> None:
        if self._owns_client:
            try:
                self._client.close()
            except Exception:
                # Cleanup cannot replace a validated result or the primary error.
                pass
            finally:
                if self._http_client is not None:
                    try:
                        self._http_client.close()
                    except Exception:
                        pass


class OpenAICompatibleInference:
    """Explicit chat-completions endpoint with strict JSON-schema output.

    Uses the standard ``max_tokens`` request field. An endpoint or model that
    does not support strict JSON schemas must fail rather than downgrade them.
    The response-size check applies after the HTTP client buffers the body.
    """

    provider = "openai-compatible"
    capabilities = frozenset({"json_schema"})

    def __init__(self, api_key: str, model: str, base_url: str, timeout: float = 20, client=None):
        self.model = _model(model)
        self.base_url = _base_url(base_url)
        self._timeout = _timeout(timeout)
        api_key = _key(api_key)
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._owns_client = client is None
        self.mode = "live" if self._owns_client else "test_double"
        self._client = client if client is not None else httpx.Client(
            timeout=self._timeout, follow_redirects=False, trust_env=False,
            transport=httpx.HTTPTransport(retries=0),
        )
        self.execution = {"provider": self.provider, "model": self.model, "mode": "not_called"}

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        self.execution = {"provider": self.provider, "model": self.model, "mode": "not_called"}
        payload = _payload(request, self.model, "max_tokens")
        self.execution.update(mode=self.mode, outcome="attempted")
        try:
            result = self._client.post(
                f"{self.base_url}/chat/completions", json=payload, headers=self._headers,
                timeout=self._timeout, follow_redirects=False,
            )
            result.raise_for_status()
            if len(result.content) > 65536:
                raise ProviderError("Inference provider returned an oversized response; no fallback was performed.")
            response = _normalize(result.json())
        except ProviderError:
            self.execution["outcome"] = "provider_error"
            raise
        except Exception as exc:
            self.execution["outcome"] = "provider_error"
            raise _failure(self.provider, exc) from None
        self.execution["outcome"] = "returned"
        return response

    def close(self) -> None:
        if self._owns_client:
            try:
                self._client.close()
            except Exception:
                pass


class MistralInference(OpenAICompatibleInference):
    """Mistral chat inference; hosted libraries and agents use separate ports."""

    provider = "mistral"

    def __init__(self, api_key: str, model: str = "mistral-small-latest",
                 timeout: float = 20, client=None):
        super().__init__(api_key, model, "https://api.mistral.ai/v1", timeout=timeout, client=client)


def create_inference(provider: str, *, model: str | None = None, env_file: Path | None = None,
                     api_key_env: str | None = None, base_url: str | None = None,
                     timeout: float = 20) -> InferencePort:
    """Validate the selected adapter before reading its credential source."""
    if not isinstance(provider, str) or provider not in {"groq", "mistral", "openai-compatible"}:
        raise InferenceError("Choose a supported inference provider: mistral, groq or openai-compatible.")
    _timeout(timeout)
    if provider == "groq":
        if base_url is not None:
            raise InferenceError("A custom base URL requires the openai-compatible provider.")
        selected_model = _model(model if model is not None else "openai/gpt-oss-20b")
        key = load_api_key(api_key_env if api_key_env is not None else "GROQ_API_KEY", env_file)
        return GroqInference(key, model=selected_model, timeout=timeout)
    if provider == "mistral":
        if base_url is not None:
            raise InferenceError("A custom base URL requires the openai-compatible provider.")
        selected_model = _model(model if model is not None else "mistral-small-latest")
        key = load_api_key(api_key_env if api_key_env is not None else "MISTRAL_API_KEY", env_file)
        return MistralInference(key, model=selected_model, timeout=timeout)
    selected_model = _model(model)
    selected_url = _base_url(base_url)
    key = load_api_key(api_key_env if api_key_env is not None else "INFERENCE_API_KEY", env_file)
    return OpenAICompatibleInference(key, selected_model, selected_url, timeout=timeout)
