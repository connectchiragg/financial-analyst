"""Question-only terminal session over reviewed, source-bound local tools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from .adapters import FileFixtureAdapter
from .agent import ToolPlanningAgent
from .cli import _render_text
from .inference import create_inference
from .service import ApplicationService


DEFAULT_CONFIG = Path(".local/analyst-config.json")
_ALLOWED = {"provider", "model", "env_file", "fixture", "database", "company", "period", "retrieval",
            "base_url", "api_key_env", "engine"}


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate configuration field.")
        result[key] = value
    return result


def _load_config(config_path):
    path = Path(config_path).expanduser().resolve()
    config = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
    if isinstance(config, dict) and config.get("engine") == "corpus":
        from .corpus_cli import load_config
        return load_config(path)
    if not isinstance(config, dict) or set(config) - _ALLOWED:
        raise ValueError("Unsupported configuration fields.")
    if config.get("engine", "legacy") != "legacy":
        raise ValueError("Unsupported chat engine.")
    for name in ("provider", "fixture", "company", "period"):
        if not isinstance(config.get(name), str) or not config[name].strip():
            raise ValueError("Required configuration is missing.")
    if config["provider"] not in {"groq", "mistral", "openai-compatible"}:
        raise ValueError("Unsupported inference provider.")
    if len(config["company"]) > 256 or re.fullmatch(r"[1-4]QFY(?:\d{2}|\d{4})", config["period"]) is None:
        raise ValueError("An explicit company and fiscal quarter are required.")
    for name in ("model", "base_url", "api_key_env"):
        value = config.get(name)
        if value is not None and (not isinstance(value, str) or not value.strip() or len(value) > 2048):
            raise ValueError("Invalid inference configuration.")
    for name in ("fixture", "database", "env_file"):
        value = config.get(name)
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Invalid configured file path.")
        selected = Path(value).expanduser()
        config[name] = selected if selected.is_absolute() else path.parent / selected
    config["retrieval"] = config.get("retrieval", "fixture")
    if config["retrieval"] not in {"fixture", "keyword"}:
        raise ValueError("The local planning session supports fixture or keyword retrieval.")
    return config


def _service(config):
    fixture = FileFixtureAdapter(config["fixture"])
    contexts = {(item.company, item.period, item.kind) for item in fixture.reviewed_observations()}
    if any((config["company"], config["period"], kind) not in contexts
           for kind in ("reported_actual", "broker_estimate")):
        raise ValueError("The configured source has no reviewed actual/estimate pair for this context.")
    analytics, passages = fixture, fixture
    if config.get("database") is not None:
        from .sqlite_adapter import SQLiteAnalyticsAdapter
        analytics = SQLiteAnalyticsAdapter(config["database"], source_sha256=fixture.source_sha256)
    if config["retrieval"] == "keyword":
        from .knowledge import KnowledgeGrowthAdapter
        passages = KnowledgeGrowthAdapter(fixture)
    return ApplicationService(analytics, fixture, passages)


def run_chat(config_path, input_fn=input, output_fn=print) -> int:
    """Keep one provider open, resetting the agent's request state each question.

    The configured financial context is never changed by a question. The agent
    still owns its existing whole-question checks and source-bound local tools.
    Errors expose no provider response, credential or configuration contents.
    """
    inference = None
    try:
        config = _load_config(config_path)
        if config.get("engine") == "corpus":
            from .corpus_cli import run_chat as run_corpus_chat
            return run_corpus_chat(config, mode=config["mode"], input_fn=input_fn, output_fn=output_fn)
        service = _service(config)
        inference = create_inference(config["provider"], model=config.get("model"),
                                     env_file=config.get("env_file"), base_url=config.get("base_url"),
                                     api_key_env=config.get("api_key_env"), timeout=20)
        agent = ToolPlanningAgent(inference, service)
    except KeyboardInterrupt:
        if inference is not None:
            try:
                inference.close()
            except Exception:
                pass
        return 0
    except Exception:
        if inference is not None:
            try:
                inference.close()
            except Exception:
                pass
        output_fn("Chat could not start. Check the private configuration, source files and provider access.")
        return 2

    try:
        output_fn(f"Available answers: {config['company']}, {config['period']}; revenue comparison, YoY and cited growth explanations only.")
        output_fn("Ask a question. Type /quit to exit.")
        while True:
            try:
                question = input_fn("Question: ")
            except (EOFError, KeyboardInterrupt):
                return 0
            except Exception:
                output_fn("The question could not be read. The session has ended.")
                return 2
            if not isinstance(question, str):
                output_fn("Please enter a question.")
                continue
            if question.strip().casefold() == "/quit":
                return 0
            if not question.strip():
                continue
            try:
                answer = agent.answer(question, config["company"], config["period"])
                output_fn(_render_text(answer))
            except KeyboardInterrupt:
                return 0
            except Exception:
                output_fn("The request could not complete. Check provider access and try another question; no answer was produced.")
    finally:
        try:
            inference.close()
        except Exception:
            pass


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ask source-backed questions in a terminal session.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Private session configuration; file paths resolve relative to this file.")
    args = parser.parse_args(argv)
    return run_chat(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
