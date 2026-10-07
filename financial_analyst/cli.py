"""Source-backed CLI; local data and live provider execution are explicit."""

import argparse
from dataclasses import asdict, is_dataclass
from decimal import Context, Decimal, ROUND_HALF_UP, localcontext
import json
from pathlib import Path
import sys

from .questions import UnsupportedQuestion, plan_question


def _json_value(value):
    if isinstance(value, Decimal):
        return str(value)
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _number(value, places=None):
    precision = max(40, len(value.as_tuple().digits), value.adjusted() + 1) + (places or 0) + 4
    with localcontext(Context(prec=precision, rounding=ROUND_HALF_UP)):
        if places is not None:
            value = value.quantize(Decimal(1).scaleb(-places))
        return format(value, ",f")


def _citation_text(refs, citations):
    by_ref = {citation.ref: citation for citation in citations}
    sources = {}
    for ref in refs:
        citation = by_ref[ref]
        sources.setdefault((citation.document_name, citation.link), set()).add(citation.page)
    selected = []
    for (document, link), pages in sources.items():
        label = f"{document}, pp. {', '.join(str(page) for page in sorted(pages))}"
        selected.append(f"{label}: {link}" if link else label)
    return " [" + "; ".join(selected) + "]"


def _render_text(answer):
    retrieval = answer.execution.get("retrieval", "not_requested").replace("_", " ")
    llm = answer.execution.get("llm", {"mode": "not_called"})
    llm_label = "none" if llm["mode"] == "not_called" else f"{llm['mode']} {llm['provider']} ({llm['model']})"
    lines = [f"Execution: {answer.execution.get('mode', 'fixture')}; analytics: fixture; "
             f"retrieval: {retrieval}; LLM: {llm_label}; database: none."]
    if answer.status != "answered":
        lines.append(f"Refused: {answer.reason}")
        if answer.citations:
            lines.append("Sources:" + _citation_text(tuple(citation.ref for citation in answer.citations), answer.citations))
        return "\n".join(lines)

    for claim in answer.claims:
        values = claim.values
        citation = _citation_text(claim.evidence_refs, answer.citations)
        if claim.kind == "comparison":
            delta, percent = values["delta"], values["beat_percent"]
            if delta > 0:
                outcome = f"beat by {_number(delta)} {values['currency']} million ({_number(percent, 2)}%)"
            elif delta < 0:
                outcome = f"missed by {_number(delta.copy_abs())} {values['currency']} million ({_number(percent.copy_abs(), 2)}%)"
            else:
                outcome = "matched the estimate (0.00% variance)"
            lines.append(
                f"{values['company']} {values['period']}: actual {values['currency']} "
                f"{_number(values['actual'])} million versus estimate {_number(values['estimate'])} million; "
                f"{outcome}. Percentages are calculated and rounded for display.{citation}"
            )
            actual, estimate = answer.comparison.actual, answer.comparison.broker_estimate
            lines.append(
                f"Source labels retained: scope {actual.scope}; {actual.source_metric_raw}; "
                f"{actual.source_unit_raw}; {actual.year_end_raw}; actual header {actual.source_header_raw} "
                f"({actual.kind}), estimate header {estimate.source_header_raw} ({estimate.kind}).{citation}"
            )
        elif claim.kind == "yoy":
            lines.append(
                f"Calculated YoY revenue change: {_number(values['yoy_percent'], 2)}% versus "
                f"{values['prior_period']} actual {values['currency']} {_number(values['prior_actual'])} million "
                f"(rounded for display).{citation}"
            )
        elif claim.kind == "growth":
            lines.append(f"Broker revenue-growth commentary: {values['excerpt']}{citation}")
    if any(claim.kind == "growth" for claim in answer.claims):
        lines.append("Growth commentary is not a quantified attribution of the estimate variance.")
    for conflict in answer.source_conflicts:
        if conflict.get("evidence_refs"):
            labels = "; ".join(
                f"{page.replace('_', ' ')}: {', '.join(values)}"
                for page, values in conflict["source_labels"].items()
            )
            lines.append("Fiscal year-end labels differ (" + labels + "). Calendar dates remain unresolved."
                         + _citation_text(conflict["evidence_refs"], answer.citations))
    return "\n\n".join(lines)


def _parser():
    parser = argparse.ArgumentParser(description="Grounded financial analyst: approved fixture iteration.")
    parser.add_argument("--mode", required=True, choices=("fixture", "live"))
    parser.add_argument("--fixture", type=Path, help="Reviewed local fixture JSON; required in fixture mode.")
    parser.add_argument("--source-pdf", type=Path, help="Optional relocation of the fixture's hash-matched PDF.")
    parser.add_argument("--retrieval", choices=("fixture", "keyword", "semantic"), default="fixture")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--llm", choices=("none", "groq"), default="none", help="Optional live selection of verified growth passages.")
    parser.add_argument("--env-file", type=Path, help="Explicit credential file; otherwise use GROQ_API_KEY from the environment.")
    parser.add_argument("--groq-model", default="openai/gpt-oss-20b", help="Explicit model for Groq evidence selection.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("compare", "combined", "growth", "ask", "beat-attribution"):
        command = commands.add_parser(name)
        command.add_argument("--company", required=True)
        command.add_argument("--period", required=True, help="Explicit fiscal label; calendar dates are not inferred.")
        if name in ("compare", "combined", "ask"):
            command.add_argument("--yoy", action="store_true")
        if name == "ask":
            command.add_argument("question")
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if args.mode != "fixture":
        parser.error("Live execution is not configured in this iteration; no fallback was performed.")
    if args.retrieval != "fixture":
        parser.error(f"{args.retrieval.capitalize()} retrieval is not implemented in this checkpoint; no fallback was performed.")
    if args.fixture is None:
        parser.error("--fixture is required in fixture mode.")

    selector = None
    try:
        operation, include_yoy = args.command, getattr(args, "yoy", False)
        if operation == "ask":
            plan = plan_question(args.question, args.company, args.period)
            operation, include_yoy = plan.operation, include_yoy or plan.include_yoy

        from .adapters import FileFixtureAdapter
        from .service import ApplicationService

        fixture = FileFixtureAdapter(args.fixture, source_path=args.source_pdf)
        if args.llm == "groq" and operation in ("growth", "combined"):
            from .selection import GroqPassageSelector, load_groq_key
            selector = GroqPassageSelector(load_groq_key(args.env_file), args.groq_model)
        elif args.llm == "groq" and operation == "compare":
            raise ValueError("Groq selection requires a growth or combined request; no model call was performed.")
        service = ApplicationService(fixture, fixture, fixture, selector=selector)
        if operation == "growth":
            answer = service.growth_answer(args.company, args.period)
        elif operation in ("beat-attribution", "beat_attribution"):
            answer = service.refuse_beat_attribution(args.company, args.period)
        else:
            answer = service.comparison_answer(
                args.company, args.period, include_growth=operation == "combined", include_yoy=include_yoy
            )
    except UnsupportedQuestion as error:
        result = {"status": "refused", "reason": str(error), "claims": [], "execution": {
            "mode": "fixture", "retrieval": "not_requested", "no_llm": True, "no_database": True
        }}
        print(json.dumps(result, indent=2) if args.format == "json" else f"Execution: fixture; LLM: none; database: none.\nRefused: {error}")
        return 1
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        print(f"Fixture execution failed: {error}", file=sys.stderr)
        return 2
    finally:
        if selector is not None:
            selector.close()

    print(json.dumps(_json_value(answer), indent=2, ensure_ascii=False) if args.format == "json" else _render_text(answer))
    return 0 if answer.status == "answered" else 1
