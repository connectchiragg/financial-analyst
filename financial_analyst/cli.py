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
    lines = [f"Execution: {answer.execution.get('mode', 'fixture')}; analytics: {answer.execution.get('analytics', 'fixture')}; "
             f"retrieval: {retrieval}; LLM: {llm_label}; database: {answer.execution.get('database', 'none')}."]
    if answer.execution.get("seed_provenance"):
        lines.append("Database seed provenance: " + ", ".join(answer.execution["seed_provenance"]) + ".")
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
    parser = argparse.ArgumentParser(description="Grounded financial analyst with explicit source and provider modes.")
    parser.add_argument("--mode", required=True, choices=("fixture", "local", "live"))
    parser.add_argument("--fixture", type=Path, help="Reviewed local fixture JSON; required in fixture mode.")
    parser.add_argument("--source-pdf", type=Path, help="Optional relocation of the fixture's hash-matched PDF.")
    parser.add_argument("--retrieval", choices=("fixture", "keyword", "semantic"), default="fixture")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--llm", choices=("none", "groq"), default="none", help="Optional live selection of verified growth passages.")
    parser.add_argument("--env-file", type=Path, help="Explicit credential file; otherwise use GROQ_API_KEY from the environment.")
    parser.add_argument("--groq-model", default="openai/gpt-oss-20b", help="Explicit model for Groq evidence selection.")
    parser.add_argument("--manifest", type=Path, help="Hash-pinned local research PDF manifest for exploratory search.")
    parser.add_argument("--database", type=Path, help="Local SQLite file for ingestion or live relational analytics.")
    commands = parser.add_subparsers(dest="command", required=True)
    search = commands.add_parser("search", help="Explore exact PDF passages with local keyword ranking.")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=5)
    commands.add_parser("ingest", help="Validate and transactionally ingest a reviewed source bundle into SQLite.")
    knowledge = commands.add_parser("kb-search", help="Search contextualized evidence using exact financial filters.")
    knowledge.add_argument("query")
    knowledge.add_argument("--company", required=True)
    knowledge.add_argument("--period", required=True)
    for name in ("scope", "metric", "kind", "currency", "unit"):
        knowledge.add_argument("--" + name)
    knowledge.add_argument("--limit", type=int, default=5)
    for name in ("compare", "combined", "growth", "ask", "beat-attribution"):
        command = commands.add_parser(name)
        command.add_argument("--company", required=True)
        command.add_argument("--period", required=True, help="Explicit fiscal label; calendar dates are not inferred.")
        if name in ("compare", "combined", "ask"):
            command.add_argument("--yoy", action="store_true")
        if name == "ask":
            command.add_argument("question")
    return parser


def _search(args, parser):
    if args.mode != "local" or args.retrieval != "keyword":
        parser.error("Exploratory search requires --mode local --retrieval keyword; no fallback was performed.")
    if args.manifest is None:
        parser.error("--manifest is required for local keyword search.")
    if args.llm != "none":
        parser.error("Exploratory keyword search does not invoke an LLM; no fallback was performed.")
    try:
        from .retrieval import LocalKeywordAdapter
        adapter = LocalKeywordAdapter(args.manifest)
        hits = adapter.search(args.query, args.limit)
        result = {"status": "retrieved" if hits else "no_matches", "claims": [],
                  "hits": [{"passage": _json_value(hit.passage), "score": hit.score,
                            "citation": _json_value(hit.passage.citation)} for hit in hits],
                  "coverage": _json_value(adapter.coverage),
                  "execution": {"mode": "local", "retrieval": "local_keyword", "context": "unreviewed_page_text",
                                "no_llm": True, "no_database": True}}
    except (ValueError, OSError, ImportError) as error:
        print(f"Local search failed: {error}", file=sys.stderr)
        return 2
    if args.format == "json":
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("Execution: local keyword retrieval; LLM: none; database: none. Exploratory passages, without reviewed financial context.")
        for hit in hits:
            passage = hit.passage
            print(f"\n{passage.document_name}, p. {passage.page}; lexical score {hit.score:.4f}; {passage.locator}")
            print(passage.excerpt)
            print(passage.url)
        if not hits:
            print("No keyword matches. This search outcome does not establish that the corpus lacks the requested fact.")
    return 0


def _knowledge_search(args, parser):
    if args.mode != "fixture" or args.retrieval != "keyword" or args.fixture is None:
        parser.error("Reviewed knowledge search requires --mode fixture --retrieval keyword and --fixture; no fallback was performed.")
    if args.llm != "none" or args.database is not None:
        parser.error("This reviewed knowledge command uses local keyword retrieval, without LLM or database execution.")
    try:
        from .adapters import FileFixtureAdapter
        from .knowledge import ReviewedKnowledgeIndex
        fixture = FileFixtureAdapter(args.fixture, source_path=args.source_pdf)
        index = ReviewedKnowledgeIndex.from_fixture(fixture)
        filters = {name: getattr(args, name) for name in ("company", "period", "scope", "metric", "kind", "currency", "unit")}
        hits = index.search(args.query, **filters, limit=args.limit)
        result = {"status": "retrieved" if hits else "no_matches", "claims": [], "filters": filters,
                  "hits": [{"record": _json_value(hit.record), "score": hit.score,
                            "matching_contexts": _json_value(hit.matching_contexts),
                            "context_prefix": hit.context_prefix} for hit in hits],
                  "execution": {"mode": "fixture", "retrieval": index.mode, "context": "pdf_validated",
                                "no_llm": True, "no_database": True}}
    except (ValueError, OSError, ImportError) as error:
        print(f"Reviewed knowledge search failed: {error}", file=sys.stderr)
        return 2
    if args.format == "json":
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("Execution: reviewed fixture knowledge; keyword retrieval; LLM: none; database: none.")
        for hit in hits:
            print("\nDerived context: " + hit.context_prefix)
            print("Source quote: " + hit.record.quote)
            citation = hit.record.citation
            print(f"{citation.document_name}, p. {citation.page}; {citation.locator}; {citation.link or ''}")
        if not hits:
            print("No matching reviewed records. This does not establish that the full corpus lacks an answer.")
    return 0


def _ingest(args, parser):
    if args.mode != "live" or args.database is None or args.fixture is None:
        parser.error("Ingestion requires --mode live, --database and --fixture; no fallback was performed.")
    if args.llm != "none":
        parser.error("Reviewed-bundle ingestion does not invoke an LLM.")
    try:
        from .adapters import FileFixtureAdapter
        from .sqlite_adapter import DocumentRecord, SQLiteStore
        fixture = FileFixtureAdapter(args.fixture, source_path=args.source_pdf)
        document = DocumentRecord(fixture.document_name, fixture.source_sha256, fixture.source_url,
                                  metadata={"source_conflicts": fixture.source_conflicts})
        store = SQLiteStore(args.database)
        store.initialize()
        ingested = store.ingest(document, fixture.reviewed_observations(), fixture.reviewed_evidence(), fixture)
        result = {"status": "ingested" if any(asdict(ingested).values()) else "unchanged",
                  "counts": asdict(ingested), "claims": [],
                  "execution": {"mode": "live", "database": "sqlite", "source_validation": "local_pdf",
                                "seed_provenance": "reviewed_fixture", "no_database": False, "no_llm": True}}
    except (ValueError, OSError, ImportError) as error:
        print(f"SQLite ingestion failed: {error}", file=sys.stderr)
        return 2
    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        print(f"SQLite ingestion: {result['status']}; input: PDF-validated reviewed fixture; LLM: none.")
        print(json.dumps(result["counts"]))
    return 0


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "search":
        return _search(args, parser)
    if args.command == "ingest":
        return _ingest(args, parser)
    if args.command == "kb-search":
        return _knowledge_search(args, parser)
    if args.mode == "live" and args.database is None:
        parser.error("Live analytics requires --database; no fallback was performed.")
    if args.mode == "local":
        parser.error("Local mode supports exploratory search only; no fallback was performed.")
    if args.mode == "fixture" and args.database is not None:
        parser.error("Select --mode live to read SQLite; no fallback was performed.")
    if args.retrieval != "fixture" and not (args.retrieval == "keyword" and args.command in {"growth", "combined"}):
        parser.error(f"{args.retrieval.capitalize()} retrieval is not implemented in this checkpoint; no fallback was performed.")
    if args.fixture is None:
        parser.error("--fixture is required for original PDF evidence validation.")

    selector = None
    try:
        operation, include_yoy = args.command, getattr(args, "yoy", False)
        if operation == "ask":
            plan = plan_question(args.question, args.company, args.period)
            operation, include_yoy = plan.operation, include_yoy or plan.include_yoy

        from .adapters import FileFixtureAdapter
        from .service import ApplicationService

        fixture = FileFixtureAdapter(args.fixture, source_path=args.source_pdf)
        passages = fixture
        if args.retrieval == "keyword":
            from .knowledge import KnowledgeGrowthAdapter
            passages = KnowledgeGrowthAdapter(fixture)
        analytics = fixture
        if args.mode == "live":
            from .sqlite_adapter import SQLiteAnalyticsAdapter
            analytics = SQLiteAnalyticsAdapter(args.database, source_sha256=fixture.source_sha256)
        if args.llm == "groq" and operation in ("growth", "combined"):
            from .selection import GroqPassageSelector, load_groq_key
            selector = GroqPassageSelector(load_groq_key(args.env_file), args.groq_model)
        elif args.llm == "groq" and operation == "compare":
            raise ValueError("Groq selection requires a growth or combined request; no model call was performed.")
        service = ApplicationService(analytics, fixture, passages, selector=selector)
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
            "mode": args.mode, "analytics": "not_requested", "retrieval": "not_requested", "no_llm": True, "no_database": True
        }}
        print(json.dumps(result, indent=2) if args.format == "json" else f"Execution: {args.mode}; LLM: none; database: none.\nRefused: {error}")
        return 1
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        print(f"Execution failed: {error}", file=sys.stderr)
        return 2
    finally:
        if selector is not None:
            selector.close()

    print(json.dumps(_json_value(answer), indent=2, ensure_ascii=False) if args.format == "json" else _render_text(answer))
    return 0 if answer.status == "answered" else 1
