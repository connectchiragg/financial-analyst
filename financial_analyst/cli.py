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
    parser.add_argument("--source-pdf", type=Path, help="Original PDF for source validation, relocation or document ingestion.")
    parser.add_argument("--retrieval", choices=("fixture", "keyword", "semantic"), default="fixture")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--llm", choices=("none", "groq", "mistral", "openai-compatible"), default="none",
                        help="Inference provider for reference selection, local tool planning or pending fact extraction.")
    parser.add_argument("--model", help="Model used by the selected inference provider.")
    parser.add_argument("--env-file", type=Path, help="Explicit credential file; otherwise use the selected provider's environment variable.")
    parser.add_argument("--api-key-env", help="Credential variable name (default: GROQ_API_KEY, MISTRAL_API_KEY or INFERENCE_API_KEY).")
    parser.add_argument("--base-url", help="Explicit API base URL for an OpenAI-compatible provider.")
    parser.add_argument("--groq-model", help="Compatibility alias for --model when --llm groq is selected.")
    parser.add_argument("--manifest", type=Path, help="Hash-pinned local research PDF manifest for exploratory search.")
    parser.add_argument("--database", type=Path, help="Local SQLite file for reviewed analytics or separate document knowledge ingestion.")
    commands = parser.add_subparsers(dest="command", required=True)
    search = commands.add_parser("search", help="Explore exact PDF passages with local keyword ranking.")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=5)
    commands.add_parser("ingest", help="Validate and transactionally ingest a reviewed source bundle into SQLite.")
    prepare = commands.add_parser("prepare-knowledge", help="Prepare exactly filtered reviewed resources for a local knowledge indexing.")
    prepare.add_argument("--company", required=True)
    prepare.add_argument("--period", required=True)
    for name in ("scope", "metric", "kind", "currency", "unit"):
        prepare.add_argument("--" + name)
    prepare.add_argument("--destination", type=Path, default=Path(".local/knowledge-ready"),
                         help="Private local export directory; use a new directory for a different bundle.")
    knowledge = commands.add_parser("kb-search", help="Search contextualized evidence using exact financial filters.")
    knowledge.add_argument("query")
    knowledge.add_argument("--company", required=True)
    knowledge.add_argument("--period", required=True)
    for name in ("scope", "metric", "kind", "currency", "unit"):
        knowledge.add_argument("--" + name)
    knowledge.add_argument("--limit", type=int, default=5)
    for name in ("store-document", "extract"):
        document = commands.add_parser(name, help="Persist authenticated source passages and optionally propose unreviewed financial facts.")
        document.add_argument("--document-id", required=True)
        document.add_argument("--document-name")
        document.add_argument("--source-url", required=True)
        document.add_argument("--company", required=True)
        document.add_argument("--agency", required=True)
        if name == "extract":
            document.add_argument("--pages", help="Comma-separated original page numbers, or all; default first three pages.")
            document.add_argument("--max-windows", type=int, default=8)
    agent = commands.add_parser("agent-ask", help="Experimental local LangGraph argument planning with canonical financial output.")
    agent.add_argument("question")
    agent.add_argument("--company", required=True)
    agent.add_argument("--period", required=True)
    for name in ("compare", "combined", "growth", "ask", "beat-attribution"):
        command = commands.add_parser(name)
        command.add_argument("--company", required=True)
        command.add_argument("--period", required=True, help="Explicit fiscal label; calendar dates are not inferred.")
        if name in ("compare", "combined", "ask"):
            command.add_argument("--yoy", action="store_true")
        if name == "ask":
            command.add_argument("question")
    return parser


def _provider_config(args):
    """Resolve CLI configuration without constructing a client or reading secrets."""
    model = args.model
    if args.groq_model is not None:
        if args.llm != "groq":
            raise ValueError("--groq-model requires --llm groq; use --model for other providers.")
        if model is not None and model != args.groq_model:
            raise ValueError("--model and --groq-model disagree; select one model.")
        model = args.groq_model
    if args.llm == "none":
        if any(value is not None for value in (model, args.base_url, args.api_key_env, args.env_file)):
            raise ValueError("Inference configuration requires an explicit --llm provider; no fallback was performed.")
    elif args.llm == "openai-compatible":
        if not model or not args.base_url:
            raise ValueError("--llm openai-compatible requires explicit --model and --base-url; no fallback was performed.")
    elif args.base_url is not None:
        raise ValueError("--base-url requires --llm openai-compatible; no fallback was performed.")
    return {"model": model, "env_file": args.env_file, "api_key_env": args.api_key_env,
            "base_url": args.base_url, "timeout": 20}


def _create_selector(args):
    from .inference import create_inference
    from .selection import PassageSelector

    inference = create_inference(args.llm, **_provider_config(args))
    try:
        return PassageSelector(inference)
    except Exception:
        try:
            inference.close()
        except Exception:
            pass
        raise


def _search(args, parser):
    if args.mode != "local" or args.retrieval != "keyword":
        parser.error("Exploratory search requires --mode local --retrieval keyword; no fallback was performed.")
    if args.manifest is None:
        parser.error("--manifest is required for local keyword search.")
    if args.llm != "none":
        parser.error("Exploratory keyword search does not invoke an LLM; no fallback was performed.")
    try:
        _provider_config(args)
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
    if args.mode != "fixture" or args.retrieval not in {"keyword", "semantic"} or args.fixture is None:
        parser.error("Reviewed knowledge search requires --mode fixture, --fixture and keyword/semantic retrieval; no fallback was performed.")
    if args.database is not None:
        parser.error("Reviewed knowledge search uses source-validated fixture records; no database read was performed.")
    if args.retrieval == "semantic" and args.llm == "none":
        parser.error("Semantic retrieval requires an explicit --llm provider; no fallback was performed.")
    if args.retrieval == "keyword" and args.llm != "none":
        parser.error("Keyword knowledge search does not invoke an LLM; no fallback was performed.")
    selector = None
    try:
        _provider_config(args)
        from .adapters import FileFixtureAdapter
        from .knowledge import ReviewedKnowledgeIndex
        fixture = FileFixtureAdapter(args.fixture, source_path=args.source_pdf)
        index = ReviewedKnowledgeIndex.from_fixture(fixture)
        filters = {name: getattr(args, name) for name in ("company", "period", "scope", "metric", "kind", "currency", "unit")}
        retriever = index
        if args.retrieval == "semantic":
            from .semantic import SemanticKnowledgeRetriever
            selector = _create_selector(args)
            retriever = SemanticKnowledgeRetriever(index, selector)
        hits = retriever.search(args.query, **filters, limit=args.limit)
        llm = dict(retriever.execution) if selector is not None else {"provider": "none", "mode": "not_called"}
        result = {"status": "retrieved" if hits else "no_matches", "claims": [], "filters": filters,
                  "hits": [{"record": _json_value(hit.record),
                            **({"rank": hit.rank} if selector is not None else {"score": hit.score}),
                            "matching_contexts": _json_value(hit.matching_contexts),
                            "context_prefix": hit.context_prefix} for hit in hits],
                  "execution": {"mode": "fixture", "retrieval": retriever.mode, "context": "pdf_validated", "llm": llm,
                                "no_llm": llm["mode"] != "live", "no_database": True}}
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        print(f"Reviewed knowledge search failed: {error}", file=sys.stderr)
        return 2
    finally:
        if selector is not None:
            selector.close()
    if args.format == "json":
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        llm_label = "none" if llm["mode"] == "not_called" else f"{llm['mode']} {llm['provider']} ({llm['model']})"
        print(f"Execution: reviewed fixture knowledge; {retriever.mode}; LLM: {llm_label}; database: none.")
        for hit in hits:
            print("\nDerived context: " + hit.context_prefix)
            print("Source quote: " + hit.record.quote)
            citation = hit.record.citation
            print(f"{citation.document_name}, p. {citation.page}; {citation.locator}; {citation.link or ''}")
        if not hits:
            print("No matching reviewed records. This does not establish that the full corpus lacks an answer.")
    return 0


def _prepare_knowledge(args, parser):
    if args.mode != "fixture" or args.fixture is None:
        parser.error("Knowledge preparation requires --mode fixture and --fixture for original PDF validation.")
    if args.database is not None or args.llm != "none" or args.retrieval != "fixture":
        parser.error("Knowledge preparation applies context filters locally; it does not read SQLite, rank evidence or invoke inference.")
    try:
        _provider_config(args)
        from .adapters import FileFixtureAdapter
        from .knowledge_preparation import prepare_knowledge_groups, write_prepared_knowledge
        fixture = FileFixtureAdapter(args.fixture, source_path=args.source_pdf)
        filters = {name: getattr(args, name) for name in ("company", "period", "scope", "metric", "kind", "currency", "unit")}
        groups = prepare_knowledge_groups(fixture, **filters)
        manifest = write_prepared_knowledge(groups, args.destination) if groups else None
        result = {
            "status": "prepared" if groups else "no_matches", "claims": [], "filters": filters,
            "manifest": str(manifest.resolve()) if manifest is not None else None,
            "counts": {"groups": len(groups), "documents": sum(len(group.documents) for group in groups)},
            "execution": {"mode": "fixture", "provenance": "reviewed_fixture", "operation": "local_preparation_only",
                          "remote_upload": False, "sanitization": "not_performed", "no_llm": True, "no_database": True},
        }
    except (ValueError, OSError, ImportError) as error:
        print(f"Knowledge preparation failed: {error}", file=sys.stderr)
        return 2
    if args.format == "json":
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif manifest is not None:
        print(f"Prepared {result['counts']['documents']} reviewed resources in {result['counts']['groups']} exact context groups.")
        print("Private local manifest: " + result["manifest"])
        print("Execution: PDF-validated reviewed fixture; preparation only; remote upload, sanitization, LLM and database: none.")
    else:
        print("No matching reviewed contexts. No export or remote upload was performed.")
    return 0


def _ingest(args, parser):
    if args.mode != "live" or args.database is None or args.fixture is None:
        parser.error("Ingestion requires --mode live, --database and --fixture; no fallback was performed.")
    if args.llm != "none":
        parser.error("Reviewed-bundle ingestion does not invoke an LLM.")
    try:
        _provider_config(args)
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


def _document_ingest(args, parser):
    with_llm = args.command == "extract"
    if args.mode != "live" or args.source_pdf is None or args.database is None:
        parser.error("Document ingestion requires --mode live, --source-pdf and a separate knowledge --database.")
    if args.fixture is not None or args.retrieval != "fixture":
        parser.error("Document ingestion uses the supplied PDF directly and does not rank evidence or read a fixture.")
    if with_llm and args.llm == "none":
        parser.error("LLM extraction requires an explicit --llm provider.")
    if not with_llm and args.llm != "none":
        parser.error("store-document persists source passages without inference; use extract for LLM proposals.")
    inference = None
    extraction_error = None
    try:
        from .document_store import DocumentStore
        from .extraction import (MAX_WINDOWS, ExtractionError, extract_document,
                                 prepare_document, validate_extracted_document)
        config = _provider_config(args)
        if with_llm:
            # Larger page/schema extraction requests use an explicit bounded
            # timeout. Selection and ordinary tool planning retain 20 seconds.
            config["timeout"] = 60
        identity = {"document_id": args.document_id, "document_name": args.document_name or args.source_pdf.name,
                    "url": args.source_url, "company": args.company, "agency": args.agency}
        if with_llm:
            if not 1 <= args.max_windows <= MAX_WINDOWS:
                raise ValueError("--max-windows must be between one and eight.")
            # Validate the input and selected pages before opening a provider or
            # creating a database. Invalid configuration is not a failed run.
            source = prepare_document(args.source_pdf, **identity)
            pages = None
            if args.pages is not None:
                if args.pages == "all":
                    pages = tuple(range(1, source.source.page_count + 1))
                else:
                    try:
                        pages = tuple(int(value) for value in args.pages.split(","))
                    except ValueError:
                        raise ValueError("--pages requires original page numbers or all.") from None
                    if (not pages or len(set(pages)) != len(pages)
                            or any(not 1 <= page <= source.source.page_count for page in pages)):
                        raise ValueError("--pages must contain unique original page numbers within the source.")
            from .inference import create_inference
            inference = create_inference(args.llm, **config)
            try:
                bundle = extract_document(args.source_pdf, **identity, inference=inference,
                                          pages=pages, max_windows=args.max_windows)
            except ExtractionError as error:
                if error.source_document is None:
                    raise
                bundle = error.source_document
                extraction_error = str(error)
        else:
            bundle = prepare_document(args.source_pdf, **identity)
        # Reauthenticate before even creating the schema; ingestion checks
        # again before its write transaction in case the source has changed.
        validate_extracted_document(bundle, args.source_pdf)
        store = DocumentStore(args.database)
        store.initialize()
        counts = store.ingest(bundle, source_path=args.source_pdf)
        execution = dict(bundle.execution)
        execution.update(database="sqlite_knowledge", no_database=False, sanitization="not_performed")
        if inference is not None:
            execution.update(llm=dict(inference.execution), no_llm=inference.execution.get("mode") != "live")
        result = {"status": "source_stored_extraction_failed" if extraction_error else
                  "extracted_pending_review" if with_llm else "source_stored_unreviewed",
                  "claims": [], "counts": _json_value(counts), "source": _json_value(bundle.source),
                  "coverage": _json_value(bundle.coverage), "execution": execution,
                  "review_state": "pending_financial_association_review", "error": extraction_error}
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        print(f"Document ingestion failed: {error}", file=sys.stderr)
        return 2
    finally:
        if inference is not None:
            inference.close()
    if args.format == "json":
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"Document ingestion: {result['status']}; stored source passages: {counts.passages_added}; proposed facts: {counts.facts_added}.")
        print("Financial associations remain unreviewed; no analytical claims were produced.")
        if extraction_error:
            print("LLM extraction failed; only authenticated source passages were stored.")
    return 2 if extraction_error else 0


def _agent_ask(args, parser):
    if args.mode not in {"fixture", "live"} or args.fixture is None:
        parser.error("agent-ask requires a reviewed --fixture and fixture/live analytics mode.")
    if args.mode == "live" and args.database is None:
        parser.error("Live agent analytics requires --database; no fallback was performed.")
    if args.mode == "fixture" and args.database is not None:
        parser.error("Select --mode live for SQLite agent analytics.")
    if args.llm == "none":
        parser.error("agent-ask requires an explicit inference provider for argument planning.")
    if args.retrieval not in {"fixture", "keyword"}:
        parser.error("agent-ask uses local curated/keyword tools; semantic selection is outside its two-call budget.")
    inference = None
    try:
        from .adapters import FileFixtureAdapter
        from .agent import ToolPlanningAgent
        from .inference import create_inference
        from .service import ApplicationService

        config = _provider_config(args)
        fixture = FileFixtureAdapter(args.fixture, source_path=args.source_pdf)
        analytics, passages = fixture, fixture
        if args.mode == "live":
            from .sqlite_adapter import SQLiteAnalyticsAdapter
            analytics = SQLiteAnalyticsAdapter(args.database, source_sha256=fixture.source_sha256)
        if args.retrieval == "keyword":
            from .knowledge import KnowledgeGrowthAdapter
            passages = KnowledgeGrowthAdapter(fixture)
        service = ApplicationService(analytics, fixture, passages)
        inference = create_inference(args.llm, **config)
        answer = ToolPlanningAgent(inference, service).answer(args.question, args.company, args.period)
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        print(f"Agent execution failed: {error}", file=sys.stderr)
        return 2
    finally:
        if inference is not None:
            inference.close()
    print(json.dumps(_json_value(answer), indent=2, ensure_ascii=False) if args.format == "json" else _render_text(answer))
    return 0 if answer.status == "answered" else 1


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "chat":
        from .chat import main as chat_main
        return chat_main(arguments[1:])
    parser = _parser()
    args = parser.parse_args(arguments)
    if args.command in {"store-document", "extract"}:
        return _document_ingest(args, parser)
    if args.command == "agent-ask":
        return _agent_ask(args, parser)
    if args.command == "search":
        return _search(args, parser)
    if args.command == "ingest":
        return _ingest(args, parser)
    if args.command == "prepare-knowledge":
        return _prepare_knowledge(args, parser)
    if args.command == "kb-search":
        return _knowledge_search(args, parser)
    if args.mode == "live" and args.database is None:
        parser.error("Live analytics requires --database; no fallback was performed.")
    if args.mode == "local":
        parser.error("Local mode supports exploratory search only; no fallback was performed.")
    if args.mode == "fixture" and args.database is not None:
        parser.error("Select --mode live to read SQLite; no fallback was performed.")
    if args.retrieval != "fixture" and args.command not in {"growth", "combined", "ask"}:
        parser.error("Keyword/semantic answer retrieval requires growth or combined intent; no fallback was performed.")
    if args.retrieval == "semantic" and args.llm == "none":
        parser.error("Semantic retrieval requires an explicit --llm provider; no fallback was performed.")
    if args.fixture is None:
        parser.error("--fixture is required for original PDF evidence validation.")

    selector = None
    try:
        operation, include_yoy = args.command, getattr(args, "yoy", False)
        if operation == "ask":
            plan = plan_question(args.question, args.company, args.period)
            operation, include_yoy = plan.operation, include_yoy or plan.include_yoy
        if args.retrieval != "fixture" and operation not in {"growth", "combined"}:
            raise ValueError("Keyword/semantic answer retrieval requires growth or combined intent; no fallback was performed.")
        _provider_config(args)
        if args.llm != "none" and operation == "compare":
            raise ValueError("Inference selection requires a growth or combined request; no model call was performed.")

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
        if args.llm != "none" and operation in ("growth", "combined"):
            selector = _create_selector(args)
        if args.retrieval == "semantic":
            from .semantic import SemanticGrowthAdapter
            passages = SemanticGrowthAdapter(fixture, selector)
        service = ApplicationService(analytics, fixture, passages,
                                     selector=selector if args.retrieval != "semantic" else None)
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
