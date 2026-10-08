# Financial Analyst

A CLI for source-backed financial facts, comparisons and commentary. It preserves source context, calculates deterministically, cites its evidence, and refuses unsupported answers.

## Current status

The question-only `financial_analyst.chat` entry point supports an explicit cross-company `corpus` engine. Its private configuration binds a reviewed proof pack, original-source catalog and SQLite database. Startup shows the companies and metrics actually available; record coverage does not imply every period or analytical operation is supported. The older revenue engine remains bound to one configured company and quarter.

The corpus engine uses LangGraph to plan typed local tools, validate their arguments, execute them, assess whole-question coverage and render canonical evidence. Tools cover source-fact/commentary lookup, compatible metric comparisons and YoY changes, revenue growth rankings, reviewed sector members, and narrowly authenticated printed-table valuation reconciliation. At most four tools and two bounded inference calls execute; model-authored values or prose are not rendered. Provider failures return no claims, with HTTP 429 reported explicitly.

Only explicitly approved source facts enter this engine. Numerical analytics additionally requires complete reviewed period, scope, status, currency and unit context. Unresolved source context remains visible and cannot be inferred by the model. Unknown calendar-year mappings, missing members and incompatible inputs refuse the whole requested calculation. Printed valuation inconsistencies are reported rather than adjusted to match a target. Full metric extraction, general redaction and comprehensive live-query accuracy remain unfinished.

## Cross-company terminal session

Create a private `.local/analyst-config.json` using your own reviewed sources:

```json
{
  "engine": "corpus",
  "mode": "live",
  "proof_pack": "reviewed-primary-corpus.json",
  "catalog": "research-manifest.json",
  "database": "reviewed-corpus.sqlite",
  "provider": "groq",
  "model": "openai/gpt-oss-20b",
  "env_file": "/absolute/path/to/private.env"
}
```

```sh
python -m financial_analyst.corpus_cli --config .local/analyst-config.json ingest
python -m financial_analyst.corpus_cli --config .local/analyst-config.json coverage
python -m financial_analyst.chat
```

The session prompts for a question; `/quit` exits. Original PDFs are required even after ingestion: live reads and final claims reauthenticate the source and canonical database records. `mode: fixture` explicitly uses reviewed in-memory records instead of SQLite. There is no fallback between modes. Source proof packs, databases, credentials and actual report-derived evaluation data belong in ignored private storage; public tests use invented facts and synthetic PDFs.

Quarterly revenue comparison, optional YoY growth and cited growth passages are implemented. Choose fixture analytics or real SQLite reads from a reviewed source bundle. Keyword retrieval and bounded Groq semantic retrieval operate over contextualized evidence. Optional inference selects references through a provider-neutral port; the application returns canonical source quotes and deterministic calculations. A separate experimental LangGraph path plans bounded local revenue-tool calls; it does not establish full cross-company coverage.

Local keyword search explores hash-verified PDFs and returns exact page passages with lexical scores and citations. Its raw passages have no reviewed financial context and do not enter calculated answers. Research PDFs, reviewed research fixtures, and private review notes are excluded from this public repository. Public tests generate synthetic PDFs and fixtures.

## Run

Python 3.10 or later is required. The calculator uses the standard library; `pdfplumber` verifies original PDF evidence. The Groq adapter uses its official SDK; Mistral and compatible endpoint adapters use HTTPX. Provider requests have bounded timeouts and no automatic retries; `python-dotenv` reads explicitly selected credential files without executing shell text.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m financial_analyst --help
```

Supply a reviewed local fixture and its original PDF. A fixture is a curated input snapshot with source locations, not automatically extracted data or a database schema. The adapter validates the PDF hash, source cells and headers, supporting passages, and citation references before returning claims.

```sh
python -m financial_analyst --mode fixture --fixture path/to/reviewed-fixture.json \
  --format json combined --company "Example Pharma" --period 1QFY27 --yoy
```

Commands are `compare`, `growth`, `combined`, `beat-attribution`, and `ask`. Use `--source-pdf` to relocate the original PDF while retaining its filename. Text output includes source links where supplied; JSON retains original observations, precise decimal strings, claim-level references, citations and execution metadata.

`ask` accepts a small, explicit question grammar, for example:

```sh
python -m financial_analyst --mode fixture --fixture path/to/reviewed-fixture.json \
  ask --company "Example Pharma" --period 1QFY27 \
  "By how much did Example Pharma's 1QFY27 revenue exceed the broker estimate?"
```

Mixed or unsupported questions are refused as a whole. The CLI does not answer a recognized subset while ignoring another company, metric, period or intent. Exit codes: `0` answered, `1` refused, `2` invalid configuration, source or provider execution. Unavailable capabilities never fall back silently to fixture execution.

## Interchangeable inference

Use `--llm groq` for `growth`, `combined` or semantic retrieval. Set `GROQ_API_KEY` in the environment, or pass `--env-file` with a private credential file. The default Groq model is `openai/gpt-oss-20b`; `--model` selects an explicit model. Keep credentials out of commands, fixtures and commits.

```sh
python -m financial_analyst --mode fixture --fixture path/to/reviewed-fixture.json \
  --llm groq --format json combined --company "Example Pharma" --period 1QFY27
```

Selection and retrieval consume a standard `InferencePort`: neutral messages, a named JSON schema and an output-token budget in; content and finish reason out. Provider adapters own wire formats, credentials, model IDs, timeouts, errors and cleanup. Financial services and the calculator do not consume provider SDK objects. Other API formats can implement this port without editing financial logic.

Switch to `--llm mistral` for its adapter (default `mistral-small-latest`, credential `MISTRAL_API_KEY`). The old `--groq-model` flag remains a Groq-only alias for `--model`. Or configure an explicit compatible endpoint:

```sh
python -m financial_analyst --mode fixture --fixture path/to/reviewed-fixture.json \
  --llm openai-compatible --model provider-model --base-url https://provider.example/v1 \
  --api-key-env PROVIDER_API_KEY combined --company "Example Pharma" --period 1QFY27
```

The compatible adapter appends `/chat/completions` to the API base URL and uses `max_tokens` with strict `json_schema`. The endpoint/model must support that contract. Incompatible responses fail explicitly without downgrading the schema, switching providers, retrying or silently falling back. Endpoints require HTTPS, or HTTP on loopback for local inference. Redirects cannot forward credentials. Its default credential variable is `INFERENCE_API_KEY`.

Output labels fixture/SQLite analytics, actual retrieval method, provider/model and actual inference invocation separately. Injected transports are labeled `test_double`. The model may select known references or abstain; malformed replies, unknown references, additional claims and API failures cannot produce an answer. Model-authored prose, citations and amounts never enter financial claims. The semantic adapter below adds query-based retrieval after context filters.

Provider-swap tests verify unchanged arithmetic, canonical quotes, citations and refusals across synthetic provider implementations. Mistral transport is tested with synthetic HTTP responses; live Mistral generation has not been verified. Documents remain local, accessed through our retrieval adapters. Hosted Mistral Libraries and Agents/Conversations are outside the current local retrieval flow.

## Exploratory keyword search

Supply a local manifest containing `documents`, each with `document_id`, `document_name`, `local_path`, `sha256` and `url`. Relative source paths resolve against the manifest directory. Include research documents rather than an exercise brief.

```sh
python -m financial_analyst --mode local --retrieval keyword \
  --manifest path/to/research-manifest.json --format json search "revenue growth" --limit 5
```

Sources are authenticated and extracted from the same bytes. Results stay within one PDF page, carry stable source-hash references, and expose extraction coverage. Ranking uses BM25 without stemming or synonyms. An empty result means no keyword matches, not proof that the corpus lacks an answer. This raw exploratory command supports keyword search only; reviewed knowledge below also supports semantic search.

## Reviewed knowledge preparation and filters

The answerable knowledge index uses only PDF-validated evidence associated with reviewed financial observations and broker commentary. It derives metadata for company, original fiscal period, scope, metric, actual/estimate/forecast kind, currency and observation source unit. A readable prefix aids interpretation; exact metadata filters determine eligibility before ranking. A comparison passage can retain multiple contexts, and every requested filter must match one complete context. Excluded contexts do not contribute to its score.

```sh
python -m financial_analyst --mode fixture --fixture path/to/reviewed-fixture.json \
  --retrieval keyword kb-search "revenue growth" --company "Example Pharma" \
  --period 1QFY27 --scope consolidated --metric net_sales --kind broker_commentary --unit INRm
```

Results expose generated `context_prefix` separately from the unchanged source `quote`, with document hash, page and supporting citations. Unit metadata describes the financial observation; a corroborating quote may contain its own rounded amount/unit, which remains in the quote. Empty results mean no matching reviewed records. General sanitation/redaction is not implemented yet; raw exploratory pages do not bypass this reviewed index.

For `growth` and `combined`, `--retrieval keyword` uses this filtered index instead of directly returning all curated passages. Optional `--llm groq` then selects canonical evidence by reference.

## Prepare local knowledge resources

Prepare a private local bundle from a reviewed, PDF-authenticated fixture. Exact filters run before export. Each resource has one complete financial context, a generated prefix separate from the unchanged quote, canonical source references, context-specific supporting citations and exact source values/labels.

```sh
python -m financial_analyst --mode fixture --fixture path/to/reviewed-fixture.json \
  prepare-knowledge --company "Example Pharma" --period 1QFY27 --kind broker_commentary \
  --destination .local/knowledge-ready/example-quarter
```

The JSON manifest labels `reviewed_fixture` provenance and `local_preparation_only`. This command performs no inference, database read, remote upload, sensitive-data redaction or general PDF sanitation. It defaults to an ignored local directory. Use a new destination for a different bundle; repeated identical content is unchanged, while conflicting files fail without replacement, including files created concurrently during export. No matching reviewed contexts produces no export.

Prepared bundles are inputs for later indexing, not authority to answer financial questions without original-source validation. Current answer retrieval still builds its reviewed index through the source adapter. The document workflow below stores unfamiliar PDFs and proposes facts, but financial associations still require review before answerable indexing.

## Bounded semantic retrieval

Use `--retrieval semantic --llm groq` with `kb-search`, `growth` or `combined`. All exact metadata filters run first; The configured inference provider sees every eligible record within the explicit budget and judges relevance by meaning. There is no keyword prefilter, embedding API or vector index. This method suits small reviewed candidate sets; exceeding 32 candidates or the 16,000-character provider request budget fails explicitly without truncating evidence or falling back.

```sh
python -m financial_analyst --mode fixture --fixture path/to/reviewed-fixture.json \
  --retrieval semantic --llm groq kb-search "What helped turnover expand?" \
  --company "Example Pharma" --period 1QFY27 --metric net_sales --kind broker_commentary
```

Semantic results contain ordinal ranks, unchanged quotes, derived context and supporting citations. They contain no invented confidence score. Abstention returns no matched reviewed records; it does not prove source absence. Malformed replies or API errors cannot yield a fallback answer. A combined semantic request performs one evidence-selection call and leaves the calculation in the deterministic analytics service.

## Intended behavior

- Retrieve cited passages with separately identified keyword and semantic search capabilities.
- Read financial observations through a read-only adapter to a user-owned relational database, then calculate comparisons deterministically.
- Combine calculated results with cited explanations while distinguishing arithmetic from source commentary.
- Refuse answers when evidence is missing or contradictory, comparison inputs are incompatible, denominators are invalid, or the source does not support an attribution.

Data sources, execution modes, and retrieval capabilities must be labeled accurately. An unavailable capability must produce an explicit limitation, without silently switching to another mode or provider. Narrative growth drivers must not be presented as the cause or quantified contribution of a numerical variance unless the evidence supports that claim.

## Boundaries and current limits

Source lookup and persistence stay behind adapters; the revenue comparator is pure. SQLite stores observations as rows, with explicit company, fiscal period, scope, metric, kind, currency and unit. New companies, metrics and periods are records rather than columns. Exact decimal text preserves monetary precision. Raw labels and optional source metadata accommodate report details. Documents, evidence, observations and supporting links form the versioned relational core; fundamental relationship changes may still require a migration.

## SQLite ingestion and analytics

Ingest an already reviewed bundle after original PDF validation. Initialization creates the local SQLite file explicitly; failed source validation creates no database. Ingestion is transactional and repeatable; identical inputs are unchanged, while conflicting source identities or observations fail instead of overwriting data.

```sh
python -m financial_analyst --mode live --database path/to/analyst.sqlite \
  --fixture path/to/reviewed-fixture.json ingest

python -m financial_analyst --mode live --database path/to/analyst.sqlite \
  --fixture path/to/reviewed-fixture.json --format json \
  combined --company "Example Pharma" --period 1QFY27 --yoy
```

Analytics opens the existing database read-only and revalidates returned observations against original PDF evidence. Output reports actual SQLite execution and `reviewed_fixture` seed provenance separately. Tampering cannot bypass evidence checks. The current CLI binds reads to one reviewed document; unbound multi-document reads need a multi-source evidence resolver. The separate document store below ingests canonical passages and pending LLM proposals. Sensitive-data redaction and promotion of those proposals into reviewed analytics are not implemented.

This iteration validates manually reviewed source bindings and retrieves supported passages. It does not automatically extract and validate financial facts from unfamiliar PDF layouts or guarantee that an entire corpus lacks an answer. A growth explanation does not establish the cause or product contribution of an estimate beat. Conflicting fiscal year-end labels remain visible; calendar dates are not inferred.

PDF hashes and reviewed source mappings check consistency with the supplied source. They are not a publisher-signature or remote Drive-authentication mechanism. The original source and its Drive identity must be reviewed when preparing a fixture.

## New-document passages and pending fact extraction

Store every extractable page passage in a separate private knowledge database. This preserves unchanged text, source hash, original page and offsets, with a readable prefix separate from the quotation. Company and research-house metadata supplied at import remain unreviewed.

```sh
python -m financial_analyst --mode live --source-pdf path/to/report.pdf \
  --database .local/knowledge.sqlite --format json store-document \
  --document-id source-document-id --source-url https://example.com/report \
  --company "Example Pharma" --agency "Example Research"
```

Use `extract` with an explicit inference provider to propose financial observations from bounded page windows. It stores all source passages even when only selected pages are sent to inference. The default selection is the first three pages; `--pages 2,6` or `--pages all` selects others, within a maximum of eight windows. The provider may return a small subset of facts; attempted pages never mean complete fact coverage. Each extraction call has a 60-second timeout and requests up to three proposals at a bounded 4,096-token output budget, with no retries.

```sh
python -m financial_analyst --mode live --source-pdf path/to/report.pdf \
  --database .local/knowledge.sqlite --llm groq --format json extract \
  --document-id source-document-id --source-url https://example.com/report \
  --company "Example Pharma" --agency "Example Research" --pages 2 --max-windows 1
```

Quotes and raw labels must match canonical source passages exactly. Decimal text must match the source number without changing its scale. Ambiguous period, scope or unit remains nullable. Actual/estimate/forecast status stays unknown in proposals; exact source status markers are retained for independent review. **Exact quote matching does not validate table-column interpretation.** All proposals remain pending review, produce no financial claims, and cannot enter the answerable index. Live-provider and synthetic-test execution are labeled separately.

Imports reauthenticate the source PDF and canonical passages before a transactional SQLite write. Identical inputs are unchanged; conflicting identities are rejected. The knowledge store uses a separate format from the reviewed analytics database. Provider or evidence failures return exit code `2`, retain only authenticated source passages, and report the failure explicitly. There is no silent fixture fallback or automatic review promotion.

This is an extraction and storage increment. Sensitive-data sanitation/redaction, an end-user review/promotion workflow and corpus-wide semantic indexing remain pending. The separate corpus engine consumes explicitly approved facts; pending extraction proposals cannot enter it automatically. Live extraction readiness must be verified against the chosen provider; synthetic tests alone do not establish it.

## Development approach

Show design checkpoints and ship small commits with verification evidence for review. Keep changes small and report what passed and what remains unverified. Do not commit research source files, private review material, credentials, or user data.

## Experimental local LangGraph planning

`agent-ask` lets the configured inference provider choose one local revenue tool and its typed arguments. LangGraph separates planning, validation, read-only execution, whole-question coverage assessment and canonical output/refusal. It uses the same neutral inference adapters as semantic retrieval; Groq is the current provider. No internet tool deployment is needed.

```sh
python -m financial_analyst --mode live --fixture path/to/reviewed-fixture.json \
  --database path/to/analyst.sqlite --llm groq --format json \
  agent-ask "How did turnover compare with the estimate and what drove revenue growth?" \
  --company "Example Pharma" --period 1QFY27
```

The graph allows at most two inference calls and one local tool call, with no retries or fallback. Tools return source-validated amounts, quotes and citations; model-authored financial prose is never rendered. Hosted tracing is disabled. Execution labels and the node trace distinguish live inference, test doubles, SQLite reads and reviewed-fixture provenance.

**This path is experimental.** Model-based whole-question coverage assessment can misinterpret requests; explicit guard tests and representative live probes supplement it. It does not establish support for every unseen question. The existing `ask` grammar and explicit deterministic commands remain available. Structured answer coverage is limited to reviewed revenue data; broad corpus keyword matches do not provide verified cross-company analytics. Provider quota failures produce zero claims and are not successful answer-quality tests.

## Question-only terminal session

```sh
python -m financial_analyst.chat
```

The session reads a private `.local/analyst-config.json` and prompts repeatedly for questions. Type `/quit`, send EOF or use Ctrl+C to exit. File paths resolve relative to the config file. Provider credentials stay in the configured external environment file. An explicit `engine: corpus` selects the cross-company path above. A legacy configuration without that engine uses its reviewed fixture, optional SQLite database, company and fiscal period, and remains limited to that bound context.

Corpus answers display readable source statements or deterministic calculations, with citations at the bottom. Use `python -m financial_analyst.chat --diagnostics` for execution, provider, database and provenance details. JSON output retains the full audit data.

OpenRouter can use the existing `openai-compatible` adapter: set `base_url` to `https://openrouter.ai/api/v1`, an explicit supported `model`, `api_key_env` to `OPENROUTER_API_KEY`, and `env_file` to a private credential file in the session configuration. Choose that configuration with `--config`; providers are never switched automatically. Successful authentication alone does not establish answer coverage.


## Temporary browser testing

The protected browser entry point uses the same reviewed corpus, inference adapter and source validation as the CLI. Set `FINANCIAL_ANALYST_ACCESS_CODE` to a private random access code and run:

```sh
python -m financial_analyst.web --config .local/openrouter-config.json --port 8765
```

The server binds to `127.0.0.1`. A temporary tunnel can make it reachable:

```sh
cloudflared tunnel --url http://127.0.0.1:8765 --no-autoupdate
```

Enter the access code and a question in the browser. The browser sends the code as an authorization header; it does not store it or put it in the URL. Reports and the SQLite database remain local. The link works only while the server, tunnel and Mac are running. This is a bounded test service, with one active answer request at a time; it does not establish production availability or full question coverage. Homebrew distribution is a separate CLI packaging task.
