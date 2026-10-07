# Financial Analyst

A CLI for source-backed revenue comparisons and growth commentary, starting with explicitly reviewed fixtures. It preserves source context, calculates deterministically, cites its evidence, and refuses unsupported answers.

## Current status

The first fixture iteration is implemented. It compares quarterly revenue actuals with broker estimates, optionally calculates YoY growth, and retrieves curated growth passages. This is **fixture execution**, with no LLM or database calls. Keyword and semantic retrieval are later capabilities and currently return an explicit unavailable error.

Research PDFs, reviewed research fixtures, and private review notes are excluded from this public repository. Public tests generate synthetic PDFs and fixtures.

## Run

Python 3.10 or later is required. The calculator uses the standard library; the fixture adapter uses `pdfplumber` to verify evidence against the original PDF.

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

Mixed or unsupported questions are refused as a whole. The CLI does not answer a recognized subset while ignoring another company, metric, period or intent. Exit codes: `0` answered, `1` refused, `2` invalid configuration or source fixture. Live, keyword and semantic requests never fall back silently to fixture execution.

## Intended behavior

- Retrieve cited passages with separately identified keyword and semantic search capabilities.
- Read financial observations through a read-only adapter to a user-owned relational database, then calculate comparisons deterministically.
- Combine calculated results with cited explanations while distinguishing arithmetic from source commentary.
- Refuse answers when evidence is missing or contradictory, comparison inputs are incompatible, denominators are invalid, or the source does not support an attribution.

Data sources, execution modes, and retrieval capabilities must be labeled accurately. An unavailable capability must produce an explicit limitation, without silently switching to another mode or provider. Narrative growth drivers must not be presented as the cause or quantified contribution of a numerical variance unless the evidence supports that claim.

## Boundaries and current limits

Source lookup stays in adapters; the revenue comparator is pure. `AnalyticsReadPort` will connect to a user-owned database through a read-only adapter. This repository contains no database schema, SQL, migrations or database engine.

This iteration validates manually reviewed source bindings and retrieves curated passages. It does not extract arbitrary PDFs, perform a general semantic assessment, or guarantee that an entire corpus lacks an answer. A growth explanation does not establish the cause or product contribution of an estimate beat. Conflicting fiscal year-end labels remain visible; calendar dates are not inferred.

PDF hashes and reviewed source mappings check consistency with the supplied source. They are not a publisher-signature or remote Drive-authentication mechanism. The original source and its Drive identity must be reviewed when preparing a fixture.

## Development approach

Show design checkpoints and ship small commits with verification evidence for review. Keep changes small and report what passed and what remains unverified. Do not commit research source files, private review material, credentials, or user data.
