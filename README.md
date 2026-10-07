# Financial Analyst

A planned CLI for answering financial questions using research documents and deterministic analytics. Answers should preserve source context, show reproducible calculations, and cite the evidence supporting each claim.

## Current status

The initial design checkpoint is awaiting approval. The CLI is **not implemented**, and no live LLM or database calls have been made. This first public iteration contains the repository scaffold only; private sources and review notes are excluded.

## Intended behavior

- Retrieve cited passages with separately identified keyword and semantic search capabilities.
- Read financial observations through a read-only adapter to a user-owned relational database, then calculate comparisons deterministically.
- Combine calculated results with cited explanations while distinguishing arithmetic from source commentary.
- Refuse answers when evidence is missing or contradictory, comparison inputs are incompatible, denominators are invalid, or the source does not support an attribution.

Data sources, execution modes, and retrieval capabilities must be labeled accurately. An unavailable capability must produce an explicit limitation, without silently switching to another mode or provider. Narrative growth drivers must not be presented as the cause or quantified contribution of a numerical variance unless the evidence supports that claim.

## Development approach

Review and approve each design checkpoint before implementation. Keep changes small, verify the important behavior, and report what was checked and what remains unverified. Do not commit research source files, private review material, credentials, or user data.
