"""Bounded semantic evidence selection after exact financial context filters.

The selector judges relevance directly; this is not an embedding or vector
index. Its only accepted output is ordered references to canonical records.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .adapters import Evidence, FileFixtureAdapter
from .knowledge import FinancialContext, KnowledgeError, KnowledgeHit, KnowledgeRecord, ReviewedKnowledgeIndex
from .retrieval import MAX_RESULTS
from .selection import PassageSelectionPort, SelectionAbstention, SelectionError, validate_selection


MAX_SEMANTIC_CANDIDATES = 32


@dataclass(frozen=True)
class SemanticHit:
    record: KnowledgeRecord
    rank: int
    matching_contexts: tuple[FinancialContext, ...]

    @property
    def context_prefix(self) -> str:
        return KnowledgeHit(self.record, 0.0, self.matching_contexts).context_prefix

    @property
    def index_text(self) -> str:
        return self.context_prefix + "\n" + self.record.quote


class SemanticKnowledgeRetriever:
    """Select meaning from all filtered records without a keyword prefilter.

    Candidate overflow fails before a selector call. Provider input-size bounds
    remain the selector's responsibility because it owns the actual request.
    Explicit result limits apply only after ordered selection, not to candidates.
    """

    mode = "llm_semantic_selection"

    def __init__(self, index: ReviewedKnowledgeIndex, selector: PassageSelectionPort):
        self.index = index
        self.selector = selector

    @property
    def execution(self) -> dict[str, Any]:
        return dict(self.selector.execution)

    def _reset_execution(self) -> None:
        # Reusing the retriever must never make a no-call request appear to have
        # invoked the model based on execution metadata from a previous query.
        self.selector.execution = {
            key: value for key, value in self.selector.execution.items()
            if key in {"provider", "model"}
        } | {"mode": "not_called"}

    def search(
        self,
        query: str,
        company: str,
        period: str,
        scope: str | None = None,
        metric: str | None = None,
        kind: str | None = None,
        currency: str | None = None,
        unit: str | None = None,
        limit: int = 5,
    ) -> tuple[SemanticHit, ...]:
        self._reset_execution()
        if not isinstance(query, str) or not query.strip():
            raise KnowledgeError("Semantic query must be a nonblank string.")
        if type(limit) is not int or not 1 <= limit <= MAX_RESULTS:
            raise KnowledgeError(f"Semantic limit must be an integer from 1 to {MAX_RESULTS}.")
        candidates = self.index.filter_records(company, period, scope, metric, kind, currency, unit)
        self.selector.query = query
        self.selector.contexts = {hit.record.ref: hit.context_prefix for hit in candidates}
        if not candidates:
            return ()
        if len(candidates) > MAX_SEMANTIC_CANDIDATES:
            raise SelectionError("Semantic retrieval exceeds the 32-candidate budget; no candidates were silently omitted.")
        passages = tuple(Evidence(
            hit.record.ref, hit.record.document_name, hit.record.page,
            hit.record.citation.locator, hit.record.quote, hit.record.citation.link,
        ) for hit in candidates)
        try:
            selected = self.selector.select(company, period, passages)
            refs = validate_selection(selected, passages)
        except SelectionAbstention:
            return ()
        by_ref = {hit.record.ref: hit for hit in candidates}
        return tuple(SemanticHit(by_ref[ref].record, rank, by_ref[ref].matching_contexts)
                     for rank, ref in enumerate(refs[:limit], 1))


class SemanticGrowthAdapter:
    """Select only the requested quarter's reviewed broker growth commentary."""

    mode = "llm_semantic_selection"

    def __init__(self, fixture: FileFixtureAdapter, selector: PassageSelectionPort):
        self._fixture = fixture
        self.source_sha256 = fixture.source_sha256
        self.index = ReviewedKnowledgeIndex.from_fixture(fixture)
        self.retriever = SemanticKnowledgeRetriever(self.index, selector)

    @property
    def execution(self) -> dict[str, Any]:
        return self.retriever.execution

    def growth_passages(self, company: str, period: str) -> tuple[Evidence, ...]:
        hits = self.retriever.search(
            "Explain the broker's reported revenue growth drivers.",
            company, period, metric="net_sales", kind="broker_commentary",
            limit=MAX_RESULTS,
        )
        return tuple(self._fixture.resolve(hit.record.ref) for hit in hits)
