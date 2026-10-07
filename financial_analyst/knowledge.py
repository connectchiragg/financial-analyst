"""Reviewed source quotes with explicit financial context and strict filters.

Generated prefixes are indexing metadata, never quotations from a PDF. This
index contains only evidence associated with validated observations or curated
growth passages. Raw corpus pages cannot enter through the fixture factory.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import re

from .adapters import Citation, Evidence, FileFixtureAdapter
from .analytics import RevenueObservation
from .retrieval import MAX_RESULTS


class KnowledgeError(ValueError):
    """A reviewed knowledge query or context is invalid."""


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise KnowledgeError(f"Knowledge {field} must be a nonblank string.")
    return value


@dataclass(frozen=True, order=True)
class FinancialContext:
    company: str
    period: str
    scope: str
    metric: str
    kind: str
    currency: str
    unit: str

    def __post_init__(self):
        for field in fields(self):
            _required_text(getattr(self, field.name), field.name)


def _prefix(contexts: tuple[FinancialContext, ...], page: int) -> str:
    def safe(value: str) -> str:
        # Keep named fields readable without allowing source labels to inject
        # a new metadata field or line. The original values remain in context.
        return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "\\n").replace("\r", "\\r")

    return "\n".join(
        " | ".join(f"{field.name}={safe(getattr(context, field.name))}" for field in fields(context))
        + f" | page={page}"
        for context in sorted(set(contexts))
    )


@dataclass(frozen=True)
class KnowledgeRecord:
    ref: str
    quote: str
    source_sha256: str
    citation: Citation
    contexts: tuple[FinancialContext, ...]
    supporting_evidence_refs: tuple[str, ...]
    supporting_citations: tuple[Citation, ...]

    @property
    def document_name(self) -> str:
        return self.citation.document_name

    @property
    def page(self) -> int:
        return self.citation.page

    @property
    def context_prefix(self) -> str:
        return _prefix(self.contexts, self.page)

    @property
    def index_text(self) -> str:
        return self.context_prefix + "\n" + self.quote


@dataclass(frozen=True)
class KnowledgeHit:
    record: KnowledgeRecord
    score: float
    matching_contexts: tuple[FinancialContext, ...]

    @property
    def context_prefix(self) -> str:
        return _prefix(self.matching_contexts, self.record.page)

    @property
    def index_text(self) -> str:
        return self.context_prefix + "\n" + self.record.quote


def _observation_context(observation: RevenueObservation, kind: str | None = None) -> FinancialContext:
    return FinancialContext(
        observation.company, observation.period, observation.scope,
        observation.metric, kind or observation.kind, observation.currency,
        observation.source_unit_raw,
    )


class ReviewedKnowledgeIndex:
    """A fixture-backed snapshot of contextualized, authenticated evidence.

    Search first requires exact company and fiscal period. All supplied filters
    must hold in one context tuple. Lexical score is the count of distinct query
    terms present in the matching prefixes plus source quote; excluded context
    prefixes do not affect ranking. No stemming or embeddings are used.
    """

    mode = "reviewed_fixture_keyword"

    def __init__(self, records: tuple[KnowledgeRecord, ...]):
        self.records = records

    @classmethod
    def from_fixture(cls, fixture: FileFixtureAdapter) -> ReviewedKnowledgeIndex:
        if not isinstance(fixture, FileFixtureAdapter):
            raise KnowledgeError("Reviewed knowledge requires a validated FileFixtureAdapter.")
        observations = fixture.reviewed_observations()
        evidence = {item.ref: item for item in fixture.reviewed_evidence()}
        contexts: dict[str, set[FinancialContext]] = {}
        supports: dict[str, set[str]] = {}

        def associate(item: Evidence, context: FinancialContext, refs: tuple[str, ...]):
            # Resolve canonical evidence rather than accepting a newly supplied
            # quote, citation, or source identity from a caller.
            canonical = fixture.resolve(item.ref)
            if canonical != evidence.get(item.ref) or canonical.document_name != fixture.document_name:
                raise KnowledgeError("Reviewed evidence source identity is inconsistent.")
            contexts.setdefault(item.ref, set()).add(context)
            supports.setdefault(item.ref, set()).update(refs)
            supports[item.ref].add(item.ref)

        for observation in observations:
            fixture.validate_observation(observation)
            context = _observation_context(observation)
            for ref in observation.evidence_refs:
                associate(fixture.resolve(ref), context, observation.evidence_refs)

        commentary_contexts: set[FinancialContext] = set()
        for observation in observations:
            if observation.kind != "reported_actual":
                continue
            reported_quarter = re.compile(
                rf"\breported {re.escape(observation.period)} revenue\b", re.I,
            )
            # Historical numeric cells do not inherit the current quarter's
            # commentary. This adapter's growth contract requires an explicit
            # reported-quarter passage; any growth validation failure propagates.
            if not any(reported_quarter.search(evidence[ref].excerpt) for ref in observation.evidence_refs):
                continue
            context = _observation_context(observation, "broker_commentary")
            if context in commentary_contexts:
                continue
            passages = fixture.growth_passages(observation.company, observation.period)
            refs = tuple(dict.fromkeys((*observation.evidence_refs, *(item.ref for item in passages))))
            for item in passages:
                associate(item, context, refs)
            commentary_contexts.add(context)

        records = []
        for ref in sorted(contexts):
            item = fixture.resolve(ref)
            support_refs = tuple(sorted(supports[ref]))
            records.append(KnowledgeRecord(
                ref, item.excerpt, fixture.source_sha256, item.citation,
                tuple(sorted(contexts[ref])), support_refs,
                tuple(fixture.resolve(support_ref).citation for support_ref in support_refs),
            ))
        return cls(tuple(records))

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
    ) -> tuple[KnowledgeHit, ...]:
        _required_text(query, "query")
        filters = {"company": company, "period": period}
        for field, value in (("scope", scope), ("metric", metric), ("kind", kind),
                             ("currency", currency), ("unit", unit)):
            if value is not None:
                filters[field] = value
        for field, value in filters.items():
            _required_text(value, field)
        query_terms = set(re.findall(r"[^\W_]+", query.casefold()))
        if not query_terms:
            raise KnowledgeError("Knowledge query must contain at least one letter or digit.")
        if type(limit) is not int or not 1 <= limit <= MAX_RESULTS:
            raise KnowledgeError(f"Knowledge limit must be an integer from 1 to {MAX_RESULTS}.")

        hits = []
        for record in self.records:
            matching = tuple(context for context in record.contexts
                             if all(getattr(context, field) == value for field, value in filters.items()))
            if not matching:
                continue
            text = _prefix(matching, record.page) + "\n" + record.quote
            terms = set(re.findall(r"[^\W_]+", text.casefold()))
            score = float(len(query_terms & terms))
            if score:
                hits.append(KnowledgeHit(record, score, matching))
        hits.sort(key=lambda hit: (-hit.score, hit.record.document_name, hit.record.page, hit.record.ref))
        return tuple(hits[:limit])


class KnowledgeGrowthAdapter:
    """Expose hard-filtered reviewed keyword results through the passage port."""

    mode = "reviewed_fixture_keyword"

    def __init__(self, fixture: FileFixtureAdapter):
        self._fixture = fixture
        self.source_sha256 = fixture.source_sha256
        self.index = ReviewedKnowledgeIndex.from_fixture(fixture)

    def growth_passages(self, company: str, period: str) -> tuple[Evidence, ...]:
        hits = self.index.search("revenue growth", company, period, metric="net_sales",
                                 kind="broker_commentary", limit=MAX_RESULTS)
        return tuple(self._fixture.resolve(hit.record.ref) for hit in hits)
