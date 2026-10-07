"""Prepare reviewed, exactly filtered evidence for our local knowledge store.

This module performs no remote upload, inference, database access, redaction or
general document sanitization. PDF authentication remains the fixture adapter's
responsibility. Generated context prefixes are separate from source quotations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import re

from .adapters import Citation, FileFixtureAdapter
from .analytics import RevenueObservation
from .knowledge import FinancialContext, KnowledgeHit, ReviewedKnowledgeIndex


class KnowledgePreparationError(ValueError):
    """A reviewed preparation or its local export is inconsistent."""


_PREPARED = object()


@dataclass(frozen=True)
class PreparedKnowledgeDocument:
    record_ref: str
    source_sha256: str
    context: FinancialContext
    content: str
    content_sha256: str
    file_name: str
    citation: Citation
    supporting_citations: tuple[Citation, ...]
    _origin: object = field(default=None, init=False, repr=False, compare=False)


@dataclass(frozen=True)
class PreparedKnowledgeGroup:
    context: FinancialContext
    context_id: str
    documents: tuple[PreparedKnowledgeDocument, ...]
    _origin: object = field(default=None, init=False, repr=False, compare=False)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _context_id(context: FinancialContext) -> str:
    return _sha256(_json(asdict(context)))


def _observation_context(observation: RevenueObservation, kind: str | None = None) -> FinancialContext:
    return FinancialContext(
        observation.company, observation.period, observation.scope,
        observation.metric, kind or observation.kind, observation.currency,
        observation.source_unit_raw,
    )


def _support(
    fixture: FileFixtureAdapter,
    hit: KnowledgeHit,
    context: FinancialContext,
    observations: tuple[RevenueObservation, ...],
) -> tuple[tuple[str, ...], tuple[dict, ...]]:
    # A table heading can support several observations. Do not export its
    # index-wide union of support from other periods or actual/estimate columns.
    commentary = context.kind == "broker_commentary"
    matching = tuple(
        observation for observation in observations
        if _observation_context(observation, "broker_commentary" if commentary else None) == context
        and (commentary and observation.kind == "reported_actual"
             or not commentary and hit.record.ref in observation.evidence_refs)
    )
    if not matching:
        raise KnowledgePreparationError("Prepared context has no reviewed observation support.")
    refs = {ref for observation in matching for ref in observation.evidence_refs}
    if commentary:
        passages = fixture.growth_passages(context.company, context.period)
        if hit.record.ref not in {item.ref for item in passages}:
            raise KnowledgePreparationError("Prepared commentary is not a reviewed growth passage.")
        refs.update(item.ref for item in passages)
    refs.add(hit.record.ref)
    raw_observations = []
    for observation in matching:
        raw = asdict(observation)
        raw["value"] = str(observation.value)
        raw_observations.append(raw)
    raw_observations.sort(key=_json)
    return tuple(sorted(refs)), tuple(raw_observations)


def prepare_knowledge_groups(
    fixture: FileFixtureAdapter,
    *,
    company: str,
    period: str,
    scope: str | None = None,
    metric: str | None = None,
    kind: str | None = None,
    currency: str | None = None,
    unit: str | None = None,
) -> tuple[PreparedKnowledgeGroup, ...]:
    """Authenticate first, then prepare one resource per complete matching context."""
    index = ReviewedKnowledgeIndex.from_fixture(fixture)
    hits = index.filter_records(company, period, scope, metric, kind, currency, unit)
    observations = fixture.reviewed_observations()
    grouped: dict[FinancialContext, list[PreparedKnowledgeDocument]] = {}
    for hit in hits:
        record = hit.record
        canonical = fixture.resolve(record.ref)
        if (canonical.excerpt != record.quote or canonical.citation != record.citation
                or record.source_sha256 != fixture.source_sha256):
            raise KnowledgePreparationError("Prepared evidence does not match the authenticated source.")
        for context in sorted(set(hit.matching_contexts)):
            refs, raw_observations = _support(fixture, hit, context, observations)
            supporting_citations = tuple(fixture.resolve(ref).citation for ref in refs)
            prefix = KnowledgeHit(record, 0.0, (context,)).context_prefix
            content = _json({
                "format_version": 1,
                "provenance": "reviewed_fixture",
                "context": asdict(context),
                "context_prefix": prefix,
                "quote": record.quote,
                "source": {
                    "sha256": fixture.source_sha256,
                    "record_ref": record.ref,
                    "citation": asdict(record.citation),
                    "supporting_evidence_refs": refs,
                    "supporting_citations": tuple(asdict(item) for item in supporting_citations),
                },
                "supporting_observations": raw_observations,
            })
            digest = _sha256(content)
            document = PreparedKnowledgeDocument(
                record.ref, fixture.source_sha256, context, content, digest,
                f"reviewed-{digest}.json", record.citation, supporting_citations,
            )
            object.__setattr__(document, "_origin", _PREPARED)
            grouped.setdefault(context, []).append(document)
    groups = []
    for context in sorted(grouped):
        documents = tuple(sorted(grouped[context], key=lambda item: item.record_ref))
        group = PreparedKnowledgeGroup(context, _context_id(context), documents)
        object.__setattr__(group, "_origin", _PREPARED)
        groups.append(group)
    return tuple(groups)


def _validate_groups(groups: tuple[PreparedKnowledgeGroup, ...]) -> dict[str, bytes]:
    if not isinstance(groups, tuple):
        raise KnowledgePreparationError("Prepared knowledge groups must be a tuple.")
    files: dict[str, bytes] = {}
    contexts = set()
    for group in groups:
        if type(group) is not PreparedKnowledgeGroup or group._origin is not _PREPARED:
            raise KnowledgePreparationError("Export requires groups prepared from an authenticated fixture.")
        if type(group.context) is not FinancialContext:
            raise KnowledgePreparationError("Prepared group requires a complete financial context.")
        if group.context_id != _context_id(group.context) or group.context in contexts:
            raise KnowledgePreparationError("Prepared knowledge group context is inconsistent or duplicated.")
        contexts.add(group.context)
        if not isinstance(group.documents, tuple) or not group.documents:
            raise KnowledgePreparationError("Prepared knowledge group requires documents.")
        refs = set()
        for document in group.documents:
            if type(document) is not PreparedKnowledgeDocument or document._origin is not _PREPARED:
                raise KnowledgePreparationError("Export requires documents prepared from an authenticated fixture.")
            if (not isinstance(document.content, str)
                    or not isinstance(document.record_ref, str) or not document.record_ref.strip()
                    or not isinstance(document.source_sha256, str)
                    or re.fullmatch(r"[a-f0-9]{64}", document.source_sha256) is None
                    or type(document.citation) is not Citation
                    or not isinstance(document.supporting_citations, tuple)
                    or any(type(item) is not Citation for item in document.supporting_citations)):
                raise KnowledgePreparationError("Prepared document fields are invalid.")
            if document.context != group.context or document.record_ref in refs:
                raise KnowledgePreparationError("Prepared document context or reference is inconsistent.")
            refs.add(document.record_ref)
            digest = _sha256(document.content)
            if (document.content_sha256 != digest or document.file_name != f"reviewed-{digest}.json"
                    or document.file_name in files):
                raise KnowledgePreparationError("Prepared document content hash or filename is inconsistent.")
            try:
                payload = json.loads(document.content)
                source = payload["source"]
                citation = asdict(document.citation)
                supports = [asdict(item) for item in document.supporting_citations]
                support_refs = [item.ref for item in document.supporting_citations]
                valid = (
                    payload["format_version"] == 1 and payload["provenance"] == "reviewed_fixture"
                    and payload["context"] == asdict(group.context)
                    and source["sha256"] == document.source_sha256
                    and source["record_ref"] == document.record_ref
                    and source["citation"] == citation
                    and source["supporting_citations"] == supports
                    and source["supporting_evidence_refs"] == support_refs
                    and document.citation.ref == document.record_ref
                    and document.record_ref in support_refs
                )
            except (KeyError, TypeError, ValueError, AttributeError, RecursionError) as exc:
                raise KnowledgePreparationError("Prepared document content is invalid.") from exc
            if not valid:
                raise KnowledgePreparationError("Prepared document metadata does not match its content.")
            files[document.file_name] = document.content.encode("utf-8")
    return files


def write_prepared_knowledge(groups: tuple[PreparedKnowledgeGroup, ...], destination: Path) -> Path:
    """Write a private local bundle; refuse replacement of conflicting files.

    A private origin marker catches accidental construction of unreviewed data.
    It is not a security boundary or serialized authority: a new process must
    authenticate the fixture and prepare records again before exporting.
    """
    files = _validate_groups(groups)
    manifest = _json({
        "format_version": 1,
        "provenance": "reviewed_fixture",
        "execution": {
            "operation": "local_preparation_only",
            "remote_upload": False,
            "llm": "none",
            "database": "none",
            "sanitization": "not_performed",
        },
        "groups": [{
            "context_id": group.context_id,
            "context": asdict(group.context),
            "documents": [{
                "file_name": document.file_name,
                "content_sha256": document.content_sha256,
                "source_sha256": document.source_sha256,
                "record_ref": document.record_ref,
            } for document in group.documents],
        } for group in groups],
    })
    files["manifest.json"] = manifest.encode("utf-8")
    destination = Path(destination)
    try:
        if destination.is_symlink() or destination.exists() and not destination.is_dir():
            raise KnowledgePreparationError("Knowledge destination must be a directory, not a symlink.")
        # Check the entire bundle before creating any files. A differing
        # manifest requires a new destination, never silent replacement.
        for name, expected in files.items():
            path = destination / name
            if path.is_symlink() or path.exists() and (not path.is_file() or path.read_bytes() != expected):
                raise KnowledgePreparationError("Knowledge destination contains conflicting prepared content.")
        destination.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            path = destination / name
            try:
                with path.open("xb") as output:
                    output.write(content)
            except FileExistsError:
                # A concurrent writer may create a file after preflight. Never
                # skip it merely because it exists: accept only identical data.
                if path.is_symlink() or not path.is_file() or path.read_bytes() != content:
                    raise KnowledgePreparationError("Knowledge destination contains conflicting prepared content.")
        # Also catch a file replaced while later resources were being written.
        # The manifest is authoritative only when the entire bundle still agrees.
        for name, expected in files.items():
            path = destination / name
            if path.is_symlink() or not path.is_file() or path.read_bytes() != expected:
                raise KnowledgePreparationError("Prepared knowledge changed during export.")
    except OSError as exc:
        raise KnowledgePreparationError("Prepared knowledge could not be exported locally.") from exc
    return destination / "manifest.json"
