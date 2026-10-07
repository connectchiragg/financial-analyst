"""Authenticated local PDF passages with deterministic keyword ranking.

BM25 scores measure lexical relevance, not financial truth or confidence. An
empty search result does not establish that a fact is absent from the corpus.
This adapter performs no semantic retrieval, LLM inference, or database reads.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re
from urllib.parse import urlsplit

from .adapters import Citation


MAX_PASSAGE_CHARACTERS = 1200
MAX_RESULTS = 50


class RetrievalError(ValueError):
    """Local source identity, extraction, or search input is invalid."""


@dataclass(frozen=True)
class Passage:
    ref: str
    document_id: str
    document_name: str
    page: int
    source_sha256: str
    excerpt: str
    start_offset: int
    end_offset: int
    url: str

    @property
    def locator(self) -> str:
        return f"Extracted page text, characters {self.start_offset}:{self.end_offset}"

    @property
    def citation(self) -> Citation:
        return Citation(self.ref, self.document_name, self.page, self.locator, self.url)


@dataclass(frozen=True)
class SearchHit:
    passage: Passage
    score: float


@dataclass(frozen=True)
class DocumentCoverage:
    document_id: str
    document_name: str
    source_sha256: str
    page_count: int
    pages_with_text: int
    passage_count: int


@dataclass(frozen=True)
class _Document:
    document_id: str
    document_name: str
    path: Path
    sha256: str
    url: str


def _nonblank(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RetrievalError(f"Manifest {field} must be a nonblank string.")
    if value != value.strip():
        raise RetrievalError(f"Manifest {field} cannot have surrounding whitespace.")
    if any(ord(character) < 32 for character in value):
        raise RetrievalError(f"Manifest {field} cannot contain control characters.")
    return value


def _tokens(text: str) -> tuple[str, ...]:
    # Unicode letters and digits are terms; punctuation is a separator. No
    # stemming or synonym expansion is implied by this keyword implementation.
    return tuple(re.findall(r"[^\W_]+", text.casefold()))


def _page_passages(text: str):
    """Yield exact, bounded slices and offsets, never crossing page boundaries."""
    start = 0
    while start < len(text):
        end = min(start + MAX_PASSAGE_CHARACTERS, len(text))
        if end < len(text):
            # Prefer a nearby whitespace boundary, but keep even a single long
            # unbroken token bounded instead of silently dropping page text.
            boundaries = [match.start() for match in re.finditer(r"\s", text[start:end])]
            if boundaries and boundaries[-1] >= MAX_PASSAGE_CHARACTERS // 2:
                end = start + boundaries[-1] + 1
        excerpt = text[start:end]
        if excerpt.strip():
            yield start, end, excerpt
        start = end


class LocalKeywordAdapter:
    """Read a hash-pinned local manifest and rank its exact PDF text passages.

    ``search`` uses BM25 (k1=1.2, b=0.75), counting each query term once.
    Ties use document identity, page, and offset for repeatable results. Sources
    are read once into memory and those same authenticated bytes are extracted,
    so a subsequent file change cannot replace the verified content.
    """

    mode = "local_keyword"

    def __init__(self, manifest_path: str | Path):
        try:
            manifest = Path(manifest_path)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise RetrievalError("Source manifest could not be read as JSON.") from exc
        documents = self._manifest_documents(payload, manifest.parent)
        try:
            import pdfplumber
        except ImportError as exc:
            raise RetrievalError("Local PDF retrieval requires pdfplumber.") from exc

        passages: list[Passage] = []
        coverage: list[DocumentCoverage] = []
        for document in documents:
            try:
                source_bytes = document.path.read_bytes()
            except OSError as exc:
                raise RetrievalError("A manifest source PDF is unavailable.") from exc
            if hashlib.sha256(source_bytes).hexdigest() != document.sha256:
                raise RetrievalError("Source PDF SHA-256 does not match the manifest.")
            page_count = 0
            pages_with_text = 0
            first_passage = len(passages)
            try:
                with pdfplumber.open(BytesIO(source_bytes)) as pdf:
                    page_count = len(pdf.pages)
                    for page_number, page in enumerate(pdf.pages, 1):
                        text = page.extract_text() or ""
                        if text.strip():
                            pages_with_text += 1
                        for start, end, excerpt in _page_passages(text):
                            ref = f"sha256:{document.sha256}:page:{page_number}:offset:{start}"
                            passages.append(Passage(
                                ref, document.document_id, document.document_name,
                                page_number, document.sha256, excerpt, start, end,
                                document.url,
                            ))
            except Exception as exc:
                raise RetrievalError("An authenticated source PDF could not be extracted.") from exc
            if page_count == 0:
                raise RetrievalError("A source PDF must contain at least one page.")
            coverage.append(DocumentCoverage(
                document.document_id, document.document_name, document.sha256,
                page_count, pages_with_text, len(passages) - first_passage,
            ))

        self.passages = tuple(passages)
        self.coverage = tuple(coverage)
        self._term_counts = tuple(Counter(_tokens(passage.excerpt)) for passage in self.passages)
        self._lengths = tuple(sum(counts.values()) for counts in self._term_counts)
        self._average_length = sum(self._lengths) / len(self._lengths) if self._lengths else 0
        self._document_frequency: Counter[str] = Counter()
        for counts in self._term_counts:
            self._document_frequency.update(counts.keys())

    @staticmethod
    def _manifest_documents(payload: object, directory: Path) -> tuple[_Document, ...]:
        if not isinstance(payload, dict) or not isinstance(payload.get("documents"), list):
            raise RetrievalError("Manifest must contain a documents list.")
        if not payload["documents"]:
            raise RetrievalError("Manifest documents list cannot be empty.")
        documents: list[_Document] = []
        ids: set[str] = set()
        names: set[str] = set()
        paths: set[Path] = set()
        hashes: set[str] = set()
        for raw in payload["documents"]:
            if not isinstance(raw, dict):
                raise RetrievalError("Each manifest document must be an object.")
            document_id = _nonblank(raw.get("document_id"), "document_id")
            name = _nonblank(raw.get("document_name"), "document_name")
            raw_path = _nonblank(raw.get("local_path"), "local_path")
            digest = _nonblank(raw.get("sha256"), "sha256").lower()
            url = _nonblank(raw.get("url"), "url")
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise RetrievalError("Manifest sha256 must contain 64 hexadecimal characters.")
            path = Path(raw_path)
            try:
                path = (path if path.is_absolute() else directory / path).resolve()
            except (OSError, RuntimeError, ValueError) as exc:
                raise RetrievalError("Manifest source path is invalid.") from exc
            if path.name != name or path.suffix.lower() != ".pdf":
                raise RetrievalError("Source PDF filename must match document_name.")
            try:
                parts = urlsplit(url)
                # Accessing port also rejects malformed ports. A citation is a
                # source link, never a credential-bearing network endpoint.
                parts.port
                valid_url = (
                    parts.scheme in {"https", "http"}
                    and bool(parts.hostname)
                    and parts.username is None
                    and parts.password is None
                    and not any(character.isspace() for character in url)
                )
            except ValueError as exc:
                raise RetrievalError("Manifest citation URL is invalid.") from exc
            if not valid_url:
                raise RetrievalError("Manifest citation URL must be an HTTP or HTTPS URL without credentials.")
            if document_id in ids or name in names or path in paths or digest in hashes:
                raise RetrievalError("Duplicate document identity, filename, source file, or digest in manifest.")
            ids.add(document_id)
            names.add(name)
            paths.add(path)
            hashes.add(digest)
            documents.append(_Document(document_id, name, path, digest, url))
        return tuple(documents)

    def search(self, query: str, limit: int = 5) -> tuple[SearchHit, ...]:
        if not isinstance(query, str) or not query.strip():
            raise RetrievalError("Keyword query must be a nonblank string.")
        query_terms = set(_tokens(query))
        if not query_terms:
            raise RetrievalError("Keyword query must contain at least one letter or digit.")
        if type(limit) is not int or not 1 <= limit <= MAX_RESULTS:
            raise RetrievalError(f"Search limit must be an integer from 1 to {MAX_RESULTS}.")
        count = len(self.passages)
        if not count or not self._average_length:
            return ()
        hits: list[SearchHit] = []
        for passage, terms, length in zip(self.passages, self._term_counts, self._lengths):
            score = 0.0
            for term in sorted(query_terms):
                frequency = terms.get(term, 0)
                if not frequency:
                    continue
                document_frequency = self._document_frequency[term]
                inverse_frequency = math.log(1 + (count - document_frequency + 0.5) / (document_frequency + 0.5))
                length_penalty = 1.2 * (1 - 0.75 + 0.75 * length / self._average_length)
                score += inverse_frequency * frequency * 2.2 / (frequency + length_penalty)
            if score > 0:
                hits.append(SearchHit(passage, score))
        hits.sort(key=lambda hit: (
            -hit.score, hit.passage.document_id, hit.passage.page,
            hit.passage.start_offset,
        ))
        return tuple(hits[:limit])
