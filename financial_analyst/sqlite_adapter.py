"""Exact, authenticated observation storage behind a read-only SQLite port."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Mapping, Sequence

from .adapters import Evidence, EvidenceError, EvidenceReadPort
from .analytics import RevenueObservation


SCHEMA_VERSION = 1


class StorageError(EvidenceError):
    """The database or proposed import cannot safely provide observations."""


@dataclass(frozen=True)
class DocumentRecord:
    document_name: str
    sha256: str
    url: str | None = None
    provenance: str = "reviewed_fixture"
    metadata: Mapping | None = None


@dataclass(frozen=True)
class IngestionResult:
    documents_added: int
    evidence_added: int
    observations_added: int


_DDL = (
    """CREATE TABLE documents (
        sha256 TEXT PRIMARY KEY,
        document_name TEXT NOT NULL,
        url TEXT,
        provenance TEXT NOT NULL,
        metadata_json TEXT NOT NULL
    )""",
    """CREATE TABLE evidence (
        document_sha256 TEXT NOT NULL REFERENCES documents(sha256),
        ref TEXT NOT NULL,
        page INTEGER NOT NULL CHECK(page > 0),
        locator TEXT NOT NULL,
        excerpt TEXT NOT NULL,
        link TEXT,
        PRIMARY KEY(document_sha256, ref)
    )""",
    """CREATE TABLE observations (
        id TEXT PRIMARY KEY,
        document_sha256 TEXT NOT NULL REFERENCES documents(sha256),
        company TEXT NOT NULL,
        period TEXT NOT NULL,
        scope TEXT NOT NULL,
        metric TEXT NOT NULL,
        kind TEXT NOT NULL,
        currency TEXT NOT NULL,
        source_unit TEXT NOT NULL,
        value_text TEXT NOT NULL,
        raw_labels_json TEXT NOT NULL,
        UNIQUE(id, document_sha256),
        UNIQUE(document_sha256, company, period, scope, metric, kind)
    )""",
    """CREATE TABLE observation_evidence (
        observation_id TEXT NOT NULL,
        document_sha256 TEXT NOT NULL,
        evidence_ref TEXT NOT NULL,
        position INTEGER NOT NULL CHECK(position >= 0),
        PRIMARY KEY(observation_id, evidence_ref),
        UNIQUE(observation_id, position),
        FOREIGN KEY(observation_id, document_sha256)
            REFERENCES observations(id, document_sha256),
        FOREIGN KEY(document_sha256, evidence_ref)
            REFERENCES evidence(document_sha256, ref)
    )""",
    "CREATE INDEX observation_filter ON observations(company, period, scope, metric, kind)",
)
_COLUMNS = {
    "documents": {"sha256", "document_name", "url", "provenance", "metadata_json"},
    "evidence": {"document_sha256", "ref", "page", "locator", "excerpt", "link"},
    "observations": {"id", "document_sha256", "company", "period", "scope", "metric", "kind",
                     "currency", "source_unit", "value_text", "raw_labels_json"},
    "observation_evidence": {"observation_id", "document_sha256", "evidence_ref", "position"},
}
_RAW_FIELDS = ("year_end_raw", "source_header_raw", "source_value_raw", "source_unit_raw",
               "source_metric_raw")
_STRING_FIELDS = ("company", "period", "scope", "metric", "currency", "source_unit", "kind",
                  *_RAW_FIELDS)


def _require_text(value: object, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise StorageError(f"{label} must be a nonempty string.")


def _json(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise StorageError("Document metadata must contain JSON values.") from exc


def _validate_sha256(value: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise StorageError("Document SHA-256 must contain 64 lowercase hexadecimal characters.")


def _connect(path: Path, mode: str) -> sqlite3.Connection:
    try:
        connection = sqlite3.connect(path.resolve().as_uri() + f"?mode={mode}", uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection
    except sqlite3.Error as exc:
        raise StorageError("An existing SQLite database could not be opened.") from exc


def _check_schema(connection: sqlite3.Connection) -> None:
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version != SCHEMA_VERSION:
        raise StorageError(f"Unsupported SQLite schema version {version}; expected {SCHEMA_VERSION}.")
    for table, required in _COLUMNS.items():
        columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
        if columns != required:
            raise StorageError("SQLite tables do not match their declared schema version.")


def _validate_observation(observation: RevenueObservation) -> None:
    if not isinstance(observation, RevenueObservation):
        raise StorageError("Import requires RevenueObservation records.")
    for name in _STRING_FIELDS:
        _require_text(getattr(observation, name), f"Observation {name}")
    if not isinstance(observation.value, Decimal) or not observation.value.is_finite():
        raise StorageError("Observation value must be a finite Decimal, never a float.")
    try:
        printed = Decimal(observation.source_value_raw.replace(",", ""))
    except InvalidOperation as exc:
        raise StorageError("Observation source value is not decimal text.") from exc
    if not printed.is_finite() or printed != observation.value:
        raise StorageError("Observation value conflicts with its printed source value.")
    refs = observation.evidence_refs
    if not isinstance(refs, tuple) or not refs:
        raise StorageError("Observation requires a tuple of supporting evidence references.")
    for ref in refs:
        _require_text(ref, "Observation evidence reference")
    if len(set(refs)) != len(refs):
        raise StorageError("Observation evidence references must be unique.")


def _observation_row(document_sha256: str, observation: RevenueObservation) -> dict:
    grain = [document_sha256, observation.company, observation.period,
             observation.scope, observation.metric, observation.kind]
    return {
        "id": hashlib.sha256(_json(grain).encode()).hexdigest(),
        "document_sha256": document_sha256,
        **{name: getattr(observation, name) for name in
           ("company", "period", "scope", "metric", "kind", "currency", "source_unit")},
        "value_text": str(observation.value),
        "raw_labels_json": _json({name: getattr(observation, name) for name in _RAW_FIELDS}),
    }


def _insert_or_compare(connection: sqlite3.Connection, table: str, row: dict,
                       key: dict) -> bool:
    # Both table and column names come only from this module's fixed schema.
    where = " AND ".join(f"{column} = ?" for column in key)
    previous = connection.execute(f"SELECT * FROM {table} WHERE {where}", tuple(key.values())).fetchone()
    if previous is not None:
        if dict(previous) != row:
            raise StorageError(f"Conflicting {table} record; existing source data was preserved.")
        return False
    columns = ", ".join(row)
    placeholders = ", ".join("?" for _ in row)
    connection.execute(f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", tuple(row.values()))
    return True


class SQLiteStore:
    """Explicitly initialize and atomically import a reviewed evidence bundle."""

    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)

    def initialize(self) -> None:
        connection = _connect(self.database_path, "rwc")
        try:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                version = connection.execute("PRAGMA user_version").fetchone()[0]
                tables = connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                if version == 0 and not tables:
                    for statement in _DDL:
                        connection.execute(statement)
                    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                _check_schema(connection)
        except sqlite3.Error as exc:
            raise StorageError("SQLite initialization failed without a partial schema.") from exc
        finally:
            connection.close()

    def ingest(self, document: DocumentRecord, observations: Sequence[RevenueObservation],
               evidence: Sequence[Evidence], validator: EvidenceReadPort) -> IngestionResult:
        if not isinstance(document, DocumentRecord):
            raise StorageError("Import requires a DocumentRecord.")
        _validate_sha256(document.sha256)
        _require_text(document.document_name, "Document name")
        _require_text(document.provenance, "Document provenance")
        if document.url is not None:
            _require_text(document.url, "Document URL")
        if document.metadata is not None and not isinstance(document.metadata, Mapping):
            raise StorageError("Document metadata must be an object.")
        authenticated_sha = getattr(validator, "source_sha256", None)
        if authenticated_sha != document.sha256:
            raise StorageError("Document identity differs from its source validator.")
        document_row = {"sha256": document.sha256, "document_name": document.document_name,
                        "url": document.url, "provenance": document.provenance,
                        "metadata_json": _json(dict(document.metadata or {}))}
        observations = tuple(observations)
        supplied = {}
        for item in evidence:
            if not isinstance(item, Evidence):
                raise StorageError("Import requires authenticated Evidence records.")
            for name in ("ref", "document_name", "locator", "excerpt"):
                _require_text(getattr(item, name), f"Evidence {name}")
            if type(item.page) is not int or item.page <= 0:
                raise StorageError("Evidence page must be a positive integer.")
            if item.document_name != document.document_name or item.link != document.url:
                raise StorageError("Evidence belongs to a different document or citation URL.")
            if validator.resolve(item.ref) != item:
                raise StorageError("Evidence differs from its authenticated source.")
            if item.ref in supplied and supplied[item.ref] != item:
                raise StorageError("Evidence references conflict within the import.")
            supplied[item.ref] = item
        for observation in observations:
            _validate_observation(observation)
            validator.validate_observation(observation)
            for ref in observation.evidence_refs:
                if ref not in supplied or validator.resolve(ref) != supplied[ref]:
                    raise StorageError("Observation evidence is missing from the authenticated import.")

        connection = _connect(self.database_path, "rw")
        try:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                _check_schema(connection)
                document_added = _insert_or_compare(connection, "documents", document_row,
                                                    {"sha256": document.sha256})
                evidence_added = 0
                for item in supplied.values():
                    row = {"document_sha256": document.sha256, "ref": item.ref,
                           "page": item.page, "locator": item.locator,
                           "excerpt": item.excerpt, "link": item.link}
                    evidence_added += _insert_or_compare(connection, "evidence", row,
                                                         {"document_sha256": document.sha256, "ref": item.ref})
                observations_added = 0
                for observation in observations:
                    row = _observation_row(document.sha256, observation)
                    added = _insert_or_compare(connection, "observations", row, {"id": row["id"]})
                    stored_refs = tuple(item[0] for item in connection.execute(
                        "SELECT evidence_ref FROM observation_evidence WHERE observation_id = ? ORDER BY position",
                        (row["id"],)))
                    if not added and stored_refs != observation.evidence_refs:
                        raise StorageError("Conflicting observation evidence; existing source data was preserved.")
                    if added:
                        for position, ref in enumerate(observation.evidence_refs):
                            connection.execute(
                                "INSERT INTO observation_evidence VALUES (?, ?, ?, ?)",
                                (row["id"], document.sha256, ref, position))
                        observations_added += 1
                return IngestionResult(int(document_added), evidence_added, observations_added)
        except sqlite3.Error as exc:
            raise StorageError("SQLite ingestion failed; the complete import was rolled back.") from exc
        finally:
            connection.close()


class SQLiteAnalyticsAdapter:
    """Read existing persisted observations without creating or writing a DB.

    Bound reads retain local references for an explicitly matching source
    validator. Unbound reads qualify references by hash and therefore require
    a multi-document evidence resolver before they can support an answer.
    """

    mode = "sqlite"

    def __init__(self, database_path: str | Path, source_sha256: str | None = None):
        self.database_path = Path(database_path)
        if source_sha256 is not None:
            _validate_sha256(source_sha256)
        self.source_sha256 = source_sha256
        connection = _connect(self.database_path, "ro")
        try:
            _check_schema(connection)
            query = "SELECT DISTINCT provenance FROM documents"
            parameters = ()
            if source_sha256 is not None:
                query += " WHERE sha256 = ?"
                parameters = (source_sha256,)
            query += " ORDER BY provenance"
            self.seed_provenance = tuple(row[0] for row in connection.execute(query, parameters))
            if source_sha256 is not None and not self.seed_provenance:
                raise StorageError("Requested source document is absent from SQLite.")
        except sqlite3.Error as exc:
            raise StorageError("SQLite database could not be inspected.") from exc
        finally:
            connection.close()

    def read_observations(self, company: str, period: str, *, scope: str | None = None,
                          metric: str | None = None, kind: str | None = None) -> tuple[RevenueObservation, ...]:
        _require_text(company, "Company filter")
        _require_text(period, "Period filter")
        query = "SELECT * FROM observations WHERE company = ? AND period = ?"
        parameters = [company, period]
        if self.source_sha256 is not None:
            query += " AND document_sha256 = ?"
            parameters.append(self.source_sha256)
        for name, value in (("scope", scope), ("metric", metric), ("kind", kind)):
            if value is not None:
                _require_text(value, f"{name} filter")
                query += f" AND {name} = ?"
                parameters.append(value)
        query += " ORDER BY document_sha256, metric, scope, kind"
        connection = _connect(self.database_path, "ro")
        try:
            _check_schema(connection)
            observations = []
            for row in connection.execute(query, parameters):
                raw = json.loads(row["raw_labels_json"])
                if not isinstance(raw, dict) or set(raw) != set(_RAW_FIELDS):
                    raise StorageError("Stored observation raw labels are invalid.")
                links = connection.execute(
                    "SELECT evidence_ref, document_sha256, position FROM observation_evidence WHERE observation_id = ? ORDER BY position",
                    (row["id"],)).fetchall()
                if any(item[1] != row["document_sha256"] or item[2] != position
                       for position, item in enumerate(links)):
                    raise StorageError("Stored observation evidence has invalid document bindings or order.")
                refs = tuple(item[0] for item in links)
                if self.source_sha256 is None:
                    refs = tuple(f"{row['document_sha256']}:{ref}" for ref in refs)
                observation = RevenueObservation(
                    **{name: row[name] for name in
                       ("company", "period", "scope", "metric", "kind", "currency", "source_unit")},
                    value=Decimal(row["value_text"]), evidence_refs=refs, **raw)
                _validate_observation(observation)
                observations.append(observation)
            return tuple(observations)
        except (sqlite3.Error, ValueError, TypeError, KeyError, InvalidOperation) as exc:
            if isinstance(exc, StorageError):
                raise
            raise StorageError("SQLite contains invalid observation records.") from exc
        finally:
            connection.close()
