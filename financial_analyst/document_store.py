"""Persistent extracted knowledge, separate from reviewed analytics storage.

Exact source text and pending fact associations are persisted independently.
Import never promotes model-extracted facts to reviewed financial evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import sqlite3


SCHEMA_VERSION = 1


class DocumentStorageError(ValueError):
    pass


@dataclass(frozen=True)
class DocumentIngestionResult:
    documents_added: int
    passages_added: int
    facts_added: int
    evidence_links_added: int
    extraction_runs_added: int


_DDL = (
    """CREATE TABLE documents (
        source_sha256 TEXT PRIMARY KEY, document_id TEXT NOT NULL,
        document_name TEXT NOT NULL, url TEXT, company TEXT NOT NULL,
        agency TEXT, page_count INTEGER NOT NULL CHECK(page_count > 0),
        review_state TEXT NOT NULL
    )""",
    """CREATE TABLE passages (
        source_sha256 TEXT NOT NULL REFERENCES documents(source_sha256),
        ref TEXT NOT NULL, page INTEGER NOT NULL CHECK(page > 0),
        start_offset INTEGER NOT NULL CHECK(start_offset >= 0),
        end_offset INTEGER NOT NULL CHECK(end_offset > start_offset),
        quote TEXT NOT NULL, context_prefix TEXT NOT NULL,
        review_state TEXT NOT NULL, PRIMARY KEY(source_sha256, ref)
    )""",
    """CREATE TABLE facts (
        source_sha256 TEXT NOT NULL REFERENCES documents(source_sha256),
        fact_id TEXT NOT NULL, company TEXT NOT NULL, period TEXT, scope TEXT,
        metric TEXT NOT NULL, kind TEXT, currency TEXT, unit TEXT,
        value_text TEXT, raw_labels_json TEXT NOT NULL,
        review_state TEXT NOT NULL, PRIMARY KEY(source_sha256, fact_id)
    )""",
    """CREATE TABLE fact_evidence (
        source_sha256 TEXT NOT NULL, fact_id TEXT NOT NULL, position INTEGER NOT NULL,
        passage_ref TEXT NOT NULL, quote TEXT NOT NULL,
        PRIMARY KEY(source_sha256, fact_id, position),
        FOREIGN KEY(source_sha256, fact_id) REFERENCES facts(source_sha256, fact_id),
        FOREIGN KEY(source_sha256, passage_ref) REFERENCES passages(source_sha256, ref)
    )""",
    """CREATE TABLE extraction_runs (
        run_id TEXT PRIMARY KEY, source_sha256 TEXT NOT NULL REFERENCES documents(source_sha256),
        coverage_json TEXT NOT NULL, execution_json TEXT NOT NULL,
        extraction_complete INTEGER NOT NULL CHECK(extraction_complete IN (0,1))
    )""",
)
_TABLES = tuple(re.search(r"CREATE TABLE (\w+)", statement).group(1) for statement in _DDL)
_RAW_FIELDS = ("source_value_raw", "source_metric_raw", "source_period_raw", "source_scope_raw", "source_unit_raw", "source_kind_raw")


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as error:
        raise DocumentStorageError("Knowledge metadata must contain finite JSON values.") from error


def _json_read(value):
    try:
        parsed = json.loads(value)
        if _json(parsed) != value:
            raise DocumentStorageError("Stored knowledge JSON differs from its canonical representation.")
        return parsed
    except (TypeError, ValueError, RecursionError) as error:
        raise DocumentStorageError("Stored knowledge JSON is invalid.") from error


def _text(value, name, nullable=False):
    if nullable and value is None:
        return
    if not isinstance(value, str) or not value.strip():
        raise DocumentStorageError(f"Knowledge {name} must be nonblank text.")


def _digest(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise DocumentStorageError("A canonical source SHA-256 is required.")


def _connect(path, mode):
    try:
        connection = sqlite3.connect(Path(path).resolve().as_uri() + f"?mode={mode}", uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection
    except sqlite3.Error as error:
        raise DocumentStorageError("The knowledge SQLite database could not be opened.") from error


def _schema(connection):
    if connection.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
        raise DocumentStorageError("Unsupported knowledge SQLite schema version.")
    actual = {row[0]: " ".join(row[1].split()).casefold() for row in connection.execute("SELECT name,sql FROM sqlite_master WHERE type='table'")}
    expected = dict(zip(_TABLES, (" ".join(statement.split()).casefold() for statement in _DDL)))
    if actual != expected:
        raise DocumentStorageError("Knowledge tables do not match the declared schema fingerprint.")
    extras = connection.execute("SELECT 1 FROM sqlite_master WHERE type NOT IN ('table','index') OR (type='index' AND sql IS NOT NULL)").fetchone()
    if extras is not None:
        raise DocumentStorageError("Knowledge schema contains undeclared triggers, views or indexes.")


def _insert(connection, table, row, key):
    previous = connection.execute(f"SELECT * FROM {table} WHERE " + " AND ".join(f"{name}=?" for name in key), tuple(key.values())).fetchone()
    if previous is not None:
        if dict(previous) != row:
            raise DocumentStorageError(f"Conflicting {table} record; existing knowledge was preserved.")
        return 0
    connection.execute(f"INSERT INTO {table} (" + ",".join(row) + ") VALUES (" + ",".join("?" for _ in row) + ")", tuple(row.values()))
    return 1


def _prefix(values):
    def safe(value):
        return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", "\\n").replace("\r", "\\r")
    return " | ".join(f"{key}={safe(value)}" for key,value in values.items() if value is not None)


def _validate_fact(fact, source_sha256, passages):
    if fact.source_sha256 != source_sha256:
        raise DocumentStorageError("Fact source identity does not match its document.")
    for field in ("fact_id", "company", "metric", "source_metric_raw"):
        _text(getattr(fact, field), field)
    for field in ("period", "scope", "kind", "currency", "unit", *_RAW_FIELDS):
        _text(getattr(fact, field), field, nullable=True)
    if fact.review_state not in {"unreviewed", "quote_validated_pending_review"}:
        raise DocumentStorageError("Extraction cannot automatically promote facts to reviewed evidence.")
    if fact.value_text is not None:
        _text(fact.value_text, "value_text")
        _text(fact.source_value_raw, "source_value_raw")
        try:
            value = Decimal(fact.value_text)
            raw = fact.source_value_raw.replace(",", "").strip()
            if raw.endswith("%"):
                raw = raw[:-1]
            if raw.startswith("(") and raw.endswith(")"):
                raw = "-" + raw[1:-1]
            printed = Decimal(raw)
        except InvalidOperation as error:
            raise DocumentStorageError("Fact values must preserve finite decimal source text.") from error
        if not value.is_finite() or not printed.is_finite() or value != printed:
            raise DocumentStorageError("Fact decimal value conflicts with its printed source value.")
    if not fact.evidence or not isinstance(fact.evidence, tuple):
        raise DocumentStorageError("Facts require canonical quoted passage evidence.")
    seen = set()
    for item in fact.evidence:
        _text(item.ref, "evidence ref")
        _text(item.quote, "evidence quote")
        if item.ref not in passages or item.quote not in passages[item.ref].excerpt:
            raise DocumentStorageError("Fact evidence is not an exact canonical source passage quote.")
        if (item.ref,item.quote) in seen:
            raise DocumentStorageError("Fact evidence must be unique.")
        seen.add((item.ref,item.quote))


def _stored_passage(row):
    passage = dict(row)
    _digest(passage["source_sha256"])
    for name in ("document_id", "document_name", "company", "quote"):
        _text(passage[name], name)
    for name in ("url", "agency"):
        _text(passage[name], name, nullable=True)
    page, start, end = (passage[name] for name in ("page", "start_offset", "end_offset"))
    if (any(type(value) is not int for value in (page, start, end, passage["page_count"]))
            or not 1 <= page <= passage["page_count"] or start < 0
            or end - start != len(passage["quote"])):
        raise DocumentStorageError("Stored passage page or exact offsets are invalid.")
    expected_ref = f"sha256:{passage['source_sha256']}:page:{page}:offset:{start}"
    expected_prefix = _prefix({"company": passage["company"], "agency": passage["agency"],
        "document": passage["document_name"], "page": page,
        "source_sha256": passage["source_sha256"], "review_state": "unreviewed"})
    if (passage["ref"] != expected_ref or passage["review_state"] != "unreviewed"
            or passage["context_prefix"] != expected_prefix):
        raise DocumentStorageError("Stored passage source identity or review state is invalid.")
    passage.pop("page_count")
    return passage


class DocumentStore:
    """Source-bound persistent knowledge with explicit, read-only review gates."""

    mode = "sqlite_extracted_knowledge"

    def __init__(self, database_path):
        self.database_path = Path(database_path)

    def initialize(self):
        connection = _connect(self.database_path, "rwc")
        try:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                if connection.execute("PRAGMA user_version").fetchone()[0] == 0 and not connection.execute("SELECT 1 FROM sqlite_master WHERE type='table'").fetchone():
                    for statement in _DDL:
                        connection.execute(statement)
                    connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                _schema(connection)
        except sqlite3.Error as error:
            raise DocumentStorageError("Knowledge schema initialization failed without partial tables.") from error
        finally:
            connection.close()

    def ingest(self, bundle, *, source_path):
        from .extraction import ExtractedDocument, validate_extracted_document
        if not isinstance(bundle, ExtractedDocument):
            raise DocumentStorageError("Knowledge import requires an ExtractedDocument.")
        # Re-open/authenticate the original before making any database writes.
        validate_extracted_document(bundle, source_path)
        source = bundle.source
        _digest(source.sha256)
        for name in ("document_id", "document_name", "company"):
            _text(getattr(source,name), name)
        passages = {item.ref:item for item in bundle.passages}
        if len(passages) != len(bundle.passages):
            raise DocumentStorageError("Canonical passage references must be unique.")
        for passage in bundle.passages:
            if passage.review_state != "unreviewed" or passage.source_sha256 != source.sha256:
                raise DocumentStorageError("Imported passages must retain their unreviewed canonical source identity.")
        fact_ids = set()
        for fact in bundle.facts:
            _validate_fact(fact, source.sha256, passages)
            if fact.fact_id in fact_ids:
                raise DocumentStorageError("Fact identifiers must be unique within an import.")
            fact_ids.add(fact.fact_id)
        coverage, execution = _json(asdict(bundle.coverage)), _json(bundle.execution)
        run_id = hashlib.sha256(_json([source.sha256, coverage, execution]).encode()).hexdigest()
        source_row = {"source_sha256":source.sha256,"document_id":source.document_id,"document_name":source.document_name,"url":source.url,"company":source.company,"agency":source.agency,"page_count":source.page_count,"review_state":"unreviewed"}
        connection = _connect(self.database_path, "rw")
        counts = [0,0,0,0,0]
        try:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                _schema(connection)
                counts[0] += _insert(connection,"documents",source_row,{"source_sha256":source.sha256})
                for passage in bundle.passages:
                    row = {"source_sha256":source.sha256,"ref":passage.ref,"page":passage.page,"start_offset":passage.start_offset,"end_offset":passage.end_offset,"quote":passage.excerpt,"context_prefix":_prefix({"company":passage.company,"agency":passage.agency,"document":passage.document_name,"page":passage.page,"source_sha256":source.sha256,"review_state":"unreviewed"}),"review_state":"unreviewed"}
                    counts[1] += _insert(connection,"passages",row,{"source_sha256":source.sha256,"ref":passage.ref})
                for fact in bundle.facts:
                    row = {"source_sha256":source.sha256,"fact_id":fact.fact_id,**{name:getattr(fact,name) for name in ("company","period","scope","metric","kind","currency","unit","value_text")},"raw_labels_json":_json({name:getattr(fact,name) for name in _RAW_FIELDS}),"review_state":fact.review_state}
                    counts[2] += _insert(connection,"facts",row,{"source_sha256":source.sha256,"fact_id":fact.fact_id})
                    for position,item in enumerate(fact.evidence):
                        row = {"source_sha256":source.sha256,"fact_id":fact.fact_id,"position":position,"passage_ref":item.ref,"quote":item.quote}
                        counts[3] += _insert(connection,"fact_evidence",row,{"source_sha256":source.sha256,"fact_id":fact.fact_id,"position":position})
                row = {"run_id":run_id,"source_sha256":source.sha256,"coverage_json":coverage,"execution_json":execution,"extraction_complete":int(set(bundle.coverage.extracted_pages)==set(range(1,source.page_count+1)))}
                counts[4] += _insert(connection,"extraction_runs",row,{"run_id":run_id})
        except sqlite3.Error as error:
            raise DocumentStorageError("Knowledge import failed; its transaction was rolled back.") from error
        finally:
            connection.close()
        return DocumentIngestionResult(*counts)

    def _read(self, operation):
        connection = _connect(self.database_path,"ro")
        try:
            _schema(connection)
            return operation(connection)
        except sqlite3.Error as error:
            raise DocumentStorageError("Stored knowledge could not be read.") from error
        finally:
            connection.close()

    @staticmethod
    def _limit(limit):
        if type(limit) is not int or not 1 <= limit <= 10000:
            raise DocumentStorageError("Knowledge limit must be from one to 10,000.")

    def explore_passages(self, *, source_sha256=None, company=None, page=None, limit=50):
        self._limit(limit)
        clauses, parameters = [], []
        if source_sha256 is not None:
            _digest(source_sha256); clauses.append("p.source_sha256=?");parameters.append(source_sha256)
        if company is not None:
            _text(company,"company");clauses.append("d.company=?");parameters.append(company)
        if page is not None:
            if type(page) is not int or page < 1:
                raise DocumentStorageError("A positive page is required.")
            clauses.append("p.page=?");parameters.append(page)
        query = "SELECT p.*,d.document_id,d.document_name,d.url,d.company,d.agency,d.page_count FROM passages p JOIN documents d ON d.source_sha256=p.source_sha256"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY p.source_sha256,p.page,p.start_offset,p.ref LIMIT ?"
        return self._read(lambda db: tuple(_stored_passage(row) for row in db.execute(query,(*parameters,limit))))

    def facts(self, *, source_sha256=None, limit=50):
        self._limit(limit)
        if source_sha256 is not None:
            _digest(source_sha256)
        query = "SELECT * FROM facts" + (" WHERE source_sha256=?" if source_sha256 is not None else "") + " ORDER BY source_sha256,fact_id LIMIT ?"
        parameters = (source_sha256,limit) if source_sha256 is not None else (limit,)
        return self._read(lambda db:self._fact_rows(db,query,parameters))

    @staticmethod
    def _fact_rows(connection, query, parameters):
        result = []
        for row in connection.execute(query,parameters):
            fact = dict(row)
            _digest(fact["source_sha256"])
            for name in ("fact_id", "company", "metric"):
                _text(fact[name], name)
            for name in ("period", "scope", "kind", "currency", "unit", "value_text"):
                _text(fact[name], name, nullable=True)
            if fact["review_state"] not in {"unreviewed", "quote_validated_pending_review", "reviewed"}:
                raise DocumentStorageError("Stored fact review state is invalid.")
            source = connection.execute("SELECT company FROM documents WHERE source_sha256=?", (fact["source_sha256"],)).fetchone()
            if source is None or source["company"] != fact["company"]:
                raise DocumentStorageError("Stored fact company conflicts with its source document.")
            raw_labels = _json_read(fact.pop("raw_labels_json"))
            if not isinstance(raw_labels, dict) or set(raw_labels) != set(_RAW_FIELDS):
                raise DocumentStorageError("Stored fact raw label fields are invalid.")
            for name, value in raw_labels.items():
                _text(value, name, nullable=name != "source_metric_raw")
            evidence = []
            for item in connection.execute("SELECT passage_ref,quote,position FROM fact_evidence WHERE source_sha256=? AND fact_id=? ORDER BY position", (fact["source_sha256"], fact["fact_id"])):
                item = dict(item)
                if type(item["position"]) is not int or item["position"] != len(evidence):
                    raise DocumentStorageError("Stored fact evidence positions are invalid.")
                _text(item["quote"], "evidence quote")
                passage = connection.execute("SELECT quote FROM passages WHERE source_sha256=? AND ref=?", (fact["source_sha256"], item["passage_ref"])).fetchone()
                if passage is None or item["quote"] not in passage["quote"]:
                    raise DocumentStorageError("Stored fact evidence conflicts with its canonical passage.")
                evidence.append(item)
            if not evidence:
                raise DocumentStorageError("Stored facts require quoted passage evidence.")
            payload = {name: fact[name] for name in ("company", "period", "scope", "metric", "kind", "currency", "unit", "value_text")}
            payload.update(raw_labels)
            payload["evidence"] = [{"ref": item["passage_ref"], "quote": item["quote"]} for item in evidence]
            # Stable extractor identity detects edits to stored values or context.
            identity = "fact:sha256:" + hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if fact["fact_id"] != identity:
                raise DocumentStorageError("Stored fact identity conflicts with its value, context or evidence.")
            fact["raw_labels"] = raw_labels
            fact["evidence"] = evidence
            fact["context_prefix"] = _prefix({name:fact[name] for name in ("company","period","scope","metric","kind","currency","unit","review_state")})
            result.append(fact)
        return tuple(result)

    def query_reviewed(self, company, period, *, scope=None, metric=None, kind=None, currency=None, unit=None, limit=50):
        """All filters bind one reviewed fact context; pending rows never qualify."""
        self._limit(limit)
        _text(company,"company");_text(period,"period")
        filters = {"company":company,"period":period}
        for name,value in (("scope",scope),("metric",metric),("kind",kind),("currency",currency),("unit",unit)):
            if value is not None:
                _text(value,name);filters[name]=value
        query = "SELECT * FROM facts WHERE review_state='reviewed' AND scope IS NOT NULL AND kind IS NOT NULL AND unit IS NOT NULL AND " + " AND ".join(f"{name}=?" for name in filters) + " ORDER BY source_sha256,fact_id LIMIT ?"
        return self._read(lambda db:self._fact_rows(db,query,(*filters.values(),limit)))

    def coverage(self, *, source_sha256=None):
        if source_sha256 is not None:
            _digest(source_sha256)
        query = "SELECT * FROM extraction_runs" + (" WHERE source_sha256=?" if source_sha256 is not None else "") + " ORDER BY source_sha256,run_id"
        parameters = (source_sha256,) if source_sha256 is not None else ()
        def operation(db):
            result = []
            for row in db.execute(query, parameters):
                coverage = _json_read(row["coverage_json"])
                execution = _json_read(row["execution_json"])
                if not isinstance(coverage, dict) or not isinstance(execution, dict):
                    raise DocumentStorageError("Stored extraction coverage and execution must be objects.")
                identity = hashlib.sha256(_json([row["source_sha256"], row["coverage_json"], row["execution_json"]]).encode()).hexdigest()
                if identity != row["run_id"]:
                    raise DocumentStorageError("Stored extraction run identity conflicts with its metadata.")
                total = coverage.get("total_pages")
                extracted = coverage.get("extracted_pages")
                source = db.execute("SELECT page_count FROM documents WHERE source_sha256=?", (row["source_sha256"],)).fetchone()
                if (source is None or type(total) is not int or total != source["page_count"]
                        or not isinstance(extracted, list)
                        or any(type(page) is not int for page in extracted)
                        or bool(row["extraction_complete"]) != (set(extracted) == set(range(1, total + 1)))):
                    raise DocumentStorageError("Stored extraction completeness conflicts with its source coverage.")
                result.append({"run_id": row["run_id"], "source_sha256": row["source_sha256"],
                    "extraction_complete": bool(row["extraction_complete"]), "coverage": coverage,
                    "execution": execution})
            return tuple(result)
        return self._read(operation)
