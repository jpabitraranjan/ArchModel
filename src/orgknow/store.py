"""SQLite-backed knowledge base: versioned documents, hybrid search, structured catalogue.

Design points
- Tenant isolation: every read takes a ``Principal`` and every SQL statement filters on tenant_id.
- Versioning: re-uploading a ``source_key`` with new content creates version N+1 and closes the old
  version's validity window; identical content is skipped (safe to re-upload whole folders).
- Time travel: search/catalogue accept ``as_of`` to see what was true on a given date.
- Two retrieval paths: free-text (BM25 via FTS5 + embeddings, fused with RRF) for policies/docs, and
  a structured catalogue (APIs, datasets/columns, glossary) for "where does X live / what does Y mean".
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
import sqlite3
from array import array
from datetime import date, datetime, timezone
from typing import Iterable

from .embeddings import Embedder, HashingEmbedder, cosine
from .models import (ApiRecord, BatchReport, DatasetRecord, DocumentRecord, Facts, FieldFact, ItemOutcome,
                     Principal, SearchHit, SectionChange, SourceItem, SourceRef, TermRecord)
from .parsers import ParseError, chunk_sections, infer_kind, parse

SCHEMA = """
CREATE TABLE IF NOT EXISTS batches(id INTEGER PRIMARY KEY, tenant_id TEXT NOT NULL, label TEXT, created_at TEXT, summary TEXT);
CREATE TABLE IF NOT EXISTS documents(
  id INTEGER PRIMARY KEY, tenant_id TEXT NOT NULL, source_key TEXT NOT NULL, version INTEGER NOT NULL,
  title TEXT, kind TEXT, content_hash TEXT, status TEXT NOT NULL, valid_from TEXT NOT NULL, valid_to TEXT,
  ingested_at TEXT, batch_id INTEGER, allowed_roles TEXT, metadata TEXT,
  UNIQUE(tenant_id, source_key, version));
CREATE INDEX IF NOT EXISTS idx_doc_tenant ON documents(tenant_id, source_key);
CREATE TABLE IF NOT EXISTS chunks(
  id INTEGER PRIMARY KEY, document_id INTEGER NOT NULL, ordinal INTEGER, heading_path TEXT, text TEXT, embedding BLOB);
CREATE INDEX IF NOT EXISTS idx_chunk_doc ON chunks(document_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, chunk_id UNINDEXED, tokenize='porter unicode61');
CREATE TABLE IF NOT EXISTS apis(id INTEGER PRIMARY KEY, document_id INTEGER, tenant_id TEXT, method TEXT, path TEXT,
  summary TEXT, description TEXT, operation_id TEXT, tags TEXT);
CREATE TABLE IF NOT EXISTS api_fields(id INTEGER PRIMARY KEY, api_id INTEGER, direction TEXT, name TEXT, type TEXT,
  required INTEGER, location TEXT, description TEXT);
CREATE TABLE IF NOT EXISTS datasets(id INTEGER PRIMARY KEY, document_id INTEGER, tenant_id TEXT, name TEXT, system TEXT, description TEXT);
CREATE TABLE IF NOT EXISTS dataset_fields(id INTEGER PRIMARY KEY, dataset_id INTEGER, name TEXT, type TEXT, description TEXT, unit TEXT, required INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS terms(id INTEGER PRIMARY KEY, document_id INTEGER, tenant_id TEXT, term TEXT, definition TEXT, synonyms TEXT);
"""

_STOP = set("the a an and or of to in for on with by is are was be as at from that this it its into over per all any".split())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[A-Za-z0-9_]+", text.lower()) if len(t) > 1 and t not in _STOP]


def _stem(t: str) -> str:
    return t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t


def _fts_query(text: str) -> str:
    toks = list(dict.fromkeys(_tokens(text)))
    return " OR ".join(f'"{t}"' for t in toks[:40])


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


class KnowledgeBase:
    def __init__(self, path: str = ":memory:", embedder: Embedder | None = None):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.embedder: Embedder = embedder or HashingEmbedder()

    # ================================================================== ingestion
    def ingest_batch(self, tenant_id: str, items: Iterable[SourceItem], label: str = "") -> BatchReport:
        """Ingest many documents. Failures are isolated per document and reported, never fatal."""
        with self.db:
            batch_id = self.db.execute("INSERT INTO batches(tenant_id,label,created_at) VALUES(?,?,?)",
                                       (tenant_id, label, _now())).lastrowid
        outcomes: list[ItemOutcome] = []
        for item in items:
            try:
                with self.db:  # one transaction per document
                    outcomes.append(self._ingest_one(tenant_id, batch_id, item))
            except ParseError as e:
                outcomes.append(ItemOutcome(source_key=item.source_key, outcome="failed", error=str(e)))
            except Exception as e:  # noqa: BLE001 - keep the batch going
                outcomes.append(ItemOutcome(source_key=item.source_key, outcome="failed",
                                            error=f"unexpected error: {type(e).__name__}: {e}"))
        report = BatchReport(batch_id=batch_id, label=label, items=outcomes)
        with self.db:
            self.db.execute("UPDATE batches SET summary=? WHERE id=?", (report.model_dump_json(), batch_id))
        return report

    def _ingest_one(self, tenant_id: str, batch_id: int, item: SourceItem) -> ItemOutcome:
        digest = hashlib.sha256(item.content).hexdigest()
        prev = self.db.execute(
            "SELECT * FROM documents WHERE tenant_id=? AND source_key=? AND status!='retired' ORDER BY version DESC LIMIT 1",
            (tenant_id, item.source_key)).fetchone()
        if prev and prev["content_hash"] == digest:
            return ItemOutcome(source_key=item.source_key, outcome="unchanged", version=prev["version"], kind=prev["kind"])
        kind = item.kind or infer_kind(item.filename, item.content)
        parsed = parse(item.filename, item.content, kind)  # may raise ParseError
        valid_from = (item.effective_date or date.today()).isoformat()
        version = (prev["version"] + 1) if prev else 1
        if prev:
            self.db.execute("UPDATE documents SET status='superseded', valid_to=? WHERE id=?",
                            (max(valid_from, prev["valid_from"]), prev["id"]))
        doc_id = self.db.execute(
            "INSERT INTO documents(tenant_id,source_key,version,title,kind,content_hash,status,valid_from,ingested_at,"
            "batch_id,allowed_roles,metadata) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (tenant_id, item.source_key, version, item.title or parsed.title, kind, digest, "current", valid_from,
             _now(), batch_id, json.dumps(item.allowed_roles), json.dumps(item.metadata))).lastrowid
        chunks = chunk_sections(parsed.sections)
        for i, (path, text) in enumerate(chunks):
            indexed = f"{path}\n{text}" if path else text
            vec = array("f", self.embedder.embed(indexed)).tobytes()
            cid = self.db.execute("INSERT INTO chunks(document_id,ordinal,heading_path,text,embedding) VALUES(?,?,?,?,?)",
                                  (doc_id, i, path, text, vec)).lastrowid
            self.db.execute("INSERT INTO chunks_fts(text,chunk_id) VALUES(?,?)", (indexed, cid))
        self._store_facts(tenant_id, doc_id, parsed.facts)
        return ItemOutcome(source_key=item.source_key, outcome="updated" if prev else "new", version=version, kind=kind,
                           chunks=len(chunks), facts=parsed.facts.counts,
                           supersedes_version=prev["version"] if prev else None)

    def _store_facts(self, tenant_id: str, doc_id: int, facts: Facts) -> None:
        for a in facts.apis:
            api_id = self.db.execute(
                "INSERT INTO apis(document_id,tenant_id,method,path,summary,description,operation_id,tags) VALUES(?,?,?,?,?,?,?,?)",
                (doc_id, tenant_id, a.method, a.path, a.summary, a.description, a.operation_id, json.dumps(a.tags))).lastrowid
            for direction, fields in (("param", a.params), ("response", a.response_fields)):
                self.db.executemany(
                    "INSERT INTO api_fields(api_id,direction,name,type,required,location,description) VALUES(?,?,?,?,?,?,?)",
                    [(api_id, direction, f.name, f.type, int(f.required), f.location, f.description) for f in fields])
        for d in facts.datasets:
            ds_id = self.db.execute("INSERT INTO datasets(document_id,tenant_id,name,system,description) VALUES(?,?,?,?,?)",
                                    (doc_id, tenant_id, d.name, d.system, d.description)).lastrowid
            self.db.executemany("INSERT INTO dataset_fields(dataset_id,name,type,description,unit,required) VALUES(?,?,?,?,?,?)",
                                [(ds_id, f.name, f.type, f.description, f.unit, int(f.required)) for f in d.fields])
        self.db.executemany("INSERT INTO terms(document_id,tenant_id,term,definition,synonyms) VALUES(?,?,?,?,?)",
                            [(doc_id, tenant_id, t.term, t.definition, json.dumps(t.synonyms)) for t in facts.terms])

    def retire(self, principal: Principal, source_key: str, as_of: date | None = None) -> int:
        """Mark a document as no longer valid (e.g. a withdrawn policy). Returns versions affected."""
        with self.db:
            cur = self.db.execute(
                "UPDATE documents SET status='retired', valid_to=? WHERE tenant_id=? AND source_key=? AND status!='retired'",
                ((as_of or date.today()).isoformat(), principal.tenant_id, source_key))
        return cur.rowcount

    # ================================================================== visibility
    @staticmethod
    def _validity(as_of: date | None, include_superseded: bool) -> tuple[str, list]:
        if as_of is not None:
            d = as_of.isoformat()
            return "(d.valid_from <= ? AND (d.valid_to IS NULL OR d.valid_to > ?))", [d, d]
        if include_superseded:
            return "d.status IN ('current','superseded')", []
        return "d.status = 'current'", []

    @staticmethod
    def _visible(principal: Principal, allowed_roles_json: str | None) -> bool:
        allowed = json.loads(allowed_roles_json or "[]")
        return not allowed or bool(set(allowed) & set(principal.roles))

    # ================================================================== search
    def search(self, principal: Principal, query: str, limit: int = 8, kinds: list[str] | None = None,
               as_of: date | None = None, include_superseded: bool = False) -> list[SearchHit]:
        """Hybrid search (BM25 + embeddings, reciprocal-rank fusion), permission-filtered, with citations."""
        vclause, vparams = self._validity(as_of, include_superseded)
        kind_sql = f" AND d.kind IN ({','.join('?' * len(kinds))})" if kinds else ""
        base = (f"FROM chunks c JOIN documents d ON d.id=c.document_id WHERE d.tenant_id=? AND {vclause}{kind_sql}")
        params = [principal.tenant_id, *vparams, *(kinds or [])]
        fetch = limit * 4
        ranked: dict[int, float] = {}
        fq = _fts_query(query)
        lexical: list[int] = []
        if fq:
            rows = self.db.execute(
                f"SELECT c.id, d.allowed_roles, bm25(chunks_fts) AS s FROM chunks_fts f JOIN chunks c ON c.id=f.chunk_id "
                f"JOIN documents d ON d.id=c.document_id WHERE chunks_fts MATCH ? AND d.tenant_id=? AND {vclause}{kind_sql} "
                f"ORDER BY s LIMIT ?", [fq, *params, fetch]).fetchall()
            lexical = [r["id"] for r in rows if self._visible(principal, r["allowed_roles"])]
        qv = self.embedder.embed(query)
        scored = []
        for r in self.db.execute(f"SELECT c.id, c.embedding, d.allowed_roles {base}", params):
            if self._visible(principal, r["allowed_roles"]):
                vec = array("f"); vec.frombytes(r["embedding"])
                scored.append((cosine(qv, list(vec)), r["id"]))
        semantic = [cid for s, cid in sorted(scored, reverse=True)[:fetch] if s > 0.05]
        for ranking in (lexical, semantic):  # reciprocal rank fusion
            for rank, cid in enumerate(ranking):
                ranked[cid] = ranked.get(cid, 0.0) + 1.0 / (60 + rank + 1)
        hits = []
        for cid, score in sorted(ranked.items(), key=lambda kv: -kv[1])[:limit]:
            r = self.db.execute("SELECT c.*, d.source_key, d.title, d.kind, d.version, d.status FROM chunks c "
                                "JOIN documents d ON d.id=c.document_id WHERE c.id=?", (cid,)).fetchone()
            hits.append(SearchHit(chunk_id=cid, document_id=r["document_id"], source_key=r["source_key"], title=r["title"],
                                  kind=r["kind"], version=r["version"], status=r["status"], heading_path=r["heading_path"],
                                  text=r["text"], score=round(score, 5)))
        return hits

    def get_chunk(self, principal: Principal, chunk_id: int) -> SearchHit | None:
        r = self.db.execute("SELECT c.*, d.source_key, d.title, d.kind, d.version, d.status, d.allowed_roles FROM chunks c "
                            "JOIN documents d ON d.id=c.document_id WHERE c.id=? AND d.tenant_id=?",
                            (chunk_id, principal.tenant_id)).fetchone()
        if not r or not self._visible(principal, r["allowed_roles"]):
            return None
        return SearchHit(chunk_id=chunk_id, document_id=r["document_id"], source_key=r["source_key"], title=r["title"],
                         kind=r["kind"], version=r["version"], status=r["status"], heading_path=r["heading_path"],
                         text=r["text"], score=0.0)

    # ================================================================== structured catalogue
    def _src(self, r: sqlite3.Row) -> SourceRef:
        return SourceRef(document_id=r["did"], source_key=r["source_key"], version=r["version"], title=r["dtitle"])

    def find_terms(self, principal: Principal, names: list[str], as_of: date | None = None) -> list[TermRecord]:
        vclause, vp = self._validity(as_of, False)
        rows = self.db.execute(
            f"SELECT t.*, d.id did, d.source_key, d.version, d.title dtitle, d.allowed_roles FROM terms t "
            f"JOIN documents d ON d.id=t.document_id WHERE t.tenant_id=? AND {vclause}", [principal.tenant_id, *vp]).fetchall()
        wanted = [_norm(n) for n in names if n.strip()]
        out: dict[str, TermRecord] = {}
        for r in rows:
            if not self._visible(principal, r["allowed_roles"]):
                continue
            syn = json.loads(r["synonyms"] or "[]")
            keys = {_norm(r["term"]), *(_norm(s) for s in syn)}
            if any(w in keys or any(len(k) > 3 and f" {k} " in f" {w} " for k in keys) for w in wanted):
                out.setdefault(r["term"].lower(), TermRecord(term=r["term"], definition=r["definition"], synonyms=syn,
                                                             source=self._src(r)))
        return list(out.values())

    def search_datasets(self, principal: Principal, keywords: list[str], as_of: date | None = None,
                        limit: int = 8) -> list[DatasetRecord]:
        vclause, vp = self._validity(as_of, False)
        kw = {_stem(t) for k in keywords for t in _tokens(k) if len(t) > 2}
        out: list[tuple[int, DatasetRecord]] = []
        for r in self.db.execute(
                f"SELECT s.*, d.id did, d.source_key, d.version, d.title dtitle, d.allowed_roles FROM datasets s "
                f"JOIN documents d ON d.id=s.document_id WHERE s.tenant_id=? AND {vclause}", [principal.tenant_id, *vp]):
            if not self._visible(principal, r["allowed_roles"]):
                continue
            fields = [FieldFact(name=f["name"], type=f["type"] or "", description=f["description"] or "", unit=f["unit"],
                                required=bool(f["required"]))
                      for f in self.db.execute("SELECT * FROM dataset_fields WHERE dataset_id=? ORDER BY id", (r["id"],))]
            hay = {_stem(t) for t in _tokens(" ".join([r["name"], r["system"] or "", r["description"] or ""] +
                                                    [f"{f.name} {f.description}" for f in fields]))}
            score = len(kw & hay)
            if score:
                out.append((score, DatasetRecord(name=r["name"], system=r["system"] or "", description=r["description"] or "",
                                                 fields=fields, source=self._src(r))))
        return [d for _, d in sorted(out, key=lambda x: -x[0])[:limit]]

    def get_dataset(self, principal: Principal, name: str, as_of: date | None = None) -> DatasetRecord | None:
        hits = self.search_datasets(principal, _tokens(name), as_of, limit=50)
        return next((d for d in hits if d.name.lower() == name.lower()), None)

    def search_apis(self, principal: Principal, keywords: list[str], as_of: date | None = None, limit: int = 8) -> list[ApiRecord]:
        vclause, vp = self._validity(as_of, False)
        kw = {_stem(t) for k in keywords for t in _tokens(k) if len(t) > 2}
        out: list[tuple[int, ApiRecord]] = []
        for r in self.db.execute(
                f"SELECT a.*, d.id did, d.source_key, d.version, d.title dtitle, d.allowed_roles FROM apis a "
                f"JOIN documents d ON d.id=a.document_id WHERE a.tenant_id=? AND {vclause}", [principal.tenant_id, *vp]):
            if not self._visible(principal, r["allowed_roles"]):
                continue
            fl = self.db.execute("SELECT * FROM api_fields WHERE api_id=? ORDER BY id", (r["id"],)).fetchall()
            mk = lambda f: FieldFact(name=f["name"], type=f["type"] or "", description=f["description"] or "",
                                     required=bool(f["required"]), location=f["location"] or "")
            params = [mk(f) for f in fl if f["direction"] == "param"]
            resp = [mk(f) for f in fl if f["direction"] == "response"]
            hay = {_stem(t) for t in _tokens(" ".join([r["path"], r["summary"] or "", r["description"] or ""] +
                                                    [f"{f.name} {f.description}" for f in params + resp]))}
            score = len(kw & hay)
            if score:
                out.append((score, ApiRecord(method=r["method"], path=r["path"], summary=r["summary"] or "",
                                             description=r["description"] or "", params=params, response_fields=resp,
                                             source=self._src(r))))
        return [a for _, a in sorted(out, key=lambda x: -x[0])[:limit]]

    # ================================================================== versions & batches
    def versions(self, principal: Principal, source_key: str) -> list[DocumentRecord]:
        rows = self.db.execute("SELECT * FROM documents WHERE tenant_id=? AND source_key=? ORDER BY version",
                               (principal.tenant_id, source_key)).fetchall()
        return [DocumentRecord(id=r["id"], source_key=r["source_key"], version=r["version"], title=r["title"], kind=r["kind"],
                               status=r["status"], valid_from=r["valid_from"], valid_to=r["valid_to"], ingested_at=r["ingested_at"],
                               batch_id=r["batch_id"], content_hash=r["content_hash"])
                for r in rows if self._visible(principal, r["allowed_roles"])]

    def documents(self, principal: Principal, include_superseded: bool = False) -> list[DocumentRecord]:
        q = "SELECT * FROM documents WHERE tenant_id=?" + ("" if include_superseded else " AND status='current'") + " ORDER BY source_key, version"
        return [DocumentRecord(id=r["id"], source_key=r["source_key"], version=r["version"], title=r["title"], kind=r["kind"],
                               status=r["status"], valid_from=r["valid_from"], valid_to=r["valid_to"], ingested_at=r["ingested_at"],
                               batch_id=r["batch_id"], content_hash=r["content_hash"])
                for r in self.db.execute(q, (principal.tenant_id,)) if self._visible(principal, r["allowed_roles"])]

    def batches(self, principal: Principal) -> list[BatchReport]:
        return [BatchReport.model_validate_json(r["summary"]) for r in self.db.execute(
            "SELECT summary FROM batches WHERE tenant_id=? AND summary IS NOT NULL ORDER BY id", (principal.tenant_id,))]

    def diff_versions(self, principal: Principal, source_key: str, old: int, new: int) -> list[SectionChange]:
        """What changed between two versions of a document, by section (e.g. policy 2025 -> 2026)."""
        def sections(version: int) -> dict[str, str]:
            rows = self.db.execute(
                "SELECT c.heading_path, c.text, d.allowed_roles FROM chunks c JOIN documents d ON d.id=c.document_id "
                "WHERE d.tenant_id=? AND d.source_key=? AND d.version=? ORDER BY c.ordinal",
                (principal.tenant_id, source_key, version)).fetchall()
            if not rows:
                raise KeyError(f"{source_key} version {version} not found")
            if not self._visible(principal, rows[0]["allowed_roles"]):
                raise PermissionError("no access to this document")
            out: dict[str, str] = {}
            for r in rows:
                out[r["heading_path"]] = (out.get(r["heading_path"], "") + "\n" + r["text"]).strip()
            return out
        a, b = sections(old), sections(new)
        changes: list[SectionChange] = []
        for path in dict.fromkeys([*a, *b]):
            if path not in a:
                changes.append(SectionChange(heading_path=path, change="added", diff=b[path]))
            elif path not in b:
                changes.append(SectionChange(heading_path=path, change="removed", diff=a[path]))
            elif a[path] != b[path]:
                d = "\n".join(difflib.unified_diff(a[path].splitlines(), b[path].splitlines(), "old", "new", lineterm="", n=0))
                changes.append(SectionChange(heading_path=path, change="modified", diff=d))
        return changes
