"""Turn uploaded files into (sections of text) + (structured facts).

Structured facts (APIs, datasets/columns, glossary terms) are extracted deterministically so the
structured catalogue never depends on an LLM guessing. Supported: .md .txt .html/.htm (Confluence
exports) .docx .pdf (needs pypdf) .json (OpenAPI) .yaml/.yml (OpenAPI, needs PyYAML) .csv .sql
"""
from __future__ import annotations

import csv
import io
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import PurePosixPath

from .models import ApiFact, DatasetFact, Facts, FieldFact, TermFact


class ParseError(Exception):
    pass


@dataclass
class Section:
    heading_path: list[str]
    text: str


@dataclass
class Parsed:
    title: str
    sections: list[Section]
    facts: Facts = field(default_factory=Facts)


# ------------------------------------------------------------------------------ kind inference
_KIND_RULES = [
    ("glossary", r"glossary|terminolog|definitions"),
    ("policy", r"polic(y|ies)|standard|guideline|procedure|sop\b|code[-_ ]of"),
    ("industry", r"regulat|compliance|industry|sebi|mifid|basel|gdpr|sox|rbi\b|finra"),
    ("historical", r"\badr\b|decision|retro|lessons|post[-_ ]?mortem|history|migration"),
    ("data_dictionary", r"data[-_ ]?dict|schema|ddl|catalog"),
    ("technical", r"architecture|design|runbook|confluence|wiki|readme|technical|infra"),
]


def infer_kind(filename: str, content: bytes) -> str:
    low = filename.lower()
    ext = PurePosixPath(low).suffix
    head = content[:4000].decode("utf-8", "ignore").lower()
    if ext in (".json", ".yaml", ".yml") and ("openapi" in head or "swagger" in head):
        return "api_spec"
    if ext == ".sql" or (ext == ".csv" and re.search(r"column|field|attribute", head.split("\n", 1)[0])):
        return "data_dictionary"
    for kind, pattern in _KIND_RULES:
        if re.search(pattern, low):
            return kind
    first_heading = re.search(r"^#{1,3}\s+(.*)$|<h[1-3][^>]*>(.*?)</h[1-3]>", head, re.M)
    if first_heading and re.search(r"glossary|terminolog|definitions", " ".join(g or "" for g in first_heading.groups())):
        return "glossary"
    if ext in (".html", ".htm"):
        return "technical"  # Confluence / wiki exports
    return "business" if ext in (".docx", ".pdf") else "other"


# ------------------------------------------------------------------------------ text / markdown
_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


def _split_markdown(text: str, fallback_title: str) -> tuple[str, list[Section]]:
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []
    title = ""

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append(Section([h for _, h in stack], body))
        buf.clear()

    for line in text.splitlines():
        m = _HEADING.match(line)
        if m:
            flush()
            level, heading = len(m.group(1)), m.group(2).strip()
            title = title or heading
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
        else:
            buf.append(line)
    flush()
    return title or fallback_title, sections


# ------------------------------------------------------------------------------ html (Confluence)
class _HtmlText(HTMLParser):
    BLOCK = {"p", "div", "li", "tr", "br", "ul", "ol", "table", "section", "pre"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sections: list[Section] = []
        self.stack: list[tuple[int, str]] = []
        self.buf: list[str] = []
        self.skip = 0
        self.heading_level: int | None = None
        self.heading_buf: list[str] = []
        self.title_tag = ""
        self.in_title = False

    def flush(self) -> None:
        body = re.sub(r"[ \t]+", " ", "".join(self.buf))
        body = "\n".join(l.strip() for l in body.splitlines() if l.strip())
        if body:
            self.sections.append(Section([h for _, h in self.stack], body))
        self.buf = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1
        elif tag == "title":
            self.in_title = True
        elif re.fullmatch(r"h[1-6]", tag):
            self.flush()
            self.heading_level, self.heading_buf = int(tag[1]), []
        elif tag in ("td", "th"):
            self.buf.append(" | ")
        elif tag in self.BLOCK:
            self.buf.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = max(0, self.skip - 1)
        elif tag == "title":
            self.in_title = False
        elif re.fullmatch(r"h[1-6]", tag) and self.heading_level:
            heading = " ".join("".join(self.heading_buf).split())
            while self.stack and self.stack[-1][0] >= self.heading_level:
                self.stack.pop()
            if heading:
                self.stack.append((self.heading_level, heading))
            self.heading_level = None
        elif tag in self.BLOCK:
            self.buf.append("\n")

    def handle_data(self, data):
        if self.skip:
            return
        if self.in_title:
            self.title_tag += data
        elif self.heading_level:
            self.heading_buf.append(data)
        else:
            self.buf.append(data)


def _parse_html(text: str, fallback_title: str) -> tuple[str, list[Section]]:
    p = _HtmlText()
    p.feed(text)
    p.flush()
    title = " ".join(p.title_tag.split()) or (p.sections[0].heading_path[0] if p.sections and p.sections[0].heading_path else fallback_title)
    return title, p.sections


# ------------------------------------------------------------------------------ docx / pdf
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _parse_docx(content: bytes, fallback_title: str) -> tuple[str, list[Section]]:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            root = ET.fromstring(z.read("word/document.xml"))
    except (zipfile.BadZipFile, KeyError, ET.ParseError) as e:
        raise ParseError(f"not a valid .docx file: {e}") from e
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []
    title = ""

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append(Section([h for _, h in stack], body))
        buf.clear()

    for para in root.iter(f"{_W}p"):
        text = "".join(t.text or "" for t in para.iter(f"{_W}t")).strip()
        if not text:
            continue
        style = para.find(f"{_W}pPr/{_W}pStyle")
        val = style.get(f"{_W}val", "") if style is not None else ""
        m = re.fullmatch(r"(?i)heading\s*(\d)", val)
        if m or val.lower() == "title":
            flush()
            level = int(m.group(1)) if m else 1
            title = title or text
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, text))
        else:
            buf.append(text)
    flush()
    return title or fallback_title, sections


def _parse_pdf(content: bytes, fallback_title: str) -> tuple[str, list[Section]]:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ParseError("PDF support needs the 'pypdf' package (pip install pypdf)") from e
    try:
        reader = PdfReader(io.BytesIO(content))
        pages = [(p.extract_text() or "").strip() for p in reader.pages]
    except Exception as e:  # pypdf raises many types
        raise ParseError(f"could not read PDF: {e}") from e
    sections = [Section([f"Page {i}"], t) for i, t in enumerate(pages, 1) if t]
    if not sections:
        raise ParseError("PDF has no extractable text (scanned? OCR is not supported yet)")
    return fallback_title, sections


# ------------------------------------------------------------------------------ glossary
_TERM_LINE = re.compile(r"^\s*(?:[-*]\s*)?\**\s*([^:*|\n]{2,60}?)\s*\**\s*(?::|\s[-–—]\s)\s*(.{10,})$")
_TERM_ROW = re.compile(r"^\s*\|\s*([^|]{2,60}?)\s*\|\s*([^|]{10,}?)\s*\|")


def _extract_terms(sections: list[Section]) -> list[TermFact]:
    terms: dict[str, TermFact] = {}
    for sec in sections:
        for line in sec.text.splitlines():
            m = _TERM_ROW.match(line) or _TERM_LINE.match(line)
            if not m:
                continue
            term, definition = m.group(1).strip("* "), m.group(2).strip("* ")
            if term.lower() in ("term", "---") or set(term) <= set("-: "):
                continue
            syn = re.search(r"\((?:also|aka|a\.k\.a\.?)[: ]+([^)]+)\)", definition, re.I)
            synonyms = [s.strip() for s in re.split(r"[,;]", syn.group(1))] if syn else []
            terms.setdefault(term.lower(), TermFact(term=term, definition=definition, synonyms=synonyms))
    return list(terms.values())


# ------------------------------------------------------------------------------ csv
_TABLE_COLS = ("table", "dataset", "entity", "table_name", "dataset_name")
_FIELD_COLS = ("column", "field", "attribute", "column_name", "field_name", "attribute_name")
_TYPE_COLS = ("type", "data_type", "datatype")
_DESC_COLS = ("description", "definition", "meaning", "business_definition")


def _pick(row: dict[str, str], names: tuple[str, ...]) -> str:
    for n in names:
        if n in row and row[n] is not None:
            return row[n].strip()
    return ""


def _parse_csv(text: str, name_stem: str) -> tuple[list[Section], Facts]:
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ParseError("CSV has no header row")
    rows = [{(k or "").strip().lower().replace(" ", "_"): (v or "") for k, v in r.items()} for r in reader]
    cols = set(rows[0]) if rows else set(reader.fieldnames)
    facts = Facts()
    if cols & set(_FIELD_COLS):
        datasets: dict[str, DatasetFact] = {}
        for r in rows:
            fname = _pick(r, _FIELD_COLS)
            if not fname:
                continue
            ds = _pick(r, _TABLE_COLS) or name_stem
            d = datasets.setdefault(ds, DatasetFact(name=ds, system=_pick(r, ("system", "source", "database"))))
            d.fields.append(FieldFact(name=fname, type=_pick(r, _TYPE_COLS), description=_pick(r, _DESC_COLS),
                                      unit=_pick(r, ("unit", "units")) or None))
        facts.datasets = list(datasets.values())
        sections = [Section([d.name], "\n".join(
            f"{f.name} ({f.type}): {f.description}" + (f" [unit: {f.unit}]" if f.unit else "")
            for f in d.fields)) for d in facts.datasets]
    elif "term" in cols and cols & set(_DESC_COLS):
        for r in rows:
            if _pick(r, ("term",)):
                syn = [s.strip() for s in re.split(r"[;,]", _pick(r, ("synonyms", "aliases"))) if s.strip()]
                facts.terms.append(TermFact(term=_pick(r, ("term",)), definition=_pick(r, _DESC_COLS), synonyms=syn))
        sections = [Section(["Glossary"], "\n".join(f"{t.term}: {t.definition}" for t in facts.terms))]
    else:
        sections = [Section([name_stem], "\n".join(" | ".join(r.values()) for r in rows))]
    return [s for s in sections if s.text.strip()], facts


# ------------------------------------------------------------------------------ sql ddl
_CREATE = re.compile(r"create\s+table\s+(?:if\s+not\s+exists\s+)?([\w.\"`\[\]]+)\s*\((.*?)\)\s*(?:;|$)", re.I | re.S)
_COMMENT_COL = re.compile(r"comment\s+on\s+column\s+([\w.\"`]+)\s+is\s+'((?:[^']|'')*)'", re.I)
_COMMENT_TBL = re.compile(r"comment\s+on\s+table\s+([\w.\"`]+)\s+is\s+'((?:[^']|'')*)'", re.I)
_CONSTRAINT = re.compile(r"^(primary|foreign|unique|constraint|check|key|index)\b", re.I)


def _parse_sql(text: str) -> tuple[list[Section], Facts]:
    clean = lambda s: s.strip('"`[]')
    datasets: dict[str, DatasetFact] = {}
    for m in _CREATE.finditer(text):
        name, body = clean(m.group(1)), m.group(2)
        ds = DatasetFact(name=name)
        for line in body.splitlines():
            line = line.strip().rstrip(",")
            if not line or line.startswith("--") or _CONSTRAINT.match(line):
                continue
            inline = re.search(r"--\s*(.+)$", line)
            sql_c = re.search(r"comment\s+'((?:[^']|'')*)'", line, re.I)
            desc = (sql_c.group(1).replace("''", "'") if sql_c else inline.group(1).strip() if inline else "")
            col = re.match(r"([\w\"`]+)\s+([\w]+(?:\s*\([^)]*\))?)", re.sub(r"--.*$", "", line))
            if col:
                ds.fields.append(FieldFact(name=clean(col.group(1)), type=col.group(2).strip(), description=desc,
                                           required="not null" in line.lower()))
        datasets[name.lower()] = ds
    for m in _COMMENT_TBL.finditer(text):
        if clean(m.group(1)).lower() in datasets:
            datasets[clean(m.group(1)).lower()].description = m.group(2).replace("''", "'")
    for m in _COMMENT_COL.finditer(text):
        full = clean(m.group(1)).lower()
        table, _, col = full.rpartition(".")
        if table in datasets:
            for f in datasets[table].fields:
                if f.name.lower() == col:
                    f.description = m.group(2).replace("''", "'")
    if not datasets:
        raise ParseError("no CREATE TABLE statements found")
    facts = Facts(datasets=list(datasets.values()))
    sections = [Section([d.name], (d.description + "\n" if d.description else "") + "\n".join(
        f"{f.name} ({f.type}): {f.description}" for f in d.fields)) for d in facts.datasets]
    return sections, facts


# ------------------------------------------------------------------------------ openapi
def _resolve(spec: dict, node: dict, depth: int = 0) -> dict:
    while isinstance(node, dict) and "$ref" in node and depth < 10:
        cur = spec
        for part in node["$ref"].lstrip("#/").split("/"):
            cur = cur.get(part, {}) if isinstance(cur, dict) else {}
        node, depth = cur, depth + 1
    return node if isinstance(node, dict) else {}


def _flatten(spec: dict, schema: dict, prefix: str = "", depth: int = 0) -> list[FieldFact]:
    schema = _resolve(spec, schema)
    out: list[FieldFact] = []
    if schema.get("type") == "array" or "items" in schema:
        items = _resolve(spec, schema.get("items", {}))
        return _flatten(spec, items, f"{prefix}[]" if prefix else "[]", depth) if depth < 5 else out
    for name, sub in (schema.get("properties") or {}).items():
        sub = _resolve(spec, sub)
        path = f"{prefix}.{name}" if prefix else name
        typ = sub.get("type", "object" if "properties" in sub else "")
        if sub.get("format"):
            typ = f"{typ}({sub['format']})"
        if sub.get("enum"):
            typ += " enum:" + "|".join(map(str, sub["enum"]))
        desc = (sub.get("description") or "").strip()
        out.append(FieldFact(name=path, type=typ, description=desc, unit=sub.get("x-unit")))
        if (sub.get("type") in ("object", "array") or "properties" in sub or "items" in sub) and depth < 4:
            out.extend(_flatten(spec, sub, path, depth + 1))
    return out


def _parse_openapi(spec: dict, fallback_title: str) -> tuple[str, list[Section], Facts]:
    if not isinstance(spec, dict) or not ("openapi" in spec or "swagger" in spec):
        raise ParseError("not an OpenAPI/Swagger document")
    title = (spec.get("info") or {}).get("title") or fallback_title
    facts = Facts()
    sections = []
    if (spec.get("info") or {}).get("description"):
        sections.append(Section([title], spec["info"]["description"].strip()))
    for path, item in (spec.get("paths") or {}).items():
        for method, op in (item or {}).items():
            if method.lower() not in ("get", "post", "put", "patch", "delete") or not isinstance(op, dict):
                continue
            params = [FieldFact(name=p.get("name", ""), type=(_resolve(spec, p.get("schema", {})).get("type", "")),
                                description=(p.get("description") or "").strip(), required=bool(p.get("required")),
                                location=p.get("in", ""))
                      for p in map(lambda x: _resolve(spec, x), (item.get("parameters", []) + op.get("parameters", [])))]
            body = (_resolve(spec, op.get("requestBody", {})).get("content", {}).get("application/json", {}) or {}).get("schema")
            if body:
                params += [FieldFact(**{**f.model_dump(), "location": "body"}) for f in _flatten(spec, body)]
            resp_fields: list[FieldFact] = []
            for code, resp in (op.get("responses") or {}).items():
                if str(code).startswith("2"):
                    r = _resolve(spec, resp)
                    schema = ((r.get("content") or {}).get("application/json") or {}).get("schema")
                    if schema:
                        resp_fields = _flatten(spec, schema)
                        break
            api = ApiFact(method=method.upper(), path=path, summary=(op.get("summary") or "").strip(),
                          description=(op.get("description") or "").strip(), operation_id=op.get("operationId", ""),
                          tags=op.get("tags", []), params=params, response_fields=resp_fields)
            facts.apis.append(api)
            lines = [f"{api.method} {api.path} — {api.summary}", api.description]
            lines += [f"param {p.name} ({p.location}, {p.type}): {p.description}" for p in params]
            lines += [f"returns {f.name} ({f.type}): {f.description}" for f in resp_fields]
            sections.append(Section([title, f"{api.method} {api.path}"], "\n".join(l for l in lines if l)))
    return title, sections, facts


# ------------------------------------------------------------------------------ entry point
def parse(filename: str, content: bytes, kind: str) -> Parsed:
    stem = PurePosixPath(filename).stem
    ext = PurePosixPath(filename.lower()).suffix
    try:
        if ext in (".md", ".markdown", ".txt", ".rst", ""):
            title, sections = _split_markdown(content.decode("utf-8", "replace"), stem)
            if not sections and content.strip():
                sections = [Section([], content.decode("utf-8", "replace").strip())]
        elif ext in (".html", ".htm"):
            title, sections = _parse_html(content.decode("utf-8", "replace"), stem)
        elif ext == ".docx":
            title, sections = _parse_docx(content, stem)
        elif ext == ".pdf":
            title, sections = _parse_pdf(content, stem)
        elif ext in (".json", ".yaml", ".yml"):
            text = content.decode("utf-8", "replace")
            try:
                if ext == ".json":
                    spec = json.loads(text)
                else:
                    import yaml  # optional
                    spec = yaml.safe_load(text)
            except ImportError as e:
                raise ParseError("YAML support needs PyYAML (pip install pyyaml)") from e
            except (ValueError, Exception) as e:
                raise ParseError(f"invalid {ext[1:]}: {e}") from e
            if isinstance(spec, dict) and ("openapi" in spec or "swagger" in spec):
                title, sections, facts = _parse_openapi(spec, stem)
                return Parsed(title, sections, facts)
            title, sections = stem, [Section([], json.dumps(spec, indent=2)[:20000])]
        elif ext == ".csv":
            sections, facts = _parse_csv(content.decode("utf-8-sig", "replace"), stem)
            return Parsed(stem, sections, facts)
        elif ext == ".sql":
            sections, facts = _parse_sql(content.decode("utf-8", "replace"))
            return Parsed(stem, sections, facts)
        else:
            raise ParseError(f"unsupported file type '{ext}'")
    except ParseError:
        raise
    except Exception as e:  # defensive: one bad file must not kill a batch
        raise ParseError(f"failed to parse {filename}: {e}") from e
    if not sections:
        raise ParseError("document is empty")
    facts = Facts(terms=_extract_terms(sections)) if kind == "glossary" else Facts()
    return Parsed(title, sections, facts)


# ------------------------------------------------------------------------------ chunking
def chunk_sections(sections: list[Section], max_chars: int = 1200) -> list[tuple[str, str]]:
    """Heading-aware chunks: returns (heading_path, text). Paragraphs are never split mid-way
    unless one alone exceeds the limit."""
    chunks: list[tuple[str, str]] = []
    for sec in sections:
        path = " > ".join(sec.heading_path)
        cur = ""
        for para in re.split(r"\n\s*\n|\n(?=[-*•]|\d+\.)", sec.text):
            para = para.strip()
            if not para:
                continue
            while len(para) > max_chars:
                cut = para.rfind(" ", 0, max_chars) or max_chars
                if cur:
                    chunks.append((path, cur)); cur = ""
                chunks.append((path, para[:cut].strip())); para = para[cut:].strip()
            if cur and len(cur) + len(para) + 1 > max_chars:
                chunks.append((path, cur)); cur = ""
            cur = f"{cur}\n{para}".strip()
        if cur:
            chunks.append((path, cur))
    return chunks
