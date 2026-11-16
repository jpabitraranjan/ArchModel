import io
import json
import zipfile
from datetime import date
from pathlib import Path

import pytest

from orgknow import KnowledgeBase, Principal, SourceItem
from orgknow.cli import load_directory

KB_DIR = Path(__file__).parent.parent / "examples" / "finance_kb"
ACME = Principal(tenant_id="acme")


def item(key, text, **kw):
    return SourceItem(source_key=key, filename=key.split("/")[-1], content=text.encode(), **kw)


@pytest.fixture
def kb():
    k = KnowledgeBase()
    k.ingest_batch("acme", load_directory(KB_DIR / "batch1_initial"), "initial")
    return k


def test_reupload_is_idempotent(kb):
    r = kb.ingest_batch("acme", load_directory(KB_DIR / "batch1_initial"), "again")
    assert r.count("unchanged") == 5 and r.count("new") == 0
    assert len(kb.documents(ACME, include_superseded=True)) == 5


def test_new_policy_version_supersedes_and_time_travel(kb):
    r = kb.ingest_batch("acme", load_directory(KB_DIR / "batch2_policy_update"), "2026 update")
    assert r.items[0].outcome == "updated" and r.items[0].supersedes_version == 1
    vs = kb.versions(ACME, "policies/data_access_policy.md")
    assert [(v.version, v.status) for v in vs] == [(1, "superseded"), (2, "current")]
    assert vs[0].valid_to == vs[1].valid_from == "2026-01-01"
    now = kb.search(ACME, "trade record retention years", kinds=["policy"])
    assert "7 years" in now[0].text and now[0].version == 2
    then = kb.search(ACME, "trade record retention years", kinds=["policy"], as_of=date(2025, 6, 1))
    assert "5 years" in then[0].text and then[0].version == 1
    assert all(h.version == 2 for h in kb.search(ACME, "retention", kinds=["policy"]))   # current only by default


def test_diff_between_versions(kb):
    kb.ingest_batch("acme", load_directory(KB_DIR / "batch2_policy_update"))
    ch = {c.heading_path.split(" > ")[-1]: c for c in kb.diff_versions(ACME, "policies/data_access_policy.md", 1, 2)}
    assert ch["Retention"].change == "modified" and "7 years" in ch["Retention"].diff
    assert ch["Data residency"].change == "added"


def test_failures_are_isolated_in_a_batch():
    kb = KnowledgeBase()
    r = kb.ingest_batch("acme", [
        item("ok.md", "# Doc\n\nSome useful business text about trading."),
        SourceItem(source_key="bad.json", filename="bad.json", content=b"{not json"),
        SourceItem(source_key="x.exe", filename="x.exe", content=b"MZ"),
        SourceItem(source_key="bad.pdf", filename="bad.pdf", content=b"%PDF-broken"),
        item("empty.md", "   "),
    ])
    out = {i.source_key: i.outcome for i in r.items}
    assert out == {"ok.md": "new", "bad.json": "failed", "x.exe": "failed", "bad.pdf": "failed", "empty.md": "failed"}
    assert len(kb.documents(ACME)) == 1          # failed items left no partial state


def test_tenant_isolation_and_roles():
    kb = KnowledgeBase()
    kb.ingest_batch("acme", [item("a.md", "# Secret\n\nAcme merger codename Bluebird.")])
    kb.ingest_batch("globex", [item("g.md", "# Other\n\nGlobex lunch menu.")])
    assert not kb.search(Principal(tenant_id="globex"), "Bluebird merger")
    assert kb.search(ACME, "Bluebird")
    kb.ingest_batch("acme", [item("hr.md", "# HR\n\nSalary bands are confidential.", allowed_roles=["hr"])])
    assert not kb.search(ACME, "salary bands")
    assert kb.search(Principal(tenant_id="acme", roles=["hr"]), "salary bands")
    chunk_id = kb.search(Principal(tenant_id="acme", roles=["hr"]), "salary bands")[0].chunk_id
    assert kb.get_chunk(ACME, chunk_id) is None                      # no direct fetch around the filter
    assert kb.get_chunk(Principal(tenant_id="globex", roles=["hr"]), chunk_id) is None


def test_catalogue_extraction(kb):
    ds = kb.get_dataset(ACME, "dwh.trades")
    side = next(f for f in ds.fields if f.name == "side")
    assert "BUY or SELL" in side.description
    assert next(f for f in ds.fields if f.name == "quantity").unit == "units"
    api = kb.search_apis(ACME, ["trades"])[0]
    assert (api.method, api.path) == ("GET", "/v1/trades")
    assert {p.name for p in api.params} >= {"from_date", "to_date"}
    resp = {f.name: f for f in api.response_fields}
    assert "items[].side" in resp and "BUY" in resp["items[].side"].type and "next_cursor" in resp
    assert kb.find_terms(ACME, ["authorized partner"])[0].term == "Authorised Participant"
    assert kb.find_terms(ACME, ["net buying"])[0].term == "Net Flow"


def test_superseded_catalogue_entries_disappear():
    kb = KnowledgeBase()
    v1 = "# Glossary\n\n- **Widget**: An old definition of a widget thing.\n"
    v2 = "# Glossary\n\n- **Widget**: A new and better definition of widget.\n"
    kb.ingest_batch("acme", [item("glossary.md", v1)], "v1")
    kb.ingest_batch("acme", [item("glossary.md", v2)], "v2")
    assert kb.find_terms(ACME, ["widget"])[0].definition.startswith("A new")
    assert kb.find_terms(ACME, ["widget"], as_of=date(2000, 1, 1)) == []


def test_html_confluence_sections(kb):
    hits = kb.search(ACME, "where is trade data stored warehouse table", kinds=["technical"])
    assert hits and "dwh.trades" in hits[0].text
    assert "Trade storage" in hits[0].heading_path
    assert "alert(" not in " ".join(h.text for h in hits)


def test_sql_ddl_and_comments():
    ddl = """CREATE TABLE risk.positions (
      position_id BIGINT NOT NULL, -- unique position
      market_value NUMERIC(18,4),
      desk VARCHAR(20) COMMENT 'Trading desk code',
      PRIMARY KEY (position_id));
    COMMENT ON COLUMN risk.positions.market_value IS 'Mark-to-market value in USD';
    COMMENT ON TABLE risk.positions IS 'Open positions';"""
    kb = KnowledgeBase()
    kb.ingest_batch("acme", [item("risk.sql", ddl)])
    d = kb.get_dataset(ACME, "risk.positions")
    f = {x.name: x for x in d.fields}
    assert d.description == "Open positions"
    assert f["position_id"].description == "unique position" and f["position_id"].required
    assert f["desk"].description == "Trading desk code"
    assert f["market_value"].description == "Mark-to-market value in USD" and f["market_value"].type == "NUMERIC(18,4)"
    assert "PRIMARY" not in f


def test_docx_headings():
    def para(text, style=None):
        ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
        return f"<w:p>{ppr}<w:r><w:t>{text}</w:t></w:r></w:p>"
    xml = ('<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'
           + para("Onboarding Guide", "Heading1") + para("Step one is to request access.")
           + para("Approvals", "Heading2") + para("Manager approval is required.") + "</w:body></w:document>")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", xml)
    kb = KnowledgeBase()
    r = kb.ingest_batch("acme", [SourceItem(source_key="guide.docx", filename="guide.docx", content=buf.getvalue())])
    assert r.items[0].outcome == "new"
    hit = kb.search(ACME, "manager approval")[0]
    assert hit.heading_path == "Onboarding Guide > Approvals" and hit.title == "Onboarding Guide"


def test_retire_removes_from_results(kb):
    assert kb.search(ACME, "retained regulatory inspection", kinds=["policy"])
    assert kb.retire(ACME, "policies/data_access_policy.md", as_of=date(2026, 3, 1)) == 1
    assert not kb.search(ACME, "retained regulatory inspection", kinds=["policy"])
    assert kb.search(ACME, "retained regulatory inspection", kinds=["policy"], as_of=date(2025, 6, 1))  # history preserved


def test_batches_are_recorded(kb):
    kb.ingest_batch("acme", load_directory(KB_DIR / "batch2_policy_update"), "2026 update")
    assert [b.label for b in kb.batches(ACME)] == ["initial", "2026 update"]
