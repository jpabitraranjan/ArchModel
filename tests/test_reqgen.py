"""End-to-end test of the requirements flow on the finance example, with a scripted LLM.

The scripted model stands in for a real one: these tests prove the PIPELINE (grounding, question
handling, validation, repair loop, persistence), not the quality of any particular model's writing."""
import copy
import json
from datetime import date
from pathlib import Path

import pytest

from archmodel import ArchitectureModel, apply_operations, lint
from orgknow import KnowledgeBase, Principal
from orgknow.cli import load_directory
from reqgen import (BlockingQuestionsError, RequirementsDocument, RequirementsSession, ScriptedLLM, to_markdown,
                    validate_document)
from reqgen.llm import LLMError, complete_json

KB_DIR = Path(__file__).parent.parent / "examples" / "finance_kb"
ACME = Principal(tenant_id="acme")
REQUEST = ("We want a dashboard for portfolio managers to see the difference between what stocks authorised partners "
           "are buying and selling, with historical trends and industry knowledge.")

UNDERSTANDING = {
    "intent": "Net buying/selling analytics of authorised participants for portfolio managers",
    "users": ["Portfolio Manager"],
    "domain_terms": ["Authorised Participant", "net buying", "GICS Sector", "Portfolio Manager"],
    "data_needs": ["purchases and sales of securities by participants", "industry classification of securities",
                   "quantity traded", "historical trades", "live trade updates"],
    "search_queries": ["where are trades stored", "data retention policy trade records", "access to participant trading data"],
}


@pytest.fixture
def kb():
    k = KnowledgeBase()
    k.ingest_batch("acme", load_directory(KB_DIR / "batch1_initial"), "initial")
    k.ingest_batch("acme", load_directory(KB_DIR / "batch2_policy_update"), "2026 policy update")
    return k


def attr(name, typ, meaning, **kw):
    return {"name": name, "type": typ, "meaning": meaning, **kw}


def good_document(bundle) -> dict:
    chunk = lambda word: next(c for c in bundle.chunks if word in c.text).ref
    retention, access, audit = chunk("retained for"), chunk("Portfolio Manager, Trader"), chunk("must be logged")
    trades = "dataset:dwh.trades"
    ident = [attr("isin", "string", "ISIN of the security, the unique international identifier.", source_field=f"{trades}.isin"),
             attr("security_name", "string", "Official name of the security.", source_field="dataset:ref.security_master.security_name")]
    window = [{"name": "from_date", "type": "date", "required": True, "description": "Start of the period (inclusive)."},
              {"name": "to_date", "type": "date", "required": True, "description": "End of the period (inclusive)."}]

    def api(i, name, method, path, satisfies, attrs, **kw):
        return {"id": f"API-{i:03d}", "name": name, "purpose": f"{name} for portfolio managers.", "method": method, "path": path,
                "satisfies": satisfies, "request_params": window, "response_attributes": attrs, "basis": "knowledge",
                "sources": [trades, retention], **kw}

    qty = lambda n, m, d: attr(n, "number", m, unit="units", derivation=d, glossary_term="Net Flow")
    return {
        "title": "Participant Trading Flow Dashboard", "summary": "Shows what authorised participants buy and sell.",
        "goals": ["Let portfolio managers see net buying and selling by security and sector"],
        "stakeholders": [{"name": "Portfolio Manager", "role": "Primary user"}],
        "glossary": [{"term": "Net Flow", "definition": "Purchases minus sales over a period.", "basis": "knowledge", "sources": ["term:Net Flow"]}],
        "functional": [
            {"id": "FR-001", "title": "View net flow by security", "description": "PMs see purchases, sales and net flow per security.",
             "actor": "Portfolio Manager", "basis": "knowledge", "sources": ["term:Net Flow", trades],
             "acceptance_criteria": [{"given": "a date range", "when": "the PM opens the dashboard", "then": "net flow per ISIN is shown within 3 s"}]},
            {"id": "FR-002", "title": "View live trades", "description": "PMs see new participant trades as they happen.", "actor": "Portfolio Manager",
             "acceptance_criteria": [{"given": "the dashboard is open", "when": "a trade executes", "then": "it appears within 5 s"}]},
            {"id": "FR-003", "title": "Browse history", "description": "PMs browse older trades and trends.", "actor": "Portfolio Manager",
             "acceptance_criteria": [{"given": "a past period", "when": "selected", "then": "historical trades load"}]},
            {"id": "FR-004", "title": "Slice by industry", "description": "Group flow by GICS sector and industry.", "actor": "Portfolio Manager",
             "basis": "knowledge", "sources": ["term:GICS Sector", "dataset:ref.security_master"],
             "acceptance_criteria": [{"given": "net flow data", "when": "grouped by sector", "then": "each ISIN maps to one sector"}]}],
        "non_functional": [
            {"id": "NFR-001", "category": "security", "statement": "Only Portfolio Manager, Trader or Compliance roles may view trading data; AP data needs an extra Compliance-approved entitlement.", "basis": "knowledge", "sources": [access]},
            {"id": "NFR-002", "category": "retention", "statement": "Trade records are retained for 7 years.", "target": "7 years", "basis": "knowledge", "sources": [retention]},
            {"id": "NFR-003", "category": "auditability", "statement": "Access to Restricted data is logged with user, timestamp and query.", "basis": "knowledge", "sources": [audit]}],
        "constraints": [{"id": "CON-001", "statement": "Trade records must be retained for 7 years.", "impact": "Historical API must serve 7 years.", "sources": [retention]},
                        {"id": "CON-002", "statement": "AP trading data is Restricted.", "impact": "Entitlement check on every API.", "sources": [access]}],
        "data_sources": [{"name": "Trading Data Warehouse trades", "ref": trades, "status": "existing", "purpose": "Source of trades"},
                         {"name": "Security master", "ref": "dataset:ref.security_master", "status": "existing", "purpose": "Industry classification"}],
        "apis": [
            api(1, "Purchases by security", "GET", "/v1/analytics/purchases", ["FR-001", "NFR-001"],
                ident + [qty("purchased_quantity", "Total units bought by participants in the period.", "SUM(quantity) WHERE side = 'BUY' AND correction_of IS NULL")]),
            api(2, "Sales by security", "GET", "/v1/analytics/sales", ["FR-001"],
                ident + [qty("sold_quantity", "Total units sold by participants in the period.", "SUM(quantity) WHERE side = 'SELL' AND correction_of IS NULL")]),
            api(3, "Net flow by security", "GET", "/v1/analytics/net-flow", ["FR-001"],
                ident + [qty("net_quantity", "Purchases minus sales; positive means net buying.", "purchased_quantity - sold_quantity")]),
            api(4, "Live trade stream", "GET", "/v1/stream/trades", ["FR-002"],
                [attr("trade_id", "string", "Unique trade identifier.", source_field=f"{trades}.trade_id"),
                 attr("side", "string", "BUY or SELL from the participant's perspective.", source_field=f"{trades}.side")],
                delivery="streaming", update_frequency="within 2 seconds of execution"),
            api(5, "Historical trades", "GET", "/v1/trades", ["FR-003"],
                [attr("items[].trade_date", "date", "Execution date.", source_field=f"{trades}.trade_date")],
                status="extend", existing_ref="api:GET /v1/trades"),
            api(6, "Industry classification", "GET", "/v1/securities/classification", ["FR-004"],
                [attr("gics_sector", "string", "GICS sector of the issuer.", source_field="dataset:ref.security_master.gics_sector"),
                 attr("gics_industry", "string", "GICS industry, more granular than sector.", source_field="dataset:ref.security_master.gics_industry")])],
        "assumptions": [{"id": "ASM-001", "statement": "Corrections are excluded from aggregates.", "reason": "Per platform page.", "needs_confirmation": True}],
    }


CLARIFY = {"questions": [
    {"question": "What does 'authorised partner' mean here?", "why_it_matters": "Defines the population", "blocking": False,
     "resolved_by_knowledge": {"answer": "Authorised Participants: licensed institutions that create/redeem ETF units.", "sources": ["term:Authorised Participant"]}},
    {"question": "Should house and client trades be included, or only Authorised Participant flow?", "why_it_matters": "Changes every aggregate",
     "options": ["AP only", "AP + house", "All"], "blocking": True, "resolved_by_knowledge": None},
    {"question": "What does the live feed need to cover?", "blocking": False,
     "resolved_by_knowledge": {"answer": "made up", "sources": ["chunk:99999"]}}]}


def make_session(kb, responses, **kw):
    llm = ScriptedLLM(responses)
    return RequirementsSession.start(kb, ACME, llm, REQUEST, **kw), llm


def test_full_flow_with_repair_loop(kb):
    holder = {}
    def bad_generation(system, user):
        d = good_document(holder["s"].state.bundle)
        d["functional"][0]["sources"].append("chunk:99999")                  # invented citation
        d["apis"][0]["response_attributes"][2]["meaning"] = ""             # attribute with no definition
        d["apis"][1]["response_attributes"][0]["source_field"] = "dataset:dwh.trades.ghost_column"
        d["apis"][3]["update_frequency"] = None                              # live API without frequency
        d["apis"][4]["existing_ref"] = "api:GET /v1/imaginary"
        d["apis"][5]["satisfies"] = []
        return json.dumps(d)
    s, llm = make_session(kb, [json.dumps(UNDERSTANDING), json.dumps(CLARIFY), bad_generation,
                               lambda sys_, user: json.dumps(good_document(holder["s"].state.bundle))])
    holder["s"] = s

    open_qs = s.analyze()
    b = s.state.bundle
    # grounding used structured + text retrieval, and only the CURRENT policy version
    assert {t.term for t in b.terms} >= {"Authorised Participant", "Net Flow", "GICS Sector"}
    assert {d.name for d in b.datasets} >= {"dwh.trades", "ref.security_master"} and b.apis[0].path == "/v1/trades"
    assert any("7 years" in c.text for c in b.chunks) and not any("5 years" in c.text for c in b.chunks)
    # the question answerable from knowledge was closed; the invented citation was NOT accepted; the real ambiguity stays open
    by_id = {q.id: q for q in s.state.questions}
    assert by_id["Q-001"].answered_by == "knowledge" and by_id["Q-001"].sources == ["term:Authorised Participant"]
    assert by_id["Q-003"].is_open
    assert [q.id for q in s.blocking_questions()] == ["Q-002"]

    with pytest.raises(BlockingQuestionsError):
        s.generate()
    s.answer("Q-002", "Authorised Participant flow only")
    res = s.generate()

    assert res.repair_rounds == 1 and res.ok and res.approvable
    repair_prompt = llm.calls[3][1]
    for code in ("unknown_source_ref", "attribute_without_meaning", "unknown_source_field", "streaming_without_frequency",
                 "unknown_existing_api", "api_without_requirement"):
        assert code in repair_prompt, code
    doc = res.document
    assert [a.id for a in doc.apis] == [f"API-00{i}" for i in range(1, 7)]
    assert {q.id: q.answered_by for q in doc.open_questions}["Q-002"] == "user"
    md = to_markdown(doc)
    assert "Response attributes (data dictionary)" in md and "purchased_quantity" in md and "SUM(quantity)" in md
    assert "Traceability" in md and "**OPEN**" in md                     # Q-003 still open, visibly

    # the requirement ids link straight into the architecture model
    ids = doc.requirement_ids()
    m = apply_operations(ArchitectureModel(), [
        {"op": "add_node", "type": "api_service", "name": "Analytics API", "ref": "svc"},
        {"op": "link_requirement", "target_id": "@svc", "requirement_id": "API-003"}]).model
    assert "uncovered_requirement" in {i.code for i in lint(m, requirement_ids=ids)}


def test_state_survives_pause_and_resume(kb):
    s, _ = make_session(kb, [json.dumps(UNDERSTANDING), json.dumps(CLARIFY)])
    s.analyze()
    saved = s.to_json()
    llm2 = ScriptedLLM([])
    s2 = RequirementsSession.resume(llm2, saved)
    s2.answer("Q-002", "AP only")
    assert s2.blocking_questions() == [] and len(s2.state.bundle.chunks) == len(s.state.bundle.chunks)
    with pytest.raises(RuntimeError):
        s2.analyze()                                                       # resumed sessions have no KB handle
    with pytest.raises(KeyError):
        s2.answer("Q-999", "x")


def test_force_generate_keeps_blocking_question_visible(kb):
    holder = {}
    s, _ = make_session(kb, [json.dumps(UNDERSTANDING), json.dumps(CLARIFY),
                             lambda a, b: json.dumps(good_document(holder["s"].state.bundle))])
    holder["s"] = s
    s.analyze()
    res = s.generate(force=True)
    assert res.ok and not res.approvable
    assert any(i.code == "blocking_question_open" and i.severity == "warning" for i in res.issues)


def test_unrepaired_errors_are_reported_not_hidden(kb):
    holder = {}
    def always_bad(a, b):
        d = good_document(holder["s"].state.bundle)
        d["functional"][0]["sources"] = ["chunk:424242"]
        return json.dumps(d)
    s, _ = make_session(kb, [json.dumps(UNDERSTANDING), json.dumps({"questions": []}), always_bad, always_bad, always_bad], max_repairs=2)
    holder["s"] = s
    s.analyze()
    res = s.generate()
    assert not res.ok and not res.approvable and res.repair_rounds == 2
    assert "unknown_source_ref" in {i.code for i in res.issues}


def test_schema_errors_trigger_repair(kb):
    holder = {}
    def broken(a, b):
        d = good_document(holder["s"].state.bundle)
        d["functional"][0]["id"] = "REQ-1"
        return json.dumps(d)
    s, llm = make_session(kb, [json.dumps(UNDERSTANDING), json.dumps({"questions": []}), broken,
                               lambda a, b: json.dumps(good_document(holder["s"].state.bundle))])
    holder["s"] = s
    s.analyze()
    res = s.generate()
    assert res.ok and res.repair_rounds == 1 and "schema_error" in llm.calls[3][1]


def test_as_of_session_sees_old_policy(kb):
    s, _ = make_session(kb, [json.dumps(UNDERSTANDING), json.dumps({"questions": []})], as_of=date(2025, 6, 1))
    s.analyze()
    assert any("5 years" in c.text for c in s.state.bundle.chunks) and not any("7 years" in c.text for c in s.state.bundle.chunks)


def test_grounding_never_crosses_tenants(kb):
    kb.ingest_batch("globex", load_directory(KB_DIR / "batch2_policy_update"))
    s = RequirementsSession.start(kb, Principal(tenant_id="initech"), ScriptedLLM([json.dumps(UNDERSTANDING), json.dumps({"questions": []})]), REQUEST)
    s.analyze()
    b = s.state.bundle
    assert not (b.chunks or b.terms or b.datasets or b.apis)


# ------------------------------------------------------------------ validator in isolation
@pytest.fixture
def bundle_and_doc(kb):
    s, _ = make_session(kb, [json.dumps(UNDERSTANDING), json.dumps({"questions": []})])
    s.analyze()
    return s.state.bundle, good_document(s.state.bundle)


def codes(doc_dict, bundle, **kw):
    return {i.code: i.severity for i in validate_document(RequirementsDocument.model_validate(doc_dict), bundle, **kw)}


def test_good_document_has_no_errors(bundle_and_doc):
    bundle, d = bundle_and_doc
    assert "error" not in codes(d, bundle).values()


@pytest.mark.parametrize("mutate,expected", [
    (lambda d: d["functional"][1].update(acceptance_criteria=[]), "no_acceptance_criteria"),
    (lambda d: d["apis"][1].update(path=d["apis"][0]["path"]), "duplicate_endpoint"),
    (lambda d: d["apis"][0].update(path="v1/no-slash"), "bad_path"),
    (lambda d: d["apis"][0]["satisfies"].append("FR-099"), "unknown_requirement_ref"),
    (lambda d: d["apis"][0].update(response_attributes=[]), "api_without_response"),
    (lambda d: d["data_sources"][0].update(ref="dataset:made.up"), "unknown_dataset"),
    (lambda d: d["functional"][0].update(sources=[], basis="knowledge"), "ungrounded_knowledge_claim"),
    (lambda d: d["apis"][0]["response_attributes"][0].update(meaning="id"), "attribute_without_meaning"),
    (lambda d: d["functional"][1].update(id="FR-001"), "duplicate_id"),
    (lambda d: d["apis"][0]["request_params"][0].update(description=" "), "param_without_description"),
])
def test_validator_catches(bundle_and_doc, mutate, expected):
    bundle, d = bundle_and_doc
    d = copy.deepcopy(d)
    mutate(d)
    assert codes(d, bundle).get(expected) == "error"


def test_validator_warnings(bundle_and_doc):
    bundle, d = bundle_and_doc
    d = copy.deepcopy(d)
    d["apis"][0]["response_attributes"].append(attr("mystery", "string", "A value of unknown origin here."))
    d["constraints"] = []
    d["non_functional"].append({"id": "NFR-004", "category": "performance", "statement": "Fast."})
    c = codes(d, bundle)
    assert c["attribute_without_origin"] == c["policies_not_applied"] == c["unmeasurable_nfr"] == "warning"


def test_complete_json_tolerates_fences_and_retries_once():
    llm = ScriptedLLM(["not json at all", 'Sure!\n```json\n{"a": 1}\n```'])
    assert complete_json(llm, "s", "u") == {"a": 1} and "not valid JSON" in llm.calls[1][1]
    with pytest.raises(LLMError):
        complete_json(ScriptedLLM(["x", "y"]), "s", "u")
