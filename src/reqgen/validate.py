"""Deterministic validation of a generated document. This is the trust layer: the LLM proposes,
these checks decide what is acceptable. Errors trigger a repair round; remaining ones are shown
to the reviewer and block approval."""
from __future__ import annotations

from archmodel.errors import Issue

from .grounding import GroundingBundle
from .schema import RequirementsDocument

_MIN_MEANING = 8


def validate_document(doc: RequirementsDocument, bundle: GroundingBundle, require_closed: bool = False) -> list[Issue]:
    issues: list[Issue] = []
    err = lambda code, msg, ref=None: issues.append(Issue(code=code, message=msg, ref=ref))
    warn = lambda code, msg, ref=None: issues.append(Issue(code=code, message=msg, severity="warning", ref=ref))

    # ---- ids unique
    all_ids = doc.requirement_ids() + [a.id for a in doc.assumptions] + [q.id for q in doc.open_questions]
    for i in {x for x in all_ids if all_ids.count(x) > 1}:
        err("duplicate_id", f"id {i} is used more than once", i)
    fr_nfr = {r.id for r in doc.functional} | {r.id for r in doc.non_functional}

    # ---- citations must exist in the grounding bundle
    def check_sources(owner: str, sources: list[str], basis: str | None) -> None:
        for s in sources:
            if not bundle.has_ref(s):
                err("unknown_source_ref", f"{owner} cites '{s}', which is not in the retrieved organisation knowledge", owner)
        if basis == "knowledge" and not sources:
            err("ungrounded_knowledge_claim", f"{owner} claims basis=knowledge but cites no source", owner)

    for r in doc.functional:
        check_sources(r.id, r.sources, r.basis)
        if not r.acceptance_criteria:
            err("no_acceptance_criteria", f"{r.id} has no acceptance criteria", r.id)
        for ac in r.acceptance_criteria:
            if not (ac.given.strip() and ac.when.strip() and ac.then.strip()):
                err("incomplete_acceptance_criterion", f"{r.id} has an empty Given/When/Then part", r.id)
    for r in doc.non_functional:
        check_sources(r.id, r.sources, r.basis)
        if r.category in ("performance", "availability", "scalability") and not r.target:
            warn("unmeasurable_nfr", f"{r.id} ({r.category}) has no measurable target", r.id)
    for g in doc.glossary:
        check_sources(f"glossary '{g.term}'", g.sources, g.basis)
    for c in doc.constraints:
        check_sources(c.id, c.sources, "knowledge")

    # ---- data sources must exist when claimed existing
    for d in doc.data_sources:
        if d.status == "existing" and not (d.ref and bundle.has_ref(d.ref)):
            err("unknown_dataset", f"data source '{d.name}' is marked existing but '{d.ref}' is not in the data catalogue", d.ref)

    # ---- API specs
    seen: dict[tuple[str, str], str] = {}
    glossary_terms = {g.term.lower() for g in doc.glossary}
    covered: set[str] = set()
    for a in doc.apis:
        check_sources(a.id, a.sources, a.basis)
        key = (a.method, a.path)
        if key in seen:
            err("duplicate_endpoint", f"{a.id} and {seen[key]} both define {a.method} {a.path}", a.id)
        seen[key] = a.id
        if not a.path.startswith("/"):
            err("bad_path", f"{a.id} path '{a.path}' must start with '/'", a.id)
        if a.status in ("existing", "extend") and not (a.existing_ref and bundle.has_ref(a.existing_ref)):
            err("unknown_existing_api", f"{a.id} is '{a.status}' but '{a.existing_ref}' is not a known existing API", a.id)
        if a.status == "new" and a.existing_ref:
            warn("new_api_with_existing_ref", f"{a.id} is 'new' but names existing_ref {a.existing_ref}", a.id)
        if not a.satisfies:
            err("api_without_requirement", f"{a.id} does not satisfy any requirement", a.id)
        for rid in a.satisfies:
            if rid not in fr_nfr:
                err("unknown_requirement_ref", f"{a.id} satisfies {rid}, which does not exist", a.id)
            covered.add(rid)
        if a.delivery == "streaming" and not a.update_frequency:
            err("streaming_without_frequency", f"{a.id} is streaming but states no update frequency", a.id)
        if not a.response_attributes:
            err("api_without_response", f"{a.id} describes no response attributes", a.id)
        names = [x.name for x in a.response_attributes]
        for n in {n for n in names if names.count(n) > 1}:
            err("duplicate_attribute", f"{a.id} lists attribute '{n}' twice", a.id)
        for attr in a.response_attributes:
            where = f"{a.id}.{attr.name}"
            if len(attr.meaning.strip()) < _MIN_MEANING:
                err("attribute_without_meaning", f"{where} has no meaningful business definition", where)
            if not attr.type.strip():
                err("attribute_without_type", f"{where} has no type", where)
            if attr.source_field and not bundle.has_ref(attr.source_field):
                err("unknown_source_field", f"{where} maps to '{attr.source_field}', which is not in the data catalogue", where)
            if not attr.source_field and not attr.derivation:
                warn("attribute_without_origin", f"{where} states neither a source column nor a derivation", where)
            if attr.glossary_term and attr.glossary_term.lower() not in glossary_terms:
                warn("glossary_term_missing", f"{where} refers to glossary term '{attr.glossary_term}' not in the document glossary", where)
        for p in a.request_params:
            if not p.description.strip():
                err("param_without_description", f"{a.id} parameter '{p.name}' has no description", a.id)

    # ---- coverage and policy application
    for r in doc.functional:
        if doc.apis and r.id not in covered:
            warn("requirement_not_served_by_api", f"{r.id} is not satisfied by any API (fine if it is UI-only)", r.id)
    if bundle.policy_chunks() and not doc.constraints:
        warn("policies_not_applied", "relevant policies were retrieved but the document has no constraints citing them")

    # ---- unresolved questions
    for q in doc.open_questions:
        if q.is_open and q.blocking:
            (err if require_closed else warn)("blocking_question_open", f"{q.id} is blocking and unanswered: {q.question}", q.id)
    return issues


def errors_only(issues: list[Issue]) -> list[Issue]:
    return [i for i in issues if i.severity == "error"]
