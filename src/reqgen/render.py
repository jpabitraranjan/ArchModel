"""Human-readable rendering of a RequirementsDocument (Markdown)."""
from __future__ import annotations

from .schema import RequirementsDocument


def _c(text: object) -> str:
    return str(text if text is not None else "").replace("|", "\\|").replace("\n", " ")


def _table(headers: list[str], rows: list[list[object]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(_c(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def _src(sources: list[str]) -> str:
    return ", ".join(f"`{s}`" for s in sources) if sources else "—"


def to_markdown(doc: RequirementsDocument) -> str:
    L: list[str] = [f"# {doc.title}", f"*Requirements document v{doc.version}*", "", doc.summary, ""]

    def section(title: str, body: list[str]) -> None:
        if body:
            L.extend([f"## {title}", *body, ""])

    section("Goals", [f"- {g}" for g in doc.goals])
    section("Stakeholders", [_table(["Name", "Role", "Interest"], [[s.name, s.role, s.interest] for s in doc.stakeholders])]
            if doc.stakeholders else [])
    section("Scope", ["**In scope**", *[f"- {x}" for x in doc.scope_in], "", "**Out of scope**", *[f"- {x}" for x in doc.scope_out]]
            if doc.scope_in or doc.scope_out else [])
    section("Glossary", [_table(["Term", "Definition", "Basis", "Sources"],
                                [[g.term, g.definition, g.basis, _src(g.sources)] for g in doc.glossary])] if doc.glossary else [])
    fr: list[str] = []
    for r in doc.functional:
        fr += [f"### {r.id} — {r.title}", f"*Actor:* {r.actor} · *Priority:* {r.priority} · *Basis:* {r.basis} · *Sources:* {_src(r.sources)}",
               "", r.description, ""]
        fr += [f"- **Given** {a.given} **when** {a.when} **then** {a.then}" for a in r.acceptance_criteria] + [""]
    section("Functional requirements", fr)
    section("Non-functional requirements", [_table(["ID", "Category", "Statement", "Target", "Basis", "Sources"],
             [[r.id, r.category, r.statement, r.target or "—", r.basis, _src(r.sources)] for r in doc.non_functional])]
            if doc.non_functional else [])
    section("Constraints from policy and regulation", [_table(["ID", "Constraint", "Impact on design", "Sources"],
             [[c.id, c.statement, c.impact, _src(c.sources)] for c in doc.constraints])] if doc.constraints else [])
    section("Data sources", [_table(["Name", "Ref", "Status", "Purpose", "Notes"],
             [[d.name, d.ref or "—", d.status, d.purpose, d.notes] for d in doc.data_sources])] if doc.data_sources else [])
    api: list[str] = []
    for a in doc.apis:
        api += [f"### {a.id} — {a.name}", f"`{a.method} {a.path}` · *{a.status}*" + (f" (`{a.existing_ref}`)" if a.existing_ref else "")
                + f" · *delivery:* {a.delivery}" + (f" · *updates:* {a.update_frequency}" if a.update_frequency else ""),
                f"*Satisfies:* {', '.join(a.satisfies) or '—'} · *Sources:* {_src(a.sources)}", "", a.purpose, ""]
        if a.request_params:
            api += ["**Request parameters**", "", _table(["Name", "Type", "Required", "Description", "Example"],
                    [[p.name, p.type, "yes" if p.required else "no", p.description, p.example or ""] for p in a.request_params]), ""]
        if a.response_attributes:
            api += ["**Response attributes (data dictionary)**", "",
                    _table(["Attribute", "Type", "Meaning", "Unit", "Nullable", "Origin", "Example"],
                           [[x.name, x.type, x.meaning, x.unit or "", "yes" if x.nullable else "no",
                             f"`{x.source_field}`" if x.source_field else (f"derived: {x.derivation}" if x.derivation else "—"),
                             x.example or ""] for x in a.response_attributes]), ""]
        meta = [f"*Pagination:* {a.pagination}" if a.pagination else "", f"*Auth:* {a.auth}" if a.auth else ""]
        api += [" · ".join(m for m in meta if m)] if any(meta) else []
        api += [f"*Errors:* {'; '.join(a.errors)}"] if a.errors else []
        api.append("")
    section("API specifications", api)
    section("Assumptions", [_table(["ID", "Assumption", "Reason", "Needs confirmation"],
             [[a.id, a.statement, a.reason, "yes" if a.needs_confirmation else "no"] for a in doc.assumptions])] if doc.assumptions else [])
    section("Open and resolved questions", [_table(["ID", "Question", "Blocking", "Answer", "Answered by"],
             [[q.id, q.question, "yes" if q.blocking else "no", q.answer or "**OPEN**", q.answered_by or ""]
              for q in doc.open_questions])] if doc.open_questions else [])
    section("Traceability", [_table(["Requirement", "Title", "Served by API"], [
        [r.id, r.title, ", ".join(a.id for a in doc.apis if r.id in a.satisfies) or "—"] for r in doc.functional])]
        if doc.functional else [])
    return "\n".join(L).rstrip() + "\n"
