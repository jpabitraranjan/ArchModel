"""Grounding: gather the organisation's knowledge that is relevant to a request.

Everything the generator may cite is collected here, each item with a verifiable reference:
    chunk:<id>             a passage of a document (policy, Confluence page, ...)
    term:<Term>            a glossary definition
    dataset:<name>         a table/dataset in the catalogue
    dataset:<name>.<col>   one column of it
    api:<METHOD> <path>    an existing API endpoint
The validator rejects any citation that is not in this bundle, which is what stops invented sources.
"""
from __future__ import annotations

from datetime import date

from orgknow import ApiRecord, DatasetRecord, KnowledgeBase, Principal, SearchHit, TermRecord
from pydantic import BaseModel, Field, PrivateAttr


class GroundingBundle(BaseModel):
    chunks: list[SearchHit] = Field(default_factory=list)
    terms: list[TermRecord] = Field(default_factory=list)
    datasets: list[DatasetRecord] = Field(default_factory=list)
    apis: list[ApiRecord] = Field(default_factory=list)
    as_of: str | None = None
    _refs: set[str] | None = PrivateAttr(default=None)

    def valid_refs(self) -> set[str]:
        refs = {c.ref for c in self.chunks} | {t.ref for t in self.terms} | {d.ref for d in self.datasets}
        refs |= {a.ref for a in self.apis}
        for d in self.datasets:
            refs |= {f"{d.ref}.{f.name}" for f in d.fields}
        return {r.lower() for r in refs}

    def has_ref(self, ref: str) -> bool:
        return ref.lower() in self._cache()

    def _cache(self) -> set[str]:
        if self._refs is None:
            self._refs = self.valid_refs()
        return self._refs

    def policy_chunks(self) -> list[SearchHit]:
        return [c for c in self.chunks if c.kind in ("policy", "industry")]

    def to_prompt(self, max_chars: int = 28000) -> str:
        parts: list[str] = []
        if self.terms:
            parts.append("## Glossary (structured)")
            parts += [f"- [{t.ref}] {t.definition}" + (f" (synonyms: {', '.join(t.synonyms)})" if t.synonyms else "")
                      for t in self.terms]
        if self.datasets:
            parts.append("\n## Existing datasets (structured catalogue)")
            for d in self.datasets:
                parts.append(f"- [{d.ref}] system={d.system or 'n/a'} {d.description}")
                for f in d.fields:
                    unit = f" unit={f.unit}" if f.unit else ""
                    req = " NOT NULL" if f.required else ""
                    parts.append(f"    - [{d.ref}.{f.name}] {f.type}{req}{unit}: {f.description or '(no description recorded)'}")
        if self.apis:
            parts.append("\n## Existing APIs (structured catalogue)")
            for a in self.apis:
                parts.append(f"- [{a.ref}] {a.summary} {a.description}")
                parts += [f"    param {p.name} ({p.location}, {p.type}{', required' if p.required else ''}): {p.description}"
                          for p in a.params]
                parts += [f"    returns {f.name} ({f.type}): {f.description}" for f in a.response_fields]
        text = "\n".join(parts)
        if self.chunks:
            text += "\n\n## Documents (cite as the [chunk:N] shown)\n"
            budget = max_chars - len(text)
            for c in self.chunks:
                entry = (f"[{c.ref}] {c.title} (v{c.version}, {c.kind}) / {c.heading_path}\n{c.text}\n\n")
                if len(entry) > budget:
                    break
                text += entry
                budget -= len(entry)
        return text


def build_grounding(kb: KnowledgeBase, who: Principal, terms: list[str], data_needs: list[str], queries: list[str],
                    as_of: date | None = None, max_chunks: int = 18) -> GroundingBundle:
    """Structured lookups first (terms, datasets, APIs), then hybrid text search, plus a dedicated
    policy/regulation pass so applicable constraints are never crowded out by other documents."""
    keywords = [*terms, *data_needs]
    bundle = GroundingBundle(
        terms=kb.find_terms(who, terms + data_needs, as_of),
        datasets=kb.search_datasets(who, keywords, as_of),
        apis=kb.search_apis(who, keywords, as_of),
        as_of=as_of.isoformat() if as_of else None)
    seen: dict[int, SearchHit] = {}
    for q in dict.fromkeys([*queries, *terms, *data_needs]):
        for h in kb.search(who, q, limit=5, as_of=as_of):
            seen.setdefault(h.chunk_id, h)
    policy_query = " ".join([*terms, *data_needs]) + " access retention audit compliance policy confidential"
    for h in kb.search(who, policy_query, limit=6, kinds=["policy", "industry"], as_of=as_of):
        seen.setdefault(h.chunk_id, h)
    policy_first = sorted(seen.values(), key=lambda h: (h.kind not in ("policy", "industry"), -h.score))
    bundle.chunks = policy_first[:max_chunks]
    return bundle
