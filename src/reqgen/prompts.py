"""Prompts. Kept in one place so they can be reviewed and versioned like code."""
import json

from .schema import RequirementsDocument

SYSTEM = (
    "You are a senior business analyst and solution architect working for the user's organisation. "
    "You turn vague requests into precise, verifiable requirements using ONLY the organisation knowledge supplied. "
    "You never invent datasets, tables, columns, APIs, policies or definitions. If something is not in the supplied "
    "knowledge, say so: mark it as an assumption or raise a question. Reply with a single JSON object and nothing else."
)

UNDERSTAND = """Request from the business:
\"\"\"{request}\"\"\"

Analyse the request. Return JSON:
{{
  "intent": "one sentence",
  "users": ["personas who will use the result"],
  "domain_terms": ["business/financial terms or concepts that must be defined precisely (include synonyms the user used)"],
  "data_needs": ["each distinct piece of data the request needs, as a short noun phrase"],
  "search_queries": ["8-12 short queries to find relevant policies, documents, existing systems, APIs and datasets in the organisation's knowledge base"]
}}"""

CLARIFY = """Request:
\"\"\"{request}\"\"\"

Understanding: {understanding}

Organisation knowledge found:
{grounding}

Identify what is still ambiguous or missing. For each point decide:
- If the knowledge above answers it, set "resolved_by_knowledge" with the answer and the exact refs that support it.
- Otherwise it must be asked to the user. Mark "blocking": true only if requirements cannot be written responsibly without it.
Ask about: scope and meaning of vague terms, which data source to treat as authoritative, time ranges and granularity,
freshness (live vs historical), users/permissions, and conflicts between the request and policies.
Do not ask questions whose answers are already in the knowledge. Return JSON:
{{"questions": [{{"question": "...", "why_it_matters": "...", "options": ["..."], "blocking": false,
  "resolved_by_knowledge": null | {{"answer": "...", "sources": ["chunk:12", "dataset:..."]}}}}]}}"""

GENERATE = """Request:
\"\"\"{request}\"\"\"

Answers to clarification questions:
{answers}

Organisation knowledge (the ONLY things you may cite, using the refs in square brackets):
{grounding}

Write the requirements document as JSON matching this JSON Schema exactly:
{schema}

Rules:
1. Cite sources only with refs that appear above (chunk:N, term:X, dataset:T, dataset:T.col, api:METHOD /path). Never invent a ref.
   basis="knowledge" requires at least one source; use "user_stated" for what the request/answers say; "assumption" otherwise
   (and also list it in "assumptions").
2. Reuse existing assets: if an existing API or dataset covers a need, reference it (status existing/extend, existing_ref) instead of
   proposing a duplicate. For new APIs say which datasets feed them. Do not invent tables or columns: if data is not in the catalogue,
   put it in data_sources with status "new" or "unknown" and raise it as an assumption.
3. APIs: one API per distinct concern (e.g. purchases, sales, quantities/net flow, live updates, historical records, classification),
   separating real-time delivery ("streaming") from historical queries ("request_response"). Every API lists "satisfies" (FR-/NFR- ids),
   request parameters, and EVERY response attribute with: type, a plain-language "meaning" (take it from the catalogue description or glossary when
   it exists), unit, nullable, and either "source_field" (dataset:T.col) or a "derivation" (the exact formula, e.g. SUM(quantity WHERE side='BUY') - SUM(...)).
4. Apply every relevant policy/regulation chunk as a constraint (CON-xxx) and as NFRs (security, retention, auditability, residency), citing it.
   Use the current wording; the knowledge supplied is already limited to what is valid now.
5. Functional requirements each need at least one Given/When/Then acceptance criterion with measurable outcomes.
6. Anything unresolved goes in "open_questions" (and "assumptions" if you proceeded anyway). Never silently guess a business meaning.
7. Use sequential ids: FR-001.., NFR-001.., API-001.., CON-001.., ASM-001.., Q-001..
Return only the JSON object."""

REPAIR = """Your requirements document has problems that must be fixed. Return the COMPLETE corrected JSON document (not a diff).

Problems:
{problems}

Previous document:
{document}

Remember: cite only refs from the knowledge listed below; do not invent datasets/columns/APIs; every API response attribute needs a meaning.

Organisation knowledge:
{grounding}"""


def schema_json() -> str:
    return json.dumps(RequirementsDocument.model_json_schema(), separators=(",", ":"))
