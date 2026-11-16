# Stage 1 — Design Engineering System (foundations)

Three packages, built so each can be used and tested on its own.

| Package | Role |
|---|---|
| `archmodel` | Structured architecture model (HLD/LLD as typed nodes and edges), component catalogue, validated operations, lint, diff, Mermaid. Source of truth for the visual editor. |
| `orgknow` | Organisation knowledge: batch ingestion, versioning, hybrid search, structured catalogue (APIs, datasets/columns, glossary). **Understands the organisation only; generates nothing.** |
| `reqgen` | Natural-language request -> grounded, validated requirements document with API specs and a data dictionary. |

## orgknow — how it handles real-world uploads
- **Batches:** `ingest_batch(tenant, items)`; one bad file never fails the batch; every item gets an outcome (new / updated / unchanged / failed + reason).
- **Idempotent:** re-uploading a folder skips unchanged files (content hash).
- **Versions:** the same `source_key` with new content becomes v2; v1 is closed with a `valid_to`. Policies that changed from year A to year B are both kept.
  Default search sees only current versions; `as_of=date` shows what was true then; `diff_versions` lists changes by section.
- **Formats:** md, txt, html (Confluence exports), docx, pdf (needs `pypdf`), OpenAPI json/yaml (yaml needs `pyyaml`), csv data dictionaries/glossaries, SQL DDL.
- **Two retrieval paths:** hybrid text search (BM25 + embeddings, RRF) for policies and documents; a structured catalogue for "where does X live" and "what does Y mean".
- **Isolation:** every query is scoped by tenant and filtered by role; no read path skips this.
- **Embeddings:** the default `HashingEmbedder` is dependency-free and NOT semantic. Implement the `Embedder` protocol with a real model for production.

## reqgen — pipeline
1. **Understand** the request (LLM): intent, users, domain terms, data needs, search queries.
2. **Ground**: structured lookups (glossary, datasets, APIs) + hybrid search + a dedicated policy/regulation pass. Everything is stored as verifiable refs.
3. **Clarify**: questions the knowledge can answer are closed (only if every cited ref really exists); the rest go to the human. Blocking questions stop generation.
4. **Generate** the document (LLM) against a strict schema.
5. **Validate** deterministically (invented citations/datasets/columns/APIs, attributes without meaning, APIs without requirements, missing acceptance criteria, ...) and **repair** (LLM) up to N rounds. Remaining errors are reported, never hidden.

## Quick start
    pip install -e ".[dev]" && pytest
    orgknow ingest --db kb.db --tenant acme --dir examples/finance_kb/batch1_initial --label initial
    orgknow ingest --db kb.db --tenant acme --dir examples/finance_kb/batch2_policy_update --label "2026 policy update"
    orgknow diff   --db kb.db --tenant acme policies/data_access_policy.md 1 2
    export ANTHROPIC_API_KEY=...        # or REQGEN_PROVIDER=openai + OPENAI_API_KEY / REQGEN_BASE_URL (Ollama etc.)
    reqgen start    --db kb.db --tenant acme --session s.json "Build a dashboard for portfolio managers ..."
    reqgen answer   --session s.json Q-002 "Authorised Participant flow only"
    reqgen generate --session s.json --out requirements.json --md requirements.md

`examples/finance_kb/sample_requirements.md` shows the output format (produced with a scripted stand-in model).
