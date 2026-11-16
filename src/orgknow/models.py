"""Data types for the organisational knowledge layer."""
from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field

DocKind = Literal["policy", "business", "technical", "api_spec", "data_dictionary",
                  "glossary", "industry", "historical", "other"]
Outcome = Literal["new", "updated", "unchanged", "failed"]


class Principal(BaseModel):
    """Who is asking. Every read is scoped to a tenant and filtered by role."""
    tenant_id: str
    roles: list[str] = Field(default_factory=list)


class SourceItem(BaseModel):
    """One uploaded document. ``source_key`` is its stable identity across versions
    (e.g. a path or Confluence page id): re-uploading the same key with new content creates a new version."""
    source_key: str
    filename: str
    content: bytes
    title: str | None = None
    kind: DocKind | None = None
    effective_date: date | None = None
    allowed_roles: list[str] = Field(default_factory=list)  # empty = visible to the whole tenant
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------- facts extracted at parse time
class FieldFact(BaseModel):
    name: str
    type: str = ""
    description: str = ""
    unit: str | None = None
    required: bool = False
    location: str = ""  # for API params: query/path/header/body


class ApiFact(BaseModel):
    method: str
    path: str
    summary: str = ""
    description: str = ""
    operation_id: str = ""
    tags: list[str] = Field(default_factory=list)
    params: list[FieldFact] = Field(default_factory=list)
    response_fields: list[FieldFact] = Field(default_factory=list)


class DatasetFact(BaseModel):
    name: str
    system: str = ""
    description: str = ""
    fields: list[FieldFact] = Field(default_factory=list)


class TermFact(BaseModel):
    term: str
    definition: str
    synonyms: list[str] = Field(default_factory=list)


class Facts(BaseModel):
    apis: list[ApiFact] = Field(default_factory=list)
    datasets: list[DatasetFact] = Field(default_factory=list)
    terms: list[TermFact] = Field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        return {"apis": len(self.apis), "datasets": len(self.datasets), "terms": len(self.terms)}


# ---------------------------------------------------------------- stored records
class SourceRef(BaseModel):
    document_id: int
    source_key: str
    version: int
    title: str


class DocumentRecord(BaseModel):
    id: int
    source_key: str
    version: int
    title: str
    kind: str
    status: str
    valid_from: str
    valid_to: str | None
    ingested_at: str
    batch_id: int
    content_hash: str


class SearchHit(BaseModel):
    chunk_id: int
    document_id: int
    source_key: str
    title: str
    kind: str
    version: int
    status: str
    heading_path: str
    text: str
    score: float

    @property
    def ref(self) -> str:
        return f"chunk:{self.chunk_id}"


class TermRecord(BaseModel):
    term: str
    definition: str
    synonyms: list[str]
    source: SourceRef

    @property
    def ref(self) -> str:
        return f"term:{self.term}"


class DatasetRecord(BaseModel):
    name: str
    system: str
    description: str
    fields: list[FieldFact]
    source: SourceRef

    @property
    def ref(self) -> str:
        return f"dataset:{self.name}"


class ApiRecord(BaseModel):
    method: str
    path: str
    summary: str
    description: str
    params: list[FieldFact]
    response_fields: list[FieldFact]
    source: SourceRef

    @property
    def ref(self) -> str:
        return f"api:{self.method.upper()} {self.path}"


class ItemOutcome(BaseModel):
    source_key: str
    outcome: Outcome
    version: int | None = None
    kind: str | None = None
    chunks: int = 0
    facts: dict[str, int] = Field(default_factory=dict)
    supersedes_version: int | None = None
    error: str | None = None


class BatchReport(BaseModel):
    batch_id: int
    label: str
    items: list[ItemOutcome]

    def count(self, outcome: Outcome) -> int:
        return sum(1 for i in self.items if i.outcome == outcome)


class SectionChange(BaseModel):
    heading_path: str
    change: Literal["added", "removed", "modified"]
    diff: str  # unified diff for modified sections; the text itself for added/removed
