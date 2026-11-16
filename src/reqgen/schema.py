"""The Requirements Document: the structured output of Part 1.

Design rules baked into the schema:
- Every claim carries a ``basis`` (knowledge / user_stated / assumption) and ``sources`` that must be
  verifiable references into the organisation's knowledge base (see grounding.py for the ref formats).
- API specs describe every response attribute: meaning, type, unit, nullability, and origin (which
  dataset column it comes from, or how it is derived). Meaning is mandatory.
- IDs (FR-001, NFR-001, API-001, CON-001, Q-001, ASM-001) are stable and are what the architecture
  model links to for traceability.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Basis = Literal["knowledge", "user_stated", "assumption"]
Priority = Literal["must", "should", "could"]


class AcceptanceCriterion(BaseModel):
    given: str
    when: str
    then: str


class FunctionalRequirement(BaseModel):
    id: str = Field(pattern=r"^FR-\d{3,}$")
    title: str
    description: str
    actor: str
    priority: Priority = "must"
    acceptance_criteria: list[AcceptanceCriterion] = Field(default_factory=list)
    basis: Basis = "user_stated"
    sources: list[str] = Field(default_factory=list)


class NonFunctionalRequirement(BaseModel):
    id: str = Field(pattern=r"^NFR-\d{3,}$")
    category: Literal["performance", "security", "compliance", "availability", "scalability", "usability",
                      "data_quality", "auditability", "retention", "other"]
    statement: str
    target: str | None = None  # measurable target, e.g. "p95 < 500 ms"
    basis: Basis = "assumption"
    sources: list[str] = Field(default_factory=list)


class Stakeholder(BaseModel):
    name: str
    role: str
    interest: str = ""


class GlossaryEntry(BaseModel):
    term: str
    definition: str
    basis: Basis = "knowledge"
    sources: list[str] = Field(default_factory=list)


class DataSource(BaseModel):
    ref: str | None = None  # dataset:<name>, required when status == existing
    name: str
    purpose: str
    status: Literal["existing", "new", "unknown"] = "unknown"
    notes: str = ""


class ParamSpec(BaseModel):
    name: str
    type: str
    required: bool = False
    description: str
    example: str | None = None


class ResponseAttribute(BaseModel):
    name: str  # dotted path for nested fields, e.g. items[].net_quantity
    type: str
    meaning: str  # business definition, in plain words — mandatory
    unit: str | None = None
    nullable: bool = False
    source_field: str | None = None  # dataset:<table>.<column> when copied from storage
    derivation: str | None = None  # formula / rule when computed (e.g. SUM(BUY qty) - SUM(SELL qty))
    glossary_term: str | None = None
    example: str | None = None


class ApiSpec(BaseModel):
    id: str = Field(pattern=r"^API-\d{3,}$")
    name: str
    purpose: str
    status: Literal["existing", "extend", "new"] = "new"
    existing_ref: str | None = None  # api:<METHOD> <path> when existing/extend
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"] = "GET"
    path: str
    delivery: Literal["request_response", "streaming", "batch"] = "request_response"
    update_frequency: str | None = None  # "real-time (< 2 s)", "daily EOD", ...
    satisfies: list[str] = Field(default_factory=list)  # FR-/NFR- ids
    request_params: list[ParamSpec] = Field(default_factory=list)
    response_attributes: list[ResponseAttribute] = Field(default_factory=list)
    pagination: str | None = None
    auth: str | None = None
    errors: list[str] = Field(default_factory=list)
    basis: Basis = "assumption"
    sources: list[str] = Field(default_factory=list)


class Constraint(BaseModel):
    id: str = Field(pattern=r"^CON-\d{3,}$")
    statement: str
    impact: str = ""  # what it forces on the design
    sources: list[str] = Field(default_factory=list)


class Assumption(BaseModel):
    id: str = Field(pattern=r"^ASM-\d{3,}$")
    statement: str
    reason: str = ""
    needs_confirmation: bool = True


class OpenQuestion(BaseModel):
    id: str = Field(pattern=r"^Q-\d{3,}$")
    question: str
    why_it_matters: str = ""
    options: list[str] = Field(default_factory=list)
    blocking: bool = False
    answer: str | None = None
    answered_by: Literal["user", "knowledge"] | None = None
    sources: list[str] = Field(default_factory=list)

    @property
    def is_open(self) -> bool:
        return self.answer is None


class RequirementsDocument(BaseModel):
    title: str
    summary: str
    version: int = 1
    goals: list[str] = Field(default_factory=list)
    stakeholders: list[Stakeholder] = Field(default_factory=list)
    scope_in: list[str] = Field(default_factory=list)
    scope_out: list[str] = Field(default_factory=list)
    glossary: list[GlossaryEntry] = Field(default_factory=list)
    functional: list[FunctionalRequirement] = Field(default_factory=list)
    non_functional: list[NonFunctionalRequirement] = Field(default_factory=list)
    data_sources: list[DataSource] = Field(default_factory=list)
    apis: list[ApiSpec] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    assumptions: list[Assumption] = Field(default_factory=list)
    open_questions: list[OpenQuestion] = Field(default_factory=list)

    def requirement_ids(self) -> list[str]:
        """All ids the architecture model may link to (feeds archmodel.lint(requirement_ids=...))."""
        return ([r.id for r in self.functional] + [r.id for r in self.non_functional]
                + [a.id for a in self.apis] + [c.id for c in self.constraints])
