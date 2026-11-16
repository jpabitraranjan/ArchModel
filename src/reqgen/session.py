"""A requirements session: understand -> ground -> clarify -> (answers) -> generate -> validate/repair.

The session state is plain JSON, so a CLI or web UI can stop after the questions, let a human answer
over hours or days, and resume. Grounding is stored in the state, so citations stay valid on resume
even if the knowledge base changes in between.
"""
from __future__ import annotations

import json
import re
from datetime import date
from typing import Any, Literal

from archmodel.errors import Issue
from orgknow import KnowledgeBase, Principal
from pydantic import BaseModel, Field, ValidationError

from . import prompts
from .grounding import GroundingBundle, build_grounding
from .llm import LLM, complete_json
from .schema import OpenQuestion, RequirementsDocument
from .validate import errors_only, validate_document


class BlockingQuestionsError(Exception):
    def __init__(self, questions: list[OpenQuestion]):
        self.questions = questions
        super().__init__("blocking questions are unanswered: " + "; ".join(f"{q.id} {q.question}" for q in questions))


class SessionState(BaseModel):
    request: str
    who: Principal
    as_of: str | None = None
    stage: Literal["new", "analyzed", "generated"] = "new"
    understanding: dict[str, Any] = Field(default_factory=dict)
    bundle: GroundingBundle = Field(default_factory=GroundingBundle)
    questions: list[OpenQuestion] = Field(default_factory=list)
    document: RequirementsDocument | None = None
    issues: list[Issue] = Field(default_factory=list)
    generations: int = 0


class GenerationResult(BaseModel):
    document: RequirementsDocument
    issues: list[Issue]
    repair_rounds: int

    @property
    def ok(self) -> bool:  # structurally valid and fully grounded
        return not errors_only(self.issues)

    @property
    def approvable(self) -> bool:  # ok and nothing blocking left open
        return self.ok and not any(q.is_open and q.blocking for q in self.document.open_questions)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


class RequirementsSession:
    def __init__(self, llm: LLM, state: SessionState, kb: KnowledgeBase | None = None, max_repairs: int = 2):
        self.llm, self.state, self.kb, self.max_repairs = llm, state, kb, max_repairs

    @classmethod
    def start(cls, kb: KnowledgeBase, who: Principal, llm: LLM, request: str, as_of: date | None = None,
              max_repairs: int = 2) -> "RequirementsSession":
        st = SessionState(request=request.strip(), who=who, as_of=as_of.isoformat() if as_of else None)
        return cls(llm, st, kb, max_repairs)

    @classmethod
    def resume(cls, llm: LLM, state_json: str, max_repairs: int = 2) -> "RequirementsSession":
        return cls(llm, SessionState.model_validate_json(state_json), None, max_repairs)

    def to_json(self) -> str:
        return self.state.model_dump_json(indent=2)

    # ------------------------------------------------------------------ 1. understand, ground, clarify
    def analyze(self) -> list[OpenQuestion]:
        if self.kb is None:
            raise RuntimeError("analyze() needs a KnowledgeBase; resume() sessions can only answer and generate")
        st = self.state
        st.understanding = complete_json(self.llm, prompts.SYSTEM, prompts.UNDERSTAND.format(request=st.request))
        as_of = date.fromisoformat(st.as_of) if st.as_of else None
        st.bundle = build_grounding(
            self.kb, st.who, terms=[str(t) for t in st.understanding.get("domain_terms", [])],
            data_needs=[str(t) for t in st.understanding.get("data_needs", [])],
            queries=[str(q) for q in st.understanding.get("search_queries", [])] + [st.request], as_of=as_of)
        raw = complete_json(self.llm, prompts.SYSTEM, prompts.CLARIFY.format(
            request=st.request, understanding=json.dumps(st.understanding), grounding=st.bundle.to_prompt()))
        st.questions = []
        for i, q in enumerate(raw.get("questions", []), 1):
            text = str(q.get("question", "")).strip()
            if not text:
                continue
            oq = OpenQuestion(id=f"Q-{i:03d}", question=text, why_it_matters=str(q.get("why_it_matters", "")),
                              options=[str(o) for o in q.get("options", [])], blocking=bool(q.get("blocking", False)))
            res = q.get("resolved_by_knowledge")
            if isinstance(res, dict) and str(res.get("answer", "")).strip():
                sources = [str(s) for s in res.get("sources", [])]
                # only accept a knowledge-based answer if every cited source really exists
                if sources and all(st.bundle.has_ref(s) for s in sources):
                    oq.answer, oq.answered_by, oq.sources = str(res["answer"]).strip(), "knowledge", sources
            st.questions.append(oq)
        st.stage = "analyzed"
        return self.open_questions()

    def open_questions(self) -> list[OpenQuestion]:
        return [q for q in self.state.questions if q.is_open]

    def blocking_questions(self) -> list[OpenQuestion]:
        return [q for q in self.open_questions() if q.blocking]

    def answer(self, question_id: str, text: str) -> None:
        for q in self.state.questions:
            if q.id == question_id:
                q.answer, q.answered_by = text.strip(), "user"
                return
        raise KeyError(f"no question {question_id}")

    # ------------------------------------------------------------------ 2. generate, validate, repair
    def _answers_block(self) -> str:
        lines = []
        for q in self.state.questions:
            if q.answer is not None:
                src = f" [from knowledge: {', '.join(q.sources)}]" if q.answered_by == "knowledge" else " [answered by the user]"
                lines.append(f"- {q.id} {q.question}\n  -> {q.answer}{src}")
            else:
                lines.append(f"- {q.id} {q.question}\n  -> UNANSWERED{' (blocking)' if q.blocking else ''}: "
                             "state an explicit assumption and keep it in open_questions")
        return "\n".join(lines) or "(none)"

    def _merge_questions(self, doc: RequirementsDocument) -> None:
        """The session is the source of truth for questions; keep any extra ones the model raised."""
        merged = [q.model_copy() for q in self.state.questions]
        known = {_norm(q.question) for q in merged}
        n = len(merged)
        for q in doc.open_questions:
            if _norm(q.question) not in known:
                n += 1
                merged.append(q.model_copy(update={"id": f"Q-{n:03d}", "answer": None, "answered_by": None}))
                known.add(_norm(q.question))
        doc.open_questions = merged

    def generate(self, force: bool = False) -> GenerationResult:
        st = self.state
        if st.stage == "new":
            raise RuntimeError("call analyze() first")
        blocking = self.blocking_questions()
        if blocking and not force:
            raise BlockingQuestionsError(blocking)
        grounding = st.bundle.to_prompt()
        prompt = prompts.GENERATE.format(request=st.request, answers=self._answers_block(), grounding=grounding,
                                         schema=prompts.schema_json())
        doc: RequirementsDocument | None = None
        issues: list[Issue] = []
        rounds = 0
        for attempt in range(self.max_repairs + 1):
            data = complete_json(self.llm, prompts.SYSTEM, prompt, max_tokens=16000)
            try:
                doc = RequirementsDocument.model_validate(data)
            except ValidationError as e:
                issues = [Issue(code="schema_error", message=f"{'.'.join(map(str, err['loc']))}: {err['msg']}")
                          for err in e.errors()[:25]]
                doc = None
            else:
                self._merge_questions(doc)
                issues = validate_document(doc, st.bundle)
            if not errors_only(issues):
                break
            if attempt == self.max_repairs:
                break
            rounds += 1
            problems = "\n".join(f"- [{i.code}] {i.message}" for i in errors_only(issues))
            prompt = prompts.REPAIR.format(problems=problems, document=json.dumps(data), grounding=grounding)
        if doc is None:
            raise ValueError("the model never produced a schema-valid document: " + "; ".join(i.message for i in issues[:5]))
        st.generations += 1
        doc.version = st.generations
        st.document, st.issues, st.stage = doc, issues, "generated"
        return GenerationResult(document=doc, issues=issues, repair_rounds=rounds)
