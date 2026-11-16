"""reqgen: natural-language request -> grounded, validated requirements document.

    reqgen start    --db kb.db --tenant acme --session s.json "Build a dashboard for portfolio managers ..."
    reqgen answer   --session s.json Q-002 "House trades are out of scope"
    reqgen generate --session s.json --out requirements.json --md requirements.md [--force]

LLM: set ANTHROPIC_API_KEY (or REQGEN_PROVIDER=openai with OPENAI_API_KEY / REQGEN_BASE_URL for
OpenAI-compatible servers such as Ollama). Override the model with REQGEN_MODEL.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from orgknow import KnowledgeBase, Principal

from .llm import LLMError, llm_from_env
from .render import to_markdown
from .session import BlockingQuestionsError, RequirementsSession


def _print_questions(qs) -> None:
    for q in qs:
        print(f"  {q.id} {'[BLOCKING] ' if q.blocking else ''}{q.question}")
        if q.why_it_matters:
            print(f"      why: {q.why_it_matters}")
        if q.options:
            print(f"      options: {' | '.join(q.options)}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="reqgen")
    p.add_argument("cmd", choices=["start", "answer", "generate"])
    p.add_argument("args", nargs="*")
    p.add_argument("--session", required=True)
    p.add_argument("--db", default="kb.db")
    p.add_argument("--tenant")
    p.add_argument("--roles", default="")
    p.add_argument("--as-of")
    p.add_argument("--out")
    p.add_argument("--md")
    p.add_argument("--force", action="store_true", help="generate even if blocking questions are unanswered")
    a = p.parse_intermixed_args(argv)
    sess_path = Path(a.session)
    try:
        if a.cmd == "start":
            if not a.tenant or not a.args:
                p.error("start needs --tenant and the request text")
            who = Principal(tenant_id=a.tenant, roles=[r for r in a.roles.split(",") if r])
            s = RequirementsSession.start(KnowledgeBase(a.db), who, llm_from_env(), " ".join(a.args),
                                          as_of=date.fromisoformat(a.as_of) if a.as_of else None)
            open_qs = s.analyze()
            sess_path.write_text(s.to_json(), encoding="utf-8")
            b = s.state.bundle
            print(f"grounded on {len(b.chunks)} passages, {len(b.terms)} terms, {len(b.datasets)} datasets, {len(b.apis)} APIs")
            answered = [q for q in s.state.questions if not q.is_open]
            for q in answered:
                print(f"  {q.id} resolved from knowledge: {q.question} -> {q.answer}")
            print(f"{len(open_qs)} question(s) need your answer:" if open_qs else "no open questions")
            _print_questions(open_qs)
            return 0
        s = RequirementsSession.resume(llm_from_env(), sess_path.read_text(encoding="utf-8"))
        if a.cmd == "answer":
            s.answer(a.args[0], " ".join(a.args[1:]))
            sess_path.write_text(s.to_json(), encoding="utf-8")
            print(f"{len(s.open_questions())} open, {len(s.blocking_questions())} blocking")
            return 0
        res = s.generate(force=a.force)
        sess_path.write_text(s.to_json(), encoding="utf-8")
        if a.out:
            Path(a.out).write_text(res.document.model_dump_json(indent=2), encoding="utf-8")
        md = to_markdown(res.document)
        if a.md:
            Path(a.md).write_text(md, encoding="utf-8")
        else:
            print(md)
        for i in res.issues:
            print(f"{i.severity.upper():7} {i.code}: {i.message}", file=sys.stderr)
        print(f"generated v{res.document.version}: ok={res.ok} approvable={res.approvable} repair_rounds={res.repair_rounds}", file=sys.stderr)
        return 0 if res.ok else 1
    except BlockingQuestionsError as e:
        print("cannot generate yet; answer these first (or use --force):", file=sys.stderr)
        _print_questions(e.questions)
        return 2
    except (LLMError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
