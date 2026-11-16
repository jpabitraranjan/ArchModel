"""Drive the three CLIs end to end with a scripted LLM standing in for the real provider."""
import json
from pathlib import Path

import pytest

from orgknow.cli import main as orgknow_main
from reqgen import ScriptedLLM, SessionState
from reqgen.cli import main as reqgen_main
from test_reqgen import CLARIFY, REQUEST, UNDERSTANDING, good_document

KB_DIR = Path(__file__).parent.parent / "examples" / "finance_kb"


def test_cli_flow(tmp_path, monkeypatch, capsys):
    db, sess = str(tmp_path / "kb.db"), str(tmp_path / "s.json")
    assert orgknow_main(["ingest", "--db", db, "--tenant", "acme", "--dir", str(KB_DIR / "batch1_initial")]) == 0
    assert orgknow_main(["ingest", "--db", db, "--tenant", "acme", "--dir", str(KB_DIR / "batch2_policy_update"), "--label", "2026"]) == 0
    capsys.readouterr()

    bundle_of = lambda: SessionState.model_validate_json(Path(sess).read_text()).bundle
    llm = ScriptedLLM([json.dumps(UNDERSTANDING), json.dumps(CLARIFY)])
    monkeypatch.setattr("reqgen.cli.llm_from_env", lambda: llm)
    assert reqgen_main(["start", "--db", db, "--tenant", "acme", "--session", sess, REQUEST]) == 0
    out = capsys.readouterr().out
    assert "Q-002 [BLOCKING]" in out and "resolved from knowledge" in out

    llm.responses = []
    assert reqgen_main(["generate", "--session", sess]) == 2            # refuses while a blocking question is open
    assert "answer these first" in capsys.readouterr().err

    assert reqgen_main(["answer", "--session", sess, "Q-002", "AP", "only"]) == 0
    llm.responses = [lambda a, b: json.dumps(good_document(bundle_of()))]
    md, js = str(tmp_path / "r.md"), str(tmp_path / "r.json")
    assert reqgen_main(["generate", "--session", sess, "--out", js, "--md", md]) == 0
    assert "approvable=True" in capsys.readouterr().err
    text = Path(md).read_text()
    assert "### API-003 — Net flow by security" in text and "AP only" in text
    assert json.loads(Path(js).read_text())["apis"][2]["path"] == "/v1/analytics/net-flow"
