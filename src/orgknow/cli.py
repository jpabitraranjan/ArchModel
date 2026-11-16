"""orgknow: batch-ingest and inspect organisational knowledge.

    orgknow ingest --db kb.db --tenant acme --dir ./docs --label "2026 policy update"
    orgknow docs --db kb.db --tenant acme
    orgknow search --db kb.db --tenant acme "retention period for trade records"
    orgknow versions --db kb.db --tenant acme policies/data_access_policy.md
    orgknow diff --db kb.db --tenant acme policies/data_access_policy.md 1 2
    orgknow catalog --db kb.db --tenant acme trades settlement
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from .models import Principal, SourceItem
from .store import KnowledgeBase


def load_directory(root: Path) -> list[SourceItem]:
    """Every file under ``root`` becomes a SourceItem keyed by its relative path. An optional
    manifest.json provides per-file overrides (kind, effective_date, allowed_roles, title) and a
    ``_default`` entry."""
    manifest: dict = {}
    mf = root / "manifest.json"
    if mf.exists():
        manifest = json.loads(mf.read_text(encoding="utf-8"))
    default = manifest.get("_default", {})
    items = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "manifest.json"):
        key = path.relative_to(root).as_posix()
        meta = {**default, **manifest.get(key, {})}
        eff = meta.get("effective_date")
        items.append(SourceItem(source_key=key, filename=path.name, content=path.read_bytes(), title=meta.get("title"),
                                kind=meta.get("kind"), effective_date=date.fromisoformat(eff) if eff else None,
                                allowed_roles=meta.get("allowed_roles", []), metadata=meta.get("metadata", {})))
    return items


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="orgknow")
    p.add_argument("cmd", choices=["ingest", "docs", "search", "versions", "diff", "catalog", "batches"])
    p.add_argument("args", nargs="*")
    p.add_argument("--db", default="kb.db")
    p.add_argument("--tenant", required=True)
    p.add_argument("--roles", default="", help="comma-separated roles of the caller")
    p.add_argument("--dir")
    p.add_argument("--label", default="")
    p.add_argument("--as-of", help="YYYY-MM-DD: see knowledge as it was on this date")
    a = p.parse_intermixed_args(argv)
    kb = KnowledgeBase(a.db)
    who = Principal(tenant_id=a.tenant, roles=[r for r in a.roles.split(",") if r])
    as_of = date.fromisoformat(a.as_of) if a.as_of else None

    if a.cmd == "ingest":
        if not a.dir:
            p.error("ingest needs --dir")
        report = kb.ingest_batch(a.tenant, load_directory(Path(a.dir)), a.label or Path(a.dir).name)
        print(f"batch {report.batch_id} '{report.label}': " + ", ".join(
            f"{n} {o}" for o in ("new", "updated", "unchanged", "failed") if (n := report.count(o))))
        for i in report.items:
            extra = f" (supersedes v{i.supersedes_version})" if i.supersedes_version else ""
            print(f"  {i.outcome:9} {i.source_key} -> v{i.version} [{i.kind}] chunks={i.chunks} facts={i.facts}{extra}"
                  if i.outcome != "failed" else f"  FAILED    {i.source_key}: {i.error}")
        return 1 if report.count("failed") else 0
    if a.cmd == "docs":
        for d in kb.documents(who, include_superseded=True):
            print(f"{d.source_key} v{d.version} [{d.kind}] {d.status} valid {d.valid_from}..{d.valid_to or 'now'}")
    elif a.cmd == "batches":
        for b in kb.batches(who):
            print(f"batch {b.batch_id} '{b.label}': {len(b.items)} items")
    elif a.cmd == "search":
        for h in kb.search(who, " ".join(a.args), as_of=as_of):
            print(f"[chunk:{h.chunk_id}] {h.title} v{h.version} / {h.heading_path}  (score {h.score})\n    {h.text[:160]!r}")
    elif a.cmd == "versions":
        for d in kb.versions(who, a.args[0]):
            print(f"v{d.version} {d.status} valid {d.valid_from}..{d.valid_to or 'now'} batch={d.batch_id}")
    elif a.cmd == "diff":
        for c in kb.diff_versions(who, a.args[0], int(a.args[1]), int(a.args[2])):
            print(f"{c.change.upper():9} {c.heading_path}\n{c.diff}\n")
    elif a.cmd == "catalog":
        kw = a.args
        for t in kb.find_terms(who, kw, as_of):
            print(f"TERM    {t.term}: {t.definition}")
        for d in kb.search_datasets(who, kw, as_of):
            print(f"DATASET {d.name}: " + ", ".join(f.name for f in d.fields))
        for x in kb.search_apis(who, kw, as_of):
            print(f"API     {x.method} {x.path}: {x.summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
