"""Small CLI for working with model files. The web UI will use the same functions via FastAPI."""
from __future__ import annotations

import argparse
import json
import sys

from .catalogue import DEFAULT_CATALOGUE
from .errors import OperationError
from .model import ArchitectureModel
from .operations import apply_operations
from .render import to_mermaid
from .validation import lint


def _load(path: str) -> ArchitectureModel:
    with open(path, encoding="utf-8") as f:
        return ArchitectureModel.from_json(f.read())


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="archmodel")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("catalogue", help="print the component catalogue as JSON")
    sub.add_parser("schema", help="print the JSON Schema of the architecture model")
    n = sub.add_parser("new", help="create an empty model file")
    n.add_argument("path")
    n.add_argument("--name", default="Untitled")
    a = sub.add_parser("apply", help="apply an operations JSON file to a model file")
    a.add_argument("model")
    a.add_argument("ops")
    a.add_argument("-o", "--output", help="output path (default: overwrite the model file)")
    v = sub.add_parser("lint", help="run design lint")
    v.add_argument("model")
    v.add_argument("--requirements", help="comma-separated requirement ids to check coverage against")
    m = sub.add_parser("mermaid", help="render Mermaid for the HLD, or an LLD scope")
    m.add_argument("model")
    m.add_argument("--scope", help="HLD node id whose LLD elements to render")
    args = p.parse_args(argv)

    if args.cmd == "catalogue":
        print(json.dumps(DEFAULT_CATALOGUE.to_dict(), indent=2))
    elif args.cmd == "schema":
        print(json.dumps(ArchitectureModel.model_json_schema(), indent=2))
    elif args.cmd == "new":
        with open(args.path, "w", encoding="utf-8") as f:
            f.write(ArchitectureModel(name=args.name).to_json())
    elif args.cmd == "apply":
        with open(args.ops, encoding="utf-8") as f:
            ops = json.load(f)
        try:
            result = apply_operations(_load(args.model), ops)
        except OperationError as e:
            print(f"REJECTED: {e}", file=sys.stderr)
            return 1
        with open(args.output or args.model, "w", encoding="utf-8") as f:
            f.write(result.model.to_json())
        print(f"applied {len(ops)} operation(s); model is now version {result.model.version}")
    elif args.cmd == "lint":
        reqs = args.requirements.split(",") if args.requirements else None
        issues = lint(_load(args.model), requirement_ids=reqs)
        for i in issues:
            print(f"{i.severity.upper():7} {i.code}: {i.message}")
        return 1 if any(i.severity == "error" for i in issues) else 0
    elif args.cmd == "mermaid":
        print(to_mermaid(_load(args.model), scope=args.scope))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
