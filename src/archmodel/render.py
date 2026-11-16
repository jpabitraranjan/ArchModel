"""Derived views. Mermaid is generated from the model, never edited by hand."""
from __future__ import annotations

import re

from .catalogue import DEFAULT_CATALOGUE, Catalogue, Level
from .model import ArchitectureModel

_SHAPES = {  # component type -> (open, close)
    "database": ("[(", ")]"), "message_queue": ("[[", "]]"), "cache": ("([", "])"),
    "ui_client": ("[/", "/]"), "external_system": ("{{", "}}"), "load_balancer": ("{", "}"),
    "table": ("[(", ")]"),
}


def _mid(node_id: str) -> str:
    return re.sub(r"\W", "_", node_id)


def _esc(text: str) -> str:
    return text.replace('"', "'")


def to_mermaid(model: ArchitectureModel, scope: str | None = None,
               catalogue: Catalogue = DEFAULT_CATALOGUE) -> str:
    """HLD flowchart when ``scope`` is None; otherwise the LLD elements inside node ``scope``."""
    if scope is None:
        nodes = [n for n in model.nodes.values() if catalogue.components[n.type].level is Level.HLD]
    else:
        if scope not in model.nodes:
            raise KeyError(f"node '{scope}' does not exist")
        nodes = model.children_of(scope)
    ids = {n.id for n in nodes}
    lines = ["flowchart LR"]
    for n in nodes:
        label = _esc(n.name) if scope is None else f"{_esc(n.name)}<br/>«{n.type}»"
        o, c = _SHAPES.get(n.type, ("[", "]"))
        lines.append(f'    {_mid(n.id)}{o}"{label}"{c}')
    for e in model.edges.values():
        if e.source in ids and e.target in ids:
            lines.append(f"    {_mid(e.source)} -->|{_esc(e.label or e.type)}| {_mid(e.target)}")
    return "\n".join(lines)
