"""The architecture model: the single source of truth for HLD and LLD.

Everything else (visual editor, Mermaid, API specs, schemas) is derived from this structure.
Mutate it only through ``operations.apply_operations``.
"""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

REQUIREMENT_ID_RE = re.compile(r"^[A-Z]{2,5}-\d+$")  # e.g. FR-001, NFR-012


class Position(BaseModel):
    x: float = 0.0
    y: float = 0.0


class Node(BaseModel):
    id: str
    type: str
    name: str
    properties: dict[str, Any] = Field(default_factory=dict)
    parent_id: str | None = None  # set for LLD elements, which live inside an HLD node
    position: Position = Field(default_factory=Position)  # layout only, not design semantics
    requirement_ids: list[str] = Field(default_factory=list)


class Edge(BaseModel):
    id: str
    type: str
    source: str
    target: str
    label: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)
    requirement_ids: list[str] = Field(default_factory=list)


class ArchitectureModel(BaseModel):
    schema_version: str = "1.0"
    name: str = "Untitled"
    tenant_id: str = "default"  # kept from day one so multi-tenancy never needs a schema change
    version: int = 0
    nodes: dict[str, Node] = Field(default_factory=dict)
    edges: dict[str, Edge] = Field(default_factory=dict)
    counters: dict[str, int] = Field(default_factory=dict)

    # ------------------------------------------------------------------ id allocation
    def next_id(self, prefix: str) -> str:
        n = self.counters.get(prefix, 0)
        while True:
            n += 1
            candidate = f"{prefix}-{n:03d}"
            if candidate not in self.nodes and candidate not in self.edges:
                self.counters[prefix] = n
                return candidate

    # ------------------------------------------------------------------ queries
    def children_of(self, node_id: str) -> list[Node]:
        return [n for n in self.nodes.values() if n.parent_id == node_id]

    def edges_of(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges.values() if node_id in (e.source, e.target)]

    # ------------------------------------------------------------------ persistence
    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "ArchitectureModel":
        return cls.model_validate_json(text)
