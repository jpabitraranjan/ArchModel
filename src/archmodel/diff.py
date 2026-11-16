"""Structural diff between two model versions (layout changes are ignored)."""
from __future__ import annotations

from .model import ArchitectureModel
from .operations import Change


def diff_models(old: ArchitectureModel, new: ArchitectureModel) -> Change:
    ch = Change()
    for nid in new.nodes.keys() - old.nodes.keys():
        ch.added_nodes.append(nid)
    for nid in old.nodes.keys() - new.nodes.keys():
        ch.removed_nodes.append(nid)
    for nid in old.nodes.keys() & new.nodes.keys():
        if old.nodes[nid].model_dump(exclude={"position"}) != new.nodes[nid].model_dump(exclude={"position"}):
            ch.updated_nodes.append(nid)
    for eid in new.edges.keys() - old.edges.keys():
        ch.added_edges.append(eid)
    for eid in old.edges.keys() - new.edges.keys():
        ch.removed_edges.append(eid)
    for eid in old.edges.keys() & new.edges.keys():
        if old.edges[eid] != new.edges[eid]:
            ch.updated_edges.append(eid)
    for name in type(ch).model_fields:
        getattr(ch, name).sort()
    return ch
