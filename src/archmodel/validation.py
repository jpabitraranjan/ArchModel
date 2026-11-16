"""Design lint: model-level checks that go beyond per-operation validation.

Operations guarantee the model is *well-formed*. Lint reports whether it is *complete and
coherent* (orphans, disconnected parts, missing requirement links, uncovered requirements).
"""
from __future__ import annotations

from collections import defaultdict

from .catalogue import DEFAULT_CATALOGUE, Catalogue, Level
from .errors import Issue
from .model import ArchitectureModel


def lint(
    model: ArchitectureModel,
    catalogue: Catalogue = DEFAULT_CATALOGUE,
    requirement_ids: list[str] | None = None,
) -> list[Issue]:
    issues: list[Issue] = []
    hld = {nid: n for nid, n in model.nodes.items() if catalogue.components[n.type].level is Level.HLD}

    # orphans: HLD nodes with no connections at all
    for nid, n in hld.items():
        if len(hld) > 1 and not model.edges_of(nid):
            issues.append(Issue(code="orphan_component", severity="warning", ref=nid,
                                message=f"{n.name} ({nid}) is not connected to anything"))

    # disconnected sub-graphs among HLD nodes (undirected)
    adj: dict[str, set[str]] = defaultdict(set)
    for e in model.edges.values():
        if e.source in hld and e.target in hld:
            adj[e.source].add(e.target)
            adj[e.target].add(e.source)
    seen: set[str] = set()
    groups = 0
    for start in hld:
        if start in seen or not adj[start]:
            continue
        groups += 1
        stack = [start]
        while stack:
            cur = stack.pop()
            if cur not in seen:
                seen.add(cur)
                stack.extend(adj[cur] - seen)
    if groups > 1:
        issues.append(Issue(code="disconnected_architecture", severity="warning",
                            message=f"the architecture has {groups} disconnected groups of components"))

    # traceability: every HLD component should answer to at least one requirement
    for nid, n in hld.items():
        if not n.requirement_ids:
            issues.append(Issue(code="no_requirement_link", severity="info", ref=nid,
                                message=f"{n.name} ({nid}) is not linked to any requirement"))

    # compute services with no detailed design yet
    for nid, n in hld.items():
        if n.type in ("api_service", "worker_service") and not model.children_of(nid):
            issues.append(Issue(code="no_lld", severity="info", ref=nid,
                                message=f"{n.name} ({nid}) has no LLD elements yet"))

    # requirement coverage (only when the caller supplies the requirement list)
    if requirement_ids is not None:
        covered = {r for n in model.nodes.values() for r in n.requirement_ids}
        covered |= {r for e in model.edges.values() for r in e.requirement_ids}
        for rid in requirement_ids:
            if rid not in covered:
                issues.append(Issue(code="uncovered_requirement", severity="warning", ref=rid,
                                    message=f"requirement {rid} is not realised by any design element"))
        for rid in sorted(covered - set(requirement_ids)):
            issues.append(Issue(code="unknown_requirement", severity="error", ref=rid,
                                message=f"design links to {rid}, which is not in the requirements list"))
    return issues
