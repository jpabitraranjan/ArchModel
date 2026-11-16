"""Validated operations: the only way a model changes.

Human edits from the visual editor and AI-agent output both go through here, so there is one
validation path. A batch is atomic: if any operation is rejected, nothing is applied.

Operations in one batch can refer to nodes created earlier in the same batch with ``@alias``:

    [{"op": "add_node", "type": "database", "name": "Orders DB", "ref": "db",
      "properties": {"engine": "postgresql"}},
     {"op": "add_node", "type": "api_service", "name": "Order API", "ref": "api"},
     {"op": "connect", "type": "db_connection", "source": "@api", "target": "@db"}]
"""
from __future__ import annotations

from typing import Annotated, Any, Callable, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter

from .catalogue import DEFAULT_CATALOGUE, Catalogue, Level, resolve_properties
from .errors import Issue, OperationError
from .model import REQUIREMENT_ID_RE, ArchitectureModel, Edge, Node, Position


# ---------------------------------------------------------------------------- operation types
class AddNode(BaseModel):
    op: Literal["add_node"] = "add_node"
    type: str
    name: str
    properties: dict[str, Any] = Field(default_factory=dict)
    parent_id: str | None = None
    position: Position = Field(default_factory=Position)
    id: str | None = None
    ref: str | None = None  # alias usable as "@ref" by later ops in the same batch


class RemoveNode(BaseModel):
    op: Literal["remove_node"] = "remove_node"
    id: str  # also removes contained LLD elements and every edge touching them


class UpdateNode(BaseModel):
    op: Literal["update_node"] = "update_node"
    id: str
    name: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)  # merged into existing
    unset: list[str] = Field(default_factory=list)  # property names to drop (revert to default)


class MoveNode(BaseModel):
    op: Literal["move_node"] = "move_node"  # layout only; does not bump the design version
    id: str
    position: Position


class Connect(BaseModel):
    op: Literal["connect"] = "connect"
    type: str
    source: str
    target: str
    label: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)
    id: str | None = None
    ref: str | None = None


class Disconnect(BaseModel):
    op: Literal["disconnect"] = "disconnect"
    id: str


class UpdateEdge(BaseModel):
    op: Literal["update_edge"] = "update_edge"
    id: str
    label: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)
    unset: list[str] = Field(default_factory=list)


class LinkRequirement(BaseModel):
    op: Literal["link_requirement"] = "link_requirement"
    target_id: str  # node or edge id
    requirement_id: str
    unlink: bool = False


Operation = Annotated[
    Union[AddNode, RemoveNode, UpdateNode, MoveNode, Connect, Disconnect, UpdateEdge, LinkRequirement],
    Field(discriminator="op"),
]
_ADAPTER: TypeAdapter[Operation] = TypeAdapter(Operation)


def parse_operation(data: dict[str, Any]) -> Operation:
    return _ADAPTER.validate_python(data)


# ---------------------------------------------------------------------------- results
class Change(BaseModel):
    added_nodes: list[str] = Field(default_factory=list)
    removed_nodes: list[str] = Field(default_factory=list)
    updated_nodes: list[str] = Field(default_factory=list)
    added_edges: list[str] = Field(default_factory=list)
    removed_edges: list[str] = Field(default_factory=list)
    updated_edges: list[str] = Field(default_factory=list)

    def _touch(self, bucket: list[str], ident: str) -> None:
        if ident not in bucket:
            bucket.append(ident)

    @property
    def is_empty(self) -> bool:
        return not any(self.model_dump().values())


class ApplyResult(BaseModel):
    model: ArchitectureModel
    changes: Change
    semantic: bool  # False when the batch only moved nodes
    created_ids: list[str | None]  # id assigned by each op (None for ops that create nothing)


class _Fail(Exception):
    def __init__(self, *issues: Issue):
        self.issues = list(issues)


def _fail(code: str, message: str, ref: str | None = None) -> _Fail:
    return _Fail(Issue(code=code, message=message, ref=ref))


# ---------------------------------------------------------------------------- helpers
def _res(value: str | None, aliases: dict[str, str]) -> str | None:
    if value is not None and value.startswith("@"):
        if value[1:] not in aliases:
            raise _fail("unknown_ref", f"alias '{value}' was not defined earlier in this batch", value)
        return aliases[value[1:]]
    return value


def _node(m: ArchitectureModel, node_id: str) -> Node:
    if node_id not in m.nodes:
        raise _fail("node_not_found", f"node '{node_id}' does not exist", node_id)
    return m.nodes[node_id]


def _check_requirement_ids(ids: list[str]) -> None:
    bad = [r for r in ids if not REQUIREMENT_ID_RE.match(r)]
    if bad:
        raise _fail("bad_requirement_id", f"invalid requirement id(s) {bad}; expected like FR-001")


def _prop_issues(errors: list[str], ref: str | None) -> list[Issue]:
    return [Issue(code="invalid_property", message=e, ref=ref) for e in errors]


# ---------------------------------------------------------------------------- handlers
Handler = Callable[[ArchitectureModel, Any, Catalogue, dict[str, str], Change], str | None]


def _add_node(m, op: AddNode, cat: Catalogue, al, ch: Change):
    comp = cat.components.get(op.type)
    if comp is None:
        raise _fail("unknown_component_type", f"component type '{op.type}' is not in the catalogue")
    if not op.name.strip():
        raise _fail("empty_name", "node name must not be empty")
    parent_id = _res(op.parent_id, al)
    if comp.level is Level.HLD:
        if parent_id:
            raise _fail("hld_cannot_have_parent", f"'{op.type}' is an HLD component and cannot be nested")
    else:
        if not parent_id:
            raise _fail("lld_requires_parent", f"'{op.type}' is an LLD element and needs a parent_id")
        parent = _node(m, parent_id)
        if parent.type not in comp.allowed_parents:
            raise _fail("invalid_parent_type",
                        f"'{op.type}' cannot live inside '{parent.type}'; allowed: {comp.allowed_parents}")
    props, errs = resolve_properties(comp.properties, op.properties)
    if errs:
        raise _Fail(*_prop_issues(errs, op.type))
    nid = op.id or m.next_id(comp.id_prefix)
    if nid in m.nodes or nid in m.edges:
        raise _fail("duplicate_id", f"id '{nid}' is already in use", nid)
    m.nodes[nid] = Node(id=nid, type=op.type, name=op.name.strip(), properties=props,
                        parent_id=parent_id, position=op.position)
    ch._touch(ch.added_nodes, nid)
    if op.ref:
        al[op.ref] = nid
    return nid


def _remove_node(m, op: RemoveNode, cat, al, ch: Change):
    nid = _res(op.id, al)
    _node(m, nid)
    doomed = {nid}
    frontier = [nid]
    while frontier:  # collect contained LLD elements (recursively, for future nesting)
        cur = frontier.pop()
        for child in m.children_of(cur):
            if child.id not in doomed:
                doomed.add(child.id)
                frontier.append(child.id)
    for eid in [e.id for e in m.edges.values() if e.source in doomed or e.target in doomed]:
        del m.edges[eid]
        ch._touch(ch.removed_edges, eid)
    for d in sorted(doomed):
        del m.nodes[d]
        ch._touch(ch.removed_nodes, d)
    return None


def _update_node(m, op: UpdateNode, cat: Catalogue, al, ch: Change):
    nid = _res(op.id, al)
    node = _node(m, nid)
    comp = cat.components[node.type]
    merged = {**node.properties, **op.properties}
    for key in op.unset:
        merged.pop(key, None)
    props, errs = resolve_properties(comp.properties, merged)
    if errs:
        raise _Fail(*_prop_issues(errs, nid))
    if op.name is not None:
        if not op.name.strip():
            raise _fail("empty_name", "node name must not be empty", nid)
        node.name = op.name.strip()
    node.properties = props
    ch._touch(ch.updated_nodes, nid)
    return None


def _move_node(m, op: MoveNode, cat, al, ch: Change):
    _node(m, _res(op.id, al)).position = op.position
    return None


def _connect(m, op: Connect, cat: Catalogue, al, ch: Change):
    ctype = cat.connections.get(op.type)
    if ctype is None:
        raise _fail("unknown_connection_type", f"connection type '{op.type}' is not in the catalogue")
    src_id, tgt_id = _res(op.source, al), _res(op.target, al)
    src, tgt = _node(m, src_id), _node(m, tgt_id)
    if src_id == tgt_id:
        raise _fail("self_connection", "a node cannot be connected to itself", src_id)
    if not ctype.allows(src.type, tgt.type):
        raise _fail("illegal_connection",
                    f"'{op.type}' cannot connect {src.type} -> {tgt.type} "
                    f"(allowed sources: {ctype.sources}; allowed targets: {ctype.targets})")
    for e in m.edges.values():
        if (e.type, e.source, e.target) == (op.type, src_id, tgt_id):
            raise _fail("duplicate_edge", f"'{op.type}' from {src_id} to {tgt_id} already exists as {e.id}", e.id)
    if ctype.level is Level.LLD and src.parent_id != tgt.parent_id:
        # An LLD relationship that crosses HLD components must be backed by an HLD edge,
        # otherwise the detailed design would silently contradict the architecture.
        a, b = src.parent_id, tgt.parent_id
        if not any({e.source, e.target} == {a, b} for e in m.edges.values()):
            raise _fail("lld_edge_without_hld_edge",
                        f"{src_id} ({a}) and {tgt_id} ({b}) are in different components that have "
                        f"no HLD connection; add that connection first")
    props, errs = resolve_properties(ctype.properties, op.properties)
    if errs:
        raise _Fail(*_prop_issues(errs, op.type))
    eid = op.id or m.next_id("E")
    if eid in m.nodes or eid in m.edges:
        raise _fail("duplicate_id", f"id '{eid}' is already in use", eid)
    m.edges[eid] = Edge(id=eid, type=op.type, source=src_id, target=tgt_id, label=op.label, properties=props)
    ch._touch(ch.added_edges, eid)
    if op.ref:
        al[op.ref] = eid
    return eid


def _edge(m: ArchitectureModel, edge_id: str) -> Edge:
    if edge_id not in m.edges:
        raise _fail("edge_not_found", f"edge '{edge_id}' does not exist", edge_id)
    return m.edges[edge_id]


def _disconnect(m, op: Disconnect, cat, al, ch: Change):
    eid = _res(op.id, al)
    _edge(m, eid)
    del m.edges[eid]
    ch._touch(ch.removed_edges, eid)
    return None


def _update_edge(m, op: UpdateEdge, cat: Catalogue, al, ch: Change):
    eid = _res(op.id, al)
    edge = _edge(m, eid)
    ctype = cat.connections[edge.type]
    merged = {**edge.properties, **op.properties}
    for key in op.unset:
        merged.pop(key, None)
    props, errs = resolve_properties(ctype.properties, merged)
    if errs:
        raise _Fail(*_prop_issues(errs, eid))
    edge.properties = props
    if op.label is not None:
        edge.label = op.label
    ch._touch(ch.updated_edges, eid)
    return None


def _link_requirement(m, op: LinkRequirement, cat, al, ch: Change):
    tid = _res(op.target_id, al)
    _check_requirement_ids([op.requirement_id])
    if tid in m.nodes:
        target, bucket = m.nodes[tid], ch.updated_nodes
    elif tid in m.edges:
        target, bucket = m.edges[tid], ch.updated_edges
    else:
        raise _fail("target_not_found", f"no node or edge '{tid}'", tid)
    reqs = target.requirement_ids
    if op.unlink:
        if op.requirement_id not in reqs:
            raise _fail("not_linked", f"{tid} is not linked to {op.requirement_id}", tid)
        reqs.remove(op.requirement_id)
    elif op.requirement_id not in reqs:
        reqs.append(op.requirement_id)
    ch._touch(bucket, tid)
    return None


_HANDLERS: dict[str, Handler] = {
    "add_node": _add_node, "remove_node": _remove_node, "update_node": _update_node,
    "move_node": _move_node, "connect": _connect, "disconnect": _disconnect,
    "update_edge": _update_edge, "link_requirement": _link_requirement,
}


# ---------------------------------------------------------------------------- entry point
def apply_operations(
    model: ArchitectureModel,
    operations: list[Operation | dict[str, Any]],
    catalogue: Catalogue = DEFAULT_CATALOGUE,
) -> ApplyResult:
    """Apply a batch atomically. Returns a new model; the input is never mutated.

    Raises ``OperationError`` (with structured ``issues``) if any operation is rejected.
    """
    ops = [parse_operation(o) if isinstance(o, dict) else o for o in operations]
    work = model.model_copy(deep=True)
    changes, aliases = Change(), {}
    created: list[str | None] = []
    semantic = False
    for i, op in enumerate(ops):
        try:
            created.append(_HANDLERS[op.op](work, op, catalogue, aliases, changes))
        except _Fail as f:
            raise OperationError(i, op.op, f.issues) from None
        if op.op != "move_node":
            semantic = True
    if semantic:
        work.version += 1
    return ApplyResult(model=work, changes=changes, semantic=semantic, created_ids=created)
