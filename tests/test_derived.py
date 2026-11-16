import pytest

from archmodel import (ArchitectureModel, DEFAULT_CATALOGUE, ModelHistory, apply_operations,
                       diff_models, lint, to_mermaid)

SHOP = [
    {"op": "add_node", "type": "ui_client", "name": "Web App", "ref": "ui"},
    {"op": "add_node", "type": "load_balancer", "name": "LB", "ref": "lb"},
    {"op": "add_node", "type": "api_service", "name": "Order API", "ref": "api"},
    {"op": "add_node", "type": "database", "name": "Orders DB", "ref": "db", "properties": {"engine": "postgresql"}},
    {"op": "connect", "type": "http_call", "source": "@ui", "target": "@lb"},
    {"op": "connect", "type": "routes_to", "source": "@lb", "target": "@api"},
    {"op": "connect", "type": "db_connection", "source": "@api", "target": "@db"},
]


def shop():
    return apply_operations(ArchitectureModel(name="Shop"), SHOP).model


def test_mermaid_hld_and_lld_scope():
    m = apply_operations(shop(), [{"op": "add_node", "type": "class", "name": "OrderService", "parent_id": "SVC-001"}]).model
    hld = to_mermaid(m)
    assert hld.startswith("flowchart LR")
    assert 'DB_001[("Orders DB")]' in hld and "LB_001 -->|routes_to| SVC_001" in hld
    assert "OrderService" not in hld                      # LLD elements stay out of the HLD view
    assert "OrderService" in to_mermaid(m, scope="SVC-001")
    with pytest.raises(KeyError):
        to_mermaid(m, scope="NOPE")


def test_lint_findings_and_coverage():
    m = apply_operations(shop(), [{"op": "add_node", "type": "cache", "name": "Lonely cache"}]).model
    found = {i.code for i in lint(m)}
    assert {"orphan_component", "no_requirement_link", "no_lld"} <= found
    m = apply_operations(m, [{"op": "link_requirement", "target_id": "DB-001", "requirement_id": "FR-001"},
                             {"op": "link_requirement", "target_id": "E-003", "requirement_id": "FR-009"}]).model
    issues = lint(m, requirement_ids=["FR-001", "FR-002"])
    by_code = {i.code: i.ref for i in issues if i.code in ("uncovered_requirement", "unknown_requirement")}
    assert by_code == {"uncovered_requirement": "FR-002", "unknown_requirement": "FR-009"}


def test_lint_detects_disconnected_groups():
    m = apply_operations(shop(), [
        {"op": "add_node", "type": "api_service", "name": "Other", "ref": "o"},
        {"op": "add_node", "type": "cache", "name": "C2", "ref": "c"},
        {"op": "connect", "type": "cache_access", "source": "@o", "target": "@c"},
    ]).model
    assert "disconnected_architecture" in {i.code for i in lint(m)}


def test_history_undo_redo_and_entries():
    h = ModelHistory()
    h.apply(SHOP[:1], author="alice", message="add UI")
    h.apply([SHOP[1]], author="agent:hld")
    assert len(h.current.nodes) == 2 and [e.author for e in h.entries] == ["alice", "agent:hld"]
    h.undo()
    assert len(h.current.nodes) == 1 and h.can_redo
    h.redo()
    assert len(h.current.nodes) == 2
    h.undo()
    h.apply([SHOP[2]])                                   # new edit discards the redo stack
    assert not h.can_redo
    with pytest.raises(IndexError):
        ModelHistory().undo()


def test_diff_ignores_layout():
    a = shop()
    b = apply_operations(a, [
        {"op": "move_node", "id": "DB-001", "position": {"x": 5, "y": 5}},
        {"op": "update_node", "id": "SVC-001", "properties": {"replicas": 4}},
        {"op": "add_node", "type": "cache", "name": "C"},
        {"op": "disconnect", "id": "E-003"},
    ]).model
    d = diff_models(a, b)
    assert d.added_nodes == ["CACHE-001"] and d.updated_nodes == ["SVC-001"] and d.removed_edges == ["E-003"]


def test_catalogue_is_self_consistent():
    cat = DEFAULT_CATALOGUE
    for c in cat.connections.values():
        for t in c.sources + c.targets:
            assert t in cat.components, f"{c.key} references unknown type {t}"
        levels = {cat.components[t].level for t in c.sources + c.targets}
        assert levels == {c.level}, f"{c.key} mixes HLD and LLD"
    for c in cat.components.values():
        for p in c.allowed_parents:
            assert p in cat.components
    assert len({c.id_prefix for c in cat.components.values()}) == len(cat.components)
    assert "components" in cat.to_dict()
