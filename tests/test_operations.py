import pytest

from archmodel import ArchitectureModel, OperationError, apply_operations


def add(m, type_, name, **kw):
    r = apply_operations(m, [{"op": "add_node", "type": type_, "name": name, **kw}])
    return r.model, r.created_ids[0]


def codes(exc: OperationError) -> set[str]:
    return {i.code for i in exc.value.issues}


def test_add_node_fills_defaults_and_assigns_stable_ids():
    m, db = add(ArchitectureModel(), "database", "Orders DB", properties={"engine": "postgresql"})
    assert db == "DB-001"
    assert m.nodes[db].properties == {"engine": "postgresql", "replication": "none"}
    m, db2 = add(m, "database", "Users DB", properties={"engine": "mysql"})
    assert db2 == "DB-002"
    assert m.version == 2


def test_input_model_is_never_mutated():
    m0 = ArchitectureModel()
    apply_operations(m0, [{"op": "add_node", "type": "cache", "name": "c"}])
    assert not m0.nodes and m0.version == 0


@pytest.mark.parametrize("props,code", [
    ({}, "invalid_property"),                          # engine is required
    ({"engine": "oracle"}, "invalid_property"),        # not in choices
    ({"engine": "mysql", "bogus": 1}, "invalid_property"),
])
def test_property_validation(props, code):
    with pytest.raises(OperationError) as e:
        apply_operations(ArchitectureModel(), [{"op": "add_node", "type": "database", "name": "d", "properties": props}])
    assert code in codes(e)


def test_unknown_type_and_empty_name():
    with pytest.raises(OperationError) as e:
        apply_operations(ArchitectureModel(), [{"op": "add_node", "type": "toaster", "name": "x"}])
    assert "unknown_component_type" in codes(e)
    with pytest.raises(OperationError) as e:
        apply_operations(ArchitectureModel(), [{"op": "add_node", "type": "cache", "name": "  "}])
    assert "empty_name" in codes(e)


def test_connection_rules_enforced():
    m, db = add(ArchitectureModel(), "database", "DB", properties={"engine": "postgresql"})
    m, ui = add(m, "ui_client", "Web")
    m, svc = add(m, "api_service", "API")
    with pytest.raises(OperationError) as e:   # a database cannot call a UI client
        apply_operations(m, [{"op": "connect", "type": "http_call", "source": db, "target": ui}])
    assert "illegal_connection" in codes(e)
    with pytest.raises(OperationError) as e:   # UI cannot talk to a database directly
        apply_operations(m, [{"op": "connect", "type": "db_connection", "source": ui, "target": db}])
    assert "illegal_connection" in codes(e)
    r = apply_operations(m, [{"op": "connect", "type": "db_connection", "source": svc, "target": db}])
    assert r.created_ids == ["E-001"]


def test_duplicate_and_self_edges_rejected():
    m, a = add(ArchitectureModel(), "api_service", "A")
    m, b = add(m, "api_service", "B")
    m = apply_operations(m, [{"op": "connect", "type": "http_call", "source": a, "target": b}]).model
    with pytest.raises(OperationError) as e:
        apply_operations(m, [{"op": "connect", "type": "http_call", "source": a, "target": b}])
    assert "duplicate_edge" in codes(e)
    with pytest.raises(OperationError) as e:
        apply_operations(m, [{"op": "connect", "type": "http_call", "source": a, "target": a}])
    assert "self_connection" in codes(e)


def test_batch_is_atomic_and_aliases_work():
    m = ArchitectureModel()
    good = [
        {"op": "add_node", "type": "api_service", "name": "API", "ref": "api"},
        {"op": "add_node", "type": "cache", "name": "Cache", "ref": "c"},
        {"op": "connect", "type": "cache_access", "source": "@api", "target": "@c"},
    ]
    r = apply_operations(m, good)
    assert r.created_ids == ["SVC-001", "CACHE-001", "E-001"] and r.model.version == 1
    bad = good + [{"op": "connect", "type": "db_connection", "source": "@api", "target": "@c"}]
    with pytest.raises(OperationError) as e:
        apply_operations(m, bad)
    assert e.value.op_index == 3
    assert not m.nodes                          # nothing leaked into the original
    with pytest.raises(OperationError) as e:
        apply_operations(m, [{"op": "connect", "type": "http_call", "source": "@nope", "target": "@nope"}])
    assert "unknown_ref" in codes(e)


def test_lld_needs_valid_parent():
    m, db = add(ArchitectureModel(), "database", "DB", properties={"engine": "postgresql"})
    m, svc = add(m, "api_service", "API")
    with pytest.raises(OperationError) as e:
        apply_operations(m, [{"op": "add_node", "type": "class", "name": "OrderService"}])
    assert "lld_requires_parent" in codes(e)
    with pytest.raises(OperationError) as e:    # a class cannot live inside a database
        apply_operations(m, [{"op": "add_node", "type": "class", "name": "X", "parent_id": db}])
    assert "invalid_parent_type" in codes(e)
    with pytest.raises(OperationError) as e:    # HLD components cannot be nested
        apply_operations(m, [{"op": "add_node", "type": "cache", "name": "c", "parent_id": svc}])
    assert "hld_cannot_have_parent" in codes(e)


def test_lld_cross_component_edge_requires_hld_edge():
    m = ArchitectureModel()
    m = apply_operations(m, [
        {"op": "add_node", "type": "api_service", "name": "API", "ref": "svc"},
        {"op": "add_node", "type": "database", "name": "DB", "ref": "db", "properties": {"engine": "postgresql"}},
        {"op": "add_node", "type": "class", "name": "OrderRepo", "parent_id": "@svc", "ref": "repo"},
        {"op": "add_node", "type": "table", "name": "orders", "parent_id": "@db", "ref": "tbl"},
    ]).model
    lld_edge = {"op": "connect", "type": "reads_writes", "source": "CLS-001", "target": "TBL-001"}
    with pytest.raises(OperationError) as e:
        apply_operations(m, [lld_edge])
    assert "lld_edge_without_hld_edge" in codes(e)
    r = apply_operations(m, [{"op": "connect", "type": "db_connection", "source": "SVC-001", "target": "DB-001"}, lld_edge])
    assert len(r.model.edges) == 2


def test_remove_node_cascades():
    m = apply_operations(ArchitectureModel(), [
        {"op": "add_node", "type": "api_service", "name": "API", "ref": "svc"},
        {"op": "add_node", "type": "cache", "name": "C", "ref": "c"},
        {"op": "add_node", "type": "class", "name": "K", "parent_id": "@svc"},
        {"op": "connect", "type": "cache_access", "source": "@svc", "target": "@c"},
    ]).model
    r = apply_operations(m, [{"op": "remove_node", "id": "SVC-001"}])
    assert set(r.changes.removed_nodes) == {"SVC-001", "CLS-001"}
    assert r.changes.removed_edges == ["E-001"]
    assert set(r.model.nodes) == {"CACHE-001"}


def test_update_node_merges_and_unsets():
    m, svc = add(ArchitectureModel(), "api_service", "API", properties={"replicas": 3})
    m = apply_operations(m, [{"op": "update_node", "id": svc, "name": "Orders API", "properties": {"language": "python"}}]).model
    assert m.nodes[svc].properties["replicas"] == 3 and m.nodes[svc].properties["language"] == "python"
    m = apply_operations(m, [{"op": "update_node", "id": svc, "unset": ["replicas"]}]).model
    assert m.nodes[svc].properties["replicas"] == 1          # back to the default
    with pytest.raises(OperationError):
        apply_operations(m, [{"op": "update_node", "id": svc, "properties": {"replicas": 0}}])


def test_move_is_layout_only():
    m, n = add(ArchitectureModel(), "cache", "C")
    r = apply_operations(m, [{"op": "move_node", "id": n, "position": {"x": 10, "y": 20}}])
    assert not r.semantic and r.model.version == m.version and r.model.nodes[n].position.x == 10


def test_requirement_links():
    m, n = add(ArchitectureModel(), "cache", "C")
    m = apply_operations(m, [{"op": "link_requirement", "target_id": n, "requirement_id": "NFR-001"}] * 2).model
    assert m.nodes[n].requirement_ids == ["NFR-001"]           # idempotent
    with pytest.raises(OperationError) as e:
        apply_operations(m, [{"op": "link_requirement", "target_id": n, "requirement_id": "bad id"}])
    assert "bad_requirement_id" in codes(e)
    m = apply_operations(m, [{"op": "link_requirement", "target_id": n, "requirement_id": "NFR-001", "unlink": True}]).model
    assert m.nodes[n].requirement_ids == []


def test_json_roundtrip():
    m, _ = add(ArchitectureModel(name="Shop"), "cache", "C")
    assert ArchitectureModel.from_json(m.to_json()) == m
