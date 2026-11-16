"""Component catalogue: what can exist in a model and how it may be connected.

The catalogue is data, not code paths. The visual editor reads it (via ``to_dict``) to build
its palette and to grey out illegal connections; the operations engine reads the same data to
enforce them, so the UI and the backend can never disagree.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Level(str, Enum):
    HLD = "hld"
    LLD = "lld"


class PropType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ENUM = "enum"
    STRING_LIST = "string_list"


class PropertySpec(BaseModel):
    name: str
    type: PropType
    required: bool = False
    default: Any = None
    choices: list[str] | None = None
    minimum: int | None = None
    description: str = ""

    def check(self, value: Any) -> str | None:
        """Return an error message if ``value`` is invalid for this property, else None."""
        t = self.type
        if t is PropType.STRING:
            return None if isinstance(value, str) else "must be a string"
        if t is PropType.INTEGER:
            if isinstance(value, bool) or not isinstance(value, int):
                return "must be an integer"
            if self.minimum is not None and value < self.minimum:
                return f"must be >= {self.minimum}"
            return None
        if t is PropType.BOOLEAN:
            return None if isinstance(value, bool) else "must be a boolean"
        if t is PropType.ENUM:
            return None if value in (self.choices or []) else f"must be one of {self.choices}"
        if t is PropType.STRING_LIST:
            ok = isinstance(value, list) and all(isinstance(v, str) for v in value)
            return None if ok else "must be a list of strings"
        return "unsupported property type"


class ComponentType(BaseModel):
    key: str
    label: str
    level: Level
    category: str
    id_prefix: str
    description: str = ""
    properties: list[PropertySpec] = Field(default_factory=list)
    allowed_parents: list[str] = Field(default_factory=list)  # LLD only: HLD types that may own it


class ConnectionType(BaseModel):
    key: str
    label: str
    level: Level
    sources: list[str]
    targets: list[str]
    description: str = ""
    properties: list[PropertySpec] = Field(default_factory=list)

    def allows(self, source_type: str, target_type: str) -> bool:
        return source_type in self.sources and target_type in self.targets


def resolve_properties(
    specs: list[PropertySpec], values: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Validate ``values`` against ``specs``; fill defaults. Returns (resolved, errors)."""
    by_name = {s.name: s for s in specs}
    errors: list[str] = []
    for key in values:
        if key not in by_name:
            errors.append(f"unknown property '{key}'")
    resolved: dict[str, Any] = {}
    for spec in specs:
        if spec.name in values:
            err = spec.check(values[spec.name])
            if err:
                errors.append(f"property '{spec.name}' {err}")
            else:
                resolved[spec.name] = values[spec.name]
        elif spec.default is not None:
            resolved[spec.name] = spec.default
        elif spec.required:
            errors.append(f"property '{spec.name}' is required")
    return resolved, errors


class Catalogue(BaseModel):
    components: dict[str, ComponentType]
    connections: dict[str, ConnectionType]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form for the editor palette."""
        return self.model_dump(mode="json")


# --------------------------------------------------------------------------- default catalogue
def _p(name: str, type_: PropType, **kw: Any) -> PropertySpec:
    return PropertySpec(name=name, type=type_, **kw)


S, I, B, E, L = (PropType.STRING, PropType.INTEGER, PropType.BOOLEAN, PropType.ENUM,
                 PropType.STRING_LIST)

_HLD = Level.HLD
_LLD = Level.LLD
SERVICES = ["api_service", "worker_service"]


def default_catalogue() -> Catalogue:
    comps = [
        ComponentType(key="ui_client", label="UI Client", level=_HLD, category="client", id_prefix="UI",
                      properties=[_p("platform", E, choices=["web", "mobile", "desktop", "cli"], default="web"),
                                  _p("framework", S)]),
        ComponentType(key="cdn", label="CDN", level=_HLD, category="edge", id_prefix="CDN",
                      properties=[_p("provider", S)]),
        ComponentType(key="api_gateway", label="API Gateway", level=_HLD, category="edge", id_prefix="GW",
                      properties=[_p("auth_required", B, default=True), _p("rate_limit_per_min", I, minimum=0)]),
        ComponentType(key="load_balancer", label="Load Balancer", level=_HLD, category="edge", id_prefix="LB",
                      properties=[_p("algorithm", E, choices=["round_robin", "least_connections", "ip_hash", "weighted"],
                                     default="round_robin"),
                                  _p("layer", E, choices=["l4", "l7"], default="l7"),
                                  _p("health_check_path", S, default="/health")]),
        ComponentType(key="api_service", label="API Service", level=_HLD, category="compute", id_prefix="SVC",
                      properties=[_p("language", S), _p("framework", S),
                                  _p("protocol", E, choices=["rest", "graphql", "grpc"], default="rest"),
                                  _p("stateless", B, default=True),
                                  _p("replicas", I, minimum=1, default=1)]),
        ComponentType(key="worker_service", label="Worker Service", level=_HLD, category="compute", id_prefix="WRK",
                      properties=[_p("language", S), _p("concurrency", I, minimum=1, default=1),
                                  _p("replicas", I, minimum=1, default=1)]),
        ComponentType(key="database", label="Database", level=_HLD, category="data", id_prefix="DB",
                      properties=[_p("engine", E, required=True,
                                     choices=["postgresql", "mysql", "mongodb", "dynamodb", "cassandra", "sqlite", "other"]),
                                  _p("replication", E, choices=["none", "primary_replica", "multi_primary", "sharded"],
                                     default="none"),
                                  _p("backup_policy", S)]),
        ComponentType(key="cache", label="Cache", level=_HLD, category="data", id_prefix="CACHE",
                      properties=[_p("engine", E, choices=["redis", "memcached", "in_memory"], default="redis"),
                                  _p("ttl_seconds", I, minimum=0)]),
        ComponentType(key="message_queue", label="Message Queue", level=_HLD, category="messaging", id_prefix="MQ",
                      properties=[_p("engine", E, required=True,
                                     choices=["kafka", "rabbitmq", "sqs", "redis_streams", "other"]),
                                  _p("delivery", E, choices=["at_least_once", "at_most_once", "exactly_once"],
                                     default="at_least_once")]),
        ComponentType(key="object_store", label="Object Store", level=_HLD, category="data", id_prefix="OBJ",
                      properties=[_p("provider", E, choices=["s3", "gcs", "azure_blob", "minio"], default="s3"),
                                  _p("public", B, default=False)]),
        ComponentType(key="auth_provider", label="Auth Provider", level=_HLD, category="security", id_prefix="AUTH",
                      properties=[_p("protocol", E, choices=["oauth2", "oidc", "saml", "api_key"], default="oauth2"),
                                  _p("provider", S)]),
        ComponentType(key="external_system", label="External System", level=_HLD, category="external", id_prefix="EXT",
                      properties=[_p("description", S), _p("protocol", S)]),
        # ---- LLD elements (always live inside an HLD node)
        ComponentType(key="class", label="Class", level=_LLD, category="code", id_prefix="CLS",
                      allowed_parents=SERVICES, properties=[_p("responsibility", S)]),
        ComponentType(key="interface", label="Interface", level=_LLD, category="code", id_prefix="IFC",
                      allowed_parents=SERVICES, properties=[_p("responsibility", S)]),
        ComponentType(key="endpoint", label="API Endpoint", level=_LLD, category="code", id_prefix="EP",
                      allowed_parents=["api_service", "api_gateway"],
                      properties=[_p("method", E, required=True, choices=["GET", "POST", "PUT", "PATCH", "DELETE"]),
                                  _p("path", S, required=True), _p("request_schema", S), _p("response_schema", S),
                                  _p("auth_required", B, default=True)]),
        ComponentType(key="table", label="Table / Collection", level=_LLD, category="data", id_prefix="TBL",
                      allowed_parents=["database"],
                      properties=[_p("columns", L, description="e.g. 'id:uuid:pk', 'email:text:unique'"),
                                  _p("primary_key", S)]),
    ]
    clients = ["ui_client", "external_system", "cdn"]
    conns = [
        ConnectionType(key="http_call", label="HTTP call", level=_HLD,
                       sources=clients + ["api_gateway", "load_balancer"] + SERVICES,
                       targets=["api_gateway", "load_balancer", "cdn", "auth_provider", "external_system"] + SERVICES),
        ConnectionType(key="routes_to", label="Routes to", level=_HLD,
                       sources=["load_balancer", "api_gateway"],
                       targets=["load_balancer", "api_gateway"] + SERVICES),
        ConnectionType(key="db_connection", label="Database connection", level=_HLD,
                       sources=SERVICES, targets=["database"]),
        ConnectionType(key="cache_access", label="Cache access", level=_HLD,
                       sources=SERVICES, targets=["cache"]),
        ConnectionType(key="publishes_to", label="Publishes to", level=_HLD,
                       sources=SERVICES, targets=["message_queue"]),
        ConnectionType(key="consumes_from", label="Consumes from", level=_HLD,
                       sources=SERVICES, targets=["message_queue"]),
        ConnectionType(key="object_access", label="Reads/writes objects", level=_HLD,
                       sources=SERVICES + ["cdn"], targets=["object_store"]),
        ConnectionType(key="authenticates_via", label="Authenticates via", level=_HLD,
                       sources=["api_gateway"] + SERVICES, targets=["auth_provider"]),
        # ---- LLD
        ConnectionType(key="implements", label="Implements", level=_LLD, sources=["class"], targets=["interface"]),
        ConnectionType(key="depends_on", label="Depends on", level=_LLD,
                       sources=["class"], targets=["class", "interface"]),
        ConnectionType(key="handled_by", label="Handled by", level=_LLD, sources=["endpoint"], targets=["class"]),
        ConnectionType(key="reads_writes", label="Reads / writes", level=_LLD, sources=["class"], targets=["table"]),
        ConnectionType(key="references", label="Foreign key to", level=_LLD, sources=["table"], targets=["table"]),
    ]
    return Catalogue(components={c.key: c for c in comps}, connections={c.key: c for c in conns})


DEFAULT_CATALOGUE = default_catalogue()
