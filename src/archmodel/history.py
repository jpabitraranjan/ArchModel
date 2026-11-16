"""In-memory edit history for an editing session (undo / redo).

This is the *editor's* history. Durable, immutable versions and approvals belong to the design
artifact repository, which will store the same operation batches.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from .catalogue import DEFAULT_CATALOGUE, Catalogue
from .model import ArchitectureModel
from .operations import ApplyResult, Change, Operation, apply_operations


class HistoryEntry(BaseModel):
    version: int
    author: str
    message: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    operations: list[dict[str, Any]]
    changes: Change


class ModelHistory:
    def __init__(self, initial: ArchitectureModel | None = None, catalogue: Catalogue = DEFAULT_CATALOGUE):
        self.catalogue = catalogue
        self._states: list[ArchitectureModel] = [initial or ArchitectureModel()]
        self._entries: list[HistoryEntry] = []
        self._cursor = 0

    @property
    def current(self) -> ArchitectureModel:
        return self._states[self._cursor]

    @property
    def entries(self) -> list[HistoryEntry]:
        return list(self._entries[: self._cursor])

    @property
    def can_undo(self) -> bool:
        return self._cursor > 0

    @property
    def can_redo(self) -> bool:
        return self._cursor < len(self._states) - 1

    def apply(self, operations: list[Operation | dict[str, Any]], author: str = "user",
              message: str | None = None) -> ApplyResult:
        result = apply_operations(self.current, operations, self.catalogue)
        del self._states[self._cursor + 1:]  # a new edit discards the redo stack
        del self._entries[self._cursor:]
        self._states.append(result.model)
        self._entries.append(HistoryEntry(
            version=result.model.version, author=author, message=message,
            operations=[o if isinstance(o, dict) else o.model_dump(mode="json", exclude_none=True)
                        for o in operations],
            changes=result.changes))
        self._cursor += 1
        return result

    def undo(self) -> ArchitectureModel:
        if not self.can_undo:
            raise IndexError("nothing to undo")
        self._cursor -= 1
        return self.current

    def redo(self) -> ArchitectureModel:
        if not self.can_redo:
            raise IndexError("nothing to redo")
        self._cursor += 1
        return self.current
