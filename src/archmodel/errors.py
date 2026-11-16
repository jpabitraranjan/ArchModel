"""Shared issue / error types."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Issue(BaseModel):
    """A single validation finding. Used for both operation errors and design lint."""

    code: str
    message: str
    severity: Literal["error", "warning", "info"] = "error"
    ref: str | None = None  # node / edge / requirement id the issue is about


class OperationError(Exception):
    """Raised when an operation batch is rejected. The batch is atomic: nothing is applied."""

    def __init__(self, op_index: int, op_name: str, issues: list[Issue]):
        self.op_index = op_index
        self.op_name = op_name
        self.issues = issues
        detail = "; ".join(f"[{i.code}] {i.message}" for i in issues)
        super().__init__(f"operation #{op_index} ({op_name}) rejected: {detail}")
