"""Fixed, non-sensitive outcomes from Orka's governed tool broker."""

from __future__ import annotations

from collections.abc import Mapping


_ORKA_TOOL_ERRORS = {
    "approval_declined": "The tool call was declined.",
    "approval_expired": "The tool approval expired.",
    "approval_cancelled": "The tool call was cancelled.",
    "approval_stale": "The tool approval is no longer valid.",
    "tool_execution_failed": "MCP tool execution failed.",
    "tool_outcome_unknown": "The tool execution outcome is unknown; do not retry.",
}


def orka_tool_error_details(value: object) -> tuple[str, str] | None:
    """Read an explicit broker code without forwarding tool-controlled text."""
    if not isinstance(value, Mapping):
        return None
    code = value.get("code")
    if not isinstance(code, str):
        return None
    message = _ORKA_TOOL_ERRORS.get(code)
    return (code, message) if message is not None else None
