from __future__ import annotations


def test_mcp_compatibility_symbols_available():
    from mcp import McpError

    assert McpError.__name__ == "McpError"
