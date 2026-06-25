"""Behavioral regression tests for the LangGraph adapter's OpenAI facade.

The HARD invariants (400 guards, single-completion, auth gate, multi-turn,
framework-agnostic shared core) are the SHARED conformance suite — imported here
so this adapter is held to the exact same contract as every other adapter. The
offline double + spec are supplied by ``conftest.py``.
"""

from __future__ import annotations

# Re-export the shared conformance suite; pytest collects each `test_*` against
# this adapter's `make_client` / `model_name` fixtures from conftest.py.
from agentkit_serve_common.conformance import *  # noqa: F401,F403


def test_unsupported_feature_error_codes(make_client):
    """The shared server returns stable OpenAI-shaped codes for v0 rejections."""
    with make_client() as c:
        base = {"model": "x", "messages": [{"role": "user", "content": "hi"}]}

        r = c.post("/v1/chat/completions", json={**base, "stream": True})
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "stream_unsupported"

        r = c.post(
            "/v1/chat/completions",
            json={**base, "tools": [{"type": "function", "function": {"name": "x"}}]},
        )
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "tools_unsupported"

        r = c.post("/v1/chat/completions", json={**base, "tool_choice": "required"})
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "tool_choice_unsupported"
