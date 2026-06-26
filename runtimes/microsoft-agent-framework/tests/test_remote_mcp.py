from __future__ import annotations

import asyncio
import os
from unittest import mock

from httpx import Request

from agentkit_serve import agent_factory
from agentkit_serve_common.config import ToolSpec


def _remote_tool(auth: dict | None = None) -> ToolSpec:
    data = {
        "name": "toolbox",
        "type": "mcp",
        "transport": "streamable-http",
        "urlEnv": "TOOLBOX_ENDPOINT",
        "headers": [{"name": "Foundry-Features", "value": "Toolboxes=V1Preview"}],
    }
    if auth is not None:
        data["auth"] = auth
    return ToolSpec.model_validate(data)


def test_build_streamable_http_mcp_tool_resolves_url_headers_and_auth():
    tool = _remote_tool({"type": "bearer", "tokenEnv": "TOOLBOX_TOKEN"})
    with mock.patch.dict(
        os.environ,
        {"TOOLBOX_ENDPOINT": "http://127.0.0.1:8765/mcp", "TOOLBOX_TOKEN": "tok"},
        clear=True,
    ):
        mcp_tool = agent_factory.build_tool(tool)

        assert mcp_tool is not None
        assert mcp_tool._httpx_client is not None
        assert "Authorization" not in mcp_tool._httpx_client.headers
        assert mcp_tool._httpx_client.timeout.connect == 120
        assert mcp_tool._httpx_client.follow_redirects is False

        hook = mcp_tool._httpx_client.event_hooks["request"][0]
        request = Request("GET", "http://127.0.0.1:8765/mcp")
        asyncio.run(hook(request))
        assert request.headers["Authorization"] == "Bearer tok"
        assert request.headers["Foundry-Features"] == "Toolboxes=V1Preview"

        redirected = Request("GET", "https://evil.example/mcp")
        asyncio.run(hook(redirected))
        assert "Authorization" not in redirected.headers
        assert "Foundry-Features" not in redirected.headers


def test_build_streamable_http_mcp_tool_refreshes_workload_headers(monkeypatch):
    tool = _remote_tool({"type": "workload-identity-token", "audience": "https://ai.azure.com/.default"})
    monkeypatch.setenv("TOOLBOX_ENDPOINT", "http://127.0.0.1:8765/mcp")
    monkeypatch.setenv("AGENTKIT_WORKLOAD_IDENTITY_TOKEN", "first-token")

    mcp_tool = agent_factory.build_tool(tool)

    assert mcp_tool._httpx_client is not None
    assert "Authorization" not in mcp_tool._httpx_client.headers
    assert mcp_tool._httpx_client.timeout.connect == 120
    assert mcp_tool._httpx_client.follow_redirects is False

    hook = mcp_tool._httpx_client.event_hooks["request"][0]
    request = Request("GET", "http://127.0.0.1:8765/mcp")
    asyncio.run(hook(request))
    assert request.headers["Authorization"] == "Bearer first-token"

    monkeypatch.setenv("AGENTKIT_WORKLOAD_IDENTITY_TOKEN", "second-token")
    refreshed = Request("GET", "http://127.0.0.1:8765/mcp")
    asyncio.run(hook(refreshed))
    assert refreshed.headers["Authorization"] == "Bearer second-token"
