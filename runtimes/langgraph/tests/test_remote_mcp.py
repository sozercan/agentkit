from __future__ import annotations

import os
from unittest import mock

from agentkit_serve import agent_factory
from agentkit_serve_common.config import ToolSpec


def test_build_streamable_http_mcp_connection_resolves_url_headers_and_auth():
    tool = ToolSpec.model_validate(
        {
            "name": "toolbox",
            "type": "mcp",
            "transport": "streamable-http",
            "urlEnv": "TOOLBOX_ENDPOINT",
            "headers": [{"name": "Foundry-Features", "value": "Toolboxes=V1Preview"}],
            "auth": {"type": "bearer", "tokenEnv": "TOOLBOX_TOKEN"},
        }
    )
    with mock.patch.dict(
        os.environ,
        {"TOOLBOX_ENDPOINT": "http://127.0.0.1:8765/mcp", "TOOLBOX_TOKEN": "tok"},
        clear=True,
    ):
        conn = agent_factory.build_mcp_connection(tool)

    assert conn["transport"] == "streamable_http"
    assert conn["url"] == "http://127.0.0.1:8765/mcp"
    assert "headers" not in conn
    assert conn["httpx_client_factory"] is not None
