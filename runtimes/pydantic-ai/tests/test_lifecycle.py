from __future__ import annotations

import asyncio
import os
import signal
import sys
import time
from pathlib import Path

import pytest

from agentkit_serve import agent_factory
from agentkit_serve_common.config import ToolSpec


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def _wait_for_pid(pid_path: Path, *, timeout: float = 3.0) -> int:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if pid_path.exists():
            return int(pid_path.read_text(encoding="utf-8"))
        await asyncio.sleep(0.01)
    raise AssertionError("stdio MCP child did not publish its PID")


async def _wait_for_process_exit(pid: int, *, timeout: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if not _process_exists(pid):
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"stdio MCP child process {pid} was not cleaned up")


def test_stdio_tool_call_honors_mcp_timeout_and_cleans_up_child(monkeypatch, tmp_path):
    server_script = tmp_path / "blocked_mcp_server.py"
    pid_path = tmp_path / "blocked_mcp_server.pid"
    server_script.write_text(
        """
import json
import os
import sys
import time
from pathlib import Path

Path(sys.argv[1]).write_text(str(os.getpid()), encoding="utf-8")

for line in sys.stdin:
    message = json.loads(line)
    method = message.get("method")
    if method == "initialize":
        response = {
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {
                "protocolVersion": message["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "blocked-tool-test", "version": "1.0"},
            },
        }
        print(json.dumps(response), flush=True)
    elif method == "tools/list":
        response = {
            "jsonrpc": "2.0",
            "id": message["id"],
            "result": {
                "tools": [
                    {
                        "name": "block_forever",
                        "description": "Never returns",
                        "inputSchema": {"type": "object", "properties": {}},
                    }
                ]
            },
        }
        print(json.dumps(response), flush=True)
    elif method == "tools/call":
        while True:
            time.sleep(3600)
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("AGENTKIT_MCP_TIMEOUT", "0.3")
    tool = ToolSpec(
        name="blocked",
        command=[sys.executable, str(server_script), str(pid_path)],
        env=[],
    )
    child_pid: int | None = None

    async def exercise() -> None:
        nonlocal child_pid
        toolset = agent_factory.build_tool_server(tool)
        wrapped = toolset.wrapped
        async with wrapped:
            child_pid = await _wait_for_pid(pid_path)
            started = time.monotonic()
            with pytest.raises(Exception):
                await asyncio.wait_for(
                    wrapped.direct_call_tool("block_forever", {}),
                    timeout=2.0,
                )
            assert time.monotonic() - started < 1.2
            assert _process_exists(child_pid)

        await _wait_for_process_exit(child_pid)

    try:
        asyncio.run(exercise())
    finally:
        if child_pid is not None and _process_exists(child_pid):
            os.kill(child_pid, signal.SIGKILL)


def test_legacy_stdio_server_receives_init_and_read_timeout(monkeypatch):
    captured: dict[str, object] = {}

    class _LegacyServer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(agent_factory, "MCPServerStdio", _LegacyServer)
    monkeypatch.setenv("AGENTKIT_MCP_TIMEOUT", "1.25")

    server = agent_factory.build_tool_server(
        ToolSpec(name="legacy", command=["legacy-mcp", "--stdio"], env=[])
    )

    assert isinstance(server, _LegacyServer)
    assert captured["timeout"] == 1.25
    assert captured["read_timeout"] == 1.25
