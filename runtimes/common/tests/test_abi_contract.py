from __future__ import annotations

from pathlib import Path

from agentkit_serve_common.config import load


def test_go_rendered_agent_yaml_golden_loads_in_python_reader():
    repo = Path(__file__).resolve().parents[3]
    golden = repo / "pkg" / "agentkit" / "abi" / "testdata" / "agent.yaml"

    spec = load(golden)

    assert spec.abi_version == "v0"
    assert spec.metadata.name == "acme-support"
    assert spec.model.provider == "openai-compatible"
    assert spec.model.base_url == "https://api.openai.com/v1"
    assert spec.model.api_key_env == "OPENAI_API_KEY"
    assert spec.instructions == "Be helpful and cite sources."
    assert len(spec.tools) == 1
    assert spec.tools[0].name == "fetch"
    assert spec.tools[0].command == ["uvx", "mcp-server-fetch"]
    assert spec.tools[0].env == ["FETCH_TIMEOUT"]
    assert spec.env == []
    assert spec.expose.openai is True
    assert spec.expose.port == 8080
