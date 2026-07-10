from __future__ import annotations

import math
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


def test_go_rendered_edge_case_agent_yaml_loads_exactly_in_python_reader():
    repo = Path(__file__).resolve().parents[3]
    golden = repo / "pkg" / "agentkit" / "abi" / "testdata" / "edge-cases.yaml"
    line_break_text = "NEL:\u0085LS:\u2028PS:\u2029end"
    property_name = "line:\u2028break"

    spec = load(golden)

    assert spec.instructions == "instructions " + line_break_text
    assert len(spec.brokered_tools) == 1
    tool = spec.brokered_tools[0]
    assert tool.description == "description " + line_break_text
    assert tool.parameters["description"] == "schema " + line_break_text
    assert tool.parameters["properties"][property_name]["description"] == "property " + line_break_text
    minimum = tool.parameters["properties"][property_name]["minimum"]
    assert isinstance(minimum, float)
    assert minimum == 0.0
    assert math.copysign(1.0, minimum) == -1.0
    assert tool.schema_digest == "sha256:c11250fadb3b86c3bdd4ced8fc9741b4fa1295b88f9faadefb56c64bbb986e2f"
