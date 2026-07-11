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
    binary_default = tool.parameters["properties"]["binary"]["default"]
    assert isinstance(binary_default, str)
    assert binary_default == "SGVsbG8="
    yaml_sensitive_default = tool.parameters["properties"]["? ask"]["default"]
    assert yaml_sensitive_default == "before\tafter"
    assert tool.parameters["properties"]["<<"]["default"] == "<<"
    assert tool.parameters["properties"]["="]["default"] == "="
    assert tool.parameters["properties"][".inf"]["default"] == ".inf"
    assert tool.parameters["properties"]["12:34:56"]["default"] == "2001-12-14 21:59:43.10 -5"
    minimum = tool.parameters["properties"][property_name]["minimum"]
    assert isinstance(minimum, float)
    assert minimum == 0.0
    assert math.copysign(1.0, minimum) == -1.0
    assert tool.schema_digest == "sha256:50d18ec3547b6ebc50aebf140716b6eac1e54f7eb4e5d84e69abef731ad7af64"
