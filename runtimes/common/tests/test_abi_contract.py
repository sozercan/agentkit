from __future__ import annotations

import json
import math
import shutil
import subprocess
import textwrap
from pathlib import Path

from _brokered_description_cases import HARMLESS_BROKERED_DESCRIPTIONS, UNSAFE_BROKERED_DESCRIPTIONS
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
    assert tool.schema_digest == "sha256:ce77aaf228491b5007ed2ee703e57180acec8def6214c84d4324719b7f4f1fb6"


def test_current_go_validation_and_renderer_match_python_brokered_description_contract(tmp_path):
    repo = Path(__file__).resolve().parents[3]
    go = shutil.which("go")
    assert go is not None, "Go is required for the cross-language ABI contract test"
    descriptions = [*HARMLESS_BROKERED_DESCRIPTIONS, *UNSAFE_BROKERED_DESCRIPTIONS]
    expected_validity = [True] * len(HARMLESS_BROKERED_DESCRIPTIONS) + [False] * len(
        UNSAFE_BROKERED_DESCRIPTIONS
    )

    source = tmp_path / "render_brokered_descriptions.go"
    source.write_text(
        textwrap.dedent(
            """
            package main

            import (
                "encoding/json"
                "fmt"
                "log"
                "os"

                "github.com/sozercan/agentkit/pkg/agentkit/abi"
                "github.com/sozercan/agentkit/pkg/agentkit/config"
                "github.com/sozercan/agentkit/pkg/agentkit/effective"
            )

            type result struct {
                Valid bool `json:"valid"`
                Rendered string `json:"rendered,omitempty"`
                Error string `json:"error,omitempty"`
            }

            func main() {
                var descriptions []string
                if err := json.NewDecoder(os.Stdin).Decode(&descriptions); err != nil {
                    log.Fatal(err)
                }
                results := make([]result, 0, len(descriptions))
                for index, description := range descriptions {
                    cfg := &config.AgentConfig{
                        APIVersion: "v1alpha1",
                        Kind: "Agent",
                        Metadata: config.Metadata{Name: fmt.Sprintf("description-regression-%d", index)},
                        Model: config.Model{
                            Provider: "openai-compatible",
                            BaseURL: "https://model.example/v1",
                            Name: "test-model",
                        },
                        Instructions: config.Source{Inline: "Test brokered descriptions."},
                        BrokeredTools: []config.BrokeredTool{{
                            Name: "safe_lookup",
                            Description: description,
                            BrokeredClass: config.BrokeredClassRead,
                            Parameters: map[string]any{"type": "object"},
                        }},
                        Expose: config.Expose{OpenAI: true, Port: 8080},
                    }
                    if err := cfg.Validate(); err != nil {
                        results = append(results, result{Valid: false, Error: err.Error()})
                        continue
                    }
                    rendered, err := abi.Render(effective.FromConfig(cfg, cfg.Instructions.Inline))
                    if err != nil {
                        log.Fatal(err)
                    }
                    results = append(results, result{Valid: true, Rendered: string(rendered)})
                }
                if err := json.NewEncoder(os.Stdout).Encode(results); err != nil {
                    log.Fatal(err)
                }
            }
            """
        ),
        encoding="utf-8",
    )

    rendered = subprocess.run(
        [go, "run", str(source)],
        cwd=repo,
        check=False,
        capture_output=True,
        input=json.dumps(descriptions),
        text=True,
    )
    assert rendered.returncode == 0, rendered.stderr
    results = json.loads(rendered.stdout)
    assert [result["valid"] for result in results] == expected_validity

    for index, description in enumerate(HARMLESS_BROKERED_DESCRIPTIONS):
        golden = tmp_path / f"go-rendered-agent-{index}.yaml"
        golden.write_text(results[index]["rendered"], encoding="utf-8")
        spec = load(golden)
        assert [tool.description for tool in spec.brokered_tools] == [description]
