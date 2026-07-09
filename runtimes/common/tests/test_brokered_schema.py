from __future__ import annotations

import yaml

from agentkit_serve_common.brokered import (
    brokered_tool_definitions,
    generate_brokered_tools_from_orka_tool_crds,
    load_orka_tool_crd_files,
    main,
    render_brokered_tools_yaml,
)
from agentkit_serve_common.config import AgentSpec, brokered_tool_schema_digest
from agentkit_serve_common.runtime import BrokeredToolDefinition


def _tool_docs() -> list[dict]:
    return [
        {
            "apiVersion": "orka.example/v1",
            "kind": "Tool",
            "metadata": {"name": "dispatch-work-order"},
            "spec": {
                "description": "Dispatch a field tech.",
                "brokeredClass": "write",
                "parameters": {"type": "object", "properties": {"incident": {"type": "string"}}, "required": ["incident"]},
                "url": "http://tool.default.svc.cluster.local",
                "secretRef": {"name": "do-not-export"},
            },
        },
        {
            "apiVersion": "orka.example/v1",
            "kind": "Tool",
            "metadata": {"name": "check-network-telemetry"},
            "spec": {
                "description": "Read telemetry.",
                "brokeredClass": "read",
                "inputSchema": {"type": "object", "properties": {"site": {"type": "string"}}},
                "headers": {"Authorization": "do-not-export"},
            },
        },
    ]


def test_generate_brokered_tools_from_orka_tool_crds_is_deterministic_and_schema_only():
    generated = generate_brokered_tools_from_orka_tool_crds(_tool_docs())

    assert [tool["name"] for tool in generated] == ["check-network-telemetry", "dispatch-work-order"]
    assert all(set(tool) == {"name", "description", "brokeredClass", "parameters", "schemaDigest"} for tool in generated)
    assert generated[0]["schemaDigest"] == brokered_tool_schema_digest(
        name="check-network-telemetry",
        description="Read telemetry.",
        brokered_class="read",
        parameters={"properties": {"site": {"type": "string"}}, "type": "object"},
    )


def test_generate_brokered_tools_can_omit_digest():
    generated = generate_brokered_tools_from_orka_tool_crds(_tool_docs(), include_digest=False)

    assert all("schemaDigest" not in entry for entry in generated)


def test_render_brokered_tools_yaml_outputs_agent_yaml_fragment():
    generated = generate_brokered_tools_from_orka_tool_crds(_tool_docs(), include_digest=False)

    rendered = render_brokered_tools_yaml(generated)

    parsed = yaml.safe_load(rendered)
    assert list(parsed) == ["brokeredTools"]
    assert [entry["name"] for entry in parsed["brokeredTools"]] == ["check-network-telemetry", "dispatch-work-order"]
    assert "url" not in rendered
    assert "secretRef" not in rendered
    assert "Authorization" not in rendered


def test_load_orka_tool_crd_files_rejects_duplicate_exported_names(tmp_path):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text(yaml.safe_dump(_tool_docs()[0]), encoding="utf-8")
    second.write_text(yaml.safe_dump(_tool_docs()[0]), encoding="utf-8")

    try:
        load_orka_tool_crd_files([first, second])
    except ValueError as exc:
        assert "duplicate brokered tool names" in str(exc)
    else:  # pragma: no cover - assertion path.
        raise AssertionError("expected duplicate brokered tool names to fail")


def test_brokered_tools_export_cli_writes_safe_fragment(tmp_path, capsys):
    src = tmp_path / "tools.yaml"
    out = tmp_path / "brokered-tools.yaml"
    src.write_text("---\n" + yaml.safe_dump(_tool_docs()[0]) + "---\n" + yaml.safe_dump(_tool_docs()[1]), encoding="utf-8")

    code = main([str(src), "--no-digest", "--output", str(out)])

    assert code == 0
    assert capsys.readouterr().out == ""
    parsed = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert [entry["name"] for entry in parsed["brokeredTools"]] == ["check-network-telemetry", "dispatch-work-order"]
    assert all("schemaDigest" not in entry for entry in parsed["brokeredTools"])


def test_brokered_tools_export_cli_reports_validation_errors(tmp_path, capsys):
    src = tmp_path / "bad.yaml"
    src.write_text(
        yaml.safe_dump(
            {
                "kind": "Tool",
                "metadata": {"name": "bad"},
                "spec": {
                    "description": "bad",
                    "brokeredClass": "read",
                    "parameters": {"type": "object", "properties": {"tokenValue": {"type": "string"}}},
                },
            }
        ),
        encoding="utf-8",
    )

    code = main([str(src)])

    assert code == 2
    assert "agentkit-brokered-tools:" in capsys.readouterr().err


def test_brokered_tool_definitions_preserve_only_safe_runtime_fields():
    spec = AgentSpec.model_validate(
        {
            "abiVersion": "v0",
            "metadata": {"name": "brokered-test"},
            "model": {"provider": "openai-compatible", "baseURL": "https://api.openai.com/v1", "name": "gpt-4o-mini"},
            "instructions": "Be helpful.",
            "tools": [],
            "brokeredTools": [
                {
                    "name": "check-network-telemetry",
                    "description": "Read telemetry.",
                    "brokeredClass": "read",
                    "parameters": {"type": "object"},
                }
            ],
            "expose": {"openai": True, "port": 8080},
        }
    )

    assert brokered_tool_definitions(spec) == [
        BrokeredToolDefinition(
            name="check-network-telemetry",
            description="Read telemetry.",
            brokered_class="read",
            parameters={"type": "object"},
        )
    ]
