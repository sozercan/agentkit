from __future__ import annotations

import pytest
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
            "apiVersion": "core.orka.ai/v1alpha1",
            "kind": "Tool",
            "metadata": {"name": "dispatch-work-order"},
            "spec": {
                "description": "Dispatch a field tech.",
                "brokeredToolClass": "write",
                "parameters": {"type": "object", "properties": {"incident": {"type": "string"}}, "required": ["incident"]},
                "http": {
                    "url": "http://tool.default.svc.cluster.local",
                    "method": "POST",
                    "authSecretRef": {"name": "do-not-export", "key": "token"},
                },
            },
        },
        {
            "apiVersion": "core.orka.ai/v1alpha1",
            "kind": "Tool",
            "metadata": {"name": "check-network-telemetry"},
            "spec": {
                "description": "Read telemetry.",
                "brokeredToolClass": "read",
                "parameters": {"type": "object", "properties": {"site": {"type": "string"}}},
                "http": {
                    "url": "http://tool.default.svc.cluster.local",
                    "method": "POST",
                    "headers": {"Authorization": "do-not-export"},
                },
            },
        },
    ]


def test_generate_brokered_tools_uses_canonical_orka_brokered_tool_class():
    documents = [
        {
            "apiVersion": "core.orka.ai/v1alpha1",
            "kind": "Tool",
            "metadata": {"name": "read-telemetry"},
            "spec": {
                "description": "Read telemetry.",
                "brokeredToolClass": "read",
                "parameters": {"type": "object"},
                "http": {"url": "http://tools.default.svc/read", "method": "POST"},
            },
        },
        {
            "apiVersion": "core.orka.ai/v1alpha1",
            "kind": "Tool",
            "metadata": {"name": "dispatch-work-order"},
            "spec": {
                "description": "Dispatch a work order.",
                "brokeredToolClass": "write",
                "parameters": {"type": "object"},
                "http": {"url": "http://tools.default.svc/write", "method": "POST"},
            },
        },
        {
            "apiVersion": "core.orka.ai/v1alpha1",
            "kind": "Tool",
            "metadata": {"name": "coordinate-response"},
            "spec": {
                "description": "Coordinate incident response.",
                "brokeredToolClass": "coordination",
                "parameters": {"type": "object"},
                "http": {"url": "http://tools.default.svc/coordinate", "method": "POST"},
            },
        },
    ]

    generated = generate_brokered_tools_from_orka_tool_crds(documents, include_digest=False)

    assert [(tool["name"], tool["brokeredClass"]) for tool in generated] == [
        ("coordinate-response", "coordination"),
        ("dispatch-work-order", "write"),
        ("read-telemetry", "read"),
    ]


def test_generate_brokered_tools_rejects_unsupported_orka_api_version():
    document = {
        "apiVersion": "core.orka.ai/v1",
        "kind": "Tool",
        "metadata": {"name": "read-telemetry"},
        "spec": {
            "description": "Read telemetry.",
            "brokeredToolClass": "read",
            "parameters": {"type": "object"},
            "http": {"url": "http://tools.default.svc/read", "method": "POST"},
        },
    }

    with pytest.raises(ValueError, match=r"unsupported Orka Tool GVK.*core\.orka\.ai/v1alpha1.*Tool"):
        generate_brokered_tools_from_orka_tool_crds([document])


def test_generate_brokered_tools_rejects_unsupported_orka_kind():
    document = {
        "apiVersion": "core.orka.ai/v1alpha1",
        "kind": "AgentRuntime",
        "metadata": {"name": "runtime"},
        "spec": {},
    }

    with pytest.raises(ValueError, match=r"unsupported Orka Tool GVK.*AgentRuntime.*Tool"):
        generate_brokered_tools_from_orka_tool_crds([document])


def test_generate_brokered_tools_rejects_non_object_documents():
    valid_tool = {
        "apiVersion": "core.orka.ai/v1alpha1",
        "kind": "Tool",
        "metadata": {"name": "read-telemetry"},
        "spec": {
            "description": "Read telemetry.",
            "brokeredToolClass": "read",
            "parameters": {"type": "object"},
            "http": {"url": "http://tools.default.svc/read", "method": "POST"},
        },
    }

    with pytest.raises(ValueError, match=r"document 0 must be an object"):
        generate_brokered_tools_from_orka_tool_crds(["not-a-resource", valid_tool])


def test_generate_brokered_tools_skips_unclassified_orka_tools():
    documents = [
        {
            "apiVersion": "core.orka.ai/v1alpha1",
            "kind": "Tool",
            "metadata": {"name": "local-only-tool"},
            "spec": {
                "description": "Run only inside Orka.",
                "parameters": {"type": "object"},
                "http": {"url": "http://tools.default.svc/local", "method": "POST"},
            },
        },
        {
            "apiVersion": "core.orka.ai/v1alpha1",
            "kind": "Tool",
            "metadata": {"name": "read-telemetry"},
            "spec": {
                "description": "Read telemetry.",
                "brokeredToolClass": "read",
                "parameters": {"type": "object"},
                "http": {"url": "http://tools.default.svc/read", "method": "POST"},
            },
        },
    ]

    generated = generate_brokered_tools_from_orka_tool_crds(documents, include_digest=False)

    assert [(tool["name"], tool["brokeredClass"]) for tool in generated] == [("read-telemetry", "read")]


def test_generate_brokered_tools_rejects_unknown_canonical_brokered_tool_class():
    document = {
        "apiVersion": "core.orka.ai/v1alpha1",
        "kind": "Tool",
        "metadata": {"name": "admin-tool"},
        "spec": {
            "description": "Attempt an unsupported operation.",
            "brokeredToolClass": "admin",
            "parameters": {"type": "object"},
            "http": {"url": "http://tools.default.svc/admin", "method": "POST"},
        },
    }

    with pytest.raises(ValueError, match=r"spec\.brokeredToolClass.*read.*write.*coordination"):
        generate_brokered_tools_from_orka_tool_crds([document])


def test_generate_brokered_tools_rejects_input_without_brokered_tools():
    document = {
        "apiVersion": "core.orka.ai/v1alpha1",
        "kind": "Tool",
        "metadata": {"name": "local-only-tool"},
        "spec": {
            "description": "Run only inside Orka.",
            "brokeredClass": "read",
            "parameters": {"type": "object"},
            "http": {"url": "http://tools.default.svc/local", "method": "POST"},
        },
    }

    with pytest.raises(ValueError, match=r"no brokered Orka Tool CRDs.*spec\.brokeredToolClass"):
        generate_brokered_tools_from_orka_tool_crds([document])


def test_generate_brokered_tools_never_lets_legacy_aliases_override_canonical_fields():
    document = {
        "apiVersion": "core.orka.ai/v1alpha1",
        "kind": "Tool",
        "metadata": {"name": "canonical-tool"},
        "spec": {
            "name": "legacy-tool",
            "description": "Canonical description.",
            "summary": "Legacy description.",
            "brokeredToolClass": "write",
            "brokeredClass": "read",
            "parameters": {"type": "object", "properties": {"canonical": {"type": "string"}}},
            "inputSchema": {"type": "object", "properties": {"legacy": {"type": "string"}}},
            "http": {"url": "http://tools.default.svc/canonical", "method": "POST"},
        },
    }

    generated = generate_brokered_tools_from_orka_tool_crds([document], include_digest=False)

    assert generated == [
        {
            "name": "canonical-tool",
            "description": "Canonical description.",
            "brokeredClass": "write",
            "parameters": {"properties": {"canonical": {"type": "string"}}, "type": "object"},
        }
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


def test_load_orka_tool_crd_files_does_not_silently_drop_invalid_documents(tmp_path):
    src = tmp_path / "tools.yaml"
    src.write_text("---\nnot-a-resource\n---\n" + yaml.safe_dump(_tool_docs()[0]), encoding="utf-8")

    with pytest.raises(ValueError, match=r"document 0 must be an object"):
        load_orka_tool_crd_files([src])


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


def test_brokered_tools_export_cli_rejects_inputs_without_brokered_tools(tmp_path, capsys):
    src = tmp_path / "tools.yaml"
    out = tmp_path / "brokered-tools.yaml"
    document = _tool_docs()[0]
    document["spec"].pop("brokeredToolClass")
    src.write_text(yaml.safe_dump(document), encoding="utf-8")

    code = main([str(src), "--output", str(out)])

    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "no brokered Orka Tool CRDs" in captured.err
    assert not out.exists()


def test_brokered_tools_export_cli_reports_validation_errors(tmp_path, capsys):
    src = tmp_path / "bad.yaml"
    src.write_text(
        yaml.safe_dump(
            {
                "apiVersion": "core.orka.ai/v1alpha1",
                "kind": "Tool",
                "metadata": {"name": "bad"},
                "spec": {
                    "description": "bad",
                    "brokeredToolClass": "read",
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
