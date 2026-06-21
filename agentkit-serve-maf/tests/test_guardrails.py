"""Guardrail regressions for the MAF adapter's non-negotiable invariants.

Two things in this adapter must never silently regress (plan §10, §12):

1. The LOCK-IN BOUNDARY — ``agent_factory.py`` must import only ``agent_framework``
   core + ``agent_framework.openai``. If wiring MAF ever pulls an Azure / Foundry /
   CopilotStudio package, that is the lock-in line and this test fails loudly.
2. The SECRET-BLEED RULE — a tool subprocess receives ONLY the env var NAMES the
   tool declares, never the full container environment (which holds the model key).
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from unittest import mock

from agentkit_serve import agent_factory
from agentkit_serve.config import ToolSpec

# Forbidden import SEGMENTS. A module is forbidden if ANY dotted segment matches
# one of these. This catches BOTH spellings of MAF's cloud integrations:
#   * separate distributions:  agent_framework_azure, agent_framework_foundry, ...
#     (their import name has the root segment "agent_framework_azure", etc.)
#   * FIRST-PARTY SUBMODULES:   agent_framework.azure / .foundry / .microsoft /
#     .copilotstudio — these live INSIDE agent_framework_core and re-export the
#     cloud surface (e.g. agent_framework.microsoft.CopilotStudioAgent). Matching
#     on segments (not just the root) is what catches these; a root-only check
#     would wave them through because their root is the allowed "agent_framework".
_FORBIDDEN_IMPORT_SEGMENTS = (
    "azure",
    "foundry",
    "copilotstudio",
    "copilot_studio",
    "microsoft",  # agent_framework.microsoft re-exports CopilotStudioAgent + Purview
    "agent_framework_azure",
    "agent_framework_foundry",
    "agent_framework_copilotstudio",
    "agent_framework_purview",
)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module)
    return mods


def _is_forbidden(mod: str) -> bool:
    """A module is forbidden if any of its dotted segments is a forbidden token."""
    return any(seg in _FORBIDDEN_IMPORT_SEGMENTS for seg in mod.split("."))


def _is_allowed_framework_module(mod: str) -> bool:
    """An agent_framework* import is allowed ONLY if it is the bare core package or
    the OpenAI client — matched EXACTLY, never as a blanket ``agent_framework.*``
    prefix (which would admit agent_framework.azure/.foundry/.microsoft).
    """
    return mod == "agent_framework" or mod == "agent_framework.openai" or mod.startswith("agent_framework.openai.")


def test_agent_factory_import_boundary():
    """agent_factory imports must not cross into Azure/Foundry/CopilotStudio.

    Covers both the separate-distribution spelling (agent_framework_azure) AND the
    first-party submodule spelling (agent_framework.azure / .microsoft), which a
    future edit is most likely to reach for.
    """
    mods = _imported_modules(Path(agent_factory.__file__))
    for mod in mods:
        assert not _is_forbidden(mod), f"agent_factory.py imports forbidden module {mod!r} (plan §12 lock-in boundary)"


def test_only_minimal_agent_framework_surface():
    """Any agent_framework import must be core or the openai client — not a cloud pkg."""
    mods = _imported_modules(Path(agent_factory.__file__))
    for mod in mods:
        if mod.startswith("agent_framework"):
            assert _is_allowed_framework_module(
                mod
            ), f"agent_factory.py imports non-minimal agent_framework module {mod!r}"


def test_guardrail_actually_rejects_cloud_submodules():
    """Meta-test: the boundary check must REJECT the cloud submodule import forms.

    Without this, the guardrail can silently rot into a false-green (the bare
    'agent_framework' prefix used to admit every agent_framework.* submodule).
    These names are the idiomatic way to pull MAF's cloud surface, so they must
    fail BOTH gates.
    """
    must_reject = [
        "agent_framework.azure",
        "agent_framework.foundry",
        "agent_framework.microsoft",  # re-exports CopilotStudioAgent
        "agent_framework_azure",
        "agent_framework_copilotstudio",
        "azure.identity",
    ]
    for mod in must_reject:
        assert _is_forbidden(mod), f"boundary should forbid {mod!r} but did not"
        assert not _is_allowed_framework_module(mod), f"boundary should NOT allow {mod!r}"
    # …and the legitimate minimal surface must still pass both gates.
    for mod in ["agent_framework", "agent_framework.openai"]:
        assert not _is_forbidden(mod), f"{mod!r} should be allowed"
        assert _is_allowed_framework_module(mod), f"{mod!r} should be allowed"


def test_server_is_framework_agnostic():
    """server.py must not import any framework/model-SDK symbol (option-B seam)."""
    from agentkit_serve import server

    mods = _imported_modules(Path(server.__file__))
    for mod in mods:
        root = mod.split(".")[0]
        assert root not in ("agent_framework", "openai", "pydantic_ai"), (
            f"server.py imports framework symbol {mod!r}; the run must go through "
            f"agent_factory.run_agent so server.py stays shareable across runtimes"
        )


def test_tool_env_passes_only_declared_names():
    """_tool_env selects ONLY declared names that are present — never the full env."""
    tool = ToolSpec(name="fetch", command=["uvx", "mcp-server-fetch"], env=["FETCH_TOKEN", "ABSENT_VAR"])
    with mock.patch.dict(
        os.environ,
        {"FETCH_TOKEN": "tok", "OPENAI_API_KEY": "sk-should-not-leak", "OTHER": "x"},
        clear=True,
    ):
        env = agent_factory._tool_env(tool)
    assert env == {"FETCH_TOKEN": "tok"}  # declared+present only
    assert "OPENAI_API_KEY" not in env  # the model secret never bleeds into a tool
    assert "ABSENT_VAR" not in env  # declared-but-absent is omitted, not empty


def test_build_tool_threads_command_and_args():
    """build_tool maps command[0]→executable, command[1:]→args, declared env only."""
    tool = ToolSpec(name="fetch", command=["uvx", "mcp-server-fetch", "--flag"], env=[])
    with mock.patch.dict(os.environ, {}, clear=True):
        mcp_tool = agent_factory.build_tool(tool)
    assert mcp_tool.command == "uvx"
    assert list(mcp_tool.args) == ["mcp-server-fetch", "--flag"]


def test_build_tool_sets_tool_name_prefix():
    """build_tool MUST namespace exposed tool names via tool_name_prefix.

    In MAF the per-server ``name`` does NOT prefix the tool functions a server
    exposes — only ``tool_name_prefix`` does. Without it, two servers exposing a
    same-named sub-tool make the agent raise "Duplicate tool name" at run time.
    This mirrors the pydantic-ai adapter's ``tool_prefix=tool.name`` guard.
    """
    tool = ToolSpec(name="fetch", command=["uvx", "mcp-server-fetch"], env=[])
    with mock.patch.dict(os.environ, {}, clear=True):
        mcp_tool = agent_factory.build_tool(tool)
    assert mcp_tool.tool_name_prefix == "fetch"


def test_build_tool_rejects_empty_command():
    tool = ToolSpec(name="broken", command=[], env=[])
    try:
        agent_factory.build_tool(tool)
    except agent_factory.AgentBuildError as exc:
        assert "empty command" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected AgentBuildError for empty command")


def test_missing_api_key_env_fails_fast():
    """A declared apiKeyEnv whose var is absent fails with a secret-free message."""
    spec_data = {
        "abiVersion": "v0",
        "metadata": {"name": "x"},
        "model": {
            "provider": "openai-compatible",
            "baseURL": "https://api.openai.com/v1",
            "name": "gpt-4o-mini",
            "apiKeyEnv": "DEFINITELY_UNSET_KEY_42",
        },
        "instructions": "hi",
        "tools": [],
        "expose": {"openai": True, "port": 8080},
    }
    from agentkit_serve.config import AgentSpec

    spec = AgentSpec.model_validate(spec_data)
    with mock.patch.dict(os.environ, {}, clear=True):
        try:
            agent_factory.build_client(spec)
        except agent_factory.AgentBuildError as exc:
            assert "DEFINITELY_UNSET_KEY_42" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected AgentBuildError for missing key env")
