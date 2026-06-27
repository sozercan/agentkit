from __future__ import annotations

import pytest

from agentkit_serve import agent_factory
from agentkit_serve_common.config import AgentSpec


def _spec_data() -> dict:
    return {
        "abiVersion": "v0",
        "metadata": {"name": "x"},
        "model": {"provider": "openai-compatible", "baseURL": "https://api.openai.com/v1", "name": "gpt-4o-mini"},
        "instructions": "hi",
        "tools": [],
        "expose": {"openai": True, "port": 8080},
    }


def test_pydantic_rejects_model_workload_identity_auth():
    data = _spec_data()
    data["model"]["auth"] = {"type": "workload-identity-token", "audience": "https://ai.azure.com/.default"}
    spec = AgentSpec.model_validate(data)

    with pytest.raises(agent_factory.AgentBuildError, match="model.auth"):
        agent_factory.build_runtime(spec)


def test_pydantic_rejects_context_providers():
    data = _spec_data()
    data["context"] = {"providers": [{"type": "skills", "source": "filesystem", "path": "/agent/skills"}]}
    spec = AgentSpec.model_validate(data)

    with pytest.raises(agent_factory.AgentBuildError, match="context providers"):
        agent_factory.build_runtime(spec)
