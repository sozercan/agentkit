"""Pytest fixtures wiring the shared conformance suite to the LangGraph adapter.

The offline double is a tiny fake compiled graph. Patching ``create_agent`` to
return it means the shared FastAPI server and LangGraph adapter run with no
network and no real model API key.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from unittest import mock

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from agentkit_serve import agent_factory
from agentkit_serve_common.config import AgentSpec
from agentkit_serve_common.server import create_app

# build_model resolves the API key from this env var at construction time before
# create_agent is patched to the fake graph. Provide a dummy so tests stay fully
# offline. NOT a real secret.
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")

_MODEL_NAME = "gpt-4o-mini"
_SPEC_DATA = {
    "abiVersion": "v0",
    "metadata": {"name": "test-agent"},
    "model": {
        "provider": "openai-compatible",
        "baseURL": "https://api.openai.com/v1",
        "name": _MODEL_NAME,
        "apiKeyEnv": "OPENAI_API_KEY",
    },
    "instructions": "Be helpful.",
    "tools": [],
    "expose": {"openai": True, "port": 8080},
}


class _FakeGraph:
    def __init__(self, output: str = "ok") -> None:
        self.output = output
        self.inputs = []

    async def ainvoke(self, state):
        self.inputs.append(state)
        return {"messages": [*state.get("messages", []), AIMessage(content=self.output)]}


class _FailingGraph:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    async def ainvoke(self, state):
        raise self.exc


def _spec() -> AgentSpec:
    return AgentSpec.model_validate(_SPEC_DATA)


@pytest.fixture
def model_name() -> str:
    return _MODEL_NAME


@pytest.fixture
def make_client():
    """Factory: a TestClient whose agent uses an offline fake compiled graph."""

    @contextmanager
    def _make(auth_token: str | None = None, output: str = "ok"):
        fake_graph = _FakeGraph(output=output)
        with mock.patch("agentkit_serve.agent_factory.create_agent", return_value=fake_graph):
            app = create_app(_spec(), agent_factory, auth_token=auth_token)
            with TestClient(app) as client:
                yield client

    return _make


@pytest.fixture
def make_failing_client():
    """Factory: a TestClient whose graph raises during ``ainvoke``."""

    @contextmanager
    def _make(exc: Exception, auth_token: str | None = None):
        fake_graph = _FailingGraph(exc)
        with mock.patch("agentkit_serve.agent_factory.create_agent", return_value=fake_graph):
            app = create_app(_spec(), agent_factory, auth_token=auth_token)
            with TestClient(app) as client:
                yield client

    return _make
