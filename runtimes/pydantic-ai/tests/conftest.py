"""Pytest fixtures wiring the shared conformance suite to the pydantic-ai adapter.

Provides the offline double via pydantic-ai's ``TestModel`` + ``agent.override``,
so the conformance tests run with no network and no real API key.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from unittest import mock

import pytest
from fastapi.testclient import TestClient
from pydantic_ai.models.test import TestModel

from agentkit_serve import agent_factory
from agentkit_serve_common.config import AgentSpec
from agentkit_serve_common.server import create_app

# build_model resolves the API key from this env var at construction time, even
# when the run is later overridden to TestModel. Provide a dummy so tests run fully
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


def _spec() -> AgentSpec:
    return AgentSpec.model_validate(_SPEC_DATA)


@pytest.fixture
def model_name() -> str:
    return _MODEL_NAME


@pytest.fixture
def make_client():
    """Factory: a TestClient whose agent model is an offline TestModel.

    Patching ``build_model`` keeps the shared server tests behind the RuntimeSession
    Interface; they do not reach through ``app.state`` to the raw framework agent.
    """

    @contextmanager
    def _make(auth_token: str | None = None, output: str = "ok"):
        with mock.patch(
            "agentkit_serve.agent_factory.build_model",
            return_value=TestModel(custom_output_text=output),
        ):
            app = create_app(_spec(), agent_factory, auth_token=auth_token)
            with TestClient(app) as client:
                yield client

    return _make


@pytest.fixture
def make_failing_client():
    """Factory: a TestClient whose agent run RAISES, to exercise the error path.

    ``make_failing_client(exc)`` patches the live agent's ``run`` to raise ``exc``,
    so the runtime session normalizes it into an ``AgentRunError`` (carrying
    the original class name as ``code``) and the server maps it to the envelope.
    """

    @contextmanager
    def _make(exc: Exception, auth_token: str | None = None):
        async def _raise(*args, **kwargs):
            raise exc

        with mock.patch(
            "agentkit_serve.agent_factory.build_model",
            return_value=TestModel(custom_output_text="unused"),
        ), mock.patch.object(agent_factory.Agent, "run", _raise):
            app = create_app(_spec(), agent_factory, auth_token=auth_token)
            with TestClient(app) as client:
                yield client

    return _make

