"""Pytest fixtures wiring the shared conformance suite to the MAF adapter.

Microsoft Agent Framework ships no ``TestModel`` analogue, so the offline double is
a tiny in-process echo client built on the framework's public ``BaseChatClient``;
patching ``agent_factory.build_client`` to return it means ``agent.run`` never
touches the network and no API key is needed.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from unittest import mock

import pytest
from agent_framework import BaseChatClient, ChatResponse, Message
from fastapi.testclient import TestClient

from agentkit_serve import agent_factory
from agentkit_serve_common.config import AgentSpec
from agentkit_serve_common.server import create_app

# Patched out below, but set a dummy so any unpatched path stays offline. NOT a
# real secret.
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


class _EchoClient(BaseChatClient):
    """Offline stand-in for the OpenAI client: returns a fixed assistant text."""

    def __init__(self, output: str = "ok", **kwargs):
        super().__init__(**kwargs)
        self._output = output

    async def _inner_get_response(self, *, messages, stream, options, **kwargs) -> ChatResponse:
        return ChatResponse(messages=[Message(role="assistant", contents=[self._output])])


def _spec() -> AgentSpec:
    return AgentSpec.model_validate(_SPEC_DATA)


@pytest.fixture
def model_name() -> str:
    return _MODEL_NAME


@pytest.fixture
def make_client():
    """Factory: a TestClient whose agent uses the offline echo client."""

    @contextmanager
    def _make(auth_token: str | None = None, output: str = "ok"):
        with mock.patch(
            "agentkit_serve.agent_factory.build_client",
            return_value=_EchoClient(output=output),
        ):
            app = create_app(_spec(), agent_factory, auth_token=auth_token)
            with TestClient(app) as client:
                yield client

    return _make


@pytest.fixture
def make_failing_client():
    """Factory: a TestClient whose agent run RAISES, to exercise the error path.

    ``make_failing_client(exc)`` patches the live agent's ``run`` to raise ``exc``,
    so ``agent_factory.run_agent`` normalizes it into an ``AgentRunError`` (carrying
    the original class name as ``code``) and the server maps it to the envelope.
    """

    @contextmanager
    def _make(exc: Exception, auth_token: str | None = None):
        with mock.patch(
            "agentkit_serve.agent_factory.build_client",
            return_value=_EchoClient(),
        ):
            app = create_app(_spec(), agent_factory, auth_token=auth_token)
            with TestClient(app) as client:
                agent = app.state.agent

                async def _raise(*args, **kwargs):
                    raise exc

                with mock.patch.object(agent, "run", _raise):
                    yield client

    return _make

