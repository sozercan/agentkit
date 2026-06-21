"""Behavioral regression tests for the OpenAI Chat-Completions facade (MAF adapter).

These lock in the v0 HARD invariants (plan §6, §10): the 400 guards, the
single-completion happy path, and the Bearer auth gate. They run fully OFFLINE.

Microsoft Agent Framework ships no ``TestModel`` analogue (verified in the M0
spike), so we substitute a tiny in-process echo client built on the framework's
public ``BaseChatClient`` — its only abstract method is ``_inner_get_response``.
Patching ``agent_factory.build_client`` to return it means ``agent.run`` never
touches the network and no API key is needed.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from unittest import mock

from agent_framework import BaseChatClient, ChatResponse, Message
from fastapi.testclient import TestClient

from agentkit_serve.config import AgentSpec
from agentkit_serve.server import create_app

# A minimal valid spec (no tools → no MCP subprocesses spawned in these tests).
_SPEC_DATA = {
    "abiVersion": "v0",
    "metadata": {"name": "test-agent"},
    "model": {
        "provider": "openai-compatible",
        "baseURL": "https://api.openai.com/v1",
        "name": "gpt-4o-mini",
        "apiKeyEnv": "OPENAI_API_KEY",
    },
    "instructions": "Be helpful.",
    "tools": [],
    "expose": {"openai": True, "port": 8080},
}

# build_client resolves the API key from this env var at construction time. We
# patch build_client out entirely below, but set a dummy so any unpatched path is
# still offline and never reads a real secret. NOT a real secret.
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")


class _EchoClient(BaseChatClient):
    """Offline stand-in for the OpenAI client: returns a fixed assistant text.

    Mirrors pydantic-ai's ``TestModel(custom_output_text=...)`` for these tests.
    """

    def __init__(self, output: str = "ok", **kwargs):
        super().__init__(**kwargs)
        self._output = output

    async def _inner_get_response(self, *, messages, stream, options, **kwargs) -> ChatResponse:
        return ChatResponse(messages=[Message(role="assistant", contents=[self._output])])


def _spec() -> AgentSpec:
    return AgentSpec.model_validate(_SPEC_DATA)


@contextmanager
def _patched_client(output: str = "ok"):
    """Patch agent_factory.build_client so the agent uses the offline echo client."""
    with mock.patch(
        "agentkit_serve.agent_factory.build_client",
        return_value=_EchoClient(output=output),
    ):
        yield


@contextmanager
def _client(auth_token: str | None = None, output: str = "ok"):
    """A TestClient whose agent is wired to the offline echo client."""
    with _patched_client(output=output):
        app = create_app(_spec(), auth_token=auth_token)
        with TestClient(app) as client:
            yield client


def test_healthz_open():
    with _client() as c:
        r = c.get("/healthz")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


def test_models_listing():
    with _client() as c:
        r = c.get("/v1/models")
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "list"
        assert body["data"][0]["id"] == "gpt-4o-mini"


def test_stream_true_rejected():
    with _client() as c:
        r = c.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        assert r.status_code == 400
        assert "stream" in r.json()["error"]["message"].lower()


def test_caller_tools_rejected():
    with _client() as c:
        r = c.post(
            "/v1/chat/completions",
            json={
                "model": "x",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "x"}}],
            },
        )
        assert r.status_code == 400
        assert "tools" in r.json()["error"]["message"].lower()


def test_caller_tool_choice_required_rejected():
    with _client() as c:
        r = c.post(
            "/v1/chat/completions",
            json={
                "model": "x",
                "messages": [{"role": "user", "content": "hi"}],
                "tool_choice": "required",
            },
        )
        assert r.status_code == 400


def test_tool_choice_auto_allowed():
    # "auto"/"none" mean "no specific tool" — must NOT be rejected.
    with _client() as c:
        r = c.post(
            "/v1/chat/completions",
            json={
                "model": "x",
                "messages": [{"role": "user", "content": "hi"}],
                "tool_choice": "auto",
            },
        )
        assert r.status_code == 200


def test_happy_path_single_completion():
    with _client(output="three bullets here") as c:
        r = c.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": [{"role": "user", "content": "summarize"}]},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert body["choices"][0]["message"]["content"] == "three bullets here"


def test_final_message_must_be_user():
    with _client() as c:
        r = c.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": [{"role": "assistant", "content": "hi"}]},
        )
        assert r.status_code == 400


def test_multi_turn_history_accepted():
    # A prior user/assistant exchange plus a final user turn → single completion.
    with _client(output="final answer") as c:
        r = c.post(
            "/v1/chat/completions",
            json={
                "model": "x",
                "messages": [
                    {"role": "system", "content": "be terse"},
                    {"role": "user", "content": "q1"},
                    {"role": "assistant", "content": "a1"},
                    {"role": "user", "content": "q2"},
                ],
            },
        )
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"] == "final answer"


def test_auth_gate_enforced():
    with _client(auth_token="secret123") as c:
        # healthz stays open
        assert c.get("/healthz").status_code == 200
        # /v1/* requires a valid bearer token
        assert c.get("/v1/models").status_code == 401
        assert c.get("/v1/models", headers={"Authorization": "Bearer nope"}).status_code == 401
        assert c.get("/v1/models", headers={"Authorization": "Bearer secret123"}).status_code == 200
