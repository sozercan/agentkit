"""Behavioral regression tests for the OpenAI Chat-Completions facade.

These lock in the v0 HARD invariants (plan §6, §10) that were verified by hand
against the built image: the 400 guards, the single-completion happy path, and
the Bearer auth gate. They run fully offline via pydantic-ai's TestModel — no
network, no real API key.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

from fastapi.testclient import TestClient
from pydantic_ai.models.test import TestModel

from agentkit_serve.config import AgentSpec
from agentkit_serve.server import create_app

# build_model resolves the API key from this env var at construction time, even
# when the run is later overridden to TestModel. Provide a dummy so tests run
# fully offline. NOT a real secret.
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")

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


def _spec() -> AgentSpec:
    return AgentSpec.model_validate(_SPEC_DATA)


def _client(auth_token: str | None = None) -> TestClient:
    """A TestClient for a fresh app (no model override; for non-run endpoints)."""
    return TestClient(create_app(_spec(), auth_token=auth_token))


@contextmanager
def _client_with_test_model(output: str = "ok", auth_token: str | None = None):
    """Enter a TestClient whose agent model is overridden to an offline TestModel.

    The agent is created inside ``create_app`` and published on ``app.state.agent``
    when the lifespan runs (i.e. inside the ``with TestClient`` block). We grab that
    instance and override *its* model so ``agent.run`` never touches the network.
    """
    app = create_app(_spec(), auth_token=auth_token)
    with TestClient(app) as client:
        agent = app.state.agent
        with agent.override(model=TestModel(custom_output_text=output)):
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
    with _client_with_test_model() as c:
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
    with _client_with_test_model(output="three bullets here") as c:
        r = c.post(
            "/v1/chat/completions",
            json={"model": "x", "messages": [{"role": "user", "content": "summarize"}]},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert isinstance(body["choices"][0]["message"]["content"], str)


def test_auth_gate_enforced():
    with _client(auth_token="secret123") as c:
        # healthz stays open
        assert c.get("/healthz").status_code == 200
        # /v1/* requires a valid bearer token
        assert c.get("/v1/models").status_code == 401
        assert c.get("/v1/models", headers={"Authorization": "Bearer nope"}).status_code == 401
        assert c.get("/v1/models", headers={"Authorization": "Bearer secret123"}).status_code == 200
