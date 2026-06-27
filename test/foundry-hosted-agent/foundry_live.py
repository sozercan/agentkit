"""Foundry Hosted Agent protocol wrapper for a real baked AgentKit runtime.

Unlike ``foundry_invocations.py`` this wrapper does not start a mock model; it
loads ``/agent/agent.yaml`` and exposes Foundry's hosted-agent protocol surfaces
against the same runtime session used by the standalone AgentKit /v1 server.
"""
from __future__ import annotations

import os

import uvicorn

from agentkit_serve import agent_factory
from agentkit_serve_common.config import load, validate_required_env
from agentkit_serve_common.foundry import create_foundry_app

spec = load("/agent/agent.yaml")
validate_required_env(spec)
app = create_foundry_app(spec, agent_factory)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8088")))
