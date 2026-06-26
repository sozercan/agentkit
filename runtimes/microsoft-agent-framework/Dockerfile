# syntax=docker/dockerfile:1
#
# agentkit-serve-maf ADAPTER image (Microsoft Agent Framework runtime).
#
# This is the runtime adapter the AgentKit Go converter uses as the LLB BASE
# image when an agentkitfile selects `runtime: microsoft-agent-framework`: the
# converter pulls this image and merges the resolved /agent/agent.yaml layer on
# top. It is also runnable standalone (ENTRYPOINT + CMD below).
#
# It bundles BOTH a Python runtime (for the MAF agent) AND Node.js + npx and
# uv/uvx, so stdio MCP servers distributed on npm (`npx -y ...`) or PyPI
# (`uvx ...`) can be spawned as tool subprocesses.
#
# No secret is ever baked in: agent.yaml carries env var NAMES only; values are
# injected at runtime via `docker run -e`.
#
# Dependency lock-in boundary (plan §12): the installed Python deps are MAF core,
# the OpenAI chat client, MCP SDK, and provider adapters needed by generic
# AgentKit capabilities (Foundry workload identity/memory and Azure AI Search).
# Do not add unrelated Copilot Studio/Purview/etc. packages.

FROM python:3.12-slim-bookworm

# --- system deps: Node.js + npx (for npm-distributed MCP servers) ------------
# ca-certificates for TLS to model endpoints; nodejs/npm provide `npx`.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        nodejs \
        npm \
    && rm -rf /var/lib/apt/lists/*

# --- uv / uvx (for PyPI-distributed MCP servers via `uvx ...`) ---------------
# Copy the static uv + uvx binaries from the official image onto PATH.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# AgentKit layout constants (must match pkg/utils/const.go and pkg/agentkit/abi):
#   AgentKitRoot = /opt/agentkit ; ServeBinary = /opt/agentkit/bin/agentkit-serve
#   abi.Path = /agent/agent.yaml
ENV AGENTKIT_ROOT=/opt/agentkit \
    PATH=/opt/agentkit/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# --- install agentkit-serve-maf into a dedicated venv at /opt/agentkit -------
# The console script lands at /opt/agentkit/bin/agentkit-serve (== ServeBinary).
# Build context is the REPO ROOT (see Makefile `build-serve-maf`), so paths are
# repo-root-relative. The shared framework-neutral core (agentkit_serve_common,
# brings fastapi/uvicorn/pydantic/pyyaml) is installed FIRST so the adapter's
# `agentkit-serve-common` dependency is already satisfied; then the adapter adds
# its framework deps (agent-framework-core/-openai/-foundry, Azure AI Search,
# mcp). The adapter image stays separate from the pydantic-ai one — disjoint
# framework deps, same shared core.
WORKDIR /src
COPY runtimes/common/pyproject.toml runtimes/common/README.md ./common/
COPY runtimes/common/agentkit_serve_common ./common/agentkit_serve_common
COPY runtimes/microsoft-agent-framework/pyproject.toml runtimes/microsoft-agent-framework/README.md ./microsoft-agent-framework/
COPY runtimes/microsoft-agent-framework/agentkit_serve ./microsoft-agent-framework/agentkit_serve

RUN python -m venv /opt/agentkit \
    && /opt/agentkit/bin/pip install --upgrade pip \
    && /opt/agentkit/bin/pip install ./common \
    && /opt/agentkit/bin/pip install ./microsoft-agent-framework \
    && { /opt/agentkit/bin/agentkit-serve --help >/dev/null 2>&1 || true; }

# --- non-root user + agent config mount point --------------------------------
# uid 1000 owns /agent so a writer/runtime can place agent.yaml there.
RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin agentkit \
    && mkdir -p /agent \
    && chown -R 1000:1000 /agent /opt/agentkit

# Loopback-only by default; binding 0.0.0.0 requires AGENTKIT_AUTH_TOKEN (§10).
ENV AGENTKIT_BIND=127.0.0.1
EXPOSE 8080

USER 1000

ENTRYPOINT ["/opt/agentkit/bin/agentkit-serve"]
CMD ["--config", "/agent/agent.yaml"]
