# AgentKit LangGraph runtime adapter

`runtimes/langgraph` builds the AgentKit runtime adapter backed by
LangChain/LangGraph. It consumes the same `/agent/agent.yaml` ABI as the other
adapters and serves the same non-streaming OpenAI-compatible surface:

- `GET /healthz`
- `GET /v1/models`
- `POST /v1/chat/completions`

Select it from an Agentkitfile with:

```yaml
runtime: langgraph
```

## Responsibilities

- Build a LangChain OpenAI chat model from `model.baseURL`, `model.name`, and the
  env var named by `model.apiKeyEnv`.
- Create a LangGraph agent from the resolved `instructions` system prompt.
- Start persistent stdio MCP sessions for ABI-declared tools during app lifespan.
- Prefix tool names by server name to avoid collisions.
- Convert the final LangChain `AIMessage` into the neutral `RunResult` contract.
- Aggregate usage metadata across every AI message in a tool-using run.

The shared server, ABI reader, CLI, network posture, auth behavior, and
conformance tests live in `runtimes/common`.

## Dependency boundary

This adapter depends on LangChain/LangGraph/OpenAI/MCP packages plus
`agentkit-serve-common`:

- `langchain`
- `langgraph`
- `langchain-openai`
- `langchain-mcp-adapters`
- `mcp`
- `agentkit-serve-common`

It intentionally does not import Azure or Foundry hosting packages. Guardrail
tests in `tests/test_guardrails.py` enforce that the generic LangGraph adapter
stays cloud-neutral.

## Tool lifecycle and secret hygiene

Each `tools:` entry is treated as one stdio MCP server. The adapter creates a
persistent `MultiServerMCPClient` session for each server during FastAPI lifespan
startup, initializes it with `AGENTKIT_MCP_TIMEOUT` (default `120` seconds), loads
LangChain tools with prefixed names, and closes sessions at shutdown.

Tool subprocess env is declared-only. If a tool declares:

```yaml
tools:
  - name: fetch
    command: ["uvx", "mcp-server-fetch"]
    env: ["FETCH_TIMEOUT"]
```

then only `FETCH_TIMEOUT` is passed to that subprocess when it is present in the
container env. Model API keys and other process env vars are not inherited unless
explicitly declared on that tool.

## Build and test

From the repo root:

```sh
cd runtimes/langgraph
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e ../common -e '.[dev]'
pytest -q
```

Build the adapter image and a LangGraph-backed test agent:

```sh
make build-serve-langgraph
make build-agentkit
make build-test-agent RUNTIME=langgraph
```

Run it with the model API key named by the fixture:

```sh
docker run --rm --platform linux/amd64 \
  -p 127.0.0.1:8080:8080 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  langgraph-agent:test
```

See `docs/runtime-adapters.md` and `docs/agent-abi.md` for the shared runtime
contract.
