# AgentKit LangGraph runtime adapter

`runtimes/langgraph` is the generic LangChain/LangGraph AgentKit runtime. It
consumes the same frozen `/agent/agent.yaml` ABI as the pydantic-ai and Microsoft
Agent Framework adapters, and serves the same non-streaming OpenAI-compatible
surface:

- `GET /healthz`
- `GET /v1/models`
- `POST /v1/chat/completions`

## Support level

`runtime: langgraph` supports AgentKit-authored single-agent LangGraph agents
generated from the AgentKit ABI:

- `model.provider: openai-compatible` via `langchain_openai.ChatOpenAI`
- `instructions` as the graph `system_prompt`
- stdio MCP `tools` loaded with `langchain-mcp-adapters`
- one final collapsed assistant message through AgentKit's `/v1` façade

Arbitrary user-authored LangGraph modules, checkpointing, streaming, multi-node
graph authoring, and Microsoft Foundry `/responses` or `/invocations` protocol
serving are intentionally out of scope for this generic adapter.

## Dependency boundary

This adapter depends on LangChain/LangGraph/OpenAI/MCP packages only:

- `langchain`
- `langgraph`
- `langchain-openai`
- `langchain-mcp-adapters`
- `mcp`
- `agentkit-serve-common`

It must not import Azure or Foundry hosting packages such as
`langchain_azure_ai` or `azure.*`. A future Foundry-native mode should be a
separate adapter/target so the generic LangGraph runtime stays cloud-neutral.

## Tool lifecycle and secret hygiene

Each `tools:` entry is treated as one stdio MCP server. The adapter creates a
persistent `MultiServerMCPClient` session for each server during FastAPI lifespan
startup, initializes it with `AGENTKIT_MCP_TIMEOUT` (default `120` seconds), loads
LangChain tools with `tool_name_prefix=True`, and closes sessions at shutdown.

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

Build the adapter image:

```sh
make build-serve-langgraph
```

Build a test AgentKit image with the LangGraph runtime:

```sh
make build-agentkit
make build-test-agent RUNTIME=langgraph
```

Run it (requires the model API key named by the fixture):

```sh
docker run --rm --platform linux/amd64 \
  -p 127.0.0.1:8080:8080 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  langgraph-agent:test
```
