# agentkit-serve

The AgentKit runtime adapter. It loads the frozen `/agent/agent.yaml` ABI
(`docs/agent-abi.md`) and serves a **non-streaming** OpenAI Chat-Completions
facade (`POST /v1/chat/completions`) backed by a [pydantic-ai](https://ai.pydantic.dev)
agent whose tools are stdio MCP servers.

This package is also published as the **adapter image** used as the LLB base by
the AgentKit Go converter.

```
agentkit-serve --config /agent/agent.yaml
```

See `docs/agent-abi.md` for the writer/reader contract.
