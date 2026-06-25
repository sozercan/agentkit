"""AgentKit runtime adapter (Microsoft Agent Framework).

Reads the frozen ``/agent/agent.yaml`` ABI (see ``docs/agent-abi.md``) and serves
a NON-STREAMING OpenAI Chat-Completions facade backed by a Microsoft Agent
Framework (MAF) agent with stdio MCP tools. It is byte-for-byte ABI-compatible
with the pydantic-ai adapter: same agent.yaml in, same /v1 contract out.
"""

__version__ = "0.0.0"
