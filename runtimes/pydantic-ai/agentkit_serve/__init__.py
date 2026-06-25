"""AgentKit runtime adapter (agentkit-serve).

Reads the frozen ``/agent/agent.yaml`` ABI (see ``docs/agent-abi.md``) and serves
a NON-STREAMING OpenAI Chat-Completions facade backed by a pydantic-ai agent with
stdio MCP tools.
"""

__version__ = "0.0.0"
