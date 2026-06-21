"""Framework-neutral core shared by AgentKit runtime adapters.

Provides the frozen ``/agent/agent.yaml`` ABI loader (:mod:`config`), the
non-streaming OpenAI ``/v1`` Chat-Completions facade (:mod:`server`), the
CLI / network-posture entry point (:mod:`cli`), and the neutral run contract
(:mod:`runtime`: :class:`RunResult`, :class:`AgentRunError`, :class:`RuntimeFactory`).

This package imports NO agent framework. Each runtime adapter supplies a thin
``agent_factory`` module that satisfies :class:`RuntimeFactory`; that module is the
only place a framework is imported (the plan §12 lock-in boundary).
"""

from .config import AgentSpec, ConfigError, ToolSpec, load, load_or_exit
from .runtime import AgentRunError, RunResult, RuntimeFactory
from .server import create_app

__all__ = [
    "AgentSpec",
    "ToolSpec",
    "ConfigError",
    "load",
    "load_or_exit",
    "RunResult",
    "AgentRunError",
    "RuntimeFactory",
    "create_app",
]

__version__ = "0.0.0"
