"""The framework-neutral run contract shared by all AgentKit runtime adapters.

This is the seam that makes the OpenAI /v1 facade (``server.py``) framework-
agnostic: the server depends only on these neutral types and the
:class:`RuntimeFactory` protocol, never on a concrete agent framework. Each
adapter ships an ``agent_factory`` module that satisfies the protocol — that
module is the ONLY place a framework (pydantic-ai, agent-framework, …) is
imported, which is what keeps the lock-in boundary (plan §12) per-adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import TracebackType
from typing import Protocol, runtime_checkable

from .config import AgentSpec
from .conversation import RunRequest


class AgentRunError(Exception):
    """A runtime/model failure during a run, carrying an HTTP status.

    The server maps this to the OpenAI error envelope WITHOUT importing any
    framework or model-SDK type — keeping ``server.py`` framework-agnostic. The
    optional ``code`` lets an adapter preserve the ORIGINAL framework exception's
    class name in the envelope's ``error.code`` field (e.g. pydantic-ai's
    ``ModelHTTPError``); when ``None``, the server falls back to this class name
    (``AgentRunError``).
    """

    def __init__(self, message: str, status: int = 502, code: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


@dataclass(frozen=True)
class RunResult:
    """Framework-neutral result of one agent run (what ``server.py`` reads)."""

    text: str
    usage: dict[str, int] = field(default_factory=dict)


@runtime_checkable
class RuntimeSession(Protocol):
    """A live runtime Adapter session owned by the adapter implementation.

    The shared server enters this session once for the FastAPI lifespan and calls
    :meth:`run` for each normalized RunRequest. Framework-specific agent objects
    and lifecycle quirks stay behind this Interface.
    """

    async def __aenter__(self) -> "RuntimeSession":
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        ...

    async def run(self, request: RunRequest) -> RunResult:
        ...


@runtime_checkable
class RuntimeFactory(Protocol):
    """The interface an adapter's ``agent_factory`` module must satisfy.

    ``server.py`` is handed a value of this shape (the adapter's module) and uses
    only ``build_runtime(spec)``, so it never imports a framework or touches raw
    framework agent lifecycle.
    """

    def build_runtime(self, spec: AgentSpec) -> RuntimeSession:
        ...
