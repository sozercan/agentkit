"""``agentkit-serve`` CLI entrypoint for the pydantic-ai adapter.

The CLI logic and network posture live in ``agentkit_serve_common.cli``; this thin
binding injects THIS adapter's framework-specific ``agent_factory`` module (which
satisfies the ``RuntimeFactory`` protocol).
"""

from __future__ import annotations

from agentkit_serve_common.cli import run

from . import agent_factory


def main(argv: list[str] | None = None) -> None:
    run(agent_factory, argv)


if __name__ == "__main__":  # pragma: no cover
    main()
