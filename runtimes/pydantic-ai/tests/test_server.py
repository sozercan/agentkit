"""Behavioral regression tests for the pydantic-ai adapter's OpenAI facade.

The HARD invariants (400 guards, single-completion, auth gate, multi-turn,
framework-agnostic shared core) are the SHARED conformance suite — imported here
so this adapter is held to the exact same contract as every other adapter. The
offline double + spec are supplied by ``conftest.py`` (pydantic-ai ``TestModel``).
"""

from __future__ import annotations

# Re-export the shared conformance suite; pytest collects each `test_*` against
# this adapter's `make_client` / `model_name` fixtures from conftest.py.
from agentkit_serve_common.conformance import *  # noqa: F401,F403
