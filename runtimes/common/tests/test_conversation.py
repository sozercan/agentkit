from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from agentkit_serve_common.conversation import (
    ConversationError,
    ConversationTurn,
    run_request_from_messages,
    text_of,
)


@dataclass(frozen=True)
class Message:
    role: str
    content: Any = None


@pytest.mark.parametrize(
    ("content", "want"),
    [
        (None, ""),
        ("plain", "plain"),
        (["a", {"type": "text", "text": "b"}, {"type": "image_url", "url": "x"}], "ab"),
        (123, "123"),
    ],
)
def test_text_of_flattens_openai_content(content: Any, want: str):
    assert text_of(content) == want


def test_run_request_from_messages_splits_history_and_prompt():
    req = run_request_from_messages(
        [
            Message("system", "be terse"),
            Message("user", "q1"),
            Message("assistant", [{"type": "text", "text": "a1"}]),
            Message("tool", "ignored"),
            Message("developer", "ignored"),
            Message("assistant", None),
            Message("user", "q2"),
        ]
    )

    assert req.prompt == "q2"
    assert req.history == (
        ConversationTurn("system", "be terse"),
        ConversationTurn("user", "q1"),
        ConversationTurn("assistant", "a1"),
    )


def test_run_request_requires_non_empty_messages():
    with pytest.raises(ConversationError, match="non-empty"):
        run_request_from_messages([])


def test_run_request_requires_final_user_message():
    with pytest.raises(ConversationError, match="final message"):
        run_request_from_messages([Message("assistant", "done")])
