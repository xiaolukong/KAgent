"""Tests for ContextWindow sliding window."""

from kagent.context.window import ContextWindow, message_tokens
from kagent.domain.entities import Message
from kagent.domain.enums import Role


class TestContextWindow:
    def test_no_trimming_under_budget(self):
        window = ContextWindow(max_tokens=10000)
        msgs = [
            Message(role=Role.SYSTEM, content="Sys"),
            Message(role=Role.USER, content="Hello"),
        ]
        result = window.trim(msgs)
        assert len(result) == 2

    def test_system_messages_never_trimmed(self):
        window = ContextWindow(max_tokens=20)
        msgs = [
            Message(role=Role.SYSTEM, content="System prompt here"),
            Message(role=Role.USER, content="A" * 400),  # ~100 tokens, over budget
            Message(role=Role.USER, content="Short"),
        ]
        result = window.trim(msgs)
        assert result[0].role == Role.SYSTEM

    def test_oldest_trimmed_first(self):
        window = ContextWindow(max_tokens=30)
        msgs = [
            Message(role=Role.SYSTEM, content="S"),
            Message(role=Role.USER, content="Old message " * 10),  # long
            Message(role=Role.USER, content="Recent"),  # short
        ]
        result = window.trim(msgs)
        # System kept, old long message trimmed, recent kept
        contents = [m.content for m in result if m.role != Role.SYSTEM]
        assert any("Recent" in (c or "") for c in contents)

    def test_empty_messages(self):
        window = ContextWindow(max_tokens=100)
        result = window.trim([])
        assert result == []

    def test_message_tokens_estimate(self):
        tokens = message_tokens(Message(role=Role.USER, content="Hello world"))
        assert tokens > 0
