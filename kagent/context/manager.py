"""ContextManager — maintains conversation context and token budget."""

from __future__ import annotations

from typing import Any

from kagent.context.window import ContextWindow
from kagent.domain.entities import AgentState, Message


class ContextManager:
    """Manages conversation history, metadata, and token budget trimming."""

    def __init__(self, max_tokens: int = 128_000) -> None:
        self._messages: list[Message] = []
        self._context: dict[str, Any] = {}
        self._window = ContextWindow(max_tokens=max_tokens)

    def add_message(self, message: Message) -> None:
        """Append a message to the history."""
        self._messages.append(message)

    def get_messages(self) -> list[Message]:
        """Return the current messages, trimmed to fit the token budget."""
        return self._window.trim(self._messages)

    def get_all_messages(self) -> list[Message]:
        """Return all messages without trimming."""
        return list(self._messages)

    def get_context(self) -> dict[str, Any]:
        return dict(self._context)

    def update_context(self, **kwargs: Any) -> None:
        self._context.update(kwargs)

    def snapshot(self) -> dict[str, Any]:
        """Create a serializable snapshot of the full context state."""
        return {
            "messages": [m.model_dump() for m in self._messages],
            "context": dict(self._context),
        }

    def restore(self, data: dict[str, Any]) -> None:
        """Restore state from a snapshot."""
        self._messages = [Message(**m) for m in data.get("messages", [])]
        self._context = dict(data.get("context", {}))

    def to_agent_state(self, turn_count: int = 0) -> AgentState:
        """Build an AgentState from current context."""
        return AgentState(
            messages=self.get_all_messages(),
            context=self._context,
            turn_count=turn_count,
        )

    def clear(self) -> None:
        """Reset all messages and context."""
        self._messages.clear()
        self._context.clear()
