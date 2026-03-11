"""ContextWindow — sliding window that trims message history by token budget."""

from __future__ import annotations

from kagent.domain.entities import Message
from kagent.domain.enums import Role


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return max(1, len(text) // 4)


def message_tokens(msg: Message) -> int:
    """Estimate token count for a single message."""
    return _estimate_tokens(msg.content or "")


class ContextWindow:
    """Manages a sliding window of messages that fits within a token budget.

    System messages are never trimmed. Oldest non-system messages are
    removed first when the budget is exceeded.
    """

    def __init__(self, max_tokens: int = 128_000) -> None:
        self._max_tokens = max_tokens

    def trim(self, messages: list[Message]) -> list[Message]:
        """Return a list of messages fitting within the token budget.

        Preserves all system messages and trims oldest non-system messages.
        """
        system_msgs = [m for m in messages if m.role == Role.SYSTEM]
        other_msgs = [m for m in messages if m.role != Role.SYSTEM]

        budget = self._max_tokens - sum(message_tokens(m) for m in system_msgs)
        if budget <= 0:
            return system_msgs

        # Keep as many recent messages as possible
        kept: list[Message] = []
        used = 0
        for msg in reversed(other_msgs):
            cost = message_tokens(msg)
            if used + cost > budget:
                break
            kept.append(msg)
            used += cost

        kept.reverse()
        return system_msgs + kept
