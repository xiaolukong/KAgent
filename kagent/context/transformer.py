"""ContextTransformer — converts App-layer messages to LLM-layer messages.

The transformer pipeline sits between ContextManager and PromptBuilder.
Each transform function receives a list of messages and returns a
(potentially filtered/modified) list.  Transforms run in priority order
(highest first) and are chained: the output of one becomes the input
of the next.

**Important**: transforms only affect the messages sent to the LLM.
ContextManager always retains the full, unmodified App-layer history.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from kagent.domain.entities import Message

# Public type alias
TransformFn = Callable[[list[Message]], Awaitable[list[Message]]]


class ContextTransformer:
    """Ordered pipeline that transforms App-layer messages into LLM-layer messages."""

    def __init__(self) -> None:
        # Each entry: (priority, sequence, fn)  — sequence breaks ties stably.
        self._transforms: list[tuple[int, int, TransformFn]] = []
        self._seq = 0

    def add(self, fn: TransformFn, *, priority: int = 0) -> None:
        """Register a transform function.

        *priority* controls execution order — **higher runs first**.
        Among transforms with equal priority, registration order is preserved.
        """
        self._transforms.append((priority, self._seq, fn))
        self._seq += 1
        # Sort descending by priority, then ascending by sequence.
        self._transforms.sort(key=lambda t: (-t[0], t[1]))

    def remove(self, fn: TransformFn) -> None:
        """Remove a previously registered transform function."""
        self._transforms = [t for t in self._transforms if t[2] is not fn]

    async def apply(self, messages: list[Message]) -> list[Message]:
        """Run all transforms in order, chaining the output of each into the next."""
        result = list(messages)
        for _priority, _seq, fn in self._transforms:
            result = await fn(result)
        return result

    @property
    def has_transforms(self) -> bool:
        return len(self._transforms) > 0


# ── Built-in default transforms ────────────────────────────────────────────


async def filter_internal_messages(messages: list[Message]) -> list[Message]:
    """Filter out messages marked as internal (metadata.internal == True).

    Internal messages are preserved in ContextManager but should not
    be sent to the LLM.
    """
    return [m for m in messages if not m.metadata.get("internal")]


async def strip_thinking_from_context(messages: list[Message]) -> list[Message]:
    """Strip thinking metadata from messages before sending to LLM.

    Thinking/reasoning content is the model's internal process and
    should not be fed back into the next LLM call.
    """
    result: list[Message] = []
    for m in messages:
        if m.metadata and "thinking" in m.metadata:
            cleaned_metadata: dict[str, Any] = {
                k: v for k, v in m.metadata.items() if k != "thinking"
            }
            result.append(m.model_copy(update={"metadata": cleaned_metadata}))
        else:
            result.append(m)
    return result
