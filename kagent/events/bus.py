"""EventBus — the central nervous system of KAgent.

Supports async publish/subscribe with glob-style pattern matching
(e.g. "tool.*" matches "tool.call.started") and priority ordering.
"""

from __future__ import annotations

import asyncio
import fnmatch
import uuid

from kagent.common.logging import get_logger
from kagent.domain.events import Event
from kagent.domain.protocols import EventHandler
from kagent.events.types import Subscription, SubscriptionGroup

logger = get_logger("events.bus")


class EventBus:
    """Async pub/sub event bus with glob-style pattern routing."""

    def __init__(self) -> None:
        self._groups: dict[str, SubscriptionGroup] = {}
        self._sub_index: dict[str, str] = {}  # subscription_id -> pattern
        self._lock = asyncio.Lock()

    # ── Publishing ───────────────────────────────────────────────────────

    async def publish(self, event: Event) -> None:
        """Publish an event to all matching subscribers."""
        event_type = event.event_type.value
        handlers: list[tuple[int, EventHandler]] = []

        for pattern, group in self._groups.items():
            if fnmatch.fnmatch(event_type, pattern):
                for sub in group.subscriptions:
                    handlers.append((sub.priority, sub.handler))

        # Sort by priority descending (highest first)
        handlers.sort(key=lambda h: h[0], reverse=True)

        for _, handler in handlers:
            try:
                await handler(event)
            except Exception:
                logger.exception(
                    "Handler error for event %s", event_type
                )

    # ── Subscribing ──────────────────────────────────────────────────────

    def subscribe(
        self,
        event_pattern: str,
        handler: EventHandler,
        *,
        priority: int = 0,
    ) -> str:
        """Subscribe a handler to events matching the given glob pattern.

        Returns a subscription ID that can be used to unsubscribe.
        """
        sub_id = uuid.uuid4().hex[:12]
        sub = Subscription(id=sub_id, pattern=event_pattern, handler=handler, priority=priority)

        if event_pattern not in self._groups:
            self._groups[event_pattern] = SubscriptionGroup(pattern=event_pattern)
        self._groups[event_pattern].add(sub)
        self._sub_index[sub_id] = event_pattern

        return sub_id

    def unsubscribe(self, subscription_id: str) -> None:
        """Remove a subscription by its ID."""
        pattern = self._sub_index.pop(subscription_id, None)
        if pattern and pattern in self._groups:
            self._groups[pattern].remove(subscription_id)
            if not self._groups[pattern].subscriptions:
                del self._groups[pattern]
