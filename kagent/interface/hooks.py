"""Lifecycle hooks — user-level callbacks for agent events."""

from __future__ import annotations

from kagent.domain.protocols import EventHandler, IEventBus


class HookRegistry:
    """Manages user-registered lifecycle hooks backed by the EventBus."""

    def __init__(self, event_bus: IEventBus) -> None:
        self._event_bus = event_bus
        self._subscription_ids: list[str] = []

    def on(self, event_pattern: str, handler: EventHandler) -> str:
        """Register a hook for the given event pattern.

        Returns a subscription ID for later removal.
        """
        sub_id = self._event_bus.subscribe(event_pattern, handler)
        self._subscription_ids.append(sub_id)
        return sub_id

    def off(self, subscription_id: str) -> None:
        """Unregister a hook by subscription ID."""
        self._event_bus.unsubscribe(subscription_id)
        if subscription_id in self._subscription_ids:
            self._subscription_ids.remove(subscription_id)

    def clear(self) -> None:
        """Remove all hooks registered through this registry."""
        for sub_id in self._subscription_ids:
            self._event_bus.unsubscribe(sub_id)
        self._subscription_ids.clear()
