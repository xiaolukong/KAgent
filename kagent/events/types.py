"""Internal types for the event system."""

from __future__ import annotations

from dataclasses import dataclass, field

from kagent.domain.protocols import EventHandler


@dataclass
class Subscription:
    """A single event subscription record."""

    id: str
    pattern: str
    handler: EventHandler
    priority: int = 0


@dataclass
class SubscriptionGroup:
    """All subscriptions for a given event pattern, kept sorted by priority."""

    pattern: str
    subscriptions: list[Subscription] = field(default_factory=list)

    def add(self, sub: Subscription) -> None:
        self.subscriptions.append(sub)
        self.subscriptions.sort(key=lambda s: s.priority, reverse=True)

    def remove(self, subscription_id: str) -> bool:
        before = len(self.subscriptions)
        self.subscriptions = [s for s in self.subscriptions if s.id != subscription_id]
        return len(self.subscriptions) < before
