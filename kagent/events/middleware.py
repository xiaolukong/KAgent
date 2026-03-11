"""Event middleware for cross-cutting concerns (logging, metrics, etc.)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from kagent.common.logging import get_logger
from kagent.domain.events import Event

logger = get_logger("events.middleware")


class EventMiddleware(ABC):
    """Base class for event middleware that wraps publish calls."""

    @abstractmethod
    async def before(self, event: Event) -> Event | None:
        """Called before handlers. Return None to suppress the event."""
        ...

    @abstractmethod
    async def after(self, event: Event) -> None:
        """Called after all handlers have been invoked."""
        ...


class LoggingMiddleware(EventMiddleware):
    """Logs every event that passes through the bus."""

    async def before(self, event: Event) -> Event | None:
        logger.debug("Event published: %s [%s]", event.event_type.value, event.correlation_id)
        return event

    async def after(self, event: Event) -> None:
        pass


class MetricsMiddleware(EventMiddleware):
    """Counts events by type (in-memory, for diagnostics)."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    async def before(self, event: Event) -> Event | None:
        key = event.event_type.value
        self.counts[key] = self.counts.get(key, 0) + 1
        return event

    async def after(self, event: Event) -> None:
        pass
