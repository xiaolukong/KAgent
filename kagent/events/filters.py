"""Event filtering utilities."""

from __future__ import annotations

import fnmatch
from collections.abc import Callable
from typing import Any

from kagent.domain.events import Event


class EventFilter:
    """Composable filter for events based on type pattern and payload conditions."""

    def __init__(
        self,
        pattern: str | None = None,
        condition: Callable[[dict[str, Any]], bool] | None = None,
    ) -> None:
        self._pattern = pattern
        self._condition = condition

    def matches(self, event: Event) -> bool:
        """Return True if the event passes this filter."""
        if self._pattern and not fnmatch.fnmatch(event.event_type.value, self._pattern):
            return False
        if self._condition and not self._condition(event.payload):
            return False
        return True

    def __and__(self, other: EventFilter) -> EventFilter:
        """Combine two filters with AND logic."""
        return EventFilter(
            condition=lambda payload: self.matches(
                Event(event_type=self._pattern or "*", payload=payload)  # type: ignore[arg-type]
            )
            and other.matches(
                Event(event_type=other._pattern or "*", payload=payload)  # type: ignore[arg-type]
            ),
        )
