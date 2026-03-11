"""Event types for the KAgent pub/sub system."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from kagent.domain.enums import EventType


class Event(BaseModel):
    """Base event transmitted through the EventBus."""

    event_type: EventType
    payload: dict[str, Any] = Field(default_factory=dict)
    source: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])


class AgentEvent(Event):
    """Event related to agent lifecycle."""
    pass


class LLMEvent(Event):
    """Event related to LLM interactions."""
    pass


class ToolEvent(Event):
    """Event related to tool registration or execution."""
    pass


class SteeringEvent(Event):
    """Event related to runtime steering directives."""
    pass
