"""Core domain entities used throughout the KAgent framework."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from kagent.domain.enums import Role, ToolCallStatus


class Message(BaseModel):
    """A single message in a conversation."""

    role: Role
    content: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ToolCall(BaseModel):
    """A tool call requested by the model."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolDefinition(BaseModel):
    """Metadata describing a registered tool."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    return_type: str | None = None


class ToolResult(BaseModel):
    """Result of a tool execution."""

    tool_call_id: str
    tool_name: str
    result: Any = None
    error: str | None = None
    status: ToolCallStatus = ToolCallStatus.COMPLETED
    duration_ms: float = 0.0


class AgentState(BaseModel):
    """Snapshot of an agent's current state."""

    messages: list[Message] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    turn_count: int = 0
