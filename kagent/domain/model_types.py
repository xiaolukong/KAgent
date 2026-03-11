"""Model request/response types for the LLM abstraction layer."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from kagent.domain.entities import Message, ToolCall, ToolDefinition
from kagent.domain.enums import StreamChunkType


class TokenUsage(BaseModel):
    """Token consumption for a single LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ModelInfo(BaseModel):
    """Metadata about a model provider."""

    provider: str
    model_name: str
    max_context_tokens: int | None = None


class ModelRequest(BaseModel):
    """Unified request sent to any model provider."""

    messages: list[Message]
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    tools: list[ToolDefinition] | None = None
    tool_choice: str | None = None
    response_format: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelResponse(BaseModel):
    """Unified response from any model provider."""

    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    usage: TokenUsage | None = None
    model: str | None = None
    parsed: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class ToolCallChunk(BaseModel):
    """Partial tool call data within a stream chunk."""

    id: str | None = None
    name: str | None = None
    arguments_delta: str | None = None


class StreamChunk(BaseModel):
    """A single chunk in a streaming LLM response."""

    chunk_type: StreamChunkType
    content: str | None = None
    tool_call: ToolCallChunk | None = None
    usage: TokenUsage | None = None
    parsed: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)
