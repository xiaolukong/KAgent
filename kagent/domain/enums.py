"""Enumerations used across the KAgent framework."""

from __future__ import annotations

from enum import StrEnum


class Role(StrEnum):
    """Message role in a conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class EventType(StrEnum):
    """All event types in the KAgent event system."""

    # Agent lifecycle
    AGENT_STARTED = "agent.started"
    AGENT_LOOP_ITERATION = "agent.loop.iteration"
    AGENT_LOOP_COMPLETED = "agent.loop.completed"
    AGENT_ERROR = "agent.error"
    AGENT_STATE_CHANGED = "agent.state.changed"

    # LLM primitives
    LLM_REQUEST_SENT = "llm.request.sent"
    LLM_STREAM_CHUNK = "llm.stream.chunk"
    LLM_STREAM_COMPLETE = "llm.stream.complete"
    LLM_RESPONSE_RECEIVED = "llm.response.received"
    LLM_ERROR = "llm.error"

    # Tool events
    TOOL_REGISTERED = "tool.registered"
    TOOL_CALL_STARTED = "tool.call.started"
    TOOL_CALL_COMPLETED = "tool.call.completed"
    TOOL_CALL_ERROR = "tool.call.error"

    # Steering events
    STEERING_REDIRECT = "steering.redirect"
    STEERING_INJECT_MESSAGE = "steering.inject_message"
    STEERING_ABORT = "steering.abort"
    STEERING_INTERRUPT = "steering.interrupt"
    STEERING_RESUME = "steering.resume"


class ModelProviderType(StrEnum):
    """Supported model provider types."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class ToolCallStatus(StrEnum):
    """Status of a tool call execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


class StreamChunkType(StrEnum):
    """Types of stream chunks emitted by model providers."""

    TEXT_DELTA = "text_delta"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_DELTA = "tool_call_delta"
    TOOL_CALL_END = "tool_call_end"
    METADATA = "metadata"
