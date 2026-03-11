"""Shared test fixtures."""

from __future__ import annotations

import asyncio
from typing import Any
from collections.abc import AsyncIterator

import pytest

from kagent.domain.entities import Message, ToolCall, ToolDefinition
from kagent.domain.enums import EventType, Role, StreamChunkType
from kagent.domain.events import Event
from kagent.domain.model_types import (
    ModelInfo,
    ModelRequest,
    ModelResponse,
    StreamChunk,
    TokenUsage,
)
from kagent.events.bus import EventBus
from kagent.models.base import BaseModelProvider
from kagent.models.config import ModelConfig
from kagent.tools.decorator import tool
from kagent.tools.registry import ToolRegistry


# ── Mock Model Provider ─────────────────────────────────────────────────────


class MockModelProvider(BaseModelProvider):
    """A mock provider that returns canned responses for testing."""

    def __init__(
        self,
        response_content: str = "Hello from mock!",
        tool_calls: list[ToolCall] | None = None,
        config: ModelConfig | None = None,
    ) -> None:
        super().__init__(config or ModelConfig(api_key="mock-key", model_name="mock-model"))
        self._response_content = response_content
        self._tool_calls = tool_calls
        self._call_count = 0

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(provider="mock", model_name="mock-model")

    async def _do_complete(self, request: ModelRequest) -> ModelResponse:
        self._call_count += 1
        # If tool_calls configured and this is the first call, return tool calls
        if self._tool_calls and self._call_count == 1:
            return ModelResponse(
                content=None,
                tool_calls=self._tool_calls,
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                model="mock-model",
            )
        return ModelResponse(
            content=self._response_content,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            model="mock-model",
        )

    async def _do_stream(self, request: ModelRequest) -> AsyncIterator[StreamChunk]:
        self._call_count += 1
        if self._tool_calls and self._call_count == 1:
            for tc in self._tool_calls:
                yield StreamChunk(
                    chunk_type=StreamChunkType.TOOL_CALL_START,
                    tool_call=__import__("kagent.domain.model_types", fromlist=["ToolCallChunk"]).ToolCallChunk(
                        id=tc.id, name=tc.name, arguments_delta=__import__("json").dumps(tc.arguments),
                    ),
                )
            return

        words = self._response_content.split()
        for i, word in enumerate(words):
            text = word if i == 0 else f" {word}"
            yield StreamChunk(chunk_type=StreamChunkType.TEXT_DELTA, content=text)

        yield StreamChunk(
            chunk_type=StreamChunkType.METADATA,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def mock_provider() -> MockModelProvider:
    return MockModelProvider()


@pytest.fixture
def tool_registry() -> ToolRegistry:
    registry = ToolRegistry()

    @tool
    async def greet(name: str) -> str:
        """Greet someone."""
        return f"Hello, {name}!"

    @tool
    async def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    registry.register(greet)
    registry.register(add)
    return registry


@pytest.fixture
def sample_messages() -> list[Message]:
    return [
        Message(role=Role.SYSTEM, content="You are helpful."),
        Message(role=Role.USER, content="What is 2+2?"),
        Message(role=Role.ASSISTANT, content="2+2 equals 4."),
    ]


@pytest.fixture
def sample_tool_definitions() -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="get_weather",
            description="Get weather for a city",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                },
                "required": ["city"],
            },
        ),
    ]
