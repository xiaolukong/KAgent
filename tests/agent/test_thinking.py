"""Tests for thinking/reasoning streaming support."""

import pytest

from kagent.agent.loop import AgentLoop
from kagent.agent.prompt_builder import PromptBuilder
from kagent.agent.steering import SteeringController
from kagent.context.manager import ContextManager
from kagent.domain.enums import Role, StreamChunkType
from kagent.domain.model_types import StreamChunk
from kagent.events.bus import EventBus
from kagent.tools.executor import ToolExecutor
from kagent.tools.registry import ToolRegistry
from tests.conftest import MockModelProvider


def _make_loop(
    provider: MockModelProvider,
    max_turns: int = 5,
) -> tuple[AgentLoop, ContextManager]:
    bus = EventBus()
    reg = ToolRegistry()
    executor = ToolExecutor(reg, bus)
    context = ContextManager()
    prompt = PromptBuilder(system_prompt="Test system.")
    steering = SteeringController(bus)
    loop = AgentLoop(
        model_provider=provider,
        event_bus=bus,
        tool_registry=reg,
        tool_executor=executor,
        context_manager=context,
        prompt_builder=prompt,
        steering=steering,
        max_turns=max_turns,
    )
    return loop, context


class TestThinkingChunkType:
    def test_thinking_delta_enum_exists(self):
        assert StreamChunkType.THINKING_DELTA == "thinking_delta"

    def test_stream_chunk_thinking_field(self):
        chunk = StreamChunk(
            chunk_type=StreamChunkType.THINKING_DELTA,
            thinking="Let me think...",
        )
        assert chunk.thinking == "Let me think..."
        assert chunk.content is None

    def test_stream_chunk_thinking_default_none(self):
        chunk = StreamChunk(
            chunk_type=StreamChunkType.TEXT_DELTA,
            content="Hello",
        )
        assert chunk.thinking is None


class TestThinkingStream:
    @pytest.mark.asyncio
    async def test_run_stream_collects_thinking(self):
        """run_stream() should collect thinking chunks and write to message metadata."""
        provider = MockModelProvider(
            response_content="The answer is 42.",
            thinking_content="Let me reason about this...",
        )
        loop, context = _make_loop(provider)

        chunks: list[StreamChunk] = []
        async for chunk in loop.run_stream("What is the meaning of life?"):
            chunks.append(chunk)

        # Verify thinking chunks were yielded
        thinking_chunks = [c for c in chunks if c.chunk_type == StreamChunkType.THINKING_DELTA]
        assert len(thinking_chunks) > 0
        thinking_text = "".join(c.thinking for c in thinking_chunks if c.thinking)
        assert thinking_text == "Let me reason about this..."

        # Verify text chunks were also yielded
        text_chunks = [c for c in chunks if c.chunk_type == StreamChunkType.TEXT_DELTA]
        assert len(text_chunks) > 0

        # Verify thinking was written to assistant message metadata
        messages = context.get_all_messages()
        assistant_msgs = [m for m in messages if m.role == Role.ASSISTANT]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].metadata.get("thinking") == "Let me reason about this..."

    @pytest.mark.asyncio
    async def test_run_stream_no_thinking(self):
        """run_stream() without thinking should work normally."""
        provider = MockModelProvider(response_content="Just text.")
        loop, context = _make_loop(provider)

        chunks: list[StreamChunk] = []
        async for chunk in loop.run_stream("Hello"):
            chunks.append(chunk)

        thinking_chunks = [c for c in chunks if c.chunk_type == StreamChunkType.THINKING_DELTA]
        assert len(thinking_chunks) == 0

        messages = context.get_all_messages()
        assistant_msgs = [m for m in messages if m.role == Role.ASSISTANT]
        assert len(assistant_msgs) == 1
        assert "thinking" not in assistant_msgs[0].metadata


class TestThinkingRun:
    @pytest.mark.asyncio
    async def test_run_propagates_thinking_metadata(self):
        """run() should propagate thinking from response.metadata to assistant message."""
        provider = MockModelProvider(
            response_content="The answer is 42.",
            thinking_content="Step by step reasoning here.",
        )
        loop, context = _make_loop(provider)

        result = await loop.run("What is the answer?")
        assert result.content == "The answer is 42."
        assert result.metadata.get("thinking") == "Step by step reasoning here."

        # Verify thinking was written to assistant message metadata
        messages = context.get_all_messages()
        assistant_msgs = [m for m in messages if m.role == Role.ASSISTANT]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].metadata.get("thinking") == "Step by step reasoning here."

    @pytest.mark.asyncio
    async def test_run_no_thinking_metadata(self):
        """run() without thinking should not add thinking to metadata."""
        provider = MockModelProvider(response_content="Plain response.")
        loop, context = _make_loop(provider)

        result = await loop.run("Hello")
        assert "thinking" not in result.metadata

        messages = context.get_all_messages()
        assistant_msgs = [m for m in messages if m.role == Role.ASSISTANT]
        assert len(assistant_msgs) == 1
        assert "thinking" not in assistant_msgs[0].metadata
