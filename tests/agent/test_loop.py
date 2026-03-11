"""Tests for AgentLoop orchestration."""

import json

import pytest

from kagent.agent.config import AgentConfig
from kagent.agent.loop import AgentLoop
from kagent.agent.prompt_builder import PromptBuilder
from kagent.agent.steering import SteeringController
from kagent.context.manager import ContextManager
from kagent.domain.entities import Message, ToolCall
from kagent.domain.enums import EventType, Role
from kagent.domain.events import Event
from kagent.events.bus import EventBus
from kagent.tools.decorator import tool
from kagent.tools.executor import ToolExecutor
from kagent.tools.registry import ToolRegistry

from tests.conftest import MockModelProvider


def _make_loop(
    provider: MockModelProvider,
    max_turns: int = 5,
    registry: ToolRegistry | None = None,
) -> tuple[AgentLoop, EventBus]:
    bus = EventBus()
    reg = registry or ToolRegistry()
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
    return loop, bus


class TestAgentLoopRun:
    @pytest.mark.asyncio
    async def test_single_turn(self):
        provider = MockModelProvider(response_content="Done")
        loop, bus = _make_loop(provider)
        result = await loop.run("Hello")
        assert result.content == "Done"

    @pytest.mark.asyncio
    async def test_tool_call_loop(self):
        """Verify the loop executes tool calls and continues."""
        registry = ToolRegistry()

        @tool
        async def double(x: int) -> int:
            """Double."""
            return x * 2

        registry.register(double)

        tc = ToolCall(id="c1", name="double", arguments={"x": 5})
        provider = MockModelProvider(response_content="Result is 10", tool_calls=[tc])
        loop, bus = _make_loop(provider, registry=registry)

        result = await loop.run("Double 5")
        assert result.content == "Result is 10"
        assert provider._call_count == 2  # first call returns tool_call, second returns text

    @pytest.mark.asyncio
    async def test_max_turns_limit(self):
        """Ensure the loop stops at max_turns even if tool calls continue."""
        tc = ToolCall(id="c1", name="infinite", arguments={})
        provider = MockModelProvider(response_content="never", tool_calls=[tc])
        # tool_calls always returned -> loop never ends naturally
        # but we override _call_count logic: always return tool_calls
        provider._tool_calls = [tc]  # type: ignore

        # Patch so it always returns tool calls
        original = provider._do_complete

        async def always_tool_calls(request):
            from kagent.domain.model_types import ModelResponse, TokenUsage
            return ModelResponse(
                content=None,
                tool_calls=[tc],
                usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            )

        provider._do_complete = always_tool_calls  # type: ignore

        registry = ToolRegistry()

        @tool
        async def infinite() -> str:
            """Never ends."""
            return "again"

        registry.register(infinite)
        loop, bus = _make_loop(provider, max_turns=3, registry=registry)

        result = await loop.run("Go")
        # Should have stopped after 3 turns
        assert result is not None


class TestAgentLoopStream:
    @pytest.mark.asyncio
    async def test_stream_collects_text(self):
        provider = MockModelProvider(response_content="Hello world")
        loop, bus = _make_loop(provider)

        chunks = []
        async for chunk in loop.run_stream("Hi"):
            chunks.append(chunk)

        text = "".join(c.content or "" for c in chunks)
        assert "Hello" in text
        assert "world" in text

    @pytest.mark.asyncio
    async def test_stream_publishes_events(self):
        provider = MockModelProvider(response_content="Test")
        loop, bus = _make_loop(provider)
        events = []

        async def collector(event: Event):
            events.append(event.event_type)

        bus.subscribe("llm.*", collector)

        async for _ in loop.run_stream("Hi"):
            pass

        assert EventType.LLM_REQUEST_SENT in events
        assert EventType.LLM_STREAM_CHUNK in events
        assert EventType.LLM_STREAM_COMPLETE in events
