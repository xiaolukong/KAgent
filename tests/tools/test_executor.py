"""Tests for ToolExecutor with event publishing."""

import pytest

from kagent.domain.enums import EventType, ToolCallStatus
from kagent.domain.events import Event
from kagent.events.bus import EventBus
from kagent.tools.decorator import tool
from kagent.tools.executor import ToolExecutor
from kagent.tools.registry import ToolRegistry


@pytest.fixture
def setup():
    bus = EventBus()
    registry = ToolRegistry()

    @tool
    async def add(a: int, b: int) -> int:
        """Add."""
        return a + b

    @tool
    async def fail_tool() -> None:
        """Fail."""
        raise RuntimeError("intentional error")

    registry.register(add)
    registry.register(fail_tool)
    executor = ToolExecutor(registry, bus)
    return executor, bus


class TestToolExecutor:
    @pytest.mark.asyncio
    async def test_execute_success(self, setup):
        executor, bus = setup
        result = await executor.execute("add", {"a": 3, "b": 7})
        assert result.result == 10
        assert result.status == ToolCallStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_with_call_id(self, setup):
        executor, bus = setup
        result = await executor.execute("add", {"a": 1, "b": 2}, call_id="my-call-123")
        assert result.tool_call_id == "my-call-123"

    @pytest.mark.asyncio
    async def test_execute_not_found(self, setup):
        executor, bus = setup
        result = await executor.execute("nonexistent", {})
        assert result.status == ToolCallStatus.ERROR
        assert "not found" in (result.error or "")

    @pytest.mark.asyncio
    async def test_execute_error(self, setup):
        executor, bus = setup
        result = await executor.execute("fail_tool", {})
        assert result.status == ToolCallStatus.ERROR
        assert "intentional error" in (result.error or "")

    @pytest.mark.asyncio
    async def test_events_published_on_success(self, setup):
        executor, bus = setup
        events: list[Event] = []

        async def collector(event: Event):
            events.append(event)

        bus.subscribe("tool.*", collector)
        await executor.execute("add", {"a": 1, "b": 2})

        event_types = [e.event_type for e in events]
        assert EventType.TOOL_CALL_STARTED in event_types
        assert EventType.TOOL_CALL_COMPLETED in event_types

    @pytest.mark.asyncio
    async def test_events_published_on_error(self, setup):
        executor, bus = setup
        events: list[Event] = []

        async def collector(event: Event):
            events.append(event)

        bus.subscribe("tool.*", collector)
        await executor.execute("fail_tool", {})

        event_types = [e.event_type for e in events]
        assert EventType.TOOL_CALL_STARTED in event_types
        assert EventType.TOOL_CALL_ERROR in event_types
