"""Tests for EventBus pub/sub, glob routing, and priority ordering."""

import pytest

from kagent.domain.enums import EventType
from kagent.domain.events import AgentEvent, Event, LLMEvent, ToolEvent
from kagent.events.bus import EventBus


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


class TestPublishSubscribe:
    @pytest.mark.asyncio
    async def test_basic_subscribe_and_publish(self, bus: EventBus):
        received = []

        async def handler(event: Event):
            received.append(event)

        bus.subscribe(EventType.AGENT_STARTED.value, handler)
        await bus.publish(AgentEvent(event_type=EventType.AGENT_STARTED))

        assert len(received) == 1
        assert received[0].event_type == EventType.AGENT_STARTED

    @pytest.mark.asyncio
    async def test_no_match(self, bus: EventBus):
        received = []

        async def handler(event: Event):
            received.append(event)

        bus.subscribe(EventType.AGENT_STARTED.value, handler)
        await bus.publish(Event(event_type=EventType.AGENT_ERROR))

        assert len(received) == 0

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self, bus: EventBus):
        results = {"a": 0, "b": 0}

        async def handler_a(event: Event):
            results["a"] += 1

        async def handler_b(event: Event):
            results["b"] += 1

        bus.subscribe(EventType.AGENT_STARTED.value, handler_a)
        bus.subscribe(EventType.AGENT_STARTED.value, handler_b)
        await bus.publish(AgentEvent(event_type=EventType.AGENT_STARTED))

        assert results["a"] == 1
        assert results["b"] == 1


class TestGlobPatternMatching:
    @pytest.mark.asyncio
    async def test_wildcard_all_agent_events(self, bus: EventBus):
        received = []

        async def handler(event: Event):
            received.append(event.event_type)

        bus.subscribe("agent.*", handler)
        await bus.publish(AgentEvent(event_type=EventType.AGENT_STARTED))
        await bus.publish(AgentEvent(event_type=EventType.AGENT_ERROR))
        await bus.publish(LLMEvent(event_type=EventType.LLM_REQUEST_SENT))

        assert len(received) == 2
        assert EventType.AGENT_STARTED in received
        assert EventType.AGENT_ERROR in received

    @pytest.mark.asyncio
    async def test_wildcard_tool_events(self, bus: EventBus):
        received = []

        async def handler(event: Event):
            received.append(event.event_type)

        bus.subscribe("tool.*", handler)
        await bus.publish(ToolEvent(event_type=EventType.TOOL_REGISTERED))
        await bus.publish(ToolEvent(event_type=EventType.TOOL_CALL_STARTED))
        await bus.publish(ToolEvent(event_type=EventType.TOOL_CALL_COMPLETED))

        assert len(received) == 3

    @pytest.mark.asyncio
    async def test_double_wildcard(self, bus: EventBus):
        received = []

        async def handler(event: Event):
            received.append(event.event_type)

        bus.subscribe("agent.loop.*", handler)
        await bus.publish(AgentEvent(event_type=EventType.AGENT_LOOP_ITERATION))
        await bus.publish(AgentEvent(event_type=EventType.AGENT_LOOP_COMPLETED))
        await bus.publish(AgentEvent(event_type=EventType.AGENT_STARTED))

        assert len(received) == 2

    @pytest.mark.asyncio
    async def test_catch_all(self, bus: EventBus):
        received = []

        async def handler(event: Event):
            received.append(event.event_type)

        bus.subscribe("*", handler)
        await bus.publish(AgentEvent(event_type=EventType.AGENT_STARTED))
        await bus.publish(LLMEvent(event_type=EventType.LLM_ERROR))

        assert len(received) == 2


class TestPriority:
    @pytest.mark.asyncio
    async def test_higher_priority_runs_first(self, bus: EventBus):
        order = []

        async def low(event: Event):
            order.append("low")

        async def high(event: Event):
            order.append("high")

        bus.subscribe(EventType.AGENT_STARTED.value, low, priority=1)
        bus.subscribe(EventType.AGENT_STARTED.value, high, priority=10)
        await bus.publish(AgentEvent(event_type=EventType.AGENT_STARTED))

        assert order == ["high", "low"]


class TestUnsubscribe:
    @pytest.mark.asyncio
    async def test_unsubscribe_stops_delivery(self, bus: EventBus):
        received = []

        async def handler(event: Event):
            received.append(event)

        sub_id = bus.subscribe(EventType.AGENT_STARTED.value, handler)
        await bus.publish(AgentEvent(event_type=EventType.AGENT_STARTED))
        assert len(received) == 1

        bus.unsubscribe(sub_id)
        await bus.publish(AgentEvent(event_type=EventType.AGENT_STARTED))
        assert len(received) == 1  # not incremented

    @pytest.mark.asyncio
    async def test_unsubscribe_nonexistent_is_noop(self, bus: EventBus):
        bus.unsubscribe("nonexistent-id")  # should not raise


class TestHandlerErrors:
    @pytest.mark.asyncio
    async def test_handler_error_does_not_stop_others(self, bus: EventBus):
        results = []

        async def bad_handler(event: Event):
            raise RuntimeError("boom")

        async def good_handler(event: Event):
            results.append("ok")

        bus.subscribe(EventType.AGENT_STARTED.value, bad_handler, priority=10)
        bus.subscribe(EventType.AGENT_STARTED.value, good_handler, priority=1)
        await bus.publish(AgentEvent(event_type=EventType.AGENT_STARTED))

        assert results == ["ok"]
