"""Tests for domain event types."""

from kagent.domain.enums import EventType
from kagent.domain.events import AgentEvent, Event, LLMEvent, SteeringEvent, ToolEvent


class TestEvent:
    def test_create_event(self):
        event = Event(event_type=EventType.AGENT_STARTED, payload={"key": "value"})
        assert event.event_type == EventType.AGENT_STARTED
        assert event.payload == {"key": "value"}
        assert event.correlation_id  # auto-generated

    def test_event_with_source(self):
        event = Event(
            event_type=EventType.LLM_REQUEST_SENT,
            source="test_module",
        )
        assert event.source == "test_module"

    def test_serialization(self):
        event = Event(event_type=EventType.TOOL_REGISTERED, payload={"name": "calc"})
        data = event.model_dump()
        restored = Event(**data)
        assert restored.event_type == event.event_type
        assert restored.correlation_id == event.correlation_id


class TestEventSubclasses:
    def test_agent_event(self):
        event = AgentEvent(event_type=EventType.AGENT_STARTED)
        assert isinstance(event, Event)

    def test_llm_event(self):
        event = LLMEvent(event_type=EventType.LLM_STREAM_CHUNK)
        assert isinstance(event, Event)

    def test_tool_event(self):
        event = ToolEvent(event_type=EventType.TOOL_CALL_STARTED)
        assert isinstance(event, Event)

    def test_steering_event(self):
        event = SteeringEvent(event_type=EventType.STEERING_ABORT)
        assert isinstance(event, Event)


class TestEventType:
    def test_all_event_types_exist(self):
        """Verify all 17 event types from the design doc are defined."""
        expected = [
            "agent.started",
            "agent.loop.iteration",
            "agent.loop.completed",
            "agent.error",
            "agent.state.changed",
            "llm.request.sent",
            "llm.stream.chunk",
            "llm.stream.complete",
            "llm.response.received",
            "llm.error",
            "tool.registered",
            "tool.call.started",
            "tool.call.completed",
            "tool.call.error",
            "steering.redirect",
            "steering.inject_message",
            "steering.abort",
        ]
        actual = [e.value for e in EventType]
        for exp in expected:
            assert exp in actual, f"Missing event type: {exp}"

    def test_event_type_string_value(self):
        assert EventType.AGENT_STARTED.value == "agent.started"
        assert EventType.TOOL_CALL_COMPLETED.value == "tool.call.completed"
