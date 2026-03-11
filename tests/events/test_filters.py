"""Tests for EventFilter."""

from kagent.domain.enums import EventType
from kagent.domain.events import Event, ToolEvent
from kagent.events.filters import EventFilter


class TestEventFilter:
    def test_pattern_match(self):
        f = EventFilter(pattern="tool.*")
        event = ToolEvent(event_type=EventType.TOOL_CALL_STARTED)
        assert f.matches(event) is True

    def test_pattern_no_match(self):
        f = EventFilter(pattern="agent.*")
        event = ToolEvent(event_type=EventType.TOOL_CALL_STARTED)
        assert f.matches(event) is False

    def test_condition_match(self):
        f = EventFilter(condition=lambda p: p.get("tool_name") == "calc")
        event = Event(
            event_type=EventType.TOOL_CALL_COMPLETED,
            payload={"tool_name": "calc"},
        )
        assert f.matches(event) is True

    def test_condition_no_match(self):
        f = EventFilter(condition=lambda p: p.get("tool_name") == "calc")
        event = Event(
            event_type=EventType.TOOL_CALL_COMPLETED,
            payload={"tool_name": "search"},
        )
        assert f.matches(event) is False

    def test_no_filter_matches_everything(self):
        f = EventFilter()
        event = Event(event_type=EventType.AGENT_STARTED)
        assert f.matches(event) is True
