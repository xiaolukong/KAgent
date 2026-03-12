"""Tests for domain entities."""

from datetime import datetime

from kagent.domain.entities import AgentState, Message, ToolCall, ToolDefinition, ToolResult
from kagent.domain.enums import Role, ToolCallStatus


class TestMessage:
    def test_create_user_message(self):
        msg = Message(role=Role.USER, content="Hello")
        assert msg.role == Role.USER
        assert msg.content == "Hello"
        assert isinstance(msg.timestamp, datetime)

    def test_create_assistant_message_with_tool_calls(self):
        tc = ToolCall(name="search", arguments={"q": "test"})
        msg = Message(role=Role.ASSISTANT, tool_calls=[tc])
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "search"

    def test_message_serialization(self):
        msg = Message(role=Role.USER, content="Hi", metadata={"key": "val"})
        data = msg.model_dump()
        restored = Message(**data)
        assert restored.role == msg.role
        assert restored.content == msg.content
        assert restored.metadata == msg.metadata


class TestToolCall:
    def test_auto_id(self):
        tc = ToolCall(name="test", arguments={"a": 1})
        assert tc.id  # auto-generated
        assert tc.name == "test"

    def test_explicit_id(self):
        tc = ToolCall(id="my-id", name="test")
        assert tc.id == "my-id"


class TestToolDefinition:
    def test_create(self):
        td = ToolDefinition(
            name="calc",
            description="Do math",
            parameters={"type": "object", "properties": {"x": {"type": "integer"}}},
        )
        assert td.name == "calc"
        assert "x" in td.parameters["properties"]


class TestToolResult:
    def test_success(self):
        tr = ToolResult(tool_call_id="123", tool_name="calc", result=42)
        assert tr.status == ToolCallStatus.COMPLETED
        assert tr.result == 42

    def test_error(self):
        tr = ToolResult(
            tool_call_id="123",
            tool_name="calc",
            error="Division by zero",
            status=ToolCallStatus.ERROR,
        )
        assert tr.error == "Division by zero"


class TestAgentState:
    def test_default(self):
        state = AgentState()
        assert state.messages == []
        assert state.turn_count == 0

    def test_with_messages(self):
        msgs = [Message(role=Role.USER, content="Hi")]
        state = AgentState(messages=msgs, turn_count=1)
        assert len(state.messages) == 1
        assert state.turn_count == 1

    def test_serialization_roundtrip(self):
        state = AgentState(
            messages=[Message(role=Role.USER, content="test")],
            context={"key": "value"},
            turn_count=3,
        )
        data = state.model_dump()
        restored = AgentState(**data)
        assert restored.turn_count == 3
        assert len(restored.messages) == 1
