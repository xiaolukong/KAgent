"""Tests for format converters."""

from pydantic import BaseModel

from kagent.domain.entities import Message, ToolCall, ToolDefinition
from kagent.domain.enums import Role
from kagent.models.converters import (
    messages_to_anthropic,
    messages_to_gemini,
    messages_to_openai,
    parse_structured_output,
    tools_to_anthropic,
    tools_to_gemini,
    tools_to_openai,
)


class TestMessagesToOpenAI:
    def test_basic_messages(self):
        msgs = [
            Message(role=Role.SYSTEM, content="Sys"),
            Message(role=Role.USER, content="Hi"),
        ]
        result = messages_to_openai(msgs)
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["content"] == "Hi"

    def test_tool_calls(self):
        tc = ToolCall(id="tc1", name="calc", arguments={"x": 1})
        msg = Message(role=Role.ASSISTANT, tool_calls=[tc])
        result = messages_to_openai([msg])
        assert result[0]["tool_calls"][0]["function"]["name"] == "calc"

    def test_tool_result_message(self):
        msg = Message(role=Role.TOOL, content="42", tool_call_id="tc1")
        result = messages_to_openai([msg])
        assert result[0]["tool_call_id"] == "tc1"


class TestToolsToOpenAI:
    def test_conversion(self):
        tools = [ToolDefinition(name="t", description="d", parameters={"type": "object"})]
        result = tools_to_openai(tools)
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "t"


class TestMessagesToAnthropic:
    def test_system_separated(self):
        msgs = [
            Message(role=Role.SYSTEM, content="System prompt"),
            Message(role=Role.USER, content="Hello"),
        ]
        system, messages = messages_to_anthropic(msgs)
        assert system == "System prompt"
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_tool_result_format(self):
        msg = Message(role=Role.TOOL, content="result", tool_call_id="tc1")
        _, messages = messages_to_anthropic([msg])
        assert messages[0]["content"][0]["type"] == "tool_result"


class TestToolsToAnthropic:
    def test_conversion(self):
        tools = [ToolDefinition(name="t", description="d", parameters={"type": "object"})]
        result = tools_to_anthropic(tools)
        assert result[0]["name"] == "t"
        assert result[0]["input_schema"] == {"type": "object"}


class TestMessagesToGemini:
    def test_system_separated(self):
        msgs = [
            Message(role=Role.SYSTEM, content="System"),
            Message(role=Role.USER, content="Hello"),
        ]
        system, contents = messages_to_gemini(msgs)
        assert system == "System"
        assert len(contents) == 1
        assert contents[0]["role"] == "user"


class TestToolsToGemini:
    def test_conversion(self):
        tools = [ToolDefinition(name="t", description="d", parameters={"type": "object"})]
        result = tools_to_gemini(tools)
        assert "functionDeclarations" in result[0]
        assert result[0]["functionDeclarations"][0]["name"] == "t"


class TestParseStructuredOutput:
    def test_valid_json(self):
        class MyModel(BaseModel):
            name: str
            age: int

        result = parse_structured_output('{"name": "Alice", "age": 30}', MyModel)
        assert result.name == "Alice"
        assert result.age == 30

    def test_json_with_code_fences(self):
        class MyModel(BaseModel):
            value: int

        result = parse_structured_output('```json\n{"value": 42}\n```', MyModel)
        assert result.value == 42

    def test_invalid_json_raises(self):
        class MyModel(BaseModel):
            x: int

        import pytest

        with pytest.raises(Exception):
            parse_structured_output("not json", MyModel)

    def test_none_content_raises(self):
        class MyModel(BaseModel):
            x: int

        import pytest

        with pytest.raises(ValueError):
            parse_structured_output(None, MyModel)
