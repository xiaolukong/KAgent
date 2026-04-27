"""Unit tests for AICoreProvider (mocked gen_ai_hub SDK)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import kagent.common.config as _cfg_module
from kagent.common.config import KAgentConfig
from kagent.common.errors import ConfigError, ModelError
from kagent.domain.entities import Message, ToolCall, ToolDefinition
from kagent.domain.enums import Role, StreamChunkType
from kagent.domain.model_types import ModelRequest, TokenUsage
from kagent.models.config import ModelConfig


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_provider(model_name: str = "gpt-4o"):
    """Create AICoreProvider with fully mocked SDK."""
    with patch("gen_ai_hub.proxy.GenAIHubProxyClient", return_value=MagicMock()), \
         patch("gen_ai_hub.orchestration_v2.service.OrchestrationService", return_value=MagicMock()):
        from kagent.models.aicore_provider import AICoreProvider
        return AICoreProvider(ModelConfig(model_name=model_name))


def _mock_completion_response(
    content: str = "Hello",
    tool_calls=None,
    request_id: str = "req-123",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
):
    """Build a minimal mock CompletionPostResponse."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or []

    choice = MagicMock()
    choice.message = msg

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens

    llm_result = MagicMock()
    llm_result.choices = [choice]
    llm_result.usage = usage

    response = MagicMock()
    response.request_id = request_id
    response.final_result = llm_result

    return response


# ── Import guard ───────────────────────────────────────────────────────────────


class TestAICoreProviderImportGuard:
    def test_missing_sdk_raises_config_error(self):
        """ImportError from gen_ai_hub → ConfigError with install hint."""
        with patch.dict("sys.modules", {"gen_ai_hub": None, "gen_ai_hub.proxy": None}):
            # Force re-import path
            import importlib
            import sys
            if "kagent.models.aicore_provider" in sys.modules:
                del sys.modules["kagent.models.aicore_provider"]
            with pytest.raises((ConfigError, ImportError)):
                from kagent.models.aicore_provider import AICoreProvider  # noqa: F401
                AICoreProvider(ModelConfig(model_name="gpt-4o"))


# ── ModelInfo ──────────────────────────────────────────────────────────────────


class TestAICoreProviderModelInfo:
    def test_get_model_info(self):
        provider = _make_provider("gpt-4o")
        info = provider.get_model_info()
        assert info.provider == "aicore"
        assert info.model_name == "gpt-4o"

    def test_model_name_from_string(self):
        provider = _make_provider("claude-3-7-sonnet")
        assert provider.get_model_info().model_name == "claude-3-7-sonnet"


# ── Non-streaming completion ───────────────────────────────────────────────────


class TestAICoreProviderComplete:
    @pytest.mark.asyncio
    async def test_complete_no_tools(self):
        provider = _make_provider()
        mock_response = _mock_completion_response(content="Paris is the capital of France.")
        provider._service.arun = AsyncMock(return_value=mock_response)

        request = ModelRequest(
            messages=[Message(role=Role.USER, content="What is the capital of France?")]
        )
        result = await provider.complete(request)

        assert result.content == "Paris is the capital of France."
        assert result.tool_calls is None
        assert result.usage is not None
        assert result.usage.total_tokens == 15
        assert result.metadata["request_id"] == "req-123"

    @pytest.mark.asyncio
    async def test_complete_with_tool_calls(self):
        provider = _make_provider()

        # Build mock tool call in the response
        func_call = MagicMock()
        func_call.name = "get_weather"
        func_call.parse_arguments.return_value = {"city": "Berlin"}

        sdk_tc = MagicMock()
        sdk_tc.id = "tc-001"
        sdk_tc.function = func_call

        mock_response = _mock_completion_response(content=None, tool_calls=[sdk_tc])
        provider._service.arun = AsyncMock(return_value=mock_response)

        request = ModelRequest(
            messages=[Message(role=Role.USER, content="What's the weather in Berlin?")],
            tools=[
                ToolDefinition(
                    name="get_weather",
                    description="Get weather",
                    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
                )
            ],
        )
        result = await provider.complete(request)

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "tc-001"
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].arguments == {"city": "Berlin"}

    @pytest.mark.asyncio
    async def test_complete_wraps_orchestration_error(self):
        provider = _make_provider()

        import httpx
        from gen_ai_hub.orchestration_v2.exceptions import OrchestrationError

        err = OrchestrationError(
            request_id="req-err",
            headers=httpx.Headers({}),
            message="Model not found",
            code=404,
            location="llm",
            intermediate_results={},
        )
        provider._service.arun = AsyncMock(side_effect=err)

        request = ModelRequest(messages=[Message(role=Role.USER, content="Hello")])
        with pytest.raises(ModelError, match="AI Core orchestration error"):
            await provider.complete(request)

    @pytest.mark.asyncio
    async def test_complete_with_system_and_user(self):
        provider = _make_provider()
        mock_response = _mock_completion_response(content="Hi!")
        provider._service.arun = AsyncMock(return_value=mock_response)

        request = ModelRequest(
            messages=[
                Message(role=Role.SYSTEM, content="You are helpful."),
                Message(role=Role.USER, content="Hello"),
            ]
        )
        result = await provider.complete(request)
        assert result.content == "Hi!"
        # Verify arun was called with no history (first turn)
        call_kwargs = provider._service.arun.call_args.kwargs
        assert call_kwargs["history"] == []


# ── Streaming ─────────────────────────────────────────────────────────────────


class TestAICoreProviderStream:
    @pytest.mark.asyncio
    async def test_stream_text_deltas(self):
        provider = _make_provider()

        # Build stream events
        def _make_stream_event(text: str, usage=None):
            delta = MagicMock()
            delta.content = text
            delta.tool_calls = []

            choice = MagicMock()
            choice.delta = delta

            llm_result = MagicMock()
            llm_result.choices = [choice]
            llm_result.usage = usage

            event = MagicMock()
            event.final_result = llm_result
            return event

        usage = MagicMock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 8
        usage.total_tokens = 18

        events = [
            _make_stream_event("Hello"),
            _make_stream_event(" world"),
            _make_stream_event("", usage=usage),
        ]

        async def _mock_aiter(self_):
            for e in events:
                yield e

        sse_ctx = MagicMock()
        sse_ctx.__aenter__ = AsyncMock(return_value=sse_ctx)
        sse_ctx.__aexit__ = AsyncMock(return_value=False)
        sse_ctx.__aiter__ = _mock_aiter
        provider._service.astream = AsyncMock(return_value=sse_ctx)

        request = ModelRequest(messages=[Message(role=Role.USER, content="Hi")])
        chunks = []
        async for chunk in provider.stream(request):
            chunks.append(chunk)

        text_chunks = [c for c in chunks if c.chunk_type == StreamChunkType.TEXT_DELTA]
        assert len(text_chunks) == 2
        assert text_chunks[0].content == "Hello"
        assert text_chunks[1].content == " world"

        metadata_chunks = [c for c in chunks if c.chunk_type == StreamChunkType.METADATA]
        assert len(metadata_chunks) == 1
        assert metadata_chunks[0].usage.total_tokens == 18


# ── Message splitting ──────────────────────────────────────────────────────────


class TestSplitMessages:
    def setup_method(self):
        self.provider = _make_provider()

    def test_system_and_single_user(self):
        messages = [
            Message(role=Role.SYSTEM, content="You are helpful."),
            Message(role=Role.USER, content="Hello"),
        ]
        template, history = self.provider._split_messages(messages)
        assert len(template) == 2
        assert len(history) == 0

    def test_only_user(self):
        messages = [Message(role=Role.USER, content="Hello")]
        template, history = self.provider._split_messages(messages)
        assert len(template) == 1
        assert len(history) == 0

    def test_multi_turn_no_tools(self):
        messages = [
            Message(role=Role.SYSTEM, content="Be helpful."),
            Message(role=Role.USER, content="First question"),
            Message(role=Role.ASSISTANT, content="First answer"),
            Message(role=Role.USER, content="Second question"),
        ]
        template, history = self.provider._split_messages(messages)
        # template: system + last USER
        assert len(template) == 2
        # history: prior USER + ASSISTANT
        assert len(history) == 2

    def test_tool_round_in_history(self):
        """ASSISTANT with tool_calls + TOOL result → history."""
        tc = ToolCall(id="tc-1", name="lookup", arguments={"q": "python"})
        messages = [
            Message(role=Role.USER, content="Look up python"),
            Message(role=Role.ASSISTANT, content=None, tool_calls=[tc]),
            Message(role=Role.TOOL, content="Python is a language.", tool_call_id="tc-1"),
            Message(role=Role.USER, content="Thanks, summarize it"),
        ]
        template, history = self.provider._split_messages(messages)
        # template: last USER only
        assert len(template) == 1
        # history: first USER + ASSISTANT(tc) + TOOL
        assert len(history) == 3

    def test_assistant_with_tool_calls_serializes_arguments(self):
        tc = ToolCall(id="tc-1", name="calc", arguments={"expr": "2+2"})
        messages = [
            Message(role=Role.USER, content="Calculate"),
            Message(role=Role.ASSISTANT, content=None, tool_calls=[tc]),
            Message(role=Role.USER, content="Result?"),
        ]
        _, history = self.provider._split_messages(messages)
        # history[1] is AssistantMessage with tool_calls
        assistant_msg = history[1]
        assert assistant_msg.tool_calls is not None
        assert len(assistant_msg.tool_calls) == 1
        # arguments are JSON-serialized strings
        assert json.loads(assistant_msg.tool_calls[0].function.arguments) == {"expr": "2+2"}


# ── Tool conversion ────────────────────────────────────────────────────────────


class TestConvertTools:
    def test_basic_conversion(self):
        tools = [
            ToolDefinition(
                name="get_weather",
                description="Get the weather for a city.",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            )
        ]
        from kagent.models.aicore_provider import AICoreProvider

        result = AICoreProvider._convert_tools(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["description"] == "Get the weather for a city."
        assert "properties" in result[0]["function"]["parameters"]

    def test_multiple_tools(self):
        from kagent.models.aicore_provider import AICoreProvider

        tools = [
            ToolDefinition(name="a", description="A", parameters={}),
            ToolDefinition(name="b", description="B", parameters={}),
        ]
        result = AICoreProvider._convert_tools(tools)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "a"
        assert result[1]["function"]["name"] == "b"


# ── Response parsing ───────────────────────────────────────────────────────────


class TestParseResponse:
    def test_text_content(self):
        from kagent.models.aicore_provider import AICoreProvider

        response = _mock_completion_response(content="Hello world")
        result = AICoreProvider._parse_response(response)
        assert result.content == "Hello world"
        assert result.tool_calls is None

    def test_usage_mapped_correctly(self):
        from kagent.models.aicore_provider import AICoreProvider

        response = _mock_completion_response(
            content="Hi", prompt_tokens=20, completion_tokens=10
        )
        result = AICoreProvider._parse_response(response)
        assert result.usage.prompt_tokens == 20
        assert result.usage.completion_tokens == 10
        assert result.usage.total_tokens == 30

    def test_request_id_in_metadata(self):
        from kagent.models.aicore_provider import AICoreProvider

        response = _mock_completion_response(request_id="req-xyz")
        result = AICoreProvider._parse_response(response)
        assert result.metadata["request_id"] == "req-xyz"

    def test_empty_choices_returns_none_content(self):
        from kagent.models.aicore_provider import AICoreProvider

        response = MagicMock()
        response.request_id = "req-empty"
        response.final_result.choices = []
        response.final_result.usage = None

        result = AICoreProvider._parse_response(response)
        assert result.content is None
        assert result.tool_calls is None
        assert result.usage is None


# ── Response format builder ────────────────────────────────────────────────────


class TestBuildResponseFormat:
    def test_none_returns_none(self):
        from kagent.models.aicore_provider import AICoreProvider

        assert AICoreProvider._build_response_format(None) is None

    def test_json_object(self):
        from kagent.models.aicore_provider import AICoreProvider
        from gen_ai_hub.orchestration_v2.models.response_format import ResponseFormatJsonObject

        result = AICoreProvider._build_response_format({"type": "json_object"})
        assert isinstance(result, ResponseFormatJsonObject)

    def test_json_schema(self):
        from kagent.models.aicore_provider import AICoreProvider
        from gen_ai_hub.orchestration_v2.models.response_format import ResponseFormatJsonSchema

        rf = {
            "type": "json_schema",
            "json_schema": {
                "name": "MyModel",
                "schema": {"type": "object", "properties": {"x": {"type": "string"}}},
                "strict": False,
            },
        }
        result = AICoreProvider._build_response_format(rf)
        assert isinstance(result, ResponseFormatJsonSchema)
        assert result.json_schema.name == "MyModel"

    def test_unknown_type_returns_none(self):
        from kagent.models.aicore_provider import AICoreProvider

        result = AICoreProvider._build_response_format({"type": "unknown"})
        assert result is None
