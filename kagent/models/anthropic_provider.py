"""Anthropic model provider implementation.

When a custom base_url is set (i.e. an OpenAI-compatible proxy), this provider
automatically delegates to the OpenAI protocol so that requests like
``anthropic:claude-opus-4-6`` work transparently against proxies that speak
the ``/chat/completions`` format.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel

from kagent.common.errors import ModelError
from kagent.domain.enums import StreamChunkType
from kagent.domain.model_types import (
    ModelInfo,
    ModelRequest,
    ModelResponse,
    StreamChunk,
    TokenUsage,
    ToolCallChunk,
)
from kagent.models.base import BaseModelProvider
from kagent.models.config import ModelConfig
from kagent.models.converters import (
    anthropic_response_to_model_response,
    messages_to_anthropic,
    tools_to_anthropic,
)
from kagent.models.openai_compat import OpenAICompatMixin, create_openai_client

try:
    import anthropic

    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

# Well-known Anthropic API hosts — requests to these use the native SDK.
_ANTHROPIC_HOSTS = {"api.anthropic.com", "anthropic.com"}


def _is_anthropic_native_url(base_url: str | None) -> bool:
    """Return True if *base_url* is None (default) or points to Anthropic's own API."""
    if base_url is None:
        return True
    from urllib.parse import urlparse

    host = urlparse(base_url).hostname or ""
    return host in _ANTHROPIC_HOSTS


class AnthropicProvider(OpenAICompatMixin, BaseModelProvider):
    """Provider for Anthropic Claude models.

    Behaviour depends on the resolved ``base_url``:

    * **No custom base_url / official Anthropic URL** — uses the native
      ``anthropic`` SDK (``/v1/messages`` format).
    * **Custom base_url (e.g. an OpenAI-compatible proxy)** — automatically
      switches to the ``openai`` SDK (``/chat/completions`` format) so that
      ``anthropic:claude-opus-4-6`` works against any proxy that speaks
      the OpenAI protocol.
    """

    _provider_label = "Anthropic (proxy)"

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)

        self._use_openai_compat = not _is_anthropic_native_url(self._config.base_url)

        if self._use_openai_compat:
            self._openai_client = create_openai_client(self._config)
        else:
            if not _HAS_ANTHROPIC:
                raise ImportError(
                    "anthropic package is required. Install with: pip install kagent[anthropic]"
                )
            client_kwargs: dict[str, Any] = {"api_key": self._config.api_key}
            if self._config.base_url:
                client_kwargs["base_url"] = self._config.base_url
            self._client = anthropic.AsyncAnthropic(**client_kwargs)

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(provider="anthropic", model_name=self._config.model_name)

    # ── Structured output ────────────────────────────────────────────────

    def _inject_response_schema(
        self,
        request: ModelRequest,
        response_model: type[BaseModel],
    ) -> ModelRequest:
        """Inject structured output via the **tool-use trick**.

        Both native and proxy modes use the same approach: define a synthetic
        tool whose ``parameters`` (input_schema) match the desired Pydantic
        model, then force the model to call it via ``tool_choice``.  The
        structured data is extracted from the tool call arguments.

        This avoids any prompt injection and works reliably because Anthropic
        models (and the proxy) fully support function calling.
        """
        from kagent.domain.entities import ToolDefinition

        schema = response_model.model_json_schema()

        struct_tool = ToolDefinition(
            name=f"structured_output_{response_model.__name__}",
            description=f"Output structured data as {response_model.__name__}",
            parameters=schema,
        )
        existing_tools = list(request.tools or [])
        existing_tools.append(struct_tool)

        if self._use_openai_compat:
            # Proxy mode — OpenAI tool_choice format
            tool_choice = {
                "type": "function",
                "function": {"name": struct_tool.name},
            }
        else:
            # Native mode — Anthropic tool_choice format
            tool_choice = {"type": "tool", "name": struct_tool.name}

        return request.model_copy(update={"tools": existing_tools, "tool_choice": tool_choice})

    # ── Non-streaming ─────────────────────────────────────────────────────

    async def _do_complete(self, request: ModelRequest) -> ModelResponse:
        if self._use_openai_compat:
            return await self._do_complete_openai(request)
        return await self._do_complete_native(request)

    async def _do_complete_native(self, request: ModelRequest) -> ModelResponse:
        kwargs = self._build_native_kwargs(request)
        try:
            response = await self._client.messages.create(**kwargs)
        except Exception as exc:
            raise ModelError(f"Anthropic completion failed: {exc}") from exc
        return anthropic_response_to_model_response(response)

    # ── Streaming ─────────────────────────────────────────────────────────

    async def _do_stream(self, request: ModelRequest) -> AsyncIterator[StreamChunk]:
        if self._use_openai_compat:
            async for chunk in self._do_stream_openai(request):
                yield chunk
        else:
            async for chunk in self._do_stream_native(request):
                yield chunk

    async def _do_stream_native(self, request: ModelRequest) -> AsyncIterator[StreamChunk]:
        kwargs = self._build_native_kwargs(request)

        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    if event.type == "content_block_start":
                        block = event.content_block
                        if block.type == "thinking":
                            pass  # thinking content will arrive in deltas
                        elif block.type == "text":
                            pass  # text will arrive in deltas
                        elif block.type == "tool_use":
                            yield StreamChunk(
                                chunk_type=StreamChunkType.TOOL_CALL_START,
                                tool_call=ToolCallChunk(
                                    id=block.id,
                                    name=block.name,
                                ),
                            )
                    elif event.type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "thinking_delta":
                            yield StreamChunk(
                                chunk_type=StreamChunkType.THINKING_DELTA,
                                thinking=delta.thinking,
                            )
                        elif delta.type == "text_delta":
                            yield StreamChunk(
                                chunk_type=StreamChunkType.TEXT_DELTA,
                                content=delta.text,
                            )
                        elif delta.type == "input_json_delta":
                            yield StreamChunk(
                                chunk_type=StreamChunkType.TOOL_CALL_DELTA,
                                tool_call=ToolCallChunk(
                                    arguments_delta=delta.partial_json,
                                ),
                            )
                    elif event.type == "content_block_stop":
                        yield StreamChunk(chunk_type=StreamChunkType.TOOL_CALL_END)
                    elif event.type == "message_delta":
                        if hasattr(event, "usage") and event.usage:
                            yield StreamChunk(
                                chunk_type=StreamChunkType.METADATA,
                                usage=TokenUsage(
                                    prompt_tokens=0,
                                    completion_tokens=event.usage.output_tokens,
                                    total_tokens=event.usage.output_tokens,
                                ),
                            )
        except Exception as exc:
            raise ModelError(f"Anthropic stream failed: {exc}") from exc

    # ── Kwargs builder ────────────────────────────────────────────────────

    def _build_native_kwargs(self, request: ModelRequest) -> dict[str, Any]:
        """Build kwargs for the native Anthropic SDK."""
        system_prompt, messages = messages_to_anthropic(request.messages)
        kwargs: dict[str, Any] = {
            "model": request.model or self._config.model_name,
            "messages": messages,
            "max_tokens": request.max_tokens or self._config.max_tokens or 4096,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        elif self._config.temperature is not None:
            kwargs["temperature"] = self._config.temperature
        if request.tools:
            kwargs["tools"] = tools_to_anthropic(request.tools)
        if request.tool_choice:
            kwargs["tool_choice"] = request.tool_choice
        return kwargs
