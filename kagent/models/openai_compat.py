"""OpenAI-compatible proxy mixin — shared logic for providers that support proxy mode.

AnthropicProvider and GeminiProvider both support an OpenAI-compatible proxy
fallback (``/chat/completions``).  The proxy-side methods (_build_openai_kwargs,
_do_complete_openai, _do_stream_openai) and client creation are identical.
This module extracts that shared code into a reusable mixin.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from kagent.common.errors import ModelError
from kagent.domain.enums import StreamChunkType
from kagent.domain.model_types import (
    ModelRequest,
    ModelResponse,
    StreamChunk,
    TokenUsage,
    ToolCallChunk,
)

if TYPE_CHECKING:
    from kagent.models.config import ModelConfig

try:
    import openai as _openai_mod

    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False


def create_openai_client(config: ModelConfig) -> Any:
    """Create an ``AsyncOpenAI`` client from a :class:`ModelConfig`.

    Raises :class:`ImportError` if the ``openai`` package is not installed.
    """
    if not _HAS_OPENAI:
        raise ImportError(
            "openai package is required for OpenAI-compatible proxy mode. "
            "Install with: pip install openai"
        )
    client_kwargs: dict[str, Any] = {"api_key": config.api_key}
    if config.base_url:
        client_kwargs["base_url"] = config.base_url
    return _openai_mod.AsyncOpenAI(**client_kwargs)


class OpenAICompatMixin:
    """Mixin providing OpenAI-compatible proxy support.

    Expects the host class to set:
    - ``self._openai_client`` — an ``AsyncOpenAI`` instance (via :func:`create_openai_client`)
    - ``self._config`` — a :class:`ModelConfig`
    - ``self._provider_label`` — a human-readable label for error messages
      (e.g. ``"Anthropic (proxy)"``)
    """

    _openai_client: Any
    _config: Any  # ModelConfig
    _provider_label: str

    # ── Non-streaming ─────────────────────────────────────────────────────

    async def _do_complete_openai(self, request: ModelRequest) -> ModelResponse:
        kwargs = self._build_openai_kwargs(request)
        try:
            response = await self._openai_client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise ModelError(f"{self._provider_label} completion failed: {exc}") from exc
        from kagent.models.converters import openai_response_to_model_response

        return openai_response_to_model_response(response)

    # ── Streaming ─────────────────────────────────────────────────────────

    async def _do_stream_openai(self, request: ModelRequest) -> AsyncIterator[StreamChunk]:
        kwargs = self._build_openai_kwargs(request)
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}

        try:
            stream = await self._openai_client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise ModelError(f"{self._provider_label} stream failed: {exc}") from exc

        async for event in stream:
            if not event.choices:
                if event.usage:
                    yield StreamChunk(
                        chunk_type=StreamChunkType.METADATA,
                        usage=TokenUsage(
                            prompt_tokens=event.usage.prompt_tokens,
                            completion_tokens=event.usage.completion_tokens,
                            total_tokens=event.usage.total_tokens,
                        ),
                    )
                continue

            delta = event.choices[0].delta

            if delta.content:
                yield StreamChunk(
                    chunk_type=StreamChunkType.TEXT_DELTA,
                    content=delta.content,
                )

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    if tc.function and tc.function.name:
                        yield StreamChunk(
                            chunk_type=StreamChunkType.TOOL_CALL_START,
                            tool_call=ToolCallChunk(
                                id=tc.id,
                                name=tc.function.name,
                                arguments_delta=tc.function.arguments,
                            ),
                        )
                    elif tc.function and tc.function.arguments:
                        yield StreamChunk(
                            chunk_type=StreamChunkType.TOOL_CALL_DELTA,
                            tool_call=ToolCallChunk(
                                arguments_delta=tc.function.arguments,
                            ),
                        )

    # ── Kwargs builder ────────────────────────────────────────────────────

    def _build_openai_kwargs(self, request: ModelRequest) -> dict[str, Any]:
        """Build kwargs for the OpenAI-compatible proxy."""
        from kagent.models.converters import messages_to_openai, tools_to_openai

        kwargs: dict[str, Any] = {
            "model": request.model or self._config.model_name,
            "messages": messages_to_openai(request.messages),
        }
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        elif self._config.temperature is not None:
            kwargs["temperature"] = self._config.temperature
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        elif self._config.max_tokens is not None:
            kwargs["max_tokens"] = self._config.max_tokens
        if request.tools:
            kwargs["tools"] = tools_to_openai(request.tools)
        if request.tool_choice:
            kwargs["tool_choice"] = request.tool_choice
        if request.response_format:
            kwargs["response_format"] = request.response_format
        return kwargs
