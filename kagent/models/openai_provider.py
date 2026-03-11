"""OpenAI model provider implementation."""

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
    messages_to_openai,
    openai_response_to_model_response,
    tools_to_openai,
)

try:
    import openai

    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False

# Models that support arbitrary temperature values.
# All other models are forced to temperature=1.
_FLEXIBLE_TEMP_MODELS = {"gpt-4o", "gpt-4.1"}


class OpenAIProvider(BaseModelProvider):
    """Provider for OpenAI and OpenAI-compatible APIs."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        if not _HAS_OPENAI:
            raise ImportError(
                "openai package is required. Install with: pip install kagent[openai]"
            )
        client_kwargs: dict[str, Any] = {"api_key": self._config.api_key}
        if self._config.base_url:
            client_kwargs["base_url"] = self._config.base_url
        self._client = openai.AsyncOpenAI(**client_kwargs)

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(provider="openai", model_name=self._config.model_name)

    async def complete(
        self,
        request: ModelRequest,
        *,
        response_model: type[BaseModel] | None = None,
    ) -> ModelResponse:
        """Non-streaming completion with optional structured output.

        When ``response_model`` is provided, uses the OpenAI SDK's
        ``chat.completions.parse(response_format=PydanticType)`` which
        handles schema injection and response parsing at the SDK level.
        """
        if response_model is not None:
            return await self._do_complete_parsed(request, response_model)
        return await super().complete(request, response_model=None)

    async def _do_complete_parsed(
        self,
        request: ModelRequest,
        response_model: type[BaseModel],
    ) -> ModelResponse:
        """Structured output via OpenAI SDK's ``chat.completions.parse``."""
        kwargs = self._build_kwargs(request)
        # Remove response_format if set — parse() handles it via response_format param
        kwargs.pop("response_format", None)
        try:
            parsed_response = await self._client.chat.completions.parse(
                response_format=response_model,
                **kwargs,
            )
        except Exception as exc:
            raise ModelError(f"OpenAI structured output failed: {exc}") from exc

        response = openai_response_to_model_response(parsed_response)
        # Extract the SDK-parsed Pydantic instance
        choice = parsed_response.choices[0]
        if choice.message.parsed is not None:
            response.parsed = choice.message.parsed
        return response

    async def _do_complete(self, request: ModelRequest) -> ModelResponse:
        kwargs = self._build_kwargs(request)
        try:
            response = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise ModelError(f"OpenAI completion failed: {exc}") from exc
        return openai_response_to_model_response(response)

    async def _do_stream(self, request: ModelRequest) -> AsyncIterator[StreamChunk]:
        kwargs = self._build_kwargs(request)
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}

        try:
            stream = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise ModelError(f"OpenAI stream failed: {exc}") from exc

        async for event in stream:
            if not event.choices:
                # Final chunk with usage info only
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

    def _build_kwargs(self, request: ModelRequest) -> dict[str, Any]:
        model_name = request.model or self._config.model_name
        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": messages_to_openai(request.messages),
        }
        # Resolve desired temperature, then clamp for models that only accept 1
        desired_temp = request.temperature if request.temperature is not None else self._config.temperature
        kwargs["temperature"] = self.clamp_temperature(model_name, desired_temp)
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
    
    def clamp_temperature(self, model_name: str, temperature: float | None) -> float:
        """Return *temperature* unchanged for models that accept it, else 1."""
        if model_name in _FLEXIBLE_TEMP_MODELS:
            return temperature if temperature is not None else 1.0
        return 1.0
