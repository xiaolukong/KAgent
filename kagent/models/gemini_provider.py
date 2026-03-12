"""Google Gemini model provider implementation.

When a custom base_url is set (i.e. an OpenAI-compatible proxy), this provider
automatically delegates to the OpenAI protocol so that requests like
``gemini:gemini-2.5-flash`` work transparently against proxies that speak
the ``/chat/completions`` format.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel

from kagent.common.errors import ModelError
from kagent.domain.entities import ToolCall
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
from kagent.models.converters import messages_to_gemini, tools_to_gemini
from kagent.models.openai_compat import OpenAICompatMixin, create_openai_client

try:
    import google.generativeai as genai

    _HAS_GEMINI = True
except ImportError:
    _HAS_GEMINI = False

# Well-known Google AI API hosts — requests to these use the native SDK.
_GOOGLE_HOSTS = {
    "generativelanguage.googleapis.com",
    "aiplatform.googleapis.com",
}


def _is_gemini_native_url(base_url: str | None) -> bool:
    """Return True if *base_url* is None (default) or points to Google's own API."""
    if base_url is None:
        return True
    from urllib.parse import urlparse

    host = urlparse(base_url).hostname or ""
    return host in _GOOGLE_HOSTS


class GeminiProvider(OpenAICompatMixin, BaseModelProvider):
    """Provider for Google Gemini models.

    Behaviour depends on the resolved ``base_url``:

    * **No custom base_url / official Google URL** — uses the native
      ``google.generativeai`` SDK (gRPC protocol).
    * **Custom base_url (e.g. an OpenAI-compatible proxy)** — automatically
      switches to the ``openai`` SDK (``/chat/completions`` format) so that
      ``gemini:gemini-2.5-flash`` works against any proxy that speaks
      the OpenAI protocol.
    """

    _provider_label = "Gemini (proxy)"

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)

        self._use_openai_compat = not _is_gemini_native_url(self._config.base_url)

        if self._use_openai_compat:
            self._openai_client = create_openai_client(self._config)
        else:
            if not _HAS_GEMINI:
                raise ImportError(
                    "google-generativeai package is required. "
                    "Install with: pip install kagent[gemini]"
                )
            if self._config.api_key:
                genai.configure(api_key=self._config.api_key)
            self._model = genai.GenerativeModel(self._config.model_name)

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(provider="gemini", model_name=self._config.model_name)

    def _inject_response_schema(
        self,
        request: ModelRequest,
        response_model: type[BaseModel],
    ) -> ModelRequest:
        """Inject structured output schema.

        * **Proxy mode**: delegates to the base class which sets
          ``response_format: json_schema`` (the proxy enforces it for Gemini).
        * **Native mode**: uses ``generation_config.response_schema``.
        """
        if self._use_openai_compat:
            return super()._inject_response_schema(request, response_model)

        schema = response_model.model_json_schema()
        return request.model_copy(
            update={
                "response_format": {
                    "response_mime_type": "application/json",
                    "response_schema": schema,
                },
            }
        )

    # ── Non-streaming ─────────────────────────────────────────────────────

    async def _do_complete(self, request: ModelRequest) -> ModelResponse:
        if self._use_openai_compat:
            return await self._do_complete_openai(request)
        return await self._do_complete_native(request)

    async def _do_complete_native(self, request: ModelRequest) -> ModelResponse:
        system_instruction, contents = messages_to_gemini(request.messages)
        kwargs = self._build_native_kwargs(request, system_instruction)

        try:
            response = await self._model.generate_content_async(contents, **kwargs)
        except Exception as exc:
            raise ModelError(f"Gemini completion failed: {exc}") from exc

        return self._parse_native_response(response)

    # ── Streaming ─────────────────────────────────────────────────────────

    async def _do_stream(self, request: ModelRequest) -> AsyncIterator[StreamChunk]:
        if self._use_openai_compat:
            async for chunk in self._do_stream_openai(request):
                yield chunk
        else:
            async for chunk in self._do_stream_native(request):
                yield chunk

    async def _do_stream_native(self, request: ModelRequest) -> AsyncIterator[StreamChunk]:
        system_instruction, contents = messages_to_gemini(request.messages)
        kwargs = self._build_native_kwargs(request, system_instruction)
        kwargs["stream"] = True

        try:
            response = await self._model.generate_content_async(contents, **kwargs)
        except Exception as exc:
            raise ModelError(f"Gemini stream failed: {exc}") from exc

        async for chunk in response:
            if chunk.text:
                yield StreamChunk(
                    chunk_type=StreamChunkType.TEXT_DELTA,
                    content=chunk.text,
                )
            if hasattr(chunk, "candidates") and chunk.candidates:
                for candidate in chunk.candidates:
                    for part in candidate.content.parts:
                        if hasattr(part, "function_call") and part.function_call:
                            fc = part.function_call
                            yield StreamChunk(
                                chunk_type=StreamChunkType.TOOL_CALL_START,
                                tool_call=ToolCallChunk(
                                    name=fc.name,
                                    arguments_delta=json.dumps(dict(fc.args)) if fc.args else None,
                                ),
                            )

    # ── Kwargs builder ────────────────────────────────────────────────────

    def _build_native_kwargs(
        self,
        request: ModelRequest,
        system_instruction: str | None,
    ) -> dict[str, Any]:
        """Build kwargs for the native Google generativeai SDK."""
        kwargs: dict[str, Any] = {}
        generation_config: dict[str, Any] = {}

        if request.temperature is not None:
            generation_config["temperature"] = request.temperature
        elif self._config.temperature is not None:
            generation_config["temperature"] = self._config.temperature
        if request.max_tokens is not None:
            generation_config["max_output_tokens"] = request.max_tokens
        elif self._config.max_tokens is not None:
            generation_config["max_output_tokens"] = self._config.max_tokens
        if request.response_format:
            generation_config.update(request.response_format)

        if generation_config:
            kwargs["generation_config"] = generation_config

        if request.tools:
            kwargs["tools"] = tools_to_gemini(request.tools)

        return kwargs

    def _parse_native_response(self, response: Any) -> ModelResponse:
        """Convert a native Gemini response to a domain ModelResponse."""
        content_text: str | None = None
        tool_calls: list[ToolCall] = []

        if response.candidates:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    content_text = (content_text or "") + part.text
                if hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    tool_calls.append(
                        ToolCall(
                            name=fc.name,
                            arguments=dict(fc.args) if fc.args else {},
                        )
                    )

        usage = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            usage = TokenUsage(
                prompt_tokens=getattr(um, "prompt_token_count", 0),
                completion_tokens=getattr(um, "candidates_token_count", 0),
                total_tokens=getattr(um, "total_token_count", 0),
            )

        return ModelResponse(
            content=content_text,
            tool_calls=tool_calls if tool_calls else None,
            usage=usage,
            model=self._config.model_name,
        )
