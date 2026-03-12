"""BaseModelProvider — abstract base with shared logic for all providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel

from kagent.common.config import get_config
from kagent.common.errors import ValidationError
from kagent.domain.model_types import (
    ModelInfo,
    ModelRequest,
    ModelResponse,
    StreamChunk,
)
from kagent.models.config import ModelConfig
from kagent.models.converters import parse_structured_output


class BaseModelProvider(ABC):
    """Abstract base for all LLM provider implementations.

    Handles:
    - Global config fallback (base_url / api_key)
    - Structured output parsing
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._resolve_global_config()

    def _resolve_global_config(self) -> None:
        """Fill in api_key / base_url from global KAgentConfig if not set locally."""
        global_cfg = get_config()
        if self._config.api_key is None and global_cfg.api_key:
            self._config.api_key = global_cfg.api_key
        if self._config.base_url is None and global_cfg.base_url:
            self._config.base_url = global_cfg.base_url

    # ── Public interface ─────────────────────────────────────────────────

    async def complete(
        self,
        request: ModelRequest,
        *,
        response_model: type[BaseModel] | None = None,
    ) -> ModelResponse:
        """Non-streaming completion with optional structured output parsing."""
        if response_model is not None:
            request = self._inject_response_schema(request, response_model)

        response = await self._do_complete(request)

        if response_model is not None:
            try:
                response.parsed = self._extract_structured_output(response, response_model)
            except Exception as exc:
                raise ValidationError(f"Failed to parse structured output: {exc}") from exc

        return response

    @staticmethod
    def _extract_structured_output(
        response: ModelResponse,
        response_model: type[BaseModel],
    ) -> BaseModel:
        """Extract structured output from a model response.

        Checks two locations:
        1. **Tool call arguments** — used by the tool-use trick (Anthropic
           native & proxy, and any provider that injects a synthetic tool).
        2. **Text content** — used by providers that rely on
           ``response_format: json_schema`` (OpenAI, Gemini).
        """
        # Check tool calls first (tool-use trick)
        if response.tool_calls:
            for tc in response.tool_calls:
                if tc.name.startswith("structured_output_"):
                    return response_model.model_validate(tc.arguments)

        # Fall back to text content
        if response.content:
            return parse_structured_output(response.content, response_model)

        raise ValueError(
            "Model response contains neither tool calls nor text content "
            "for structured output parsing"
        )

    async def stream(
        self,
        request: ModelRequest,
        *,
        response_model: type[BaseModel] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Streaming completion with optional structured output on final chunk."""
        if response_model is not None:
            request = self._inject_response_schema(request, response_model)

        collected_content: list[str] = []

        async for chunk in self._do_stream(request):
            if chunk.content:
                collected_content.append(chunk.content)
            yield chunk

        # Parse structured output from the full collected content
        if response_model is not None and collected_content:
            full_text = "".join(collected_content)
            try:
                parsed = parse_structured_output(full_text, response_model)
                # Yield a final metadata chunk with the parsed result
                yield StreamChunk(
                    chunk_type="metadata",  # type: ignore[arg-type]
                    parsed=parsed,
                )
            except Exception as exc:
                raise ValidationError(f"Failed to parse structured output: {exc}") from exc

    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """Return metadata about this provider."""
        ...

    # ── Subclass implementation points ───────────────────────────────────

    @abstractmethod
    async def _do_complete(self, request: ModelRequest) -> ModelResponse:
        """Provider-specific non-streaming call."""
        ...

    @abstractmethod
    async def _do_stream(self, request: ModelRequest) -> AsyncIterator[StreamChunk]:
        """Provider-specific streaming call."""
        ...

    # ── Structured output schema injection ───────────────────────────────

    @staticmethod
    def _add_additional_properties_false(schema: dict[str, Any]) -> None:
        """Recursively add ``"additionalProperties": false`` to all object nodes.

        OpenAI's structured-output strict mode requires every ``type: "object"``
        node in the JSON Schema to carry this flag.  Pydantic's
        ``model_json_schema()`` does not emit it by default.
        """
        if schema.get("type") == "object":
            schema.setdefault("additionalProperties", False)

        # Recurse into properties
        for prop in schema.get("properties", {}).values():
            BaseModelProvider._add_additional_properties_false(prop)

        # Recurse into items (array elements)
        if "items" in schema and isinstance(schema["items"], dict):
            BaseModelProvider._add_additional_properties_false(schema["items"])

        # Recurse into $defs / definitions (nested Pydantic models)
        for defn in schema.get("$defs", {}).values():
            BaseModelProvider._add_additional_properties_false(defn)
        for defn in schema.get("definitions", {}).values():
            BaseModelProvider._add_additional_properties_false(defn)

        # Recurse into allOf / anyOf / oneOf combinators
        for key in ("allOf", "anyOf", "oneOf"):
            for sub_schema in schema.get(key, []):
                if isinstance(sub_schema, dict):
                    BaseModelProvider._add_additional_properties_false(sub_schema)

    def _inject_response_schema(
        self,
        request: ModelRequest,
        response_model: type[BaseModel],
    ) -> ModelRequest:
        """Default implementation: inject JSON Schema into response_format.

        Subclasses can override to use vendor-specific mechanisms.
        """
        schema = response_model.model_json_schema()
        self._add_additional_properties_false(schema)
        updated = request.model_copy(
            update={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "schema": schema,
                        "strict": True,
                    },
                }
            }
        )
        return updated
