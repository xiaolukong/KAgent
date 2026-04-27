"""AICoreProvider — wraps SAP generative-ai-hub-sdk orchestration_v2.

Activated transparently when ``configure(backend="aicore")`` is set (the default).
Users keep their existing model strings unchanged:

    configure()  # backend="aicore" by default
    KAgent(model="openai:gpt-4o")       # same string, now routed via AI Core
    KAgent(model="anthropic:claude-3-7-sonnet")

Credentials are read from AICORE_* env vars (preferred) or via configure():
    AICORE_AUTH_URL, AICORE_CLIENT_ID, AICORE_CLIENT_SECRET,
    AICORE_BASE_URL, AICORE_RESOURCE_GROUP
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel

from kagent.common.config import get_config
from kagent.common.errors import ConfigError, ModelError
from kagent.domain.entities import Message, ToolCall
from kagent.domain.enums import Role, StreamChunkType
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


class AICoreProvider(BaseModelProvider):
    """Model provider backed by SAP AI Core (generative-ai-hub-sdk orchestration_v2).

    Tool calling, streaming, and structured output all flow through
    OrchestrationService — no secondary client is needed.
    The KAgent AgentLoop manages the agentic loop; this provider handles
    exactly one LLM call per invocation.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._proxy_client = self._build_proxy_client()
        self._service = self._build_service()

    # ── Client construction ────────────────────────────────────────────────

    def _build_proxy_client(self) -> Any:
        try:
            from gen_ai_hub.proxy import GenAIHubProxyClient
        except ImportError as exc:
            raise ConfigError(
                "generative-ai-hub-sdk is not installed. "
                "Run: pip install 'kagent[aicore]'"
            ) from exc

        global_cfg = get_config()
        kwargs: dict[str, str] = {}
        if global_cfg.aicore_auth_url:
            kwargs["auth_url"] = global_cfg.aicore_auth_url
        if global_cfg.aicore_client_id:
            kwargs["client_id"] = global_cfg.aicore_client_id
        if global_cfg.aicore_client_secret:
            kwargs["client_secret"] = global_cfg.aicore_client_secret
        if global_cfg.aicore_base_url:
            kwargs["base_url"] = global_cfg.aicore_base_url
        if global_cfg.aicore_resource_group:
            kwargs["resource_group"] = global_cfg.aicore_resource_group
        # If no kwargs, GenAIHubProxyClient reads AICORE_* env vars itself
        return GenAIHubProxyClient(**kwargs)

    def _build_service(self) -> Any:
        from gen_ai_hub.orchestration_v2.service import OrchestrationService

        return OrchestrationService(proxy_client=self._proxy_client)

    # ── ModelInfo ─────────────────────────────────────────────────────────

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(provider="aicore", model_name=self._config.model_name)

    # ── Public entry points ───────────────────────────────────────────────

    async def _do_complete(self, request: ModelRequest) -> ModelResponse:
        from gen_ai_hub.orchestration_v2.exceptions import OrchestrationError

        orch_config, history = self._build_orch_request(request)
        try:
            response = await self._service.arun(
                config=orch_config,
                history=history,
                timeout=self._config.timeout,
            )
        except OrchestrationError as exc:
            raise ModelError(
                f"AI Core orchestration error [{exc.code}] "
                f"at {exc.location}: {exc.message}"
            ) from exc
        return self._parse_response(response)

    async def _do_stream(self, request: ModelRequest) -> AsyncIterator[StreamChunk]:
        from gen_ai_hub.orchestration_v2.exceptions import OrchestrationError

        orch_config, history = self._build_orch_request(request, stream=True)
        try:
            sse = await self._service.astream(
                config=orch_config,
                history=history,
                timeout=self._config.timeout,
            )
            async for event in sse:
                if not event.final_result or not event.final_result.choices:
                    continue
                delta = event.final_result.choices[0].delta

                if delta.content:
                    yield StreamChunk(
                        chunk_type=StreamChunkType.TEXT_DELTA,
                        content=delta.content,
                    )

                for tc in delta.tool_calls or []:
                    if tc.function and tc.function.name:
                        yield StreamChunk(
                            chunk_type=StreamChunkType.TOOL_CALL_START,
                            tool_call=ToolCallChunk(
                                id=tc.id or "",
                                name=tc.function.name,
                                arguments_delta=tc.function.arguments or "",
                            ),
                        )
                    elif tc.function and tc.function.arguments:
                        yield StreamChunk(
                            chunk_type=StreamChunkType.TOOL_CALL_DELTA,
                            tool_call=ToolCallChunk(
                                arguments_delta=tc.function.arguments,
                            ),
                        )

                if event.final_result.usage:
                    u = event.final_result.usage
                    yield StreamChunk(
                        chunk_type=StreamChunkType.METADATA,
                        usage=TokenUsage(
                            prompt_tokens=u.prompt_tokens,
                            completion_tokens=u.completion_tokens,
                            total_tokens=u.total_tokens,
                        ),
                    )
        except OrchestrationError as exc:
            raise ModelError(
                f"AI Core stream error [{exc.code}]: {exc.message}"
            ) from exc

    # ── Structured output schema injection ────────────────────────────────

    def _inject_response_schema(
        self,
        request: ModelRequest,
        response_model: type[BaseModel],
    ) -> ModelRequest:
        """Override: pass JSON Schema via response_format dict.

        _build_orch_request() reads request.response_format and converts it
        to the SDK's ResponseFormatJsonSchema before building the Template.
        """
        schema = response_model.model_json_schema()
        self._add_additional_properties_false(schema)
        return request.model_copy(
            update={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "schema": schema,
                        "strict": False,
                    },
                }
            }
        )

    # ── Request builder ───────────────────────────────────────────────────

    def _build_orch_request(
        self, request: ModelRequest, *, stream: bool = False
    ) -> tuple[Any, list[Any]]:
        """Convert a KAgent ModelRequest into (OrchestrationConfig, history)."""
        from gen_ai_hub.orchestration_v2.models.config import (
            ModuleConfig,
            OrchestrationConfig,
        )
        from gen_ai_hub.orchestration_v2.models.llm_model_details import LLMModelDetails
        from gen_ai_hub.orchestration_v2.models.streaming import GlobalStreamOptions
        from gen_ai_hub.orchestration_v2.models.template import (
            PromptTemplatingModuleConfig,
            Template,
        )

        template_messages, history = self._split_messages(request.messages)

        params: dict[str, Any] = {}
        temperature = request.temperature if request.temperature is not None else self._config.temperature
        if temperature is not None:
            params["temperature"] = temperature
        max_tokens = request.max_tokens if request.max_tokens is not None else self._config.max_tokens
        if max_tokens is not None:
            params["max_completion_tokens"] = max_tokens

        llm = LLMModelDetails(name=self._config.model_name, params=params or None)

        tools = self._convert_tools(request.tools) if request.tools else None
        response_format = self._build_response_format(request.response_format)

        template = Template(
            template=template_messages,
            tools=tools,
            response_format=response_format,
        )
        module_config = ModuleConfig(
            prompt_templating=PromptTemplatingModuleConfig(prompt=template, model=llm)
        )
        stream_opts = GlobalStreamOptions(enabled=True) if stream else None
        orch_config = OrchestrationConfig(modules=module_config, stream=stream_opts)

        return orch_config, history

    @staticmethod
    def _build_response_format(response_format: dict[str, Any] | None) -> Any:
        """Convert KAgent response_format dict to orchestration_v2 model."""
        if response_format is None:
            return None

        from gen_ai_hub.orchestration_v2.models.response_format import (
            JSONResponseSchema,
            ResponseFormatJsonObject,
            ResponseFormatJsonSchema,
        )

        rf_type = response_format.get("type")
        if rf_type == "json_object":
            return ResponseFormatJsonObject()
        if rf_type == "json_schema" and "json_schema" in response_format:
            js = response_format["json_schema"]
            return ResponseFormatJsonSchema(
                json_schema=JSONResponseSchema(
                    name=js.get("name", "output"),
                    schema=js.get("schema", {}),
                    strict=js.get("strict", False),
                )
            )
        return None

    # ── Message splitting ─────────────────────────────────────────────────

    def _split_messages(
        self, messages: list[Message]
    ) -> tuple[list[Any], list[Any]]:
        """Split KAgent messages into (template_messages, history).

        Template receives: system message (if any) + last user message.
        History receives:  all prior turns (enables multi-turn tool calling).
        """
        from gen_ai_hub.orchestration_v2.models.message import (
            AssistantMessage,
            FunctionCall,
            MessageToolCall,
            SystemMessage,
            ToolChatMessage,
            UserMessage,
        )

        # Find index of last USER message
        last_user_idx: int | None = None
        for i, msg in enumerate(messages):
            if msg.role == Role.USER:
                last_user_idx = i

        template_messages: list[Any] = []
        history_messages: list[Any] = []

        for i, msg in enumerate(messages):
            if msg.role == Role.SYSTEM:
                template_messages.append(SystemMessage(content=msg.content or ""))
                continue

            if msg.role == Role.USER and i == last_user_idx:
                template_messages.append(UserMessage(content=msg.content or ""))
                continue

            # All other messages → history
            if msg.role == Role.USER:
                history_messages.append(UserMessage(content=msg.content or ""))

            elif msg.role == Role.ASSISTANT:
                if msg.tool_calls:
                    sdk_tool_calls = [
                        MessageToolCall(
                            id=tc.id,
                            function=FunctionCall(
                                name=tc.name,
                                arguments=json.dumps(tc.arguments),
                            ),
                        )
                        for tc in msg.tool_calls
                    ]
                    history_messages.append(
                        AssistantMessage(content=msg.content, tool_calls=sdk_tool_calls)
                    )
                else:
                    history_messages.append(AssistantMessage(content=msg.content))

            elif msg.role == Role.TOOL:
                history_messages.append(
                    ToolChatMessage(
                        tool_call_id=msg.tool_call_id or "",
                        content=msg.content or "",
                    )
                )

        return template_messages, history_messages

    # ── Tool definition conversion ────────────────────────────────────────

    @staticmethod
    def _convert_tools(tools: list[Any]) -> list[dict[str, Any]]:
        """Convert KAgent ToolDefinitions to AI Core JSON Schema dicts.

        Uses the plain dict format supported by orchestration_v2 Template(tools=[...]).
        """
        result: list[dict[str, Any]] = []
        for t in tools:
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                        "strict": False,
                    },
                }
            )
        return result

    # ── Response parser ───────────────────────────────────────────────────

    @staticmethod
    def _parse_response(response: Any) -> ModelResponse:
        """Convert CompletionPostResponse → KAgent ModelResponse."""
        llm_result = response.final_result
        choice = llm_result.choices[0] if llm_result.choices else None

        content: str | None = None
        tool_calls: list[ToolCall] | None = None

        if choice:
            msg = choice.message
            content = msg.content if msg.content else None
            if msg.tool_calls:
                tool_calls = [
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=tc.function.parse_arguments(),
                    )
                    for tc in msg.tool_calls
                ]

        usage: TokenUsage | None = None
        if llm_result.usage:
            u = llm_result.usage
            usage = TokenUsage(
                prompt_tokens=u.prompt_tokens,
                completion_tokens=u.completion_tokens,
                total_tokens=u.total_tokens,
            )

        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            metadata={"request_id": response.request_id},
        )
