"""AgentLoop — core orchestration cycle for an agent run."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel

from kagent.agent.interceptor import InterceptBlockedError, InterceptorPipeline
from kagent.agent.prompt_builder import PromptBuilder
from kagent.agent.steering import SteeringController
from kagent.common.logging import get_logger
from kagent.context.manager import ContextManager
from kagent.context.transformer import ContextTransformer
from kagent.domain.entities import Message, ToolCall
from kagent.domain.enums import EventType, Role, StreamChunkType
from kagent.domain.events import AgentEvent, LLMEvent
from kagent.domain.model_types import ModelResponse, StreamChunk
from kagent.domain.protocols import IEventBus, IModelProvider
from kagent.tools.executor import ToolExecutor
from kagent.tools.registry import ToolRegistry

logger = get_logger("agent.loop")


class AgentLoop:
    """The main orchestration loop: prompt -> model -> parse -> tool -> repeat."""

    def __init__(
        self,
        *,
        model_provider: IModelProvider,
        event_bus: IEventBus,
        tool_registry: ToolRegistry,
        tool_executor: ToolExecutor,
        context_manager: ContextManager,
        prompt_builder: PromptBuilder,
        steering: SteeringController,
        pipeline: InterceptorPipeline | None = None,
        transformer: ContextTransformer | None = None,
        max_turns: int = 10,
    ) -> None:
        self._provider = model_provider
        self._event_bus = event_bus
        self._tool_registry = tool_registry
        self._tool_executor = tool_executor
        self._context = context_manager
        self._prompt_builder = prompt_builder
        self._steering = steering
        self._pipeline = pipeline or InterceptorPipeline()
        self._transformer = transformer or ContextTransformer()
        self._max_turns = max_turns

    async def run(
        self,
        user_input: str,
        *,
        response_model: type[BaseModel] | None = None,
    ) -> ModelResponse:
        """Execute the agent loop (non-streaming) and return the final response."""
        self._context.add_message(Message(role=Role.USER, content=user_input))

        final_response: ModelResponse | None = None

        for turn in range(self._max_turns):
            # Check steering
            if self._steering.is_aborted:
                break
            directive = self._steering.get_pending_directive()
            if directive:
                logger.info("Steering directive received: %s", directive.event_type)
                if directive.event_type == EventType.STEERING_REDIRECT:
                    redirect_text = directive.payload.get("directive", "")
                    if redirect_text:
                        self._context.add_message(Message(role=Role.USER, content=redirect_text))
                else:
                    break

            await self._event_bus.publish(
                AgentEvent(
                    event_type=EventType.AGENT_LOOP_ITERATION,
                    payload={"turn_number": turn},
                    source="agent_loop",
                )
            )

            # ① before_prompt_build
            messages = self._context.get_messages()
            messages = await self._transformer.apply(messages)
            tool_defs = self._tool_registry.list_definitions() or []
            prompt_ctx: dict[str, Any] = {"messages": messages, "tool_definitions": tool_defs}
            prompt_ctx = await self._pipeline.run("before_prompt_build", prompt_ctx)
            messages = prompt_ctx["messages"]
            tool_defs = prompt_ctx["tool_definitions"]

            request = self._prompt_builder.build(messages, tool_defs or None)

            # ② before_llm_request
            request = await self._pipeline.run("before_llm_request", request)

            await self._event_bus.publish(
                LLMEvent(
                    event_type=EventType.LLM_REQUEST_SENT,
                    payload={"model": request.model, "message_count": len(request.messages)},
                    source="agent_loop",
                )
            )

            response = await self._provider.complete(request, response_model=response_model)

            # ③ after_llm_response
            response = await self._pipeline.run("after_llm_response", response)

            await self._event_bus.publish(
                LLMEvent(
                    event_type=EventType.LLM_RESPONSE_RECEIVED,
                    payload={
                        "has_tool_calls": response.has_tool_calls,
                        "usage": response.usage.model_dump() if response.usage else None,
                    },
                    source="agent_loop",
                )
            )

            # Add assistant message to context
            assistant_msg = Message(
                role=Role.ASSISTANT,
                content=response.content,
                tool_calls=response.tool_calls,
                metadata={"thinking": response.metadata["thinking"]}
                if "thinking" in response.metadata
                else {},
            )
            self._context.add_message(assistant_msg)

            # If no tool calls, we're done
            if not response.has_tool_calls:
                final_response = response
                break

            # Execute tool calls
            tool_results: list[dict[str, Any]] = []
            for tc in response.tool_calls or []:
                # ④ before_tool_call
                tc_ctx: dict[str, Any] = {
                    "tool_name": tc.name,
                    "arguments": tc.arguments,
                    "call_id": tc.id,
                }
                try:
                    tc_ctx = await self._pipeline.run("before_tool_call", tc_ctx)
                except InterceptBlockedError as exc:
                    logger.info("Tool '%s' blocked by interceptor: %s", tc.name, exc.reason)
                    tool_msg = Message(
                        role=Role.TOOL,
                        content=json.dumps({"blocked": True, "reason": exc.reason}),
                        tool_call_id=tc.id,
                        metadata={"tool_name": tc.name},
                    )
                    self._context.add_message(tool_msg)
                    tool_results.append({"tool_name": tc.name, "blocked": True})
                    continue

                result = await self._tool_executor.execute(
                    tc_ctx["tool_name"], tc_ctx["arguments"], call_id=tc_ctx["call_id"]
                )

                # ⑤ after_tool_call
                result = await self._pipeline.run("after_tool_call", result)

                tool_content = (
                    json.dumps(result.result) if result.result is not None else result.error
                )
                tool_msg = Message(
                    role=Role.TOOL,
                    content=tool_content,
                    tool_call_id=tc.id,
                    metadata={"tool_name": tc.name},
                )
                self._context.add_message(tool_msg)
                tool_results.append(
                    {"tool_name": tc.name, "result": result.result, "error": result.error}
                )

            # ⑥ after_tool_round
            tool_results = await self._pipeline.run("after_tool_round", tool_results)

            # Inject any pending follow-up messages
            for msg in self._steering.get_pending_messages():
                self._context.add_message(msg)

            # Continue loop to get model's response to tool results
            final_response = response

        result = final_response or ModelResponse(content="Max turns reached without completion.")

        # ⑦ before_return
        result = await self._pipeline.run("before_return", result)

        return result

    async def run_stream(
        self,
        user_input: str,
        *,
        response_model: type[BaseModel] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Execute the agent loop with streaming output."""
        self._context.add_message(Message(role=Role.USER, content=user_input))

        for turn in range(self._max_turns):
            if self._steering.is_aborted:
                break
            directive = self._steering.get_pending_directive()
            if directive:
                if directive.event_type == EventType.STEERING_REDIRECT:
                    redirect_text = directive.payload.get("directive", "")
                    if redirect_text:
                        self._context.add_message(Message(role=Role.USER, content=redirect_text))
                else:
                    break

            await self._event_bus.publish(
                AgentEvent(
                    event_type=EventType.AGENT_LOOP_ITERATION,
                    payload={"turn_number": turn},
                    source="agent_loop",
                )
            )

            # ① before_prompt_build
            messages = self._context.get_messages()
            messages = await self._transformer.apply(messages)
            tool_defs = self._tool_registry.list_definitions() or []
            prompt_ctx: dict[str, Any] = {"messages": messages, "tool_definitions": tool_defs}
            prompt_ctx = await self._pipeline.run("before_prompt_build", prompt_ctx)
            messages = prompt_ctx["messages"]
            tool_defs = prompt_ctx["tool_definitions"]

            request = self._prompt_builder.build(messages, tool_defs or None)

            # ② before_llm_request
            request = await self._pipeline.run("before_llm_request", request)

            await self._event_bus.publish(
                LLMEvent(
                    event_type=EventType.LLM_REQUEST_SENT,
                    payload={"model": request.model, "message_count": len(request.messages)},
                    source="agent_loop",
                )
            )

            # Collect the full response while streaming chunks out
            collected_content: list[str] = []
            collected_thinking: list[str] = []
            collected_tool_calls: dict[int, dict[str, Any]] = {}
            current_tool_idx = -1

            async for chunk in self._provider.stream(request, response_model=response_model):
                await self._event_bus.publish(
                    LLMEvent(
                        event_type=EventType.LLM_STREAM_CHUNK,
                        payload={
                            "chunk_type": (
                                chunk.chunk_type.value
                                if hasattr(chunk.chunk_type, "value")
                                else str(chunk.chunk_type)
                            ),
                        },
                        source="agent_loop",
                    )
                )
                yield chunk

                if chunk.chunk_type == StreamChunkType.TEXT_DELTA and chunk.content:
                    collected_content.append(chunk.content)
                elif chunk.chunk_type == StreamChunkType.THINKING_DELTA and chunk.thinking:
                    collected_thinking.append(chunk.thinking)
                elif chunk.chunk_type == StreamChunkType.TOOL_CALL_START and chunk.tool_call:
                    current_tool_idx += 1
                    collected_tool_calls[current_tool_idx] = {
                        "id": chunk.tool_call.id or "",
                        "name": chunk.tool_call.name or "",
                        "arguments": chunk.tool_call.arguments_delta or "",
                    }
                elif chunk.chunk_type == StreamChunkType.TOOL_CALL_DELTA and chunk.tool_call:
                    if current_tool_idx in collected_tool_calls:
                        collected_tool_calls[current_tool_idx]["arguments"] += (
                            chunk.tool_call.arguments_delta or ""
                        )

            await self._event_bus.publish(
                LLMEvent(
                    event_type=EventType.LLM_STREAM_COMPLETE,
                    payload={"content_length": sum(len(c) for c in collected_content)},
                    source="agent_loop",
                )
            )

            # Build tool calls from collected data
            tool_calls = []
            for idx in sorted(collected_tool_calls):
                tc_data = collected_tool_calls[idx]
                try:
                    args = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(id=tc_data["id"], name=tc_data["name"], arguments=args))

            full_content = "".join(collected_content) if collected_content else None

            # Build response for after_llm_response hook
            stream_metadata: dict[str, Any] = {}
            if collected_thinking:
                stream_metadata["thinking"] = "".join(collected_thinking)
            stream_response = ModelResponse(
                content=full_content,
                tool_calls=tool_calls if tool_calls else None,
                metadata=stream_metadata,
            )

            # ③ after_llm_response
            stream_response = await self._pipeline.run("after_llm_response", stream_response)
            full_content = stream_response.content
            tool_calls = list(stream_response.tool_calls) if stream_response.tool_calls else []

            # Add assistant message to context
            assistant_msg = Message(
                role=Role.ASSISTANT,
                content=full_content,
                tool_calls=tool_calls if tool_calls else None,
                metadata={"thinking": stream_response.metadata["thinking"]}
                if "thinking" in stream_response.metadata
                else {},
            )
            self._context.add_message(assistant_msg)

            # If no tool calls, we're done
            if not tool_calls:
                break

            # Execute tool calls
            tool_results: list[dict[str, Any]] = []
            for tc in tool_calls:
                # ④ before_tool_call
                tc_ctx: dict[str, Any] = {
                    "tool_name": tc.name,
                    "arguments": tc.arguments,
                    "call_id": tc.id,
                }
                try:
                    tc_ctx = await self._pipeline.run("before_tool_call", tc_ctx)
                except InterceptBlockedError as exc:
                    logger.info("Tool '%s' blocked by interceptor: %s", tc.name, exc.reason)
                    tool_msg = Message(
                        role=Role.TOOL,
                        content=json.dumps({"blocked": True, "reason": exc.reason}),
                        tool_call_id=tc.id,
                        metadata={"tool_name": tc.name},
                    )
                    self._context.add_message(tool_msg)
                    tool_results.append({"tool_name": tc.name, "blocked": True})
                    continue

                result = await self._tool_executor.execute(
                    tc_ctx["tool_name"], tc_ctx["arguments"], call_id=tc_ctx["call_id"]
                )

                # ⑤ after_tool_call
                result = await self._pipeline.run("after_tool_call", result)

                tool_content = (
                    json.dumps(result.result) if result.result is not None else result.error
                )
                tool_msg = Message(
                    role=Role.TOOL,
                    content=tool_content,
                    tool_call_id=tc.id,
                    metadata={"tool_name": tc.name},
                )
                self._context.add_message(tool_msg)
                tool_results.append(
                    {"tool_name": tc.name, "result": result.result, "error": result.error}
                )

            # ⑥ after_tool_round
            tool_results = await self._pipeline.run("after_tool_round", tool_results)

            for msg in self._steering.get_pending_messages():
                self._context.add_message(msg)
