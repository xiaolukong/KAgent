"""AgentLoop — core orchestration cycle for an agent run."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel

from kagent.agent.prompt_builder import PromptBuilder
from kagent.agent.steering import SteeringController
from kagent.common.logging import get_logger
from kagent.context.manager import ContextManager
from kagent.domain.entities import Message
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
        max_turns: int = 10,
    ) -> None:
        self._provider = model_provider
        self._event_bus = event_bus
        self._tool_registry = tool_registry
        self._tool_executor = tool_executor
        self._context = context_manager
        self._prompt_builder = prompt_builder
        self._steering = steering
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
                    # Redirect: inject the directive as a user message and continue
                    redirect_text = directive.payload.get("directive", "")
                    if redirect_text:
                        self._context.add_message(Message(role=Role.USER, content=redirect_text))
                    # Continue the loop — the model will respond to the new instruction
                else:
                    break

            await self._event_bus.publish(
                AgentEvent(
                    event_type=EventType.AGENT_LOOP_ITERATION,
                    payload={"turn_number": turn},
                    source="agent_loop",
                )
            )

            request = self._prompt_builder.build(
                self._context.get_messages(),
                self._tool_registry.list_definitions() or None,
            )

            await self._event_bus.publish(
                LLMEvent(
                    event_type=EventType.LLM_REQUEST_SENT,
                    payload={"model": request.model, "message_count": len(request.messages)},
                    source="agent_loop",
                )
            )

            response = await self._provider.complete(request, response_model=response_model)

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
            )
            self._context.add_message(assistant_msg)

            # If no tool calls, we're done
            if not response.has_tool_calls:
                final_response = response
                break

            # Execute tool calls
            for tc in response.tool_calls or []:
                result = await self._tool_executor.execute(tc.name, tc.arguments, call_id=tc.id)
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

            # Inject any pending follow-up messages
            for msg in self._steering.get_pending_messages():
                self._context.add_message(msg)

            # Continue loop to get model's response to tool results
            final_response = response

        return final_response or ModelResponse(content="Max turns reached without completion.")

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

            request = self._prompt_builder.build(
                self._context.get_messages(),
                self._tool_registry.list_definitions() or None,
            )

            await self._event_bus.publish(
                LLMEvent(
                    event_type=EventType.LLM_REQUEST_SENT,
                    payload={"model": request.model, "message_count": len(request.messages)},
                    source="agent_loop",
                )
            )

            # Collect the full response while streaming chunks out
            collected_content: list[str] = []
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
            from kagent.domain.entities import ToolCall

            tool_calls = []
            for idx in sorted(collected_tool_calls):
                tc_data = collected_tool_calls[idx]
                try:
                    args = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(id=tc_data["id"], name=tc_data["name"], arguments=args))

            full_content = "".join(collected_content) if collected_content else None

            # Add assistant message to context
            assistant_msg = Message(
                role=Role.ASSISTANT,
                content=full_content,
                tool_calls=tool_calls if tool_calls else None,
            )
            self._context.add_message(assistant_msg)

            # If no tool calls, we're done
            if not tool_calls:
                break

            # Execute tool calls
            for tc in tool_calls:
                result = await self._tool_executor.execute(tc.name, tc.arguments, call_id=tc.id)
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

            for msg in self._steering.get_pending_messages():
                self._context.add_message(msg)
