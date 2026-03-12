"""Agent — the core agent class that owns and wires all components."""

from __future__ import annotations

from collections.abc import AsyncIterator

from pydantic import BaseModel

from kagent.agent.config import AgentConfig
from kagent.agent.loop import AgentLoop
from kagent.agent.prompt_builder import PromptBuilder
from kagent.agent.steering import SteeringController
from kagent.context.manager import ContextManager
from kagent.domain.enums import EventType
from kagent.domain.events import AgentEvent, SteeringEvent
from kagent.domain.model_types import ModelResponse, StreamChunk
from kagent.domain.protocols import IEventBus, IModelProvider
from kagent.tools.executor import ToolExecutor
from kagent.tools.registry import ToolRegistry


class Agent:
    """Core agent that orchestrates model calls, tool execution, and state."""

    def __init__(
        self,
        *,
        config: AgentConfig,
        model_provider: IModelProvider,
        event_bus: IEventBus,
        tool_registry: ToolRegistry | None = None,
        context_manager: ContextManager | None = None,
    ) -> None:
        self._config = config
        self._event_bus = event_bus
        self._tool_registry = tool_registry if tool_registry is not None else ToolRegistry()
        self._context = context_manager if context_manager is not None else ContextManager(
            max_tokens=config.max_context_tokens
        )
        self._tool_executor = ToolExecutor(self._tool_registry, event_bus)
        self._steering = SteeringController(event_bus)
        self._prompt_builder = PromptBuilder(
            system_prompt=config.system_prompt,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            tool_choice=config.tool_choice,
        )
        self._loop = AgentLoop(
            model_provider=model_provider,
            event_bus=event_bus,
            tool_registry=self._tool_registry,
            tool_executor=self._tool_executor,
            context_manager=self._context,
            prompt_builder=self._prompt_builder,
            steering=self._steering,
            max_turns=config.max_turns,
        )

    @property
    def tool_registry(self) -> ToolRegistry:
        return self._tool_registry

    @property
    def event_bus(self) -> IEventBus:
        return self._event_bus

    @property
    def context(self) -> ContextManager:
        return self._context

    async def run(
        self,
        user_input: str,
        *,
        response_model: type[BaseModel] | None = None,
    ) -> ModelResponse:
        """Run the agent (non-streaming) and return the final response."""
        await self._event_bus.publish(
            AgentEvent(
                event_type=EventType.AGENT_STARTED,
                payload={"input": user_input, "mode": "complete"},
                source="agent",
            )
        )

        try:
            result = await self._loop.run(user_input, response_model=response_model)
            await self._event_bus.publish(
                AgentEvent(
                    event_type=EventType.AGENT_LOOP_COMPLETED,
                    payload={"has_content": result.content is not None},
                    source="agent",
                )
            )
            return result
        except Exception as exc:
            await self._event_bus.publish(
                AgentEvent(
                    event_type=EventType.AGENT_ERROR,
                    payload={"error_type": type(exc).__name__, "message": str(exc)},
                    source="agent",
                )
            )
            raise

    async def stream(
        self,
        user_input: str,
        *,
        response_model: type[BaseModel] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Run the agent with streaming output."""
        await self._event_bus.publish(
            AgentEvent(
                event_type=EventType.AGENT_STARTED,
                payload={"input": user_input, "mode": "stream"},
                source="agent",
            )
        )

        try:
            async for chunk in self._loop.run_stream(
                user_input, response_model=response_model
            ):
                yield chunk

            await self._event_bus.publish(
                AgentEvent(
                    event_type=EventType.AGENT_LOOP_COMPLETED,
                    payload={"mode": "stream"},
                    source="agent",
                )
            )
        except Exception as exc:
            await self._event_bus.publish(
                AgentEvent(
                    event_type=EventType.AGENT_ERROR,
                    payload={"error_type": type(exc).__name__, "message": str(exc)},
                    source="agent",
                )
            )
            raise

    async def steer(self, directive: str) -> None:
        """Inject a mid-turn steering directive."""
        await self._event_bus.publish(
            AgentEvent(
                event_type=EventType.STEERING_REDIRECT,
                payload={"directive": directive},
                source="agent",
            )
        )

    async def abort(self, reason: str = "User requested abort") -> None:
        """Request immediate abort of the current run."""
        await self._event_bus.publish(
            AgentEvent(
                event_type=EventType.STEERING_ABORT,
                payload={"reason": reason},
                source="agent",
            )
        )

    async def interrupt(self, prompt: str) -> None:
        """Pause the agent loop and request user input.

        The agent loop will suspend at the next turn boundary and wait
        until ``resume()`` is called with the user's reply.
        """
        await self._event_bus.publish(
            SteeringEvent(
                event_type=EventType.STEERING_INTERRUPT,
                payload={"prompt": prompt},
                source="agent",
            )
        )

    async def resume(self, user_input: str) -> None:
        """Provide user input to resume a paused agent loop.

        Must be called after ``interrupt()`` to unblock the loop.
        """
        await self._event_bus.publish(
            SteeringEvent(
                event_type=EventType.STEERING_RESUME,
                payload={"user_input": user_input},
                source="agent",
            )
        )
