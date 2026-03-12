"""KAgent — the user-facing Facade that assembles and wires everything."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any

from pydantic import BaseModel

from kagent.agent.agent import Agent
from kagent.agent.config import AgentConfig
from kagent.context.manager import ContextManager
from kagent.domain.events import Event
from kagent.domain.model_types import ModelResponse, StreamChunk
from kagent.domain.protocols import EventHandler
from kagent.events.bus import EventBus
from kagent.interface.hooks import HookRegistry
from kagent.models.factory import create_provider
from kagent.tools.decorator import ToolWrapper, tool as tool_decorator
from kagent.tools.registry import ToolRegistry


class KAgent:
    """Main entry point for the KAgent framework.

    Usage:
        agent = KAgent(model="openai:gpt-4o")
        result = await agent.run("Hello!")
    """

    def __init__(
        self,
        model: str = "openai:gpt-4o",
        *,
        system_prompt: str = "You are a helpful assistant.",
        max_turns: int = 10,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tool_choice: str | None = None,
        max_context_tokens: int = 128_000,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        # 1. EventBus
        self._event_bus = EventBus()

        # 2. Model provider (Infrastructure)
        self._model_provider = create_provider(
            model, api_key=api_key, base_url=base_url
        )

        # 3. Tool registry
        self._tool_registry = ToolRegistry()

        # 4. Context manager
        self._context_manager = ContextManager(max_tokens=max_context_tokens)

        # 5. Agent config
        config = AgentConfig(
            model=model,
            system_prompt=system_prompt,
            max_turns=max_turns,
            temperature=temperature,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
            max_context_tokens=max_context_tokens,
        )

        # 6. Core agent (Application)
        self._agent = Agent(
            config=config,
            model_provider=self._model_provider,
            event_bus=self._event_bus,
            tool_registry=self._tool_registry,
            context_manager=self._context_manager,
        )

        # 7. Hook registry
        self._hooks = HookRegistry(self._event_bus)

    # ── Tool registration ────────────────────────────────────────────────

    def tool(
        self,
        func: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Any:
        """Decorator to register a function as a tool on this agent.

        Usage:
            @agent.tool
            async def my_func(x: int) -> str: ...

            @agent.tool(name="custom")
            async def my_func(x: int) -> str: ...
        """
        def decorator(fn: Callable[..., Any]) -> ToolWrapper:
            wrapper = tool_decorator(fn, name=name, description=description)
            self._tool_registry.register(wrapper)
            return wrapper

        if func is not None:
            return decorator(func)
        return decorator

    # ── Event hooks ──────────────────────────────────────────────────────

    def on(
        self,
        event_pattern: str,
        func: EventHandler | None = None,
    ) -> Any:
        """Register an event handler. Can be used as a decorator.

        Usage:
            @agent.on("tool.call.completed")
            async def handler(event): ...

            agent.on("tool.*", some_handler)
        """
        if func is not None:
            self._hooks.on(event_pattern, func)
            return func

        def decorator(fn: EventHandler) -> EventHandler:
            self._hooks.on(event_pattern, fn)
            return fn

        return decorator

    # ── Running ──────────────────────────────────────────────────────────

    async def run(
        self,
        user_input: str,
        *,
        response_model: type[BaseModel] | None = None,
    ) -> ModelResponse:
        """Run the agent (non-streaming)."""
        return await self._agent.run(user_input, response_model=response_model)

    async def stream(
        self,
        user_input: str,
        *,
        response_model: type[BaseModel] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Run the agent with streaming output."""
        async for chunk in self._agent.stream(
            user_input, response_model=response_model
        ):
            yield chunk

    # ── Steering ─────────────────────────────────────────────────────────

    async def steer(self, directive: str) -> None:
        """Inject a mid-turn steering directive."""
        await self._agent.steer(directive)

    async def abort(self, reason: str = "User requested abort") -> None:
        """Abort the current run."""
        await self._agent.abort(reason)

    async def interrupt(self, prompt: str) -> None:
        """Pause the agent loop and request user input.

        The loop suspends at the next turn boundary. Call ``resume()``
        from another coroutine to provide user input and continue.
        """
        await self._agent.interrupt(prompt)

    async def resume(self, user_input: str) -> None:
        """Provide user input to resume a paused agent loop."""
        await self._agent.resume(user_input)
