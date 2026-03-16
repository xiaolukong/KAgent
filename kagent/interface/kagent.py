"""KAgent — the user-facing Facade that assembles and wires everything."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any

from pydantic import BaseModel

from kagent.agent.agent import Agent
from kagent.agent.config import AgentConfig
from kagent.context.manager import ContextManager
from kagent.domain.model_types import ModelResponse, StreamChunk
from kagent.domain.protocols import EventHandler
from kagent.events.bus import EventBus
from kagent.interface.hooks import HookRegistry
from kagent.models.factory import create_provider
from kagent.tools.decorator import ToolWrapper
from kagent.tools.decorator import tool as tool_decorator
from kagent.tools.registry import ToolRegistry


class KAgent:
    """Main entry point for the KAgent framework.

    A single-agent async runtime engine with tool calling, streaming,
    interceptors, context transforms, and steering support.

    Args:
        model: Provider and model in ``"provider:model"`` format.
            Providers: ``"openai:gpt-4o"``, ``"anthropic:claude-sonnet-4-20250514"``,
            ``"gemini:gemini-2.0-flash"``.
        system_prompt: The system message prepended to every request.
        max_turns: Maximum agent loop iterations before stopping.
        max_tool_retries: Consecutive failures on the same tool before
            the circuit breaker fires and stops retries.
        temperature: LLM sampling temperature (``None`` = provider default).
        max_tokens: Maximum response tokens (``None`` = provider default).
        tool_choice: Tool selection strategy (``None`` = auto).
        max_context_tokens: Context window budget for message truncation.
        api_key: Override the globally-configured API key for this agent.
        base_url: Override the provider's base URL (e.g. for proxies).

    Example::

        from kagent import KAgent, configure

        configure(api_key="sk-...")
        agent = KAgent(model="openai:gpt-4o", system_prompt="You are helpful.")
        result = await agent.run("Hello!")
        print(result.content)
    """

    def __init__(
        self,
        model: str = "openai:gpt-4o",
        *,
        system_prompt: str = "You are a helpful assistant.",
        max_turns: int = 1,
        max_tool_retries: int = 3,
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
        self._model_provider = create_provider(model, api_key=api_key, base_url=base_url)

        # 3. Tool registry
        self._tool_registry = ToolRegistry()

        # 4. Context manager
        self._context_manager = ContextManager(max_tokens=max_context_tokens)

        # 5. Agent config
        config = AgentConfig(
            model=model,
            system_prompt=system_prompt,
            max_turns=max_turns,
            max_tool_retries=max_tool_retries,
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

        The function's signature is auto-converted to a JSON Schema for the
        LLM's tool-calling interface.  Supports both bare and parameterized
        decorator forms.

        Args:
            func: The function (when used as bare ``@agent.tool``).
            name: Override the tool name (defaults to ``func.__name__``).
            description: Override the tool description (defaults to the
                function's docstring).

        Returns:
            The wrapped ``ToolWrapper`` (when used as decorator).

        Example::

            @agent.tool
            async def get_weather(city: str) -> str:
                \"\"\"Get current weather for a city.\"\"\"
                return f"Sunny in {city}"

            @agent.tool(name="search", description="Web search")
            async def search_web(query: str) -> str:
                return await do_search(query)
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

        Event handlers are **read-only** observers (Pub/Sub). They receive
        events but cannot modify the data flowing through the agent loop.
        Use ``intercept()`` for mutable hooks.

        Args:
            event_pattern: Glob-style pattern to match event types.
                Examples: ``"tool.call.completed"``, ``"tool.*"``,
                ``"llm.response.received"``, ``"agent.*"``.
            func: The handler function (when used as ``agent.on(pattern, fn)``).

        Returns:
            The original function (when used as decorator).

        Example::

            @agent.on("tool.call.completed")
            async def log_tool(event):
                print(f"Tool {event.payload['tool_name']} completed")

            @agent.on("llm.*")
            async def log_llm(event):
                print(f"LLM event: {event.event_type}")
        """
        if func is not None:
            self._hooks.on(event_pattern, func)
            return func

        def decorator(fn: EventHandler) -> EventHandler:
            self._hooks.on(event_pattern, fn)
            return fn

        return decorator

    # ── Interceptors ──────────────────────────────────────────────────

    def intercept(
        self,
        hook: str,
        func: Callable[..., Any] | None = None,
        *,
        priority: int = 0,
    ) -> Any:
        """Register an interceptor on a hook point. Can be used as a decorator.

        Interceptors are **mutable** hooks — they receive data, can modify it,
        and return it to the next handler in the chain. Unlike event hooks
        (``@agent.on()``), interceptors directly affect the agent loop.

        Args:
            hook: The pipeline hook point. One of:
                ``before_prompt_build``, ``before_llm_request``,
                ``after_llm_response``, ``before_tool_call``,
                ``after_tool_call``, ``after_tool_round``, ``before_return``.
            func: The interceptor function (when used as
                ``agent.intercept(hook, fn)``).
            priority: Execution order. Higher priority runs first. Default 0.

        Returns:
            The original function (when used as decorator).

        Example::

            @agent.intercept("before_llm_request")
            async def add_trace(request):
                request.metadata["trace_id"] = str(uuid.uuid4())
                return request

            @agent.intercept("before_tool_call", priority=10)
            async def block_dangerous(ctx):
                if ctx["tool_name"] == "rm_rf":
                    from kagent.agent.interceptor import InterceptResult
                    return InterceptResult.block("Dangerous tool blocked")
                return ctx
        """
        if func is not None:
            self._agent.intercept(hook, func, priority=priority)
            return func

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            self._agent.intercept(hook, fn, priority=priority)
            return fn

        return decorator

    # ── Context transforms ───────────────────────────────────────────

    def transform(
        self,
        func: Callable[..., Any] | None = None,
        *,
        priority: int = 0,
    ) -> Any:
        """Register a context transform function. Can be used as a decorator.

        Transforms filter or modify messages before they are sent to the LLM.
        The original messages in ContextManager are never modified.

        Args:
            func: The transform function (when used as bare ``@agent.transform``).
            priority: Higher priority runs first. Default 0.
                Built-in transforms (internal message filter, thinking stripper)
                use priority -100, so user transforms always run first.

        Returns:
            The original function (when used as decorator).

        Example::

            from kagent import Message, Role

            @agent.transform
            async def inject_time(messages: list[Message]) -> list[Message]:
                from datetime import datetime, UTC
                time_msg = Message(
                    role=Role.SYSTEM,
                    content=f"Current time: {datetime.now(UTC).isoformat()}",
                )
                return [messages[0], time_msg] + messages[1:]

            @agent.transform(priority=10)
            async def hide_debug(messages: list[Message]) -> list[Message]:
                return [m for m in messages if m.metadata.get("type") != "debug"]
        """
        if func is not None:
            self._agent.add_transform(func, priority=priority)
            return func

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            self._agent.add_transform(fn, priority=priority)
            return fn

        return decorator

    # ── Running ──────────────────────────────────────────────────────────

    async def run(
        self,
        user_input: str,
        *,
        response_model: type[BaseModel] | None = None,
    ) -> ModelResponse:
        """Run the agent loop (non-streaming) and return the final response.

        Args:
            user_input: The user message to send to the agent.
            response_model: Optional Pydantic model for structured output.
                When provided, the LLM is instructed to return JSON matching
                the schema, and the parsed result is available as
                ``response.parsed``.

        Returns:
            ModelResponse with ``.content`` (str), ``.tool_calls``,
            ``.usage``, ``.parsed`` (if ``response_model`` was given),
            and ``.metadata`` (may contain ``"thinking"``).

        Example::

            result = await agent.run("What is 2+2?")
            print(result.content)  # "4"

            # Structured output:
            from pydantic import BaseModel
            class Answer(BaseModel):
                value: int
            result = await agent.run("What is 2+2?", response_model=Answer)
            print(result.parsed.value)  # 4
        """
        return await self._agent.run(user_input, response_model=response_model)

    async def stream(
        self,
        user_input: str,
        *,
        response_model: type[BaseModel] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Run the agent with streaming output.

        Yields ``StreamChunk`` objects as the LLM generates tokens. Each
        chunk has a ``.chunk_type`` indicating what it contains.

        Args:
            user_input: The user message to send to the agent.
            response_model: Optional Pydantic model for structured output.

        Yields:
            StreamChunk — check ``chunk.chunk_type`` to distinguish:
            - ``TEXT_DELTA``: ``chunk.content`` contains text.
            - ``THINKING_DELTA``: ``chunk.thinking`` contains reasoning.
            - ``TOOL_CALL_START/DELTA/END``: ``chunk.tool_call`` metadata.

        Example::

            from kagent import StreamChunkType

            async for chunk in agent.stream("Tell me a story"):
                if chunk.chunk_type == StreamChunkType.THINKING_DELTA:
                    print(f"[think] {chunk.thinking}", end="")
                elif chunk.content:
                    print(chunk.content, end="", flush=True)
        """
        async for chunk in self._agent.stream(user_input, response_model=response_model):
            yield chunk

    # ── Steering ─────────────────────────────────────────────────────────

    async def steer(self, directive: str) -> None:
        """Inject a mid-turn steering directive."""
        await self._agent.steer(directive)

    async def abort(self, reason: str = "User requested abort") -> None:
        """Abort the current run."""
        await self._agent.abort(reason)

    async def interrupt(self, prompt: str) -> str:
        """Pause execution and wait for user input.

        Publishes a ``steering.interrupt`` event, then blocks until
        ``resume()`` is called.  Returns the user's reply string.

        Can be called from within a tool to implement human-in-the-loop
        confirmation::

            @agent.tool
            async def transfer(amount: int, account: str) -> str:
                if amount > 1000:
                    reply = await agent.interrupt(f"Transfer {amount}?")
                    if reply.strip().lower() != "yes":
                        return "Cancelled."
                return f"Transferred {amount} to {account}"
        """
        return await self._agent.interrupt(prompt)

    async def resume(self, user_input: str) -> None:
        """Provide user input to resume a paused agent loop."""
        await self._agent.resume(user_input)
