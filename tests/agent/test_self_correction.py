"""Tests for Tool Self-Correction and Circuit Breaker."""

import pytest

from kagent.agent.agent import Agent
from kagent.agent.config import AgentConfig
from kagent.agent.loop import AgentLoop
from kagent.agent.prompt_builder import PromptBuilder
from kagent.agent.steering import SteeringController
from kagent.context.manager import ContextManager
from kagent.domain.entities import ToolCall
from kagent.domain.enums import Role, ToolCallStatus
from kagent.domain.model_types import ModelResponse, TokenUsage
from kagent.events.bus import EventBus
from kagent.tools.decorator import tool
from kagent.tools.executor import ToolExecutor
from kagent.tools.registry import ToolRegistry
from tests.conftest import MockModelProvider

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_loop(
    provider: MockModelProvider,
    registry: ToolRegistry | None = None,
    max_turns: int = 10,
    max_tool_retries: int = 3,
) -> tuple[AgentLoop, ContextManager, EventBus]:
    bus = EventBus()
    reg = registry or ToolRegistry()
    executor = ToolExecutor(reg, bus)
    context = ContextManager()
    prompt = PromptBuilder(system_prompt="Test system.")
    steering = SteeringController(bus)
    loop = AgentLoop(
        model_provider=provider,
        event_bus=bus,
        tool_registry=reg,
        tool_executor=executor,
        context_manager=context,
        prompt_builder=prompt,
        steering=steering,
        max_turns=max_turns,
        max_tool_retries=max_tool_retries,
    )
    return loop, context, bus


# ── ToolWrapper: ValidationError no longer crashes ───────────────────────────


class TestValidationErrorSelfCorrection:
    @pytest.mark.asyncio
    async def test_validation_error_returns_tool_result(self):
        """ValidationError should produce a ToolResult with ERROR status, not raise."""

        @tool
        async def typed_tool(x: int) -> int:
            """Requires int."""
            return x * 2

        # Pass a string that cannot be coerced to int
        result = await typed_tool.execute({"x": "not-a-number"})
        assert result.status == ToolCallStatus.ERROR
        assert "Validation failed" in (result.error or "")
        assert result.tool_name == "typed_tool"

    @pytest.mark.asyncio
    async def test_validation_error_via_executor(self):
        """ToolExecutor should enrich validation errors with arguments context."""
        bus = EventBus()
        registry = ToolRegistry()

        @tool
        async def typed_tool(user_id: int) -> str:
            """Look up user."""
            return f"user-{user_id}"

        registry.register(typed_tool)
        executor = ToolExecutor(registry, bus)

        result = await executor.execute("typed_tool", {"user_id": "abc"}, call_id="test-call")

        assert result.status == ToolCallStatus.ERROR
        assert "parameter validation failed" in (result.error or "")
        assert "You provided" in (result.error or "")
        assert '"user_id": "abc"' in (result.error or "")
        assert "Please fix" in (result.error or "")

    @pytest.mark.asyncio
    async def test_validation_error_publishes_event(self):
        """Validation errors should still publish TOOL_CALL_ERROR events."""
        from kagent.domain.enums import EventType
        from kagent.domain.events import Event

        bus = EventBus()
        registry = ToolRegistry()

        @tool
        async def typed_tool(x: int) -> int:
            """Typed."""
            return x

        registry.register(typed_tool)
        executor = ToolExecutor(registry, bus)

        events: list[Event] = []

        async def collector(event: Event):
            events.append(event)

        bus.subscribe("tool.*", collector)
        await executor.execute("typed_tool", {"x": "not-a-number"})

        event_types = [e.event_type for e in events]
        assert EventType.TOOL_CALL_STARTED in event_types
        assert EventType.TOOL_CALL_ERROR in event_types

    @pytest.mark.asyncio
    async def test_validation_error_fed_back_to_llm(self):
        """In AgentLoop, validation error becomes a Role.TOOL message, not a crash."""
        registry = ToolRegistry()

        @tool
        async def typed_tool(x: int) -> int:
            """Typed."""
            return x

        registry.register(typed_tool)

        # LLM first calls typed_tool with bad params, then responds with text
        call_count = 0

        class SelfCorrectingProvider(MockModelProvider):
            async def _do_complete(self, request):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # First call: LLM tries to call tool with wrong type
                    return ModelResponse(
                        content=None,
                        tool_calls=[
                            ToolCall(
                                id="c1",
                                name="typed_tool",
                                arguments={"x": "not-a-number"},
                            )
                        ],
                        usage=TokenUsage(
                            prompt_tokens=10,
                            completion_tokens=5,
                            total_tokens=15,
                        ),
                    )
                # Second call: LLM sees error, gives text response
                return ModelResponse(
                    content="I see the error, the answer is 42.",
                    usage=TokenUsage(
                        prompt_tokens=20,
                        completion_tokens=10,
                        total_tokens=30,
                    ),
                )

        provider = SelfCorrectingProvider()
        loop, context, bus = _make_loop(provider, registry=registry)

        result = await loop.run("Double 5")
        assert result.content == "I see the error, the answer is 42."
        assert call_count == 2

        # Verify the error message is in context as a TOOL message
        all_msgs = context.get_all_messages()
        tool_msgs = [m for m in all_msgs if m.role == Role.TOOL]
        assert len(tool_msgs) == 1
        assert "Validation failed" in (tool_msgs[0].content or "")


# ── Circuit Breaker ──────────────────────────────────────────────────────────


class TestCircuitBreaker:
    @pytest.mark.asyncio
    async def test_circuit_breaker_triggers_after_max_retries(self):
        """After max_tool_retries consecutive failures, the circuit breaker fires."""
        registry = ToolRegistry()

        @tool
        async def bad_tool(x: int) -> int:
            """Always fails validation."""
            return x

        registry.register(bad_tool)

        turn_count = 0

        class AlwaysCallBadTool(MockModelProvider):
            async def _do_complete(self, request):
                nonlocal turn_count
                turn_count += 1
                if turn_count <= 4:
                    return ModelResponse(
                        content=None,
                        tool_calls=[
                            ToolCall(
                                id=f"c{turn_count}",
                                name="bad_tool",
                                arguments={"x": "invalid"},
                            )
                        ],
                        usage=TokenUsage(
                            prompt_tokens=10,
                            completion_tokens=5,
                            total_tokens=15,
                        ),
                    )
                return ModelResponse(
                    content="Gave up on bad_tool.",
                    usage=TokenUsage(
                        prompt_tokens=10,
                        completion_tokens=5,
                        total_tokens=15,
                    ),
                )

        provider = AlwaysCallBadTool()
        loop, context, bus = _make_loop(
            provider, registry=registry, max_tool_retries=3, max_turns=10
        )

        result = await loop.run("Try it")
        assert result is not None

        # The circuit breaker message should be in context
        all_msgs = context.get_all_messages()
        breaker_msgs = [
            m for m in all_msgs if m.role == Role.TOOL and "consecutive times" in (m.content or "")
        ]
        assert len(breaker_msgs) >= 1
        assert "3 consecutive times" in (breaker_msgs[0].content or "")

    @pytest.mark.asyncio
    async def test_success_resets_error_counter(self):
        """A successful call resets the consecutive error count."""
        registry = ToolRegistry()

        @tool
        async def sometimes_works(x: int) -> int:
            """Works when x is valid."""
            return x * 2

        registry.register(sometimes_works)

        call_sequence = [
            # Turn 1: bad params (error #1)
            {"x": "bad"},
            # Turn 2: bad params (error #2)
            {"x": "bad"},
            # Turn 3: good params (success → resets counter)
            {"x": 5},
            # Turn 4: bad params (error #1 again, not #3)
            {"x": "bad"},
            # Turn 5: bad params (error #2 again, not #4)
            {"x": "bad"},
        ]
        turn_idx = 0

        class SequenceProvider(MockModelProvider):
            async def _do_complete(self, request):
                nonlocal turn_idx
                if turn_idx < len(call_sequence):
                    args = call_sequence[turn_idx]
                    turn_idx += 1
                    return ModelResponse(
                        content=None,
                        tool_calls=[
                            ToolCall(
                                id=f"c{turn_idx}",
                                name="sometimes_works",
                                arguments=args,
                            )
                        ],
                        usage=TokenUsage(
                            prompt_tokens=10,
                            completion_tokens=5,
                            total_tokens=15,
                        ),
                    )
                return ModelResponse(
                    content="Done.",
                    usage=TokenUsage(
                        prompt_tokens=10,
                        completion_tokens=5,
                        total_tokens=15,
                    ),
                )

        provider = SequenceProvider()
        loop, context, bus = _make_loop(
            provider, registry=registry, max_tool_retries=3, max_turns=10
        )

        result = await loop.run("Go")
        assert result is not None

        # Should NOT have triggered circuit breaker (max errors=2, then reset, then 2)
        all_msgs = context.get_all_messages()
        breaker_msgs = [
            m for m in all_msgs if m.role == Role.TOOL and "consecutive times" in (m.content or "")
        ]
        assert len(breaker_msgs) == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_in_stream(self):
        """Circuit breaker works in streaming mode too."""
        registry = ToolRegistry()

        @tool
        async def bad_tool(x: int) -> int:
            """Always fails."""
            return x

        registry.register(bad_tool)

        turn_count = 0

        class AlwaysCallBadTool(MockModelProvider):
            async def _do_complete(self, request):
                nonlocal turn_count
                turn_count += 1
                if turn_count <= 4:
                    return ModelResponse(
                        content=None,
                        tool_calls=[
                            ToolCall(
                                id=f"c{turn_count}",
                                name="bad_tool",
                                arguments={"x": "invalid"},
                            )
                        ],
                        usage=TokenUsage(
                            prompt_tokens=10,
                            completion_tokens=5,
                            total_tokens=15,
                        ),
                    )
                return ModelResponse(
                    content="Gave up.",
                    usage=TokenUsage(
                        prompt_tokens=10,
                        completion_tokens=5,
                        total_tokens=15,
                    ),
                )

            async def _do_stream(self, request):
                response = await self._do_complete(request)
                if response.tool_calls:
                    import json as _json

                    from kagent.domain.enums import StreamChunkType
                    from kagent.domain.model_types import StreamChunk, ToolCallChunk

                    for tc in response.tool_calls:
                        yield StreamChunk(
                            chunk_type=StreamChunkType.TOOL_CALL_START,
                            tool_call=ToolCallChunk(
                                id=tc.id,
                                name=tc.name,
                                arguments_delta=_json.dumps(tc.arguments),
                            ),
                        )
                else:
                    from kagent.domain.enums import StreamChunkType
                    from kagent.domain.model_types import StreamChunk

                    for word in (response.content or "").split():
                        yield StreamChunk(
                            chunk_type=StreamChunkType.TEXT_DELTA,
                            content=word + " ",
                        )

        provider = AlwaysCallBadTool()
        loop, context, bus = _make_loop(
            provider, registry=registry, max_tool_retries=3, max_turns=10
        )

        chunks = []
        async for chunk in loop.run_stream("Try it"):
            chunks.append(chunk)

        all_msgs = context.get_all_messages()
        breaker_msgs = [
            m for m in all_msgs if m.role == Role.TOOL and "consecutive times" in (m.content or "")
        ]
        assert len(breaker_msgs) >= 1

    @pytest.mark.asyncio
    async def test_different_tools_have_separate_counters(self):
        """Each tool has its own consecutive error counter."""
        registry = ToolRegistry()

        @tool
        async def tool_a(x: int) -> int:
            """Tool A."""
            return x

        @tool
        async def tool_b(x: int) -> int:
            """Tool B."""
            return x

        registry.register(tool_a)
        registry.register(tool_b)

        turn_count = 0

        class AlternatingProvider(MockModelProvider):
            async def _do_complete(self, request):
                nonlocal turn_count
                turn_count += 1
                if turn_count <= 4:
                    # Alternate between tool_a and tool_b with bad params
                    name = "tool_a" if turn_count % 2 == 1 else "tool_b"
                    return ModelResponse(
                        content=None,
                        tool_calls=[
                            ToolCall(
                                id=f"c{turn_count}",
                                name=name,
                                arguments={"x": "invalid"},
                            )
                        ],
                        usage=TokenUsage(
                            prompt_tokens=10,
                            completion_tokens=5,
                            total_tokens=15,
                        ),
                    )
                return ModelResponse(
                    content="Done.",
                    usage=TokenUsage(
                        prompt_tokens=10,
                        completion_tokens=5,
                        total_tokens=15,
                    ),
                )

        provider = AlternatingProvider()
        loop, context, bus = _make_loop(
            provider, registry=registry, max_tool_retries=3, max_turns=10
        )

        result = await loop.run("Go")
        assert result is not None

        # Neither tool should have hit 3 consecutive failures
        # (each only fails 2 times: a, b, a, b)
        all_msgs = context.get_all_messages()
        breaker_msgs = [
            m for m in all_msgs if m.role == Role.TOOL and "consecutive times" in (m.content or "")
        ]
        assert len(breaker_msgs) == 0


# ── AgentConfig ──────────────────────────────────────────────────────────────


class TestAgentConfigMaxToolRetries:
    def test_default_value(self):
        config = AgentConfig()
        assert config.max_tool_retries == 3

    def test_custom_value(self):
        config = AgentConfig(max_tool_retries=5)
        assert config.max_tool_retries == 5

    @pytest.mark.asyncio
    async def test_wired_through_agent(self):
        """max_tool_retries flows from AgentConfig → Agent → AgentLoop."""
        provider = MockModelProvider(response_content="ok")
        config = AgentConfig(model="mock:mock", max_tool_retries=5)
        agent = Agent(
            config=config,
            model_provider=provider,
            event_bus=EventBus(),
        )
        assert agent._loop._max_tool_retries == 5
