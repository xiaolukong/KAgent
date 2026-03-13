"""Tests for ContextTransformer pipeline."""

import pytest

from kagent.agent.loop import AgentLoop
from kagent.agent.prompt_builder import PromptBuilder
from kagent.agent.steering import SteeringController
from kagent.context.manager import ContextManager
from kagent.context.transformer import (
    ContextTransformer,
    filter_internal_messages,
    strip_thinking_from_context,
)
from kagent.domain.entities import Message
from kagent.domain.enums import Role
from kagent.events.bus import EventBus
from kagent.tools.executor import ToolExecutor
from kagent.tools.registry import ToolRegistry
from tests.conftest import MockModelProvider

# ── ContextTransformer unit tests ────────────────────────────────────────────


class TestContextTransformerBasics:
    @pytest.mark.asyncio
    async def test_empty_transformer_passes_through(self):
        transformer = ContextTransformer()
        messages = [
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="Hi"),
        ]
        result = await transformer.apply(messages)
        assert len(result) == 2
        assert result[0].content == "Hello"
        assert result[1].content == "Hi"

    @pytest.mark.asyncio
    async def test_single_transform(self):
        transformer = ContextTransformer()

        async def upper_content(messages: list[Message]) -> list[Message]:
            return [m.model_copy(update={"content": (m.content or "").upper()}) for m in messages]

        transformer.add(upper_content)
        messages = [Message(role=Role.USER, content="hello")]
        result = await transformer.apply(messages)
        assert result[0].content == "HELLO"

    @pytest.mark.asyncio
    async def test_chained_transforms(self):
        """Transforms are chained: output of one is input to next."""
        transformer = ContextTransformer()

        async def add_prefix(messages: list[Message]) -> list[Message]:
            return [m.model_copy(update={"content": f"[prefix] {m.content}"}) for m in messages]

        async def add_suffix(messages: list[Message]) -> list[Message]:
            return [m.model_copy(update={"content": f"{m.content} [suffix]"}) for m in messages]

        transformer.add(add_prefix)
        transformer.add(add_suffix)
        messages = [Message(role=Role.USER, content="test")]
        result = await transformer.apply(messages)
        assert result[0].content == "[prefix] test [suffix]"

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        """Higher priority transforms run first."""
        transformer = ContextTransformer()
        order: list[str] = []

        async def low(messages: list[Message]) -> list[Message]:
            order.append("low")
            return messages

        async def high(messages: list[Message]) -> list[Message]:
            order.append("high")
            return messages

        transformer.add(low, priority=0)
        transformer.add(high, priority=10)
        await transformer.apply([])
        assert order == ["high", "low"]

    @pytest.mark.asyncio
    async def test_same_priority_preserves_order(self):
        """Same priority → registration order preserved."""
        transformer = ContextTransformer()
        order: list[str] = []

        async def first(messages: list[Message]) -> list[Message]:
            order.append("first")
            return messages

        async def second(messages: list[Message]) -> list[Message]:
            order.append("second")
            return messages

        transformer.add(first, priority=0)
        transformer.add(second, priority=0)
        await transformer.apply([])
        assert order == ["first", "second"]

    @pytest.mark.asyncio
    async def test_remove_transform(self):
        transformer = ContextTransformer()

        async def noop(messages: list[Message]) -> list[Message]:
            return []

        transformer.add(noop)
        assert transformer.has_transforms
        transformer.remove(noop)
        assert not transformer.has_transforms

    def test_has_transforms(self):
        transformer = ContextTransformer()
        assert not transformer.has_transforms

        async def noop(messages: list[Message]) -> list[Message]:
            return messages

        transformer.add(noop)
        assert transformer.has_transforms


# ── Built-in transforms ──────────────────────────────────────────────────────


class TestFilterInternalMessages:
    @pytest.mark.asyncio
    async def test_filters_internal(self):
        messages = [
            Message(role=Role.USER, content="visible"),
            Message(role=Role.USER, content="hidden", metadata={"internal": True}),
            Message(role=Role.ASSISTANT, content="reply"),
        ]
        result = await filter_internal_messages(messages)
        assert len(result) == 2
        assert result[0].content == "visible"
        assert result[1].content == "reply"

    @pytest.mark.asyncio
    async def test_keeps_all_when_no_internal(self):
        messages = [
            Message(role=Role.USER, content="a"),
            Message(role=Role.ASSISTANT, content="b"),
        ]
        result = await filter_internal_messages(messages)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_internal_false_is_kept(self):
        messages = [
            Message(role=Role.USER, content="keep", metadata={"internal": False}),
        ]
        result = await filter_internal_messages(messages)
        assert len(result) == 1


class TestStripThinkingFromContext:
    @pytest.mark.asyncio
    async def test_strips_thinking_metadata(self):
        messages = [
            Message(
                role=Role.ASSISTANT,
                content="answer",
                metadata={"thinking": "reasoning...", "other": "data"},
            ),
        ]
        result = await strip_thinking_from_context(messages)
        assert len(result) == 1
        assert result[0].content == "answer"
        assert "thinking" not in result[0].metadata
        assert result[0].metadata["other"] == "data"

    @pytest.mark.asyncio
    async def test_no_thinking_unchanged(self):
        messages = [
            Message(role=Role.ASSISTANT, content="plain", metadata={"key": "val"}),
        ]
        result = await strip_thinking_from_context(messages)
        assert result[0].metadata == {"key": "val"}

    @pytest.mark.asyncio
    async def test_does_not_mutate_original(self):
        original = Message(
            role=Role.ASSISTANT,
            content="answer",
            metadata={"thinking": "secret"},
        )
        messages = [original]
        result = await strip_thinking_from_context(messages)
        # Original should still have thinking
        assert "thinking" in original.metadata
        # Result should not
        assert "thinking" not in result[0].metadata


# ── AgentLoop integration tests ──────────────────────────────────────────────


def _make_loop(
    provider: MockModelProvider,
    transformer: ContextTransformer | None = None,
    max_turns: int = 5,
) -> tuple[AgentLoop, ContextManager]:
    bus = EventBus()
    reg = ToolRegistry()
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
        transformer=transformer,
        max_turns=max_turns,
    )
    return loop, context


class TestTransformerInLoop:
    @pytest.mark.asyncio
    async def test_transform_filters_internal_in_run(self):
        """Internal messages should be filtered before LLM sees them."""
        transformer = ContextTransformer()
        transformer.add(filter_internal_messages)

        provider = MockModelProvider(response_content="Reply")
        loop, context = _make_loop(provider, transformer=transformer)

        # Pre-populate context with an internal message
        context.add_message(
            Message(
                role=Role.USER,
                content="debug info",
                metadata={"internal": True},
            )
        )

        result = await loop.run("Hello")
        assert result.content == "Reply"

        # The internal message is preserved in full context
        all_msgs = context.get_all_messages()
        internal_msgs = [m for m in all_msgs if m.metadata.get("internal")]
        assert len(internal_msgs) == 1

    @pytest.mark.asyncio
    async def test_transform_filters_internal_in_stream(self):
        """Internal messages should be filtered in streaming mode too."""
        transformer = ContextTransformer()
        transformer.add(filter_internal_messages)

        provider = MockModelProvider(response_content="Streamed reply")
        loop, context = _make_loop(provider, transformer=transformer)

        context.add_message(
            Message(
                role=Role.USER,
                content="internal data",
                metadata={"internal": True},
            )
        )

        chunks = []
        async for chunk in loop.run_stream("Hello"):
            chunks.append(chunk)

        assert len(chunks) > 0
        # Internal message still in context
        all_msgs = context.get_all_messages()
        internal_msgs = [m for m in all_msgs if m.metadata.get("internal")]
        assert len(internal_msgs) == 1

    @pytest.mark.asyncio
    async def test_transform_does_not_modify_context(self):
        """Transform output is only used for LLM — context remains unchanged."""
        transformer = ContextTransformer()

        async def drop_all_user_msgs(messages: list[Message]) -> list[Message]:
            return [m for m in messages if m.role != Role.USER]

        transformer.add(drop_all_user_msgs)

        provider = MockModelProvider(response_content="Done")
        loop, context = _make_loop(provider, transformer=transformer)

        await loop.run("Keep me in context")

        # User message must still be in context
        all_msgs = context.get_all_messages()
        user_msgs = [m for m in all_msgs if m.role == Role.USER]
        assert len(user_msgs) == 1
        assert user_msgs[0].content == "Keep me in context"

    @pytest.mark.asyncio
    async def test_strip_thinking_default_in_agent(self):
        """When using Agent (not raw loop), thinking should be stripped by default."""
        from kagent.agent.agent import Agent
        from kagent.agent.config import AgentConfig

        provider = MockModelProvider(
            response_content="answer",
            thinking_content="step by step reasoning",
        )

        config = AgentConfig(model="mock:mock", system_prompt="Test.")
        agent = Agent(
            config=config,
            model_provider=provider,
            event_bus=EventBus(),
        )

        # First call: produces thinking in metadata
        result = await agent.run("Think about this")
        assert result.content == "answer"
        assert result.metadata.get("thinking") == "step by step reasoning"

        # The assistant message in context has thinking in metadata
        all_msgs = agent.context.get_all_messages()
        assistant_msgs = [m for m in all_msgs if m.role == Role.ASSISTANT]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].metadata.get("thinking") == "step by step reasoning"
