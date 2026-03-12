"""Tests for the Agent class (end-to-end with mock provider)."""

import pytest

from kagent.agent.agent import Agent
from kagent.agent.config import AgentConfig
from kagent.domain.entities import ToolCall
from kagent.domain.enums import EventType
from kagent.domain.events import Event
from kagent.events.bus import EventBus
from kagent.tools.decorator import tool
from kagent.tools.registry import ToolRegistry
from tests.conftest import MockModelProvider


@pytest.fixture
def basic_agent():
    bus = EventBus()
    provider = MockModelProvider(response_content="Hello from agent!")
    config = AgentConfig(model="mock:model", system_prompt="Be helpful.", max_turns=5)
    agent = Agent(config=config, model_provider=provider, event_bus=bus)
    return agent, bus


@pytest.fixture
def tool_agent():
    """Agent with a tool that the mock provider will call."""
    bus = EventBus()
    registry = ToolRegistry()

    @tool
    async def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    registry.register(add)

    tc = ToolCall(id="call-1", name="add", arguments={"a": 3, "b": 5})
    provider = MockModelProvider(
        response_content="The result is 8.",
        tool_calls=[tc],
    )

    config = AgentConfig(model="mock:model", system_prompt="Be helpful.", max_turns=5)
    agent = Agent(config=config, model_provider=provider, event_bus=bus, tool_registry=registry)
    return agent, bus


class TestAgentRun:
    @pytest.mark.asyncio
    async def test_basic_run(self, basic_agent):
        agent, bus = basic_agent
        result = await agent.run("Hello")
        assert result.content == "Hello from agent!"

    @pytest.mark.asyncio
    async def test_run_publishes_lifecycle_events(self, basic_agent):
        agent, bus = basic_agent
        events = []

        async def collector(event: Event):
            events.append(event.event_type)

        bus.subscribe("agent.*", collector)
        await agent.run("Hello")

        assert EventType.AGENT_STARTED in events
        assert EventType.AGENT_LOOP_COMPLETED in events

    @pytest.mark.asyncio
    async def test_run_with_tools(self, tool_agent):
        agent, bus = tool_agent
        events = []

        async def collector(event: Event):
            events.append(event.event_type)

        bus.subscribe("tool.*", collector)
        result = await agent.run("What is 3+5?")

        # Provider returns tool call first, then text on second call
        assert EventType.TOOL_CALL_STARTED in events
        assert EventType.TOOL_CALL_COMPLETED in events
        assert result.content == "The result is 8."


class TestAgentStream:
    @pytest.mark.asyncio
    async def test_basic_stream(self, basic_agent):
        agent, bus = basic_agent
        chunks = []
        async for chunk in agent.stream("Hello"):
            chunks.append(chunk)
        assert len(chunks) > 0
        text = "".join(c.content or "" for c in chunks)
        assert "Hello" in text

    @pytest.mark.asyncio
    async def test_stream_publishes_events(self, basic_agent):
        agent, bus = basic_agent
        events = []

        async def collector(event: Event):
            events.append(event.event_type)

        bus.subscribe("agent.*", collector)
        bus.subscribe("llm.*", collector)

        async for _ in agent.stream("Hello"):
            pass

        assert EventType.AGENT_STARTED in events
        assert EventType.LLM_REQUEST_SENT in events


class TestAgentStructuredOutput:
    @pytest.mark.asyncio
    async def test_run_with_response_model(self):
        from pydantic import BaseModel

        class Answer(BaseModel):
            answer: str
            confidence: float

        bus = EventBus()
        provider = MockModelProvider(response_content='{"answer": "42", "confidence": 0.99}')
        config = AgentConfig(model="mock:model")
        agent = Agent(config=config, model_provider=provider, event_bus=bus)

        result = await agent.run("What is the answer?", response_model=Answer)
        assert result.parsed is not None
        assert result.parsed.answer == "42"
        assert result.parsed.confidence == 0.99
