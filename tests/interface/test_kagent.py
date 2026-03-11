"""Tests for KAgent Facade (end-to-end integration)."""

import pytest

from kagent.domain.enums import EventType
from kagent.domain.events import Event
from kagent.interface.kagent import KAgent

# We need to monkey-patch the factory to return a mock provider
from unittest.mock import patch
from tests.conftest import MockModelProvider


def _mock_create_provider(model_string, **kwargs):
    """Return a MockModelProvider instead of a real one."""
    return MockModelProvider(response_content="KAgent response")


class TestKAgentRun:
    @pytest.mark.asyncio
    @patch("kagent.interface.kagent.create_provider", side_effect=_mock_create_provider)
    async def test_basic_run(self, mock_factory):
        agent = KAgent(model="mock:model", system_prompt="Test")
        result = await agent.run("Hello")
        assert result.content == "KAgent response"

    @pytest.mark.asyncio
    @patch("kagent.interface.kagent.create_provider", side_effect=_mock_create_provider)
    async def test_tool_registration(self, mock_factory):
        agent = KAgent(model="mock:model")

        @agent.tool
        async def greet(name: str) -> str:
            """Greet."""
            return f"Hello, {name}!"

        # Verify tool is registered
        defs = agent._tool_registry.list_definitions()
        assert any(d.name == "greet" for d in defs)

    @pytest.mark.asyncio
    @patch("kagent.interface.kagent.create_provider", side_effect=_mock_create_provider)
    async def test_event_hook(self, mock_factory):
        agent = KAgent(model="mock:model")
        events_received = []

        @agent.on("agent.*")
        async def handler(event: Event):
            events_received.append(event.event_type)

        await agent.run("Hello")
        assert EventType.AGENT_STARTED in events_received
        assert EventType.AGENT_LOOP_COMPLETED in events_received


class TestKAgentStream:
    @pytest.mark.asyncio
    @patch("kagent.interface.kagent.create_provider", side_effect=_mock_create_provider)
    async def test_basic_stream(self, mock_factory):
        agent = KAgent(model="mock:model")
        chunks = []
        async for chunk in agent.stream("Hello"):
            chunks.append(chunk)
        assert len(chunks) > 0
        text = "".join(c.content or "" for c in chunks)
        assert "KAgent" in text


class TestKAgentStructured:
    @pytest.mark.asyncio
    async def test_structured_output(self):
        from pydantic import BaseModel

        class Info(BaseModel):
            name: str
            score: float

        mock = MockModelProvider(response_content='{"name": "test", "score": 9.5}')

        with patch("kagent.interface.kagent.create_provider", return_value=mock):
            agent = KAgent(model="mock:model")
            result = await agent.run("Get info", response_model=Info)
            assert result.parsed.name == "test"
            assert result.parsed.score == 9.5
