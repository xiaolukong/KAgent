"""Tests for Anthropic provider (base behavior via mock)."""

import pytest

from kagent.domain.entities import Message
from kagent.domain.enums import Role
from kagent.domain.model_types import ModelRequest
from tests.conftest import MockModelProvider


class TestAnthropicProviderBase:
    @pytest.mark.asyncio
    async def test_stream_collects_full_response(self):
        provider = MockModelProvider(response_content="I am Claude")
        request = ModelRequest(messages=[Message(role=Role.USER, content="Who are you?")])

        collected = []
        async for chunk in provider.stream(request):
            if chunk.content:
                collected.append(chunk.content)

        full = "".join(collected)
        assert "Claude" in full

    @pytest.mark.asyncio
    async def test_model_info(self):
        provider = MockModelProvider()
        info = provider.get_model_info()
        assert info.provider == "mock"
        assert info.model_name == "mock-model"
