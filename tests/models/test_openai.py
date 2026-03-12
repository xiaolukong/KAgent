"""Tests for OpenAI provider (using mock provider for unit tests)."""

import pytest

from kagent.domain.entities import Message
from kagent.domain.enums import Role
from kagent.domain.model_types import ModelRequest

# Use MockModelProvider from conftest to test base provider logic
from tests.conftest import MockModelProvider


class TestOpenAIProviderBase:
    """Tests using the mock to verify base provider behavior."""

    @pytest.mark.asyncio
    async def test_complete_returns_response(self):
        provider = MockModelProvider(response_content="Test response")
        request = ModelRequest(messages=[Message(role=Role.USER, content="Hello")])
        response = await provider.complete(request)
        assert response.content == "Test response"
        assert response.usage is not None

    @pytest.mark.asyncio
    async def test_stream_returns_chunks(self):
        provider = MockModelProvider(response_content="Hello world test")
        request = ModelRequest(messages=[Message(role=Role.USER, content="Hi")])
        chunks = []
        async for chunk in provider.stream(request):
            chunks.append(chunk)
        assert len(chunks) > 0
        # At least one text delta
        text_chunks = [c for c in chunks if c.content]
        assert len(text_chunks) > 0

    @pytest.mark.asyncio
    async def test_complete_with_response_model(self):
        from pydantic import BaseModel

        class Answer(BaseModel):
            text: str
            confidence: float

        provider = MockModelProvider(response_content='{"text": "hello", "confidence": 0.95}')
        request = ModelRequest(messages=[Message(role=Role.USER, content="test")])
        response = await provider.complete(request, response_model=Answer)
        assert response.parsed is not None
        assert response.parsed.text == "hello"
        assert response.parsed.confidence == 0.95

    @pytest.mark.asyncio
    async def test_complete_with_bad_json_raises(self):
        from pydantic import BaseModel

        from kagent.common.errors import ValidationError

        class Answer(BaseModel):
            text: str

        provider = MockModelProvider(response_content="not json at all")
        request = ModelRequest(messages=[Message(role=Role.USER, content="test")])
        with pytest.raises(ValidationError):
            await provider.complete(request, response_model=Answer)
