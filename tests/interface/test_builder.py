"""Tests for KAgentBuilder."""

from unittest.mock import patch

from kagent.interface.builder import KAgentBuilder
from kagent.interface.kagent import KAgent
from tests.conftest import MockModelProvider


def _mock_create_provider(model_string, **kwargs):
    return MockModelProvider()


class TestKAgentBuilder:
    @patch("kagent.interface.kagent.create_provider", side_effect=_mock_create_provider)
    def test_build_with_defaults(self, mock_factory):
        agent = KAgentBuilder().build()
        assert isinstance(agent, KAgent)

    @patch("kagent.interface.kagent.create_provider", side_effect=_mock_create_provider)
    def test_builder_chaining(self, mock_factory):
        agent = (
            KAgentBuilder()
            .model("mock:model")
            .system_prompt("Custom prompt")
            .max_turns(20)
            .temperature(0.5)
            .max_tokens(1000)
            .max_context_tokens(64_000)
            .build()
        )
        assert isinstance(agent, KAgent)

    @patch("kagent.interface.kagent.create_provider", side_effect=_mock_create_provider)
    def test_builder_with_api_key(self, mock_factory):
        agent = (
            KAgentBuilder()
            .model("mock:model")
            .api_key("sk-test")
            .base_url("https://proxy.example.com")
            .build()
        )
        assert isinstance(agent, KAgent)
