"""Tests for ModelProviderFactory."""

import pytest

from kagent.common.errors import ConfigError
from kagent.models.factory import create_provider


class TestCreateProvider:
    def test_invalid_format_raises(self):
        with pytest.raises(ConfigError, match="Invalid model string"):
            create_provider("no-colon-here")

    def test_unknown_provider_raises(self):
        with pytest.raises(ConfigError, match="Unknown provider"):
            create_provider("unknown:model-123")

    def test_openai_provider_requires_package(self):
        """OpenAI provider should either succeed or raise ImportError."""
        try:
            provider = create_provider("openai:gpt-4o", api_key="test-key")
            info = provider.get_model_info()
            assert info.provider == "openai"
            assert info.model_name == "gpt-4o"
        except ImportError:
            pytest.skip("openai package not installed")

    def test_anthropic_provider_requires_package(self):
        try:
            provider = create_provider("anthropic:claude-3-sonnet", api_key="test-key")
            info = provider.get_model_info()
            assert info.provider == "anthropic"
        except ImportError:
            pytest.skip("anthropic package not installed")

    def test_gemini_provider_requires_package(self):
        try:
            provider = create_provider("gemini:gemini-pro", api_key="test-key")
            info = provider.get_model_info()
            assert info.provider == "gemini"
        except ImportError:
            pytest.skip("google-generativeai package not installed")
