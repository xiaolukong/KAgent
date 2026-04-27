"""Tests for ModelProviderFactory."""

import pytest

from kagent.common.config import KAgentConfig, _config as _orig_config
from kagent.common.errors import ConfigError
from kagent.models.factory import create_provider
import kagent.common.config as _cfg_module


def _set_aiproxy():
    """Force aiproxy backend for tests that verify direct provider routing."""
    _cfg_module._config = KAgentConfig(backend="aiproxy", api_key="test-key")


def _reset():
    """Reset global config to default."""
    _cfg_module._config = None


class TestCreateProvider:
    def setup_method(self):
        _set_aiproxy()

    def teardown_method(self):
        _reset()

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


class TestCreateProviderAICore:
    def setup_method(self):
        _cfg_module._config = KAgentConfig(backend="aicore")

    def teardown_method(self):
        _reset()

    def test_aicore_backend_routes_openai_string(self):
        """backend=aicore routes 'openai:...' to AICoreProvider."""
        try:
            from unittest.mock import patch, MagicMock
            with patch("gen_ai_hub.proxy.GenAIHubProxyClient", return_value=MagicMock()), \
                 patch("gen_ai_hub.orchestration_v2.service.OrchestrationService", return_value=MagicMock()):
                provider = create_provider("openai:gpt-4o")
                info = provider.get_model_info()
                assert info.provider == "aicore"
                assert info.model_name == "gpt-4o"
        except ImportError:
            pytest.skip("generative-ai-hub-sdk not installed")

    def test_aicore_backend_routes_anthropic_string(self):
        """backend=aicore routes 'anthropic:...' to AICoreProvider."""
        try:
            from unittest.mock import patch, MagicMock
            with patch("gen_ai_hub.proxy.GenAIHubProxyClient", return_value=MagicMock()), \
                 patch("gen_ai_hub.orchestration_v2.service.OrchestrationService", return_value=MagicMock()):
                provider = create_provider("anthropic:claude-3-7-sonnet")
                info = provider.get_model_info()
                assert info.provider == "aicore"
                assert info.model_name == "claude-3-7-sonnet"
        except ImportError:
            pytest.skip("generative-ai-hub-sdk not installed")
