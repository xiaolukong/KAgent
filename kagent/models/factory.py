"""ModelProviderFactory — create providers from 'provider:model' strings."""

from __future__ import annotations

from kagent.common.errors import ConfigError
from kagent.domain.enums import ModelProviderType
from kagent.models.base import BaseModelProvider
from kagent.models.config import ModelConfig


def create_provider(
    model_string: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: object,
) -> BaseModelProvider:
    """Create a model provider from a string like 'openai:gpt-4o'.

    The provider-specific SDK must be installed. Extra kwargs are
    passed through to ModelConfig.
    """
    parts = model_string.split(":", 1)
    if len(parts) != 2:
        raise ConfigError(
            f"Invalid model string '{model_string}'. "
            "Expected format: 'provider:model_name' (e.g. 'openai:gpt-4o')"
        )

    provider_name, model_name = parts
    config = ModelConfig(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        **kwargs,  # type: ignore[arg-type]
    )

    if provider_name == ModelProviderType.OPENAI:
        from kagent.models.openai_provider import OpenAIProvider

        return OpenAIProvider(config)

    if provider_name == ModelProviderType.ANTHROPIC:
        from kagent.models.anthropic_provider import AnthropicProvider

        return AnthropicProvider(config)

    if provider_name == ModelProviderType.GEMINI:
        from kagent.models.gemini_provider import GeminiProvider

        return GeminiProvider(config)

    raise ConfigError(
        f"Unknown provider '{provider_name}'. "
        f"Supported: {', '.join(t.value for t in ModelProviderType)}"
    )
