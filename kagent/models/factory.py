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

    When ``configure(backend="aicore")`` is active (the default), all model
    strings are routed to :class:`AICoreProvider` regardless of the provider
    prefix.  Set ``configure(backend="aiproxy")`` to use the direct provider
    implementations (OpenAI, Anthropic, Gemini).

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

    # ── AI Core backend routing ────────────────────────────────────────────
    from kagent.common.config import get_config

    if get_config().backend == ModelProviderType.AICORE:
        from kagent.models.aicore_provider import AICoreProvider

        return AICoreProvider(config)

    # ── Direct provider routing (backend="aiproxy") ────────────────────────
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
