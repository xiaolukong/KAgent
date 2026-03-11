"""Global configuration for KAgent.

Provides a singleton KAgentConfig that holds base_url and api_key.
All model providers fall back to this config when not configured individually.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings

# Module-level singleton
_config: KAgentConfig | None = None


class KAgentConfig(BaseSettings):
    """Global KAgent configuration.

    Values are resolved in order: explicit configure() call > env vars.
    Environment variable names: KAGENT_BASE_URL, KAGENT_API_KEY.
    """

    base_url: str | None = None
    api_key: str | None = None

    model_config = {"env_prefix": "KAGENT_"}


def configure(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
) -> KAgentConfig:
    """Set the global KAgent configuration.

    Calling this overwrites any previously set values.
    Fields left as None will still fall back to environment variables.
    """
    global _config
    _config = KAgentConfig(base_url=base_url, api_key=api_key)
    return _config


def get_config() -> KAgentConfig:
    """Return the current global config, creating a default one if needed."""
    global _config
    if _config is None:
        _config = KAgentConfig()
    return _config
