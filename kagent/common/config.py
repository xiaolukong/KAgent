"""Global configuration for KAgent.

Provides a singleton KAgentConfig that holds base_url and api_key.
All model providers fall back to this config when not configured individually.

Resolution order (highest priority first):
    1. Explicit ``configure(api_key=...)`` call
    2. Environment variables (``KAGENT_API_KEY``, ``KAGENT_BASE_URL``)
    3. ``.env`` file in the project root (auto-loaded by pydantic-settings)
"""

from __future__ import annotations

from pydantic_settings import BaseSettings

# Module-level singleton
_config: KAgentConfig | None = None


class KAgentConfig(BaseSettings):
    """Global KAgent configuration.

    Values are resolved in order:
        explicit configure() call > env vars > .env file.

    Environment variable names: KAGENT_BASE_URL, KAGENT_API_KEY.
    """

    base_url: str | None = None
    api_key: str | None = None

    model_config = {"env_prefix": "KAGENT_", "env_file": ".env", "env_file_encoding": "utf-8"}


def configure(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
) -> KAgentConfig:
    """Set the global KAgent configuration.

    Calling this overwrites any previously set values.
    Fields left as None will fall back to environment variables / .env file.
    """
    global _config
    kwargs: dict[str, str] = {}
    if base_url is not None:
        kwargs["base_url"] = base_url
    if api_key is not None:
        kwargs["api_key"] = api_key
    _config = KAgentConfig(**kwargs)
    return _config


def get_config() -> KAgentConfig:
    """Return the current global config, creating a default one if needed."""
    global _config
    if _config is None:
        _config = KAgentConfig()
    return _config
