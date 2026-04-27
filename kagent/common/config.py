"""Global configuration for KAgent.

Provides a singleton KAgentConfig that holds backend selection and credentials.
All model providers fall back to this config when not configured individually.

Resolution order (highest priority first):
    1. Explicit ``configure(...)`` call
    2. Environment variables (``KAGENT_*`` prefix for proxy fields)
    3. ``.env`` file in the project root (auto-loaded by pydantic-settings)

Backend modes:
    ``"aicore"`` (default) — route all model calls through SAP AI Core
                             Orchestration Service. Credentials are read from
                             AICORE_* env vars or explicit configure() params.
    ``"aiproxy"``           — direct connection to OpenAI / Anthropic / Gemini
                             APIs. Uses KAGENT_API_KEY and KAGENT_BASE_URL.
"""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings

# Module-level singleton
_config: KAgentConfig | None = None


class KAgentConfig(BaseSettings):
    """Global KAgent configuration.

    Values are resolved in order:
        explicit configure() call > env vars > .env file.
    """

    # ── Backend selection ──────────────────────────────────────────────────
    backend: Literal["aicore", "aiproxy"] = "aicore"

    # ── AI Proxy mode (backend="aiproxy") ─────────────────────────────────
    # Environment variable names: KAGENT_API_KEY, KAGENT_BASE_URL
    base_url: str | None = None
    api_key: str | None = None

    # ── AI Core mode (backend="aicore") ───────────────────────────────────
    # GenAIHubProxyClient reads AICORE_* env vars directly (no KAGENT_ prefix).
    # These fields are only needed when overriding via configure() explicitly.
    aicore_auth_url: str | None = None
    aicore_client_id: str | None = None
    aicore_client_secret: str | None = None
    aicore_base_url: str | None = None
    aicore_resource_group: str | None = None

    model_config = {"env_prefix": "KAGENT_", "env_file": ".env", "env_file_encoding": "utf-8"}


def configure(
    *,
    backend: Literal["aicore", "aiproxy"] | None = None,
    # AI Proxy params (original)
    base_url: str | None = None,
    api_key: str | None = None,
    # AI Core params (new)
    aicore_auth_url: str | None = None,
    aicore_client_id: str | None = None,
    aicore_client_secret: str | None = None,
    aicore_base_url: str | None = None,
    aicore_resource_group: str | None = None,
) -> KAgentConfig:
    """Set the global KAgent configuration.

    Calling this overwrites any previously set values.
    Fields left as None fall back to environment variables / .env file.

    Args:
        backend: ``"aicore"`` (default) or ``"aiproxy"``.
        base_url: AI Proxy base URL (aiproxy mode).
        api_key: AI Proxy API key (aiproxy mode).
        aicore_auth_url: AI Core OAuth token URL.
        aicore_client_id: AI Core client ID.
        aicore_client_secret: AI Core client secret.
        aicore_base_url: AI Core API base URL.
        aicore_resource_group: AI Core resource group name.
    """
    global _config
    kwargs: dict[str, object] = {}
    if backend is not None:
        kwargs["backend"] = backend
    if base_url is not None:
        kwargs["base_url"] = base_url
    if api_key is not None:
        kwargs["api_key"] = api_key
    if aicore_auth_url is not None:
        kwargs["aicore_auth_url"] = aicore_auth_url
    if aicore_client_id is not None:
        kwargs["aicore_client_id"] = aicore_client_id
    if aicore_client_secret is not None:
        kwargs["aicore_client_secret"] = aicore_client_secret
    if aicore_base_url is not None:
        kwargs["aicore_base_url"] = aicore_base_url
    if aicore_resource_group is not None:
        kwargs["aicore_resource_group"] = aicore_resource_group
    _config = KAgentConfig(**kwargs)
    return _config


def get_config() -> KAgentConfig:
    """Return the current global config, creating a default one if needed."""
    global _config
    if _config is None:
        _config = KAgentConfig()
    return _config
