"""Model-specific configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for a specific model provider instance.

    Fields left as None will fall back to the global KAgentConfig values.
    """

    api_key: str | None = None
    base_url: str | None = None
    model_name: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int | None = None
    timeout: float = 120.0
    max_retries: int = 3
    extra: dict[str, object] = Field(default_factory=dict)
