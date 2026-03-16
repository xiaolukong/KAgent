"""Agent configuration."""

from __future__ import annotations

from pydantic import BaseModel


class AgentConfig(BaseModel):
    """Configuration for an Agent instance."""

    model: str = "openai:gpt-4o"
    system_prompt: str = "You are a helpful assistant."
    max_turns: int = 1
    temperature: float | None = None
    max_tokens: int | None = None
    tool_choice: str | None = None
    max_context_tokens: int = 128_000
    max_tool_retries: int = 3
