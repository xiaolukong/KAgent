"""PromptBuilder — assembles the full prompt from system prompt, context, tools, and history."""

from __future__ import annotations

from kagent.domain.entities import Message, ToolDefinition
from kagent.domain.enums import Role
from kagent.domain.model_types import ModelRequest


class PromptBuilder:
    """Builds a ModelRequest by combining system prompt, messages, and tool definitions."""

    def __init__(
        self,
        system_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tool_choice: str | None = None,
    ) -> None:
        self._system_prompt = system_prompt
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._tool_choice = tool_choice

    def build(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
    ) -> ModelRequest:
        """Assemble a complete ModelRequest."""
        # Ensure system message is at the front
        has_system = any(m.role == Role.SYSTEM for m in messages)
        if not has_system:
            system_msg = Message(role=Role.SYSTEM, content=self._system_prompt)
            messages = [system_msg] + messages

        return ModelRequest(
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            tools=tools if tools else None,
            tool_choice=self._tool_choice,
        )
