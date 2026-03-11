"""KAgentBuilder — chainable builder for constructing KAgent instances."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kagent.agent.config import AgentConfig

if TYPE_CHECKING:
    from kagent.interface.kagent import KAgent


class KAgentBuilder:
    """Builder pattern for configuring and constructing a KAgent."""

    def __init__(self) -> None:
        self._model: str = "openai:gpt-4o"
        self._system_prompt: str = "You are a helpful assistant."
        self._max_turns: int = 10
        self._temperature: float | None = None
        self._max_tokens: int | None = None
        self._tool_choice: str | None = None
        self._max_context_tokens: int = 128_000
        self._api_key: str | None = None
        self._base_url: str | None = None

    def model(self, model: str) -> KAgentBuilder:
        self._model = model
        return self

    def system_prompt(self, prompt: str) -> KAgentBuilder:
        self._system_prompt = prompt
        return self

    def max_turns(self, n: int) -> KAgentBuilder:
        self._max_turns = n
        return self

    def temperature(self, t: float) -> KAgentBuilder:
        self._temperature = t
        return self

    def max_tokens(self, n: int) -> KAgentBuilder:
        self._max_tokens = n
        return self

    def tool_choice(self, choice: str) -> KAgentBuilder:
        self._tool_choice = choice
        return self

    def max_context_tokens(self, n: int) -> KAgentBuilder:
        self._max_context_tokens = n
        return self

    def api_key(self, key: str) -> KAgentBuilder:
        self._api_key = key
        return self

    def base_url(self, url: str) -> KAgentBuilder:
        self._base_url = url
        return self

    def build(self) -> KAgent:
        from kagent.interface.kagent import KAgent

        return KAgent(
            model=self._model,
            system_prompt=self._system_prompt,
            max_turns=self._max_turns,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            tool_choice=self._tool_choice,
            max_context_tokens=self._max_context_tokens,
            api_key=self._api_key,
            base_url=self._base_url,
        )
