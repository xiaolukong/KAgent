"""Protocol interfaces — the contracts that all layers depend on."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from kagent.domain.entities import ToolDefinition, ToolResult
from kagent.domain.events import Event
from kagent.domain.model_types import ModelInfo, ModelRequest, ModelResponse, StreamChunk


# ── Event Bus ────────────────────────────────────────────────────────────────

EventHandler = Callable[[Event], Coroutine[Any, Any, None]]


@runtime_checkable
class IEventBus(Protocol):
    """Pub/sub event bus interface."""

    async def publish(self, event: Event) -> None: ...

    def subscribe(
        self,
        event_pattern: str,
        handler: EventHandler,
        *,
        priority: int = 0,
    ) -> str: ...

    def unsubscribe(self, subscription_id: str) -> None: ...


# ── Model Provider ───────────────────────────────────────────────────────────


@runtime_checkable
class IModelProvider(Protocol):
    """LLM model provider interface supporting both streaming and non-streaming."""

    async def complete(
        self,
        request: ModelRequest,
        *,
        response_model: type[BaseModel] | None = None,
    ) -> ModelResponse: ...

    async def stream(
        self,
        request: ModelRequest,
        *,
        response_model: type[BaseModel] | None = None,
    ) -> AsyncIterator[StreamChunk]: ...

    def get_model_info(self) -> ModelInfo: ...


# ── Tool ─────────────────────────────────────────────────────────────────────


@runtime_checkable
class ITool(Protocol):
    """Interface for a registered tool."""

    def get_definition(self) -> ToolDefinition: ...

    async def validate_input(self, arguments: dict[str, Any]) -> dict[str, Any]: ...

    async def execute(self, arguments: dict[str, Any]) -> ToolResult: ...


# ── State Store ──────────────────────────────────────────────────────────────


@runtime_checkable
class IStateStore(Protocol):
    """Key-value state storage interface."""

    async def get(self, key: str) -> Any: ...

    async def set(self, key: str, value: Any) -> None: ...

    async def update(self, key: str, value: Any) -> None: ...

    async def delete(self, key: str) -> None: ...


# ── Context Manager ─────────────────────────────────────────────────────────


@runtime_checkable
class IContextManager(Protocol):
    """Interface for managing conversation context and history."""

    def get_context(self) -> dict[str, Any]: ...

    def update_context(self, **kwargs: Any) -> None: ...

    def snapshot(self) -> dict[str, Any]: ...

    def restore(self, snapshot: dict[str, Any]) -> None: ...
