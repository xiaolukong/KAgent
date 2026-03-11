"""StateManager — runtime state CRUD with EventBus broadcasting."""

from __future__ import annotations

from typing import Any

from kagent.domain.enums import EventType
from kagent.domain.events import AgentEvent
from kagent.domain.protocols import IEventBus, IStateStore


class StateManager:
    """Manages agent runtime state, broadcasting changes via EventBus."""

    def __init__(self, store: IStateStore, event_bus: IEventBus) -> None:
        self._store = store
        self._event_bus = event_bus

    async def get(self, key: str) -> Any:
        return await self._store.get(key)

    async def set(self, key: str, value: Any) -> None:
        old = await self._store.get(key)
        await self._store.set(key, value)
        await self._event_bus.publish(
            AgentEvent(
                event_type=EventType.AGENT_STATE_CHANGED,
                payload={"key": key, "old_value": old, "new_value": value},
                source="state_manager",
            )
        )

    async def update(self, key: str, value: Any) -> None:
        old = await self._store.get(key)
        await self._store.update(key, value)
        await self._event_bus.publish(
            AgentEvent(
                event_type=EventType.AGENT_STATE_CHANGED,
                payload={"key": key, "old_value": old, "new_value": value},
                source="state_manager",
            )
        )

    async def delete(self, key: str) -> None:
        old = await self._store.get(key)
        await self._store.delete(key)
        await self._event_bus.publish(
            AgentEvent(
                event_type=EventType.AGENT_STATE_CHANGED,
                payload={"key": key, "old_value": old, "new_value": None},
                source="state_manager",
            )
        )
