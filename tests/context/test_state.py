"""Tests for StateManager and InMemoryStateStore."""

import pytest

from kagent.context.state import StateManager
from kagent.context.stores.memory import InMemoryStateStore
from kagent.domain.events import Event
from kagent.events.bus import EventBus


@pytest.fixture
def state_setup():
    bus = EventBus()
    store = InMemoryStateStore()
    manager = StateManager(store, bus)
    return manager, bus, store


class TestInMemoryStateStore:
    @pytest.mark.asyncio
    async def test_set_and_get(self):
        store = InMemoryStateStore()
        await store.set("key", "value")
        assert await store.get("key") == "value"

    @pytest.mark.asyncio
    async def test_get_missing(self):
        store = InMemoryStateStore()
        assert await store.get("missing") is None

    @pytest.mark.asyncio
    async def test_update(self):
        store = InMemoryStateStore()
        await store.set("key", "old")
        await store.update("key", "new")
        assert await store.get("key") == "new"

    @pytest.mark.asyncio
    async def test_delete(self):
        store = InMemoryStateStore()
        await store.set("key", "value")
        await store.delete("key")
        assert await store.get("key") is None

    @pytest.mark.asyncio
    async def test_delete_missing_is_noop(self):
        store = InMemoryStateStore()
        await store.delete("missing")  # should not raise


class TestStateManager:
    @pytest.mark.asyncio
    async def test_set_publishes_event(self, state_setup):
        manager, bus, store = state_setup
        events = []

        async def collector(event: Event):
            events.append(event)

        bus.subscribe("agent.state.changed", collector)
        await manager.set("counter", 42)

        assert await store.get("counter") == 42
        assert len(events) == 1
        assert events[0].payload["new_value"] == 42

    @pytest.mark.asyncio
    async def test_update_publishes_event(self, state_setup):
        manager, bus, store = state_setup
        events = []

        async def collector(event: Event):
            events.append(event)

        bus.subscribe("agent.state.changed", collector)
        await manager.set("counter", 1)
        await manager.update("counter", 2)

        assert len(events) == 2
        assert events[1].payload["old_value"] == 1
        assert events[1].payload["new_value"] == 2

    @pytest.mark.asyncio
    async def test_delete_publishes_event(self, state_setup):
        manager, bus, store = state_setup
        events = []

        async def collector(event: Event):
            events.append(event)

        bus.subscribe("agent.state.changed", collector)
        await manager.set("key", "val")
        await manager.delete("key")

        assert len(events) == 2
        assert events[1].payload["new_value"] is None
