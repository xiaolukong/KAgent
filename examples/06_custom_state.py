"""Example 06: Custom state management.

This example demonstrates:
- Using StateManager with InMemoryStateStore
- State changes broadcast through EventBus
- Snapshot and restore of ContextManager

Note: This example exercises the KAgent context/state layer directly and
does not make LLM calls, so no AI Core credentials are required.

Usage:
    python examples/06_custom_state.py
"""

import asyncio

from kagent.context.manager import ContextManager
from kagent.context.state import StateManager
from kagent.context.stores.memory import InMemoryStateStore
from kagent.domain.entities import Message
from kagent.domain.enums import EventType, Role
from kagent.domain.events import Event
from kagent.events.bus import EventBus


async def main():
    # Set up event bus and state manager
    bus = EventBus()
    store = InMemoryStateStore()
    state_mgr = StateManager(store, bus)

    # Listen for state change events
    async def on_state_change(event: Event):
        key = event.payload.get("key")
        old = event.payload.get("old_value")
        new = event.payload.get("new_value")
        print(f"  State changed: {key} = {old} -> {new}")

    bus.subscribe(EventType.AGENT_STATE_CHANGED.value, on_state_change)

    # Demonstrate state CRUD
    print("=== State Management ===")
    await state_mgr.set("user_name", "Alice")
    await state_mgr.set("turn_count", 0)
    await state_mgr.update("turn_count", 1)
    await state_mgr.delete("user_name")

    print(f"\n  Final turn_count: {await state_mgr.get('turn_count')}")
    print(f"  Deleted user_name: {await state_mgr.get('user_name')}")

    # Demonstrate ContextManager snapshot/restore
    print("\n=== Context Snapshot/Restore ===")
    ctx = ContextManager(max_tokens=10000)
    ctx.add_message(Message(role=Role.SYSTEM, content="You are helpful."))
    ctx.add_message(Message(role=Role.USER, content="Hello!"))
    ctx.add_message(Message(role=Role.ASSISTANT, content="Hi there!"))
    ctx.update_context(session_id="abc123")

    # Take snapshot
    snapshot = ctx.snapshot()
    print(f"  Snapshot taken: {len(snapshot['messages'])} messages")

    # Restore into a new context manager
    ctx2 = ContextManager()
    ctx2.restore(snapshot)
    print(f"  Restored: {len(ctx2.get_all_messages())} messages")
    print(f"  Context: {ctx2.get_context()}")

    # Verify state
    state = ctx2.to_agent_state(turn_count=3)
    print(f"  AgentState turn_count: {state.turn_count}")
    print(f"  AgentState messages: {len(state.messages)}")


if __name__ == "__main__":
    asyncio.run(main())
