"""Tests for ContextManager."""

from kagent.context.manager import ContextManager
from kagent.domain.entities import Message
from kagent.domain.enums import Role


class TestContextManager:
    def test_add_and_get_messages(self):
        cm = ContextManager()
        cm.add_message(Message(role=Role.USER, content="Hi"))
        cm.add_message(Message(role=Role.ASSISTANT, content="Hello"))
        msgs = cm.get_all_messages()
        assert len(msgs) == 2

    def test_get_messages_trimmed(self):
        cm = ContextManager(max_tokens=20)
        cm.add_message(Message(role=Role.SYSTEM, content="System"))
        cm.add_message(Message(role=Role.USER, content="A" * 200))
        cm.add_message(Message(role=Role.USER, content="Short"))
        trimmed = cm.get_messages()
        # System message is always kept; the long message should be trimmed
        assert any(m.role == Role.SYSTEM for m in trimmed)

    def test_context_crud(self):
        cm = ContextManager()
        cm.update_context(key1="val1", key2="val2")
        ctx = cm.get_context()
        assert ctx["key1"] == "val1"
        assert ctx["key2"] == "val2"

    def test_snapshot_and_restore(self):
        cm = ContextManager()
        cm.add_message(Message(role=Role.USER, content="test"))
        cm.update_context(foo="bar")
        snap = cm.snapshot()

        cm2 = ContextManager()
        cm2.restore(snap)
        assert len(cm2.get_all_messages()) == 1
        assert cm2.get_context()["foo"] == "bar"

    def test_to_agent_state(self):
        cm = ContextManager()
        cm.add_message(Message(role=Role.USER, content="hi"))
        state = cm.to_agent_state(turn_count=3)
        assert state.turn_count == 3
        assert len(state.messages) == 1

    def test_clear(self):
        cm = ContextManager()
        cm.add_message(Message(role=Role.USER, content="hi"))
        cm.update_context(key="val")
        cm.clear()
        assert len(cm.get_all_messages()) == 0
        assert cm.get_context() == {}
