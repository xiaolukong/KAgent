"""Tests for SteeringController dual-queue system."""

import asyncio

import pytest

from kagent.agent.steering import SteeringController
from kagent.domain.entities import Message
from kagent.domain.enums import EventType, Role
from kagent.domain.events import SteeringEvent
from kagent.events.bus import EventBus


@pytest.fixture
def steering_setup():
    bus = EventBus()
    controller = SteeringController(bus)
    return controller, bus


class TestSteeringAbort:
    @pytest.mark.asyncio
    async def test_abort_sets_flag(self, steering_setup):
        controller, bus = steering_setup
        assert controller.is_aborted is False

        await bus.publish(
            SteeringEvent(
                event_type=EventType.STEERING_ABORT,
                payload={"reason": "test abort"},
            )
        )

        assert controller.is_aborted is True

    @pytest.mark.asyncio
    async def test_abort_queues_directive(self, steering_setup):
        controller, bus = steering_setup
        await bus.publish(
            SteeringEvent(
                event_type=EventType.STEERING_ABORT,
                payload={"reason": "stop"},
            )
        )
        directive = controller.get_pending_directive()
        assert directive is not None
        assert directive.event_type == EventType.STEERING_ABORT


class TestSteeringRedirect:
    @pytest.mark.asyncio
    async def test_redirect_queued(self, steering_setup):
        controller, bus = steering_setup
        await bus.publish(
            SteeringEvent(
                event_type=EventType.STEERING_REDIRECT,
                payload={"directive": "change topic"},
            )
        )
        directive = controller.get_pending_directive()
        assert directive is not None
        assert directive.payload["directive"] == "change topic"


class TestSteeringInjectMessage:
    @pytest.mark.asyncio
    async def test_inject_message_queued(self, steering_setup):
        controller, bus = steering_setup
        msg = Message(role=Role.USER, content="injected message")
        await bus.publish(
            SteeringEvent(
                event_type=EventType.STEERING_INJECT_MESSAGE,
                payload={"message": msg},
            )
        )
        messages = controller.get_pending_messages()
        assert len(messages) == 1
        assert messages[0].content == "injected message"


class TestSteeringInterrupt:
    @pytest.mark.asyncio
    async def test_interrupt_sets_flag(self, steering_setup):
        controller, bus = steering_setup
        assert controller.is_interrupted is False

        await bus.publish(
            SteeringEvent(
                event_type=EventType.STEERING_INTERRUPT,
                payload={"prompt": "Need confirmation"},
            )
        )

        assert controller.is_interrupted is True

    @pytest.mark.asyncio
    async def test_interrupt_does_not_queue_directive(self, steering_setup):
        """Interrupt no longer queues a steering directive — it blocks inline."""
        controller, bus = steering_setup
        await bus.publish(
            SteeringEvent(
                event_type=EventType.STEERING_INTERRUPT,
                payload={"prompt": "Please confirm"},
            )
        )
        # Interrupt is handled inline (via wait_for_resume), not via the queue
        directive = controller.get_pending_directive()
        assert directive is None

    @pytest.mark.asyncio
    async def test_interrupt_stores_prompt(self, steering_setup):
        controller, bus = steering_setup
        await bus.publish(
            SteeringEvent(
                event_type=EventType.STEERING_INTERRUPT,
                payload={"prompt": "What should I do?"},
            )
        )
        assert controller._interrupt_prompt == "What should I do?"

    @pytest.mark.asyncio
    async def test_resume_clears_interrupt(self, steering_setup):
        controller, bus = steering_setup
        await bus.publish(
            SteeringEvent(
                event_type=EventType.STEERING_INTERRUPT,
                payload={"prompt": "Confirm?"},
            )
        )
        assert controller.is_interrupted is True

        await bus.publish(
            SteeringEvent(
                event_type=EventType.STEERING_RESUME,
                payload={"user_input": "Yes"},
            )
        )
        assert controller.is_interrupted is False

    @pytest.mark.asyncio
    async def test_wait_for_resume_returns_user_input(self, steering_setup):
        controller, bus = steering_setup

        await bus.publish(
            SteeringEvent(
                event_type=EventType.STEERING_INTERRUPT,
                payload={"prompt": "Pick a color"},
            )
        )

        # Resume from another task after a small delay
        async def provide_input():
            await asyncio.sleep(0.05)
            await bus.publish(
                SteeringEvent(
                    event_type=EventType.STEERING_RESUME,
                    payload={"user_input": "blue"},
                )
            )

        asyncio.create_task(provide_input())
        result = await controller.wait_for_resume()
        assert result == "blue"

    @pytest.mark.asyncio
    async def test_wait_for_resume_clears_state(self, steering_setup):
        controller, bus = steering_setup

        await bus.publish(
            SteeringEvent(
                event_type=EventType.STEERING_INTERRUPT,
                payload={"prompt": "Choose"},
            )
        )

        async def provide_input():
            await asyncio.sleep(0.05)
            await bus.publish(
                SteeringEvent(
                    event_type=EventType.STEERING_RESUME,
                    payload={"user_input": "done"},
                )
            )

        asyncio.create_task(provide_input())
        await controller.wait_for_resume()

        # After resume, prompt and response should be cleared
        assert controller._interrupt_prompt is None
        assert controller._interrupt_response is None
        assert controller.is_interrupted is False


class TestSteeringReset:
    @pytest.mark.asyncio
    async def test_reset_clears_all(self, steering_setup):
        controller, bus = steering_setup
        await bus.publish(
            SteeringEvent(
                event_type=EventType.STEERING_ABORT,
                payload={"reason": "test"},
            )
        )
        assert controller.is_aborted is True

        controller.reset()
        assert controller.is_aborted is False
        assert controller.get_pending_directive() is None
        assert controller.get_pending_messages() == []

    @pytest.mark.asyncio
    async def test_reset_clears_interrupt(self, steering_setup):
        controller, bus = steering_setup
        await bus.publish(
            SteeringEvent(
                event_type=EventType.STEERING_INTERRUPT,
                payload={"prompt": "Waiting..."},
            )
        )
        assert controller.is_interrupted is True

        controller.reset()
        assert controller.is_interrupted is False
        assert controller._interrupt_prompt is None
        assert controller._interrupt_response is None


class TestSteeringEmpty:
    def test_no_pending_directive(self, steering_setup):
        controller, bus = steering_setup
        assert controller.get_pending_directive() is None

    def test_no_pending_messages(self, steering_setup):
        controller, bus = steering_setup
        assert controller.get_pending_messages() == []

    def test_not_interrupted_initially(self, steering_setup):
        controller, bus = steering_setup
        assert controller.is_interrupted is False
