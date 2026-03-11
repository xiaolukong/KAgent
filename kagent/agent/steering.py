"""SteeringController — dual-queue system for runtime agent intervention."""

from __future__ import annotations

import asyncio

from kagent.domain.entities import Message
from kagent.domain.enums import EventType
from kagent.domain.events import SteeringEvent
from kagent.domain.protocols import IEventBus


class SteeringController:
    """Manages two queues for runtime steering:

    - steering_queue: high-priority directives (abort, redirect)
      checked at the START of each loop iteration
    - message_queue: follow-up messages injected into the conversation
      processed AFTER the current tool execution completes
    """

    def __init__(self, event_bus: IEventBus) -> None:
        self._event_bus = event_bus
        self._steering_queue: asyncio.Queue[SteeringEvent] = asyncio.Queue()
        self._message_queue: asyncio.Queue[Message] = asyncio.Queue()
        self._aborted = False

        # Subscribe to steering events
        event_bus.subscribe(EventType.STEERING_ABORT.value, self._on_abort)
        event_bus.subscribe(EventType.STEERING_REDIRECT.value, self._on_redirect)
        event_bus.subscribe(EventType.STEERING_INJECT_MESSAGE.value, self._on_inject)

    @property
    def is_aborted(self) -> bool:
        return self._aborted

    async def _on_abort(self, event: SteeringEvent) -> None:  # type: ignore[override]
        self._aborted = True
        await self._steering_queue.put(event)

    async def _on_redirect(self, event: SteeringEvent) -> None:  # type: ignore[override]
        await self._steering_queue.put(event)

    async def _on_inject(self, event: SteeringEvent) -> None:  # type: ignore[override]
        msg_data = event.payload.get("message")
        if isinstance(msg_data, Message):
            await self._message_queue.put(msg_data)

    def get_pending_directive(self) -> SteeringEvent | None:
        """Non-blocking check for a high-priority steering directive."""
        try:
            return self._steering_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def get_pending_messages(self) -> list[Message]:
        """Drain all pending follow-up messages."""
        messages: list[Message] = []
        while not self._message_queue.empty():
            try:
                messages.append(self._message_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return messages

    def reset(self) -> None:
        """Clear all queues and reset abort flag."""
        self._aborted = False
        while not self._steering_queue.empty():
            try:
                self._steering_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        while not self._message_queue.empty():
            try:
                self._message_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
