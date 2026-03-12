"""SteeringController — dual-queue system for runtime agent intervention."""

from __future__ import annotations

import asyncio

from kagent.domain.entities import Message
from kagent.domain.enums import EventType
from kagent.domain.events import SteeringEvent
from kagent.domain.protocols import IEventBus


class SteeringController:
    """Manages two queues for runtime steering:

    - steering_queue: high-priority directives (abort, redirect, interrupt)
      checked at the START of each loop iteration
    - message_queue: follow-up messages injected into the conversation
      processed AFTER the current tool execution completes
    """

    def __init__(self, event_bus: IEventBus) -> None:
        self._event_bus = event_bus
        self._steering_queue: asyncio.Queue[SteeringEvent] = asyncio.Queue()
        self._message_queue: asyncio.Queue[Message] = asyncio.Queue()
        self._aborted = False

        # Interrupt state
        self._interrupt_event = asyncio.Event()
        self._interrupt_event.set()  # Not interrupted by default
        self._interrupt_prompt: str | None = None
        self._interrupt_response: str | None = None
        self._interrupted = False

        # Subscribe to steering events
        event_bus.subscribe(EventType.STEERING_ABORT.value, self._on_abort)
        event_bus.subscribe(EventType.STEERING_REDIRECT.value, self._on_redirect)
        event_bus.subscribe(EventType.STEERING_INJECT_MESSAGE.value, self._on_inject)
        event_bus.subscribe(EventType.STEERING_INTERRUPT.value, self._on_interrupt)
        event_bus.subscribe(EventType.STEERING_RESUME.value, self._on_resume)

    @property
    def is_aborted(self) -> bool:
        return self._aborted

    @property
    def is_interrupted(self) -> bool:
        return self._interrupted

    async def _on_abort(self, event: SteeringEvent) -> None:  # type: ignore[override]
        self._aborted = True
        await self._steering_queue.put(event)

    async def _on_redirect(self, event: SteeringEvent) -> None:  # type: ignore[override]
        await self._steering_queue.put(event)

    async def _on_inject(self, event: SteeringEvent) -> None:  # type: ignore[override]
        msg_data = event.payload.get("message")
        if isinstance(msg_data, Message):
            await self._message_queue.put(msg_data)

    async def _on_interrupt(self, event: SteeringEvent) -> None:  # type: ignore[override]
        self._interrupt_prompt = event.payload.get("prompt", "")
        self._interrupt_response = None
        self._interrupted = True
        self._interrupt_event.clear()

    async def _on_resume(self, event: SteeringEvent) -> None:  # type: ignore[override]
        self._interrupt_response = event.payload.get("user_input", "")
        self._interrupted = False
        self._interrupt_event.set()

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

    async def wait_for_resume(self) -> str:
        """Block until a resume event provides user input. Returns the user's reply."""
        await self._interrupt_event.wait()
        response = self._interrupt_response or ""
        self._interrupt_prompt = None
        self._interrupt_response = None
        return response

    def reset(self) -> None:
        """Clear all queues and reset abort/interrupt flags."""
        self._aborted = False
        self._interrupted = False
        self._interrupt_prompt = None
        self._interrupt_response = None
        self._interrupt_event.set()
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
