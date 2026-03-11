"""ToolExecutor — validates, runs, and wraps tool results with event publishing."""

from __future__ import annotations

from typing import Any

from kagent.common.errors import ToolError
from kagent.common.utils import Timer
from kagent.domain.entities import ToolResult
from kagent.domain.enums import EventType, ToolCallStatus
from kagent.domain.events import ToolEvent
from kagent.domain.protocols import IEventBus
from kagent.tools.registry import ToolRegistry


class ToolExecutor:
    """Executes registered tools with validation, timing, and event publishing."""

    def __init__(self, registry: ToolRegistry, event_bus: IEventBus) -> None:
        self._registry = registry
        self._event_bus = event_bus

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        call_id: str | None = None,
    ) -> ToolResult:
        """Look up a tool by name, execute it, and publish lifecycle events."""
        # Publish started event
        await self._event_bus.publish(
            ToolEvent(
                event_type=EventType.TOOL_CALL_STARTED,
                payload={"tool_name": tool_name, "arguments": arguments, "call_id": call_id},
                source="tool_executor",
            )
        )

        try:
            wrapper = self._registry.get(tool_name)
        except ToolError:
            result = ToolResult(
                tool_call_id=call_id or "",
                tool_name=tool_name,
                error=f"Tool '{tool_name}' not found",
                status=ToolCallStatus.ERROR,
            )
            await self._event_bus.publish(
                ToolEvent(
                    event_type=EventType.TOOL_CALL_ERROR,
                    payload={"tool_name": tool_name, "error": result.error},
                    source="tool_executor",
                )
            )
            return result

        timer = Timer()
        with timer:
            result = await wrapper.execute(arguments)

        # Override call_id if provided externally (e.g. from model's tool_call.id)
        if call_id:
            result.tool_call_id = call_id
        result.duration_ms = timer.elapsed_ms

        if result.status == ToolCallStatus.COMPLETED:
            await self._event_bus.publish(
                ToolEvent(
                    event_type=EventType.TOOL_CALL_COMPLETED,
                    payload={
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "result": result.result,
                        "duration_ms": result.duration_ms,
                    },
                    source="tool_executor",
                )
            )
        else:
            await self._event_bus.publish(
                ToolEvent(
                    event_type=EventType.TOOL_CALL_ERROR,
                    payload={
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "error": result.error,
                        "duration_ms": result.duration_ms,
                    },
                    source="tool_executor",
                )
            )

        return result
