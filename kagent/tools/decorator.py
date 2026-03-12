"""@tool decorator for registering functions as agent tools."""

from __future__ import annotations

import asyncio
import functools
import inspect
import uuid
from collections.abc import Callable
from typing import Any

from pydantic import TypeAdapter

from kagent.common.errors import ValidationError
from kagent.common.utils import Timer
from kagent.domain.entities import ToolDefinition, ToolResult
from kagent.domain.enums import ToolCallStatus
from kagent.tools.schema_gen import function_to_json_schema


class ToolWrapper:
    """Wraps a plain function as a tool with schema, validation, and execution."""

    def __init__(
        self,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        self._func = func
        self._name = name or func.__name__
        self._description = description or (inspect.getdoc(func) or "")
        self._parameters = function_to_json_schema(func)
        self._is_async = asyncio.iscoroutinefunction(func)

        # Build type adapters for parameter validation
        hints = {}
        try:
            hints = func.__annotations__
        except AttributeError:
            pass
        self._validators: dict[str, TypeAdapter[Any]] = {}
        sig = inspect.signature(func)
        for pname, param in sig.parameters.items():
            if pname in ("self", "cls", "return"):
                continue
            ann = hints.get(pname)
            if ann is not None:
                self._validators[pname] = TypeAdapter(ann)

    def get_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self._name,
            description=self._description,
            parameters=self._parameters,
        )

    async def validate_input(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Validate arguments against the function's type hints."""
        validated: dict[str, Any] = {}
        for key, value in arguments.items():
            adapter = self._validators.get(key)
            if adapter is not None:
                try:
                    validated[key] = adapter.validate_python(value)
                except Exception as exc:
                    raise ValidationError(
                        f"Validation failed for parameter '{key}': {exc}"
                    ) from exc
            else:
                validated[key] = value
        return validated

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Validate inputs, run the function, and return a ToolResult."""
        call_id = uuid.uuid4().hex[:12]
        timer = Timer()
        try:
            validated = await self.validate_input(arguments)
            with timer:
                if self._is_async:
                    result = await self._func(**validated)
                else:
                    result = self._func(**validated)
            return ToolResult(
                tool_call_id=call_id,
                tool_name=self._name,
                result=result,
                status=ToolCallStatus.COMPLETED,
                duration_ms=timer.elapsed_ms,
            )
        except ValidationError:
            raise
        except Exception as exc:
            return ToolResult(
                tool_call_id=call_id,
                tool_name=self._name,
                error=str(exc),
                status=ToolCallStatus.ERROR,
                duration_ms=timer.elapsed_ms,
            )


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Any:
    """Decorator that turns a function into a ToolWrapper.

    Usage:
        @tool
        async def my_tool(x: int) -> str: ...

        @tool(name="custom_name", description="Does something")
        async def my_tool(x: int) -> str: ...
    """

    def decorator(fn: Callable[..., Any]) -> ToolWrapper:
        wrapper = ToolWrapper(fn, name=name, description=description)
        functools.update_wrapper(wrapper, fn)
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator
