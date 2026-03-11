"""Tests for @tool decorator."""

import pytest

from kagent.domain.enums import ToolCallStatus
from kagent.tools.decorator import ToolWrapper, tool


class TestToolDecorator:
    def test_bare_decorator(self):
        @tool
        async def my_tool(x: int) -> int:
            """My tool."""
            return x * 2

        assert isinstance(my_tool, ToolWrapper)
        defn = my_tool.get_definition()
        assert defn.name == "my_tool"
        assert defn.description == "My tool."

    def test_decorator_with_args(self):
        @tool(name="custom_name", description="Custom desc")
        async def my_tool(x: int) -> int:
            return x

        defn = my_tool.get_definition()
        assert defn.name == "custom_name"
        assert defn.description == "Custom desc"

    def test_sync_function(self):
        @tool
        def sync_tool(x: int) -> int:
            """Sync."""
            return x + 1

        assert isinstance(sync_tool, ToolWrapper)


class TestToolExecution:
    @pytest.mark.asyncio
    async def test_execute_success(self):
        @tool
        async def add(a: int, b: int) -> int:
            """Add."""
            return a + b

        result = await add.execute({"a": 3, "b": 5})
        assert result.result == 8
        assert result.status == ToolCallStatus.COMPLETED
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_execute_sync_tool(self):
        @tool
        def multiply(a: int, b: int) -> int:
            """Multiply."""
            return a * b

        result = await multiply.execute({"a": 4, "b": 5})
        assert result.result == 20

    @pytest.mark.asyncio
    async def test_execute_error(self):
        @tool
        async def failing():
            """Fail."""
            raise ValueError("boom")

        result = await failing.execute({})
        assert result.status == ToolCallStatus.ERROR
        assert "boom" in (result.error or "")

    @pytest.mark.asyncio
    async def test_execute_with_default_params(self):
        @tool
        async def greet(name: str, greeting: str = "Hello") -> str:
            """Greet."""
            return f"{greeting}, {name}!"

        result = await greet.execute({"name": "World"})
        assert result.result == "Hello, World!"

        result2 = await greet.execute({"name": "World", "greeting": "Hi"})
        assert result2.result == "Hi, World!"


class TestToolValidation:
    @pytest.mark.asyncio
    async def test_validate_input_types(self):
        @tool
        async def typed(x: int, y: str) -> str:
            """Typed."""
            return f"{y}={x}"

        validated = await typed.validate_input({"x": 5, "y": "val"})
        assert validated == {"x": 5, "y": "val"}

    @pytest.mark.asyncio
    async def test_validate_coercion(self):
        @tool
        async def typed(x: int) -> int:
            """Typed."""
            return x

        # String "5" should be coerced to int 5
        validated = await typed.validate_input({"x": "5"})
        assert validated["x"] == 5

    @pytest.mark.asyncio
    async def test_validate_bad_type(self):
        from kagent.common.errors import ValidationError

        @tool
        async def typed(x: int) -> int:
            """Typed."""
            return x

        with pytest.raises(ValidationError):
            await typed.validate_input({"x": "not-a-number"})
