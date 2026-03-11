"""Tests for ToolRegistry."""

import pytest

from kagent.common.errors import ToolError
from kagent.tools.decorator import tool
from kagent.tools.registry import ToolRegistry


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry()


class TestRegister:
    def test_register_tool(self, registry: ToolRegistry):
        @tool
        async def my_tool(x: int) -> int:
            """A tool."""
            return x

        registry.register(my_tool)
        assert "my_tool" in registry
        assert len(registry) == 1

    def test_register_duplicate_raises(self, registry: ToolRegistry):
        @tool
        async def dup(x: int) -> int:
            """Dup."""
            return x

        registry.register(dup)
        with pytest.raises(ToolError, match="already registered"):
            registry.register(dup)


class TestUnregister:
    def test_unregister(self, registry: ToolRegistry):
        @tool
        async def temp(x: int) -> int:
            """Temp."""
            return x

        registry.register(temp)
        assert "temp" in registry
        registry.unregister("temp")
        assert "temp" not in registry

    def test_unregister_missing_raises(self, registry: ToolRegistry):
        with pytest.raises(ToolError, match="not found"):
            registry.unregister("nonexistent")


class TestGet:
    def test_get_existing(self, registry: ToolRegistry):
        @tool
        async def existing(x: int) -> int:
            """Existing."""
            return x

        registry.register(existing)
        wrapper = registry.get("existing")
        assert wrapper.get_definition().name == "existing"

    def test_get_missing_raises(self, registry: ToolRegistry):
        with pytest.raises(ToolError, match="not found"):
            registry.get("missing")


class TestListDefinitions:
    def test_list(self, registry: ToolRegistry):
        @tool
        async def a() -> str:
            """A."""
            return "a"

        @tool
        async def b() -> str:
            """B."""
            return "b"

        registry.register(a)
        registry.register(b)
        defs = registry.list_definitions()
        names = {d.name for d in defs}
        assert names == {"a", "b"}
