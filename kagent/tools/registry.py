"""ToolRegistry — global and per-agent tool registration."""

from __future__ import annotations

from kagent.common.errors import ToolError
from kagent.domain.entities import ToolDefinition
from kagent.tools.decorator import ToolWrapper


class ToolRegistry:
    """A registry that holds named ToolWrapper instances."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolWrapper] = {}

    def register(self, wrapper: ToolWrapper) -> None:
        """Register a tool. Raises ToolError if a tool with the same name exists."""
        defn = wrapper.get_definition()
        if defn.name in self._tools:
            raise ToolError(f"Tool '{defn.name}' is already registered")
        self._tools[defn.name] = wrapper

    def unregister(self, name: str) -> None:
        """Remove a tool by name."""
        if name not in self._tools:
            raise ToolError(f"Tool '{name}' not found")
        del self._tools[name]

    def get(self, name: str) -> ToolWrapper:
        """Retrieve a tool by name. Raises ToolError if not found."""
        wrapper = self._tools.get(name)
        if wrapper is None:
            raise ToolError(f"Tool '{name}' not found")
        return wrapper

    def list_definitions(self) -> list[ToolDefinition]:
        """Return definitions of all registered tools."""
        return [w.get_definition() for w in self._tools.values()]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
