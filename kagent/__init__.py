"""KAgent — A clean-architecture async Python agent framework.

Quick start::

    from kagent import KAgent, configure, Message, Role, StreamChunk, StreamChunkType

    configure(api_key="sk-...")
    agent = KAgent(model="openai:gpt-4o")
    result = await agent.run("Hello!")

See CLAUDE.md for the full AI coding-agent cheatsheet,
or docs/AI_MANIFEST.json for intent-to-code mapping.
"""

from kagent._version import __version__
from kagent.common.config import configure, get_config
from kagent.domain.entities import Message, ToolCall, ToolResult
from kagent.domain.enums import EventType, Role, StreamChunkType
from kagent.domain.model_types import ModelResponse, StreamChunk
from kagent.interface.builder import KAgentBuilder
from kagent.interface.kagent import KAgent
from kagent.tools.decorator import tool

__all__ = [
    # ── Core ──
    "KAgent",
    "KAgentBuilder",
    "configure",
    "get_config",
    "tool",
    "__version__",
    # ── Domain Types (frequently needed by user code) ──
    "Message",
    "Role",
    "ModelResponse",
    "StreamChunk",
    "StreamChunkType",
    "EventType",
    "ToolCall",
    "ToolResult",
]
