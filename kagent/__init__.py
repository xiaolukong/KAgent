"""KAgent — A clean-architecture async Python agent framework.

Quick start::

    from kagent import KAgent, configure, Message, Role, StreamChunk, StreamChunkType

    configure()  # reads KAGENT_API_KEY from .env / environment
    agent = KAgent(model="openai:gpt-4o")
    result = await agent.run("Hello!")

AI coding agents: for the full usage cheatsheet, read ``kagent/_ai/CLAUDE.md``.
Locate it programmatically::

    import kagent
    guide_dir = kagent.get_ai_guide_path()
    # guide_dir / "CLAUDE.md"         — coding guidelines
    # guide_dir / "AI_MANIFEST.json"  — intent-to-code mapping
    # guide_dir / "examples/"         — runnable examples
"""

from pathlib import Path

from kagent._version import __version__
from kagent.common.config import configure, get_config
from kagent.domain.entities import Message, ToolCall, ToolResult
from kagent.domain.enums import EventType, Role, StreamChunkType
from kagent.domain.model_types import ModelResponse, StreamChunk
from kagent.interface.builder import KAgentBuilder
from kagent.interface.kagent import KAgent
from kagent.tools.decorator import tool


def get_ai_guide_path() -> Path:
    """Return the filesystem path to the bundled AI guidance directory.

    The directory contains:
    - ``CLAUDE.md`` — AI coding agent guidelines and API cheatsheet
    - ``AI_MANIFEST.json`` — intent-to-code mapping for 12 examples
    - ``examples/`` — runnable example scripts

    Returns:
        Path to the ``kagent/_ai/`` directory inside the installed package.

    Raises:
        FileNotFoundError: If the bundled files are not present (e.g. in an
            editable install). In that case, read the root-level ``CLAUDE.md``
            directly from the repository.

    Example::

        import kagent
        guide_dir = kagent.get_ai_guide_path()
        print((guide_dir / "CLAUDE.md").read_text())
    """
    import importlib.resources

    ai_dir = importlib.resources.files("kagent._ai")
    ai_path = Path(str(ai_dir))
    if (ai_path / "CLAUDE.md").is_file():
        return ai_path
    raise FileNotFoundError(
        "kagent._ai bundle not found. "
        "If you are in a development (editable) install, read the root-level "
        "CLAUDE.md directly. Otherwise reinstall with: pip install kagent"
    )


__all__ = [
    # ── Core ──
    "KAgent",
    "KAgentBuilder",
    "configure",
    "get_config",
    "tool",
    "get_ai_guide_path",
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
