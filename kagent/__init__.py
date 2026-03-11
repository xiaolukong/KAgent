"""KAgent — A clean-architecture Python agent framework with pub/sub scheduling."""

from kagent._version import __version__
from kagent.common.config import configure, get_config
from kagent.interface.builder import KAgentBuilder
from kagent.interface.kagent import KAgent
from kagent.tools.decorator import tool

__all__ = [
    "KAgent",
    "KAgentBuilder",
    "__version__",
    "configure",
    "get_config",
    "tool",
]
