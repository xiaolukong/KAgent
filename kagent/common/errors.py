"""Unified exception hierarchy for KAgent."""


class KAgentError(Exception):
    """Base exception for all KAgent errors."""


class ModelError(KAgentError):
    """Raised when a model provider call fails."""


class ToolError(KAgentError):
    """Raised when a tool execution fails."""


class ValidationError(KAgentError):
    """Raised when schema or input validation fails."""


class ConfigError(KAgentError):
    """Raised when configuration is invalid or missing."""


class TimeoutError(KAgentError):
    """Raised when an operation exceeds its timeout."""
