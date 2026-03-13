"""InterceptorPipeline — mutable hook system for the agent loop.

Unlike the EventBus (fire-and-forget Pub/Sub), the interceptor pipeline is
a sequential chain where each handler **receives data, may modify it, and
returns it** to the next handler.  This enables pre/post processing of LLM
requests, tool calls, and prompt construction.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from kagent.common.errors import KAgentError
from kagent.common.logging import get_logger

logger = get_logger("agent.interceptor")

T = TypeVar("T")


class InterceptBlockedError(KAgentError):
    """Raised when an interceptor blocks execution (e.g. before_tool_call)."""

    def __init__(self, reason: str = "Blocked by interceptor") -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass
class InterceptResult(Generic[T]):
    """Wrapper for interceptor return values that need blocking semantics.

    Normal interceptors simply return the (possibly modified) data.  Use
    ``InterceptResult`` only when you need to **block** execution::

        @agent.intercept("before_tool_call")
        async def block_rm(ctx):
            if ctx["tool_name"] == "rm_rf":
                return InterceptResult(data=ctx, blocked=True, reason="Dangerous")
            return ctx
    """

    data: T
    blocked: bool = False
    reason: str | None = None


# The interceptor callable: takes data, returns data or InterceptResult.
InterceptorFn = Callable[[Any], Awaitable[Any]]


@dataclass
class _Registration:
    """Internal record for a registered interceptor."""

    id: str
    hook: str
    fn: InterceptorFn
    priority: int


class InterceptorPipeline:
    """Priority-ordered handler chains for each hook point.

    Handlers registered with higher *priority* values run first.  Within
    the same priority, execution order is insertion order.
    """

    # All valid hook names.
    HOOKS: frozenset[str] = frozenset(
        {
            "before_prompt_build",
            "before_llm_request",
            "after_llm_response",
            "before_tool_call",
            "after_tool_call",
            "after_tool_round",
            "before_return",
        }
    )

    def __init__(self) -> None:
        self._registrations: dict[str, list[_Registration]] = {h: [] for h in self.HOOKS}

    def add(
        self,
        hook: str,
        fn: InterceptorFn,
        *,
        priority: int = 0,
    ) -> str:
        """Register an interceptor on *hook*.  Returns a unique ID for removal."""
        if hook not in self.HOOKS:
            raise ValueError(f"Unknown hook '{hook}'. Valid hooks: {sorted(self.HOOKS)}")
        reg_id = uuid.uuid4().hex[:12]
        reg = _Registration(id=reg_id, hook=hook, fn=fn, priority=priority)
        self._registrations[hook].append(reg)
        # Keep sorted by descending priority (stable within same priority).
        self._registrations[hook].sort(key=lambda r: r.priority, reverse=True)
        logger.debug("Interceptor %s registered on '%s' (priority=%d)", reg_id, hook, priority)
        return reg_id

    def remove(self, interceptor_id: str) -> None:
        """Remove a previously registered interceptor by its ID."""
        for hook, regs in self._registrations.items():
            for i, reg in enumerate(regs):
                if reg.id == interceptor_id:
                    regs.pop(i)
                    logger.debug("Interceptor %s removed from '%s'", interceptor_id, hook)
                    return

    async def run(self, hook: str, data: T) -> T:
        """Execute all handlers for *hook*, threading *data* through the chain.

        Raises ``InterceptBlockedError`` if any handler returns a blocked result.
        """
        regs = self._registrations.get(hook)
        if not regs:
            return data

        for reg in regs:
            try:
                result = await reg.fn(data)
            except InterceptBlockedError:
                raise
            except Exception:
                logger.exception("Interceptor %s on '%s' raised an exception", reg.id, hook)
                raise

            if isinstance(result, InterceptResult):
                if result.blocked:
                    raise InterceptBlockedError(result.reason or "Blocked by interceptor")
                data = result.data
            else:
                data = result

        return data

    def has_handlers(self, hook: str) -> bool:
        """Check whether *hook* has any registered handlers."""
        return bool(self._registrations.get(hook))
