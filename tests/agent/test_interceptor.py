"""Tests for InterceptorPipeline and its integration with AgentLoop."""

import pytest

from kagent.agent.interceptor import InterceptBlockedError, InterceptorPipeline, InterceptResult
from kagent.agent.loop import AgentLoop
from kagent.agent.prompt_builder import PromptBuilder
from kagent.agent.steering import SteeringController
from kagent.context.manager import ContextManager
from kagent.domain.entities import ToolCall, ToolResult
from kagent.domain.model_types import ModelRequest, ModelResponse
from kagent.events.bus import EventBus
from kagent.tools.decorator import tool
from kagent.tools.executor import ToolExecutor
from kagent.tools.registry import ToolRegistry
from tests.conftest import MockModelProvider

# ── Pipeline unit tests ──────────────────────────────────────────────────


class TestPipelineBasics:
    @pytest.mark.asyncio
    async def test_empty_pipeline_passes_data_through(self):
        pipeline = InterceptorPipeline()
        data = {"key": "value"}
        result = await pipeline.run("before_llm_request", data)
        assert result == data

    @pytest.mark.asyncio
    async def test_single_handler_modifies_data(self):
        pipeline = InterceptorPipeline()

        async def add_field(data):
            data["added"] = True
            return data

        pipeline.add("before_llm_request", add_field)
        result = await pipeline.run("before_llm_request", {"original": True})
        assert result == {"original": True, "added": True}

    @pytest.mark.asyncio
    async def test_chain_threads_data(self):
        """Each handler receives the output of the previous one."""
        pipeline = InterceptorPipeline()

        async def step1(x):
            return x + 1

        async def step2(x):
            return x * 10

        pipeline.add("before_return", step1)
        pipeline.add("before_return", step2)
        # Both at priority 0 → insertion order: step1 then step2
        result = await pipeline.run("before_return", 5)
        assert result == (5 + 1) * 10

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        """Higher priority handlers run first."""
        pipeline = InterceptorPipeline()
        order: list[str] = []

        async def low(data):
            order.append("low")
            return data

        async def high(data):
            order.append("high")
            return data

        pipeline.add("before_return", low, priority=0)
        pipeline.add("before_return", high, priority=10)
        await pipeline.run("before_return", None)
        assert order == ["high", "low"]


class TestPipelineBlocking:
    @pytest.mark.asyncio
    async def test_intercept_result_blocked_raises(self):
        pipeline = InterceptorPipeline()

        async def blocker(data):
            return InterceptResult(data=data, blocked=True, reason="Nope")

        pipeline.add("before_tool_call", blocker)
        with pytest.raises(InterceptBlockedError, match="Nope"):
            await pipeline.run("before_tool_call", {"tool_name": "rm"})

    @pytest.mark.asyncio
    async def test_intercept_result_not_blocked_passes_data(self):
        pipeline = InterceptorPipeline()

        async def passer(data):
            return InterceptResult(data={"modified": True}, blocked=False)

        pipeline.add("before_tool_call", passer)
        result = await pipeline.run("before_tool_call", {})
        assert result == {"modified": True}


class TestPipelineRegistration:
    def test_invalid_hook_raises(self):
        pipeline = InterceptorPipeline()
        with pytest.raises(ValueError, match="Unknown hook"):

            async def noop(d):
                return d

            pipeline.add("nonexistent_hook", noop)

    def test_remove_handler(self):
        pipeline = InterceptorPipeline()

        async def noop(d):
            return d

        rid = pipeline.add("before_return", noop)
        assert pipeline.has_handlers("before_return")
        pipeline.remove(rid)
        assert not pipeline.has_handlers("before_return")

    def test_has_handlers(self):
        pipeline = InterceptorPipeline()
        assert not pipeline.has_handlers("before_return")

        async def noop(d):
            return d

        pipeline.add("before_return", noop)
        assert pipeline.has_handlers("before_return")


class TestPipelineErrorPropagation:
    @pytest.mark.asyncio
    async def test_handler_exception_propagates(self):
        pipeline = InterceptorPipeline()

        async def explode(data):
            raise RuntimeError("Boom")

        pipeline.add("before_llm_request", explode)
        with pytest.raises(RuntimeError, match="Boom"):
            await pipeline.run("before_llm_request", {})


# ── Integration tests with AgentLoop ─────────────────────────────────────


def _make_loop_with_pipeline(
    provider: MockModelProvider,
    pipeline: InterceptorPipeline,
    max_turns: int = 5,
    registry: ToolRegistry | None = None,
) -> tuple[AgentLoop, EventBus]:
    bus = EventBus()
    reg = registry or ToolRegistry()
    executor = ToolExecutor(reg, bus)
    context = ContextManager()
    prompt = PromptBuilder(system_prompt="Test system.")
    steering = SteeringController(bus)
    loop = AgentLoop(
        model_provider=provider,
        event_bus=bus,
        tool_registry=reg,
        tool_executor=executor,
        context_manager=context,
        prompt_builder=prompt,
        steering=steering,
        pipeline=pipeline,
        max_turns=max_turns,
    )
    return loop, bus


class TestLoopBeforeLlmRequest:
    @pytest.mark.asyncio
    async def test_interceptor_modifies_request(self):
        """Verify before_llm_request can modify the ModelRequest."""
        pipeline = InterceptorPipeline()
        captured_temps: list[float | None] = []

        async def set_temp(request: ModelRequest) -> ModelRequest:
            captured_temps.append(request.temperature)
            request.temperature = 0.0
            return request

        pipeline.add("before_llm_request", set_temp)

        provider = MockModelProvider(response_content="OK")
        loop, bus = _make_loop_with_pipeline(provider, pipeline)
        await loop.run("Hello")
        assert len(captured_temps) == 1


class TestLoopAfterLlmResponse:
    @pytest.mark.asyncio
    async def test_interceptor_modifies_response(self):
        """Verify after_llm_response can rewrite the model content."""
        pipeline = InterceptorPipeline()

        async def censor(response: ModelResponse) -> ModelResponse:
            if response.content:
                response.content = response.content.replace("bad", "***")
            return response

        pipeline.add("after_llm_response", censor)

        provider = MockModelProvider(response_content="This is bad content")
        loop, _ = _make_loop_with_pipeline(provider, pipeline)
        result = await loop.run("Say something")
        assert result.content == "This is *** content"


class TestLoopBeforeToolCall:
    @pytest.mark.asyncio
    async def test_block_tool_call(self):
        """Verify before_tool_call can block a tool from executing."""
        pipeline = InterceptorPipeline()
        executed: list[str] = []

        async def block_dangerous(ctx):
            if ctx["tool_name"] == "dangerous":
                return InterceptResult(data=ctx, blocked=True, reason="Blocked")
            return ctx

        pipeline.add("before_tool_call", block_dangerous)

        registry = ToolRegistry()

        @tool
        async def dangerous() -> str:
            """A dangerous tool."""
            executed.append("dangerous")
            return "done"

        registry.register(dangerous)

        tc = ToolCall(id="c1", name="dangerous", arguments={})
        provider = MockModelProvider(response_content="OK", tool_calls=[tc])
        loop, _ = _make_loop_with_pipeline(provider, pipeline, registry=registry)

        result = await loop.run("Do something dangerous")
        # Tool should NOT have been executed
        assert executed == []
        # But the loop should still complete (blocked tool result is added to context)
        assert result.content == "OK"

    @pytest.mark.asyncio
    async def test_modify_tool_arguments(self):
        """Verify before_tool_call can modify arguments."""
        pipeline = InterceptorPipeline()

        async def override_x(ctx):
            if ctx["tool_name"] == "double":
                ctx["arguments"] = {"x": 100}
            return ctx

        pipeline.add("before_tool_call", override_x)

        registry = ToolRegistry()
        results: list[int] = []

        @tool
        async def double(x: int) -> int:
            """Double."""
            results.append(x * 2)
            return x * 2

        registry.register(double)

        tc = ToolCall(id="c1", name="double", arguments={"x": 5})
        provider = MockModelProvider(response_content="200", tool_calls=[tc])
        loop, _ = _make_loop_with_pipeline(provider, pipeline, registry=registry)

        await loop.run("Double 5")
        # Interceptor changed x from 5 to 100
        assert results == [200]


class TestLoopAfterToolCall:
    @pytest.mark.asyncio
    async def test_modify_tool_result(self):
        """Verify after_tool_call can modify the ToolResult."""
        pipeline = InterceptorPipeline()

        async def redact(result: ToolResult) -> ToolResult:
            if result.result and "secret" in str(result.result):
                result.result = "[REDACTED]"
            return result

        pipeline.add("after_tool_call", redact)

        registry = ToolRegistry()

        @tool
        async def get_secret() -> str:
            """Return a secret."""
            return "secret-password-123"

        registry.register(get_secret)

        tc = ToolCall(id="c1", name="get_secret", arguments={})
        provider = MockModelProvider(response_content="OK", tool_calls=[tc])
        loop, _ = _make_loop_with_pipeline(provider, pipeline, registry=registry)

        await loop.run("Get the secret")
        # The interceptor should have redacted the result before it was added to context


class TestLoopBeforeReturn:
    @pytest.mark.asyncio
    async def test_modify_final_response(self):
        """Verify before_return can modify the final ModelResponse."""
        pipeline = InterceptorPipeline()

        async def add_metadata(response: ModelResponse) -> ModelResponse:
            response.metadata["intercepted"] = True
            return response

        pipeline.add("before_return", add_metadata)

        provider = MockModelProvider(response_content="Done")
        loop, _ = _make_loop_with_pipeline(provider, pipeline)
        result = await loop.run("Hello")
        assert result.metadata.get("intercepted") is True


class TestLoopBeforePromptBuild:
    @pytest.mark.asyncio
    async def test_filter_tools(self):
        """Verify before_prompt_build can filter tool definitions."""
        pipeline = InterceptorPipeline()

        async def remove_all_tools(ctx):
            ctx["tool_definitions"] = []
            return ctx

        pipeline.add("before_prompt_build", remove_all_tools)

        registry = ToolRegistry()

        @tool
        async def some_tool() -> str:
            """A tool."""
            return "result"

        registry.register(some_tool)

        provider = MockModelProvider(response_content="No tools available")
        loop, _ = _make_loop_with_pipeline(provider, pipeline, registry=registry)
        result = await loop.run("Use a tool")
        # Even though tool is registered, interceptor removed it from the request
        assert result.content == "No tools available"
