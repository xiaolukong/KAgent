"""Example 09: Interceptors — mutable hooks for the agent loop via SAP AI Core.

This example demonstrates:
- Using @agent.intercept() to modify data flowing through the agent loop
- Filtering tools dynamically with before_prompt_build
- Logging and modifying LLM requests/responses
- Blocking dangerous tool calls with before_tool_call
- Post-processing the final response with before_return

Unlike @agent.on() (read-only Pub/Sub), interceptors receive data,
can modify it, and return it to the next handler in the chain.

Credentials are read automatically from AICORE_* environment variables.

Usage:
    # 1. Copy .env.example to .env and fill in your AICORE_* credentials
    # 2. Run:
    python examples/09_interceptors.py
"""

import asyncio

from kagent import KAgent, configure
from kagent.agent.interceptor import InterceptResult


async def demo_tool_filter() -> None:
    """Demo 1: Dynamically filter tools based on a "safe mode" flag."""
    print("--- Demo 1: Dynamic tool filtering ---\n")

    agent = KAgent(
        model="openai:gpt-4o",
        system_prompt="You are a helpful assistant with access to tools.",
        max_turns=5,
    )

    safe_mode = True

    @agent.tool
    async def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"Weather in {city}: 22°C, sunny"

    @agent.tool
    async def delete_file(path: str) -> str:
        """Delete a file from the filesystem."""
        return f"Deleted {path}"

    @agent.intercept("before_prompt_build")
    async def filter_tools_in_safe_mode(ctx):
        if safe_mode:
            allowed = {"get_weather"}
            original = [t.name for t in ctx["tool_definitions"]]
            ctx["tool_definitions"] = [t for t in ctx["tool_definitions"] if t.name in allowed]
            filtered = [t.name for t in ctx["tool_definitions"]]
            print(f"  [interceptor] Safe mode ON: {original} -> {filtered}")
        return ctx

    result = await agent.run("What's the weather in Berlin?")
    print(f"  Result: {result.content}\n")


async def demo_request_logging() -> None:
    """Demo 2: Log and modify LLM requests."""
    print("\n--- Demo 2: Request logging & modification ---\n")

    agent = KAgent(
        model="openai:gpt-4o",
        system_prompt="You are helpful.",
        temperature=0.7,
        max_turns=3,
    )

    @agent.intercept("before_llm_request")
    async def log_and_fix_temp(request):
        print(f"  [interceptor] Model: {request.model}, messages: {len(request.messages)}")
        print(f"  [interceptor] Original temp: {request.temperature} -> forcing 0.0")
        request.temperature = 0.0
        return request

    @agent.intercept("after_llm_response")
    async def log_response(response):
        tokens = response.usage.total_tokens if response.usage else "?"
        print(f"  [interceptor] Response tokens: {tokens}, has_tools: {response.has_tool_calls}")
        return response

    result = await agent.run("Say hello in 3 languages.")
    print(f"  Result: {result.content}\n")


async def demo_tool_blocking() -> None:
    """Demo 3: Block dangerous tool calls."""
    print("\n--- Demo 3: Tool call blocking ---\n")

    agent = KAgent(
        model="openai:gpt-4o",
        system_prompt=(
            "You are a file assistant. Use the tools to help the user. "
            "If a tool is blocked, explain that to the user."
        ),
        max_turns=5,
    )

    @agent.tool
    async def read_file(path: str) -> str:
        """Read a file's contents."""
        print(f"  [tool] read_file('{path}')")
        return f"Contents of {path}: Hello World"

    @agent.tool
    async def delete_file(path: str) -> str:
        """Delete a file."""
        print(f"  [tool] delete_file('{path}') — THIS SHOULD NOT EXECUTE")
        return f"Deleted {path}"

    blocked_tools = {"delete_file"}

    @agent.intercept("before_tool_call")
    async def block_dangerous_tools(ctx):
        if ctx["tool_name"] in blocked_tools:
            print(f"  [interceptor] BLOCKED: {ctx['tool_name']}")
            return InterceptResult(
                data=ctx,
                blocked=True,
                reason=f"Tool '{ctx['tool_name']}' is not allowed in this environment.",
            )
        print(f"  [interceptor] ALLOWED: {ctx['tool_name']}")
        return ctx

    result = await agent.run("Read the file at /tmp/hello.txt, then delete it.")
    print(f"  Result: {result.content}\n")


async def demo_response_postprocessing() -> None:
    """Demo 4: Post-process the final response."""
    print("\n--- Demo 4: Response post-processing ---\n")

    agent = KAgent(
        model="openai:gpt-4o",
        system_prompt="You are a database assistant. Use tools to query data.",
        max_turns=5,
    )

    @agent.tool
    async def query_db(sql: str) -> str:
        """Execute a SQL query and return results."""
        print(f"  [tool] query_db: {sql}")
        return "user: alice, email: alice@secret.com, ssn: 123-45-6789"

    @agent.intercept("after_tool_call")
    async def redact_pii(result):
        if result.result and "ssn" in str(result.result):
            original = str(result.result)
            result.result = original.replace("123-45-6789", "***-**-****")
            print("  [interceptor] PII redacted from tool result")
        return result

    @agent.intercept("before_return")
    async def add_audit_metadata(response):
        response.metadata["audited"] = True
        response.metadata["pii_check"] = "passed"
        print(f"  [interceptor] Audit metadata added: {response.metadata}")
        return response

    result = await agent.run("Look up alice's information in the database.")
    print(f"  Result: {result.content}")
    print(f"  Metadata: {result.metadata}\n")


async def main():
    configure()  # backend="aicore" by default; reads AICORE_* from .env

    print("=== Interceptor Examples ===\n")
    await demo_tool_filter()
    await demo_request_logging()
    await demo_tool_blocking()
    await demo_response_postprocessing()
    print("\n=== All interceptor demos complete ===")


if __name__ == "__main__":
    asyncio.run(main())
