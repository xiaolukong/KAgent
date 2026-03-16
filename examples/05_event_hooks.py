"""Example 05: Event hooks for observability.

This example demonstrates:
- Subscribing to agent lifecycle events
- Subscribing to LLM events for monitoring
- Subscribing to tool events for auditing
- Using glob patterns to match multiple event types
- Running event hooks across multiple model providers

Usage:
    # 1. Copy .env.example to .env and fill in your API key
    # 2. Run:
    python examples/05_event_hooks.py
"""

import asyncio
import time

from kagent import KAgent, configure
from kagent.domain.events import Event

MODELS = [
    "openai:gpt-5",
    "anthropic:claude-opus-4-5-20251101",
    "gemini:gemini-2.5-pro",
]


async def run_with_model(model: str) -> None:
    """Run an agent with event hooks using the given model."""
    try:
        agent = KAgent(
            model=model,
            system_prompt="You are helpful.",
        )

        start_time = time.time()

        @agent.on("agent.*")
        async def on_agent_event(event: Event):
            elapsed = time.time() - start_time
            print(f"    [{elapsed:.2f}s] AGENT: {event.event_type.value}")

        @agent.on("llm.request.sent")
        async def on_llm_request(event: Event):
            msg_count = event.payload.get("message_count", "?")
            print(f"    LLM request sent ({msg_count} messages)")

        @agent.on("llm.response.received")
        async def on_llm_response(event: Event):
            usage = event.payload.get("usage")
            if usage:
                print(f"    LLM response received (tokens: {usage.get('total_tokens', '?')})")

        @agent.on("tool.call.started")
        async def on_tool_start(event: Event):
            name = event.payload.get("tool_name", "?")
            print(f"    TOOL started: {name}")

        @agent.on("tool.call.completed")
        async def on_tool_done(event: Event):
            name = event.payload.get("tool_name", "?")
            ms = event.payload.get("duration_ms", 0)
            print(f"    TOOL completed: {name} ({ms:.1f}ms)")

        @agent.tool
        async def lookup(topic: str) -> str:
            """Look up information about a topic."""
            return f"Here are some facts about {topic}..."

        print(f"\n[{model}]")
        result = await agent.run("Look up information about Python programming.")
        print(f"  Final response: {result.content}")
    except Exception as e:
        print(f"\n[{model}] Error: {e}")


async def main():
    configure()  # reads KAGENT_API_KEY and KAGENT_BASE_URL from .env

    print("=== Event Hooks — Multi-Model ===")
    for model in MODELS:
        await run_with_model(model)


if __name__ == "__main__":
    asyncio.run(main())
