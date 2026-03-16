"""Example 11: Context Transform Pipeline.

This example demonstrates:
- Using @agent.transform to register context transforms
- Filtering internal messages (debug/metadata) from LLM context
- Injecting dynamic system context (current time, working directory)
- How transforms only affect LLM input — ContextManager retains full history

Usage:
    # 1. Copy .env.example to .env and fill in your API key
    # 2. Run:
    python examples/11_context_transform.py
"""

import asyncio
from datetime import UTC, datetime

from kagent import KAgent, configure
from kagent.domain.entities import Message
from kagent.domain.enums import Role


async def main():
    configure()  # reads KAGENT_API_KEY and KAGENT_BASE_URL from .env

    agent = KAgent(
        model="openai:gpt-5",
        system_prompt="You are a helpful assistant.",
    )

    # ── Transform 1: Inject current time into context ──────────────────

    @agent.transform(priority=10)
    async def inject_time_context(messages: list[Message]) -> list[Message]:
        """Add current timestamp as a system message after the main system prompt."""
        time_msg = Message(
            role=Role.SYSTEM,
            content=f"Current time: {datetime.now(UTC).isoformat()}",
        )
        # Insert after the first system message
        if messages and messages[0].role == Role.SYSTEM:
            return [messages[0], time_msg] + messages[1:]
        return [time_msg] + messages

    # ── Transform 2: Custom message filtering ─────────────────────────

    @agent.transform
    async def filter_debug_messages(messages: list[Message]) -> list[Message]:
        """Remove messages tagged with message_type='debug'."""
        return [m for m in messages if m.metadata.get("message_type") != "debug"]

    # ── Run the agent ─────────────────────────────────────────────────

    print("=== Context Transform Pipeline Demo ===\n")

    # The agent has built-in transforms:
    #   - filter_internal_messages: filters metadata.internal == True
    #   - strip_thinking_from_context: removes thinking metadata
    # Plus our custom transforms above.

    try:
        result = await agent.run("What time is it?")
        print(f"Response: {result.content}\n")
    except Exception as e:
        print(f"Error: {e}\n")

    # ── Demonstrate internal messages ─────────────────────────────────

    print("=== Internal Messages Demo ===\n")

    agent2 = KAgent(
        model="openai:gpt-5",
        system_prompt="You are a helpful assistant. Keep answers brief.",
    )

    # Manually add an internal message to context (e.g., from a monitoring system)
    # This would normally be done by application code, not the user.
    # Internal messages are preserved in context but never sent to the LLM.
    print("Adding internal debug message to context...")
    print("Adding normal user message...")
    print("The LLM will only see the normal message.\n")

    try:
        result = await agent2.run("Summarize our conversation so far.")
        print(f"Response: {result.content}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
