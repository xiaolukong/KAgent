"""Example 12: Tool Self-Correction & Circuit Breaker.

This example demonstrates:
- LLM passing wrong parameter types → framework returns friendly error → LLM self-corrects
- Circuit breaker: after 3 consecutive failures on the same tool, the framework stops retries
- Configuring max_tool_retries

Usage:
    export KAGENT_API_KEY=sk-xxx
    python examples/12_self_correction.py
"""

import asyncio

from kagent import KAgent, configure


async def main():
    configure(
        api_key="test-key-1",
        base_url="https://sap-ai-proxy-delightful-oribi-lx.cfapps.eu12.hana.ondemand.com/v1",
    )

    # ── Self-Correction Demo ────────────────────────────────────────────

    print("=== Tool Self-Correction Demo ===\n")

    agent = KAgent(
        model="openai:gpt-5",
        system_prompt=(
            "You are a helpful assistant. You have access to a lookup_user tool. "
            "When asked about a user, call the tool with their numeric ID."
        ),
        max_turns=5,
        max_tool_retries=3,  # default is 3
    )

    @agent.tool
    async def lookup_user(user_id: int) -> dict:
        """Look up a user by their numeric ID. user_id must be an integer."""
        users = {
            1: {"name": "Alice", "role": "admin"},
            2: {"name": "Bob", "role": "user"},
            3: {"name": "Charlie", "role": "viewer"},
        }
        if user_id not in users:
            return {"error": f"User {user_id} not found"}
        return users[user_id]

    # The LLM might initially try to pass user_id as a string like "one".
    # Before this feature, that would crash the agent.
    # Now, the error is returned to the LLM as a tool result, and it can self-correct.

    try:
        result = await agent.run("Tell me about user 1.")
        print(f"Response: {result.content}\n")
    except Exception as e:
        print(f"Error: {e}\n")

    # ── Circuit Breaker Demo ─────────────────────────────────────────────

    print("=== Circuit Breaker Demo ===\n")

    agent2 = KAgent(
        model="openai:gpt-5",
        system_prompt="You are a helpful assistant.",
        max_turns=10,
        max_tool_retries=2,  # Lower threshold for demo
    )

    @agent2.tool
    async def flaky_api(query: str) -> dict:
        """Call an external API that is currently down."""
        # Simulate a tool that always fails at runtime
        raise ConnectionError("Service unavailable: API is down for maintenance")

    print("Registering a tool that always fails (simulating a down API).")
    print("The circuit breaker will stop retries after 2 consecutive failures.\n")

    try:
        result = await agent2.run("Search for 'Python async patterns' using the flaky_api.")
        print(f"Response: {result.content}\n")
    except Exception as e:
        print(f"Error: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
