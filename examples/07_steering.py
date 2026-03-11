"""Example 07: Steering — mid-turn abort and redirect.

This example demonstrates:
- Using agent.abort() to cancel a running agent from a concurrent coroutine
- Observing steering events via event hooks
- How the agent loop checks for abort at the start of each turn

Steering is designed for concurrent use: the agent runs in the foreground
while another coroutine (or an event hook) calls steer() / abort() to
intervene.  The agent loop checks for pending directives at the start of
each turn and terminates gracefully when an abort is detected.

Usage:
    export KAGENT_API_KEY=sk-xxx
    python examples/07_steering.py
"""

import asyncio

from kagent import KAgent, configure
from kagent.domain.events import Event


async def demo_abort() -> None:
    """Demonstrate aborting an agent mid-run.

    The agent is given a task that requires multiple tool calls.  After the
    first tool call completes, a concurrent coroutine fires agent.abort().
    The agent loop detects the abort flag at the next turn boundary and
    stops early.
    """
    print("\n--- Demo 1: Abort mid-run ---\n")

    agent = KAgent(
        model="openai:gpt-5",
        system_prompt=(
            "You are a research assistant. "
            "Always call the research tool for EACH topic separately, one at a time."
        ),
        max_turns=10,
    )

    call_count = 0

    @agent.tool
    async def research(topic: str) -> str:
        """Research a topic and return findings."""
        nonlocal call_count
        call_count += 1
        print(f"  [tool] research('{topic}') — call #{call_count}")
        # Simulate a slow API call
        await asyncio.sleep(0.5)
        return f"Findings about {topic}: it is very interesting."

    # Track steering events
    @agent.on("steering.*")
    async def on_steering(event: Event):
        print(f"  [steering event] {event.event_type.value}: {event.payload}")

    @agent.on("agent.loop.iteration")
    async def on_turn(event: Event):
        turn = event.payload.get("turn_number", "?")
        print(f"  [loop] turn {turn} starting")

    @agent.on("agent.loop.completed")
    async def on_done(event: Event):
        print("  [loop] completed")

    # Run agent in background; abort after the first tool call finishes
    async def abort_after_first_tool():
        # Wait until at least one tool call has completed
        while call_count < 1:
            await asyncio.sleep(0.1)
        print("  >>> Sending abort signal!")
        await agent.abort("User decided to stop early")

    # Launch both concurrently
    agent_task = asyncio.create_task(
        agent.run("Research these three topics: quantum computing, black holes, and DNA.")
    )
    abort_task = asyncio.create_task(abort_after_first_tool())

    # Wait for both
    result, _ = await asyncio.gather(agent_task, abort_task, return_exceptions=True)

    if isinstance(result, Exception):
        print(f"  Agent ended with exception: {result}")
    else:
        content = result.content or "(no text — aborted before model finished)"
        print(f"\n  Agent response: {content}")
    print(f"  Tool was called {call_count} time(s) before abort (3 were requested).")


async def demo_abort_streaming() -> None:
    """Demonstrate aborting a streaming agent run.

    Same concept, but using agent.stream() instead of agent.run().
    """
    print("\n\n--- Demo 2: Abort during streaming ---\n")

    agent = KAgent(
        model="openai:gpt-5",
        system_prompt=(
            "You are an assistant. "
            "When asked to research, call the research tool for each topic."
        ),
        max_turns=10,
    )

    call_count = 0

    @agent.tool
    async def research(topic: str) -> str:
        """Research a topic."""
        nonlocal call_count
        call_count += 1
        print(f"  [tool] research('{topic}') — call #{call_count}")
        await asyncio.sleep(0.3)
        return f"Results for {topic}: fascinating stuff."

    @agent.on("steering.abort")
    async def on_abort(event: Event):
        reason = event.payload.get("reason", "unknown")
        print(f"  [steering event] abort received: {reason}")

    collected_text: list[str] = []

    async def run_stream():
        async for chunk in agent.stream(
            "Research: neural networks, relativity."
        ):
            if chunk.content:
                collected_text.append(chunk.content)

    async def abort_after_delay():
        while call_count < 1:
            await asyncio.sleep(0.1)
        print("  >>> Sending abort during stream!")
        await agent.abort("Enough research for now")

    stream_task = asyncio.create_task(run_stream())
    abort_task = asyncio.create_task(abort_after_delay())

    await asyncio.gather(stream_task, abort_task, return_exceptions=True)

    streamed = "".join(collected_text)
    if streamed:
        print(f"\n  Streamed text (partial): {streamed[:200]}{'...' if len(streamed) > 200 else ''}")
    else:
        print("\n  No text streamed (aborted before model produced final text).")
    print(f"  Tool was called {call_count} time(s) before abort (2 were requested).")


async def main():
    configure(
        api_key="test-key-1",
        base_url="https://sap-ai-proxy-delightful-oribi-lx.cfapps.eu12.hana.ondemand.com/v1",
    )

    print("=== Steering Example ===")
    await demo_abort()
    await demo_abort_streaming()

    print("\n\n=== All steering demos complete ===")


if __name__ == "__main__":
    asyncio.run(main())
