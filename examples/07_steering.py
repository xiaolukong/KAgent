"""Example 07: Steering — mid-turn redirect and abort via SAP AI Core.

This example demonstrates:
- Using agent.steer() to redirect a running agent to a new instruction
- Using agent.abort() to cancel a running agent
- Observing steering events via event hooks
- How the agent loop handles redirect vs abort differently

Steering is designed for concurrent use: the agent runs in the foreground
while another coroutine (or an event hook) calls steer() / abort() to
intervene.

- steer(directive): injects the directive as a new user message and the
  agent CONTINUES the loop, responding to the new instruction.
- abort(reason): sets the abort flag and the agent STOPS at the next
  turn boundary.

Credentials are read automatically from AICORE_* environment variables.

Usage:
    # 1. Copy .env.example to .env and fill in your AICORE_* credentials
    # 2. Run:
    python examples/07_steering.py
"""

import asyncio

from kagent import KAgent, configure
from kagent.domain.events import Event


async def demo_steer() -> None:
    """Demonstrate redirecting an agent mid-run with steer().

    The agent starts researching topic A using a tool.  After the first tool
    call completes, a concurrent coroutine fires agent.steer() with a new
    instruction.  The agent loop injects this as a user message and
    continues — the model now responds to the new instruction instead.
    """
    print("\n--- Demo 1: Redirect with steer() ---\n")

    agent = KAgent(
        model="openai:gpt-4o",
        system_prompt=(
            "You are a research assistant. "
            "Always call the research tool for each topic. "
            "After researching, provide a brief summary."
        ),
        max_turns=10,
    )

    topics_researched: list[str] = []

    @agent.tool
    async def research(topic: str) -> str:
        """Research a topic and return findings."""
        topics_researched.append(topic)
        print(f"  [tool] research('{topic}')")
        await asyncio.sleep(0.3)
        return f"Key finding: {topic} is a rapidly evolving field with major breakthroughs in 2025."

    @agent.on("steering.redirect")
    async def on_redirect(event: Event):
        directive = event.payload.get("directive", "")
        print(f'  [steering event] redirect: "{directive}"')

    @agent.on("agent.loop.iteration")
    async def on_turn(event: Event):
        turn = event.payload.get("turn_number", "?")
        print(f"  [loop] turn {turn}")

    async def redirect_after_first_tool():
        while len(topics_researched) < 1:
            await asyncio.sleep(0.1)
        print("  >>> Redirecting agent to a new topic!")
        await agent.steer(
            "Actually, stop what you're doing. "
            "Research 'artificial intelligence' instead and summarize that."
        )

    agent_task = asyncio.create_task(agent.run("Research 'quantum computing' and 'black holes'."))
    steer_task = asyncio.create_task(redirect_after_first_tool())

    result, _ = await asyncio.gather(agent_task, steer_task, return_exceptions=True)

    if isinstance(result, Exception):
        print(f"  Agent ended with exception: {result}")
    else:
        print(f"\n  Final response: {result.content}")
    print(f"  Topics researched: {topics_researched}")
    print("  (Notice the agent switched to the redirected topic!)")


async def demo_abort() -> None:
    """Demonstrate aborting an agent mid-run.

    The agent is given a task that requires multiple tool calls.  After the
    first tool call completes, a concurrent coroutine fires agent.abort().
    The agent loop detects the abort flag at the next turn boundary and
    stops early.
    """
    print("\n\n--- Demo 2: Abort mid-run ---\n")

    agent = KAgent(
        model="openai:gpt-4o",
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
        await asyncio.sleep(0.5)
        return f"Findings about {topic}: it is very interesting."

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

    async def abort_after_first_tool():
        while call_count < 1:
            await asyncio.sleep(0.1)
        print("  >>> Sending abort signal!")
        await agent.abort("User decided to stop early")

    agent_task = asyncio.create_task(
        agent.run("Research these three topics: quantum computing, black holes, and DNA.")
    )
    abort_task = asyncio.create_task(abort_after_first_tool())

    result, _ = await asyncio.gather(agent_task, abort_task, return_exceptions=True)

    if isinstance(result, Exception):
        print(f"  Agent ended with exception: {result}")
    else:
        content = result.content or "(no text — aborted before model finished)"
        print(f"\n  Agent response: {content}")
    print(f"  Tool was called {call_count} time(s) before abort (3 were requested).")


async def demo_abort_streaming() -> None:
    """Demonstrate aborting a streaming agent run."""
    print("\n\n--- Demo 3: Abort during streaming ---\n")

    agent = KAgent(
        model="openai:gpt-4o",
        system_prompt="You are an assistant. When asked to research, call the research tool.",
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
        async for chunk in agent.stream("Research: neural networks, relativity."):
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
        truncated = streamed[:200] + ("..." if len(streamed) > 200 else "")
        print(f"\n  Streamed text (partial): {truncated}")
    else:
        print("\n  No text streamed (aborted before model produced final text).")
    print(f"  Tool was called {call_count} time(s) before abort (2 were requested).")


async def main():
    configure()  # backend="aicore" by default; reads AICORE_* from .env

    print("=== Steering Example ===")
    await demo_steer()
    await demo_abort()
    await demo_abort_streaming()

    print("\n\n=== All steering demos complete ===")


if __name__ == "__main__":
    asyncio.run(main())
