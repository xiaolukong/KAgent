"""Example 08: Interrupt — pause the agent loop and wait for user input.

This example demonstrates:
- Using agent.interrupt() to pause a running agent and send a prompt to the user
- Using agent.resume() to provide user input and continue the agent loop
- How the agent incorporates user feedback into its conversation context
- Observing interrupt/resume events via event hooks

Interrupt is designed for concurrent use: the agent runs in one asyncio task
while another task (driven by a steering.interrupt event hook) collects user
input via stdin and calls resume() to unblock the loop.

Usage:
    export KAGENT_API_KEY=sk-xxx
    python examples/08_interrupt.py
"""

import asyncio
import sys

from kagent import KAgent, configure
from kagent.domain.events import Event


async def async_input(prompt: str = "") -> str:
    """Read a line from stdin without blocking the event loop."""
    loop = asyncio.get_running_loop()
    if prompt:
        sys.stdout.write(prompt)
        sys.stdout.flush()
    return await loop.run_in_executor(None, sys.stdin.readline)


async def demo_interrupt_basic() -> None:
    """Demonstrate basic interrupt/resume flow with real user input.

    The agent starts researching multiple topics.  After the first tool
    call completes, an external coroutine fires agent.interrupt() to
    pause the agent.  The user is prompted in the terminal; once they
    type a response, agent.resume() is called and the loop continues.
    """
    print("\n--- Demo 1: Basic interrupt and resume ---\n")

    agent = KAgent(
        model="openai:gpt-5",
        system_prompt=(
            "You are a research assistant. "
            "Always call the research tool for each topic. "
            "After researching, provide a brief summary."
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
        await asyncio.sleep(0.3)
        return f"Key finding: {topic} is a rapidly evolving field with major breakthroughs in 2025."

    # When the agent is interrupted, prompt the user and resume
    @agent.on("steering.interrupt")
    async def on_interrupt(event: Event):
        prompt = event.payload.get("prompt", "")
        print(f"\n  ╔══ AGENT PAUSED ═══════════════════════════════════")
        print(f"  ║ {prompt}")
        print(f"  ╚══════════════════════════════════════════════════\n")
        user_reply = await async_input("  Your response > ")
        user_reply = user_reply.strip()
        print()
        await agent.resume(user_reply)

    @agent.on("agent.loop.iteration")
    async def on_turn(event: Event):
        turn = event.payload.get("turn_number", "?")
        print(f"  [loop] turn {turn}")

    # After the first tool call, interrupt and ask the user
    async def interrupt_after_first_tool():
        while call_count < 1:
            await asyncio.sleep(0.1)
        await agent.interrupt(
            "I've researched the first topic. "
            "Should I continue with the remaining topics, "
            "or would you like me to focus on something else?"
        )

    agent_task = asyncio.create_task(
        agent.run("Research 'quantum computing' and 'machine learning'.")
    )
    interrupt_task = asyncio.create_task(interrupt_after_first_tool())

    result, _ = await asyncio.gather(
        agent_task, interrupt_task, return_exceptions=True
    )

    if isinstance(result, Exception):
        print(f"  Agent ended with exception: {result}")
    else:
        print(f"\n  Final response: {result.content}")
    print(f"  Tool calls made: {call_count}")


async def demo_interrupt_confirmation() -> None:
    """Demonstrate using interrupt for user confirmation before a dangerous action.

    The agent is asked to clean up files.  After listing the files,
    the agent is interrupted so the user can choose which files to delete.
    """
    print("\n\n--- Demo 2: Interrupt for user confirmation ---\n")

    agent = KAgent(
        model="openai:gpt-5",
        system_prompt=(
            "You are a file management assistant. "
            "When asked to clean up files, first list them with list_files, "
            "then delete the ones the user tells you to delete with delete_file. "
            "Always call the appropriate tools."
        ),
        max_turns=10,
    )

    deleted_files: list[str] = []
    list_done = False

    @agent.tool
    async def list_files(directory: str) -> list[str]:
        """List files in a directory."""
        nonlocal list_done
        print(f"  [tool] list_files('{directory}')")
        list_done = True
        return ["report_draft.docx", "temp_data.csv", "important_notes.txt", "cache.tmp"]

    @agent.tool
    async def delete_file(filename: str) -> str:
        """Delete a file."""
        print(f"  [tool] delete_file('{filename}')")
        deleted_files.append(filename)
        return f"Deleted {filename}"

    @agent.on("steering.interrupt")
    async def on_interrupt(event: Event):
        prompt = event.payload.get("prompt", "")
        print(f"\n  ╔══ AGENT PAUSED ═══════════════════════════════════")
        print(f"  ║ {prompt}")
        print(f"  ╚══════════════════════════════════════════════════\n")
        user_reply = await async_input("  Your response > ")
        user_reply = user_reply.strip()
        print()
        await agent.resume(user_reply)

    # After listing files, interrupt to ask which ones to delete
    async def confirm_before_delete():
        while not list_done:
            await asyncio.sleep(0.1)
        await agent.interrupt(
            "I found these files: report_draft.docx, temp_data.csv, "
            "important_notes.txt, cache.tmp. Which ones should I delete?"
        )

    agent_task = asyncio.create_task(
        agent.run("Clean up the files in /tmp/workspace.")
    )
    confirm_task = asyncio.create_task(confirm_before_delete())

    result, _ = await asyncio.gather(
        agent_task, confirm_task, return_exceptions=True
    )

    if isinstance(result, Exception):
        print(f"  Agent ended with exception: {result}")
    else:
        print(f"\n  Final response: {result.content}")
    print(f"  Files deleted: {deleted_files}")


async def main():
    configure(
        api_key="test-key-1",
        base_url="https://sap-ai-proxy-delightful-oribi-lx.cfapps.eu12.hana.ondemand.com/v1",
    )

    print("=== Interrupt Example ===")
    print("(The agent will pause and ask for your input in the terminal.)\n")
    await demo_interrupt_basic()
    await demo_interrupt_confirmation()

    print("\n\n=== All interrupt demos complete ===")


if __name__ == "__main__":
    asyncio.run(main())
