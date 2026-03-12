"""Example 08: Interrupt — human-in-the-loop confirmation from inside tools.

This example demonstrates:
- Using await agent.interrupt(prompt) inside a tool to pause and ask the user
- The interrupt blocks until the user replies, then returns their response
- The tool uses the response to decide whether to proceed or cancel
- A steering.interrupt event hook collects user input from the terminal

Key pattern:
    interrupt() is called INSIDE a tool.  It publishes a steering.interrupt
    event (which your event hook handles to collect user input), then blocks
    until resume() is called.  The return value is the user's reply string.

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


async def demo_transfer() -> None:
    """Demo 1: Large transfer requires human confirmation.

    The agent is asked to transfer money.  The transfer_money tool checks
    the amount — if it exceeds the threshold, it calls agent.interrupt()
    to ask the user for confirmation before proceeding.
    """
    print("\n--- Demo 1: Transfer confirmation ---\n")

    agent = KAgent(
        model="openai:gpt-5",
        system_prompt=(
            "You are a banking assistant. "
            "When the user asks you to transfer money, call the transfer_money tool. "
            "Report the result back to the user."
        ),
        max_turns=10,
    )

    @agent.tool
    async def transfer_money(amount: int, account: str) -> str:
        """Transfer money to an account."""
        if amount > 1000:
            print(f"  [tool] Large transfer detected: {amount} -> {account}")
            reply = await agent.interrupt(
                f"Large transfer: {amount} to {account}. Approve? (yes/no)"
            )
            if reply.strip().lower() != "yes":
                return f"Transfer of {amount} to {account} was cancelled by user."
        return f"Successfully transferred {amount} to {account}."

    # Event hook: when an interrupt fires, prompt the user in the terminal
    @agent.on("steering.interrupt")
    async def on_interrupt(event: Event):
        prompt = event.payload.get("prompt", "")
        print("\n  ╔══ CONFIRMATION REQUIRED ═══════════════════════════")
        print(f"  ║ {prompt}")
        print("  ╚════════════════════════════════════════════════════\n")
        user_reply = await async_input("  Your answer > ")
        print()
        await agent.resume(user_reply.strip())

    # Small transfer — no interrupt
    print("  Request: Transfer 500 to Alice")
    result = await agent.run("Transfer 500 to Alice's account.")
    print(f"  Result: {result.content}\n")

    # Large transfer — triggers interrupt
    print("  Request: Transfer 5000 to Bob")
    result = await agent.run("Transfer 5000 to Bob's account.")
    print(f"  Result: {result.content}")


async def demo_delete_files() -> None:
    """Demo 2: Destructive operation requires confirmation.

    The agent lists files, then the delete tool asks for user confirmation
    before actually deleting anything.
    """
    print("\n\n--- Demo 2: Destructive action confirmation ---\n")

    agent = KAgent(
        model="openai:gpt-5",
        system_prompt=(
            "You are a file management assistant. "
            "When asked to delete files, call delete_files with the list of filenames. "
            "Report which files were deleted and which were kept."
        ),
        max_turns=10,
    )

    @agent.tool
    async def delete_files(filenames: list[str]) -> str:
        """Delete the specified files."""
        file_list = ", ".join(filenames)
        print(f"  [tool] delete_files requested: {file_list}")

        reply = await agent.interrupt(
            f"About to delete {len(filenames)} file(s): {file_list}\n"
            f"  ║ Type the filenames to keep (comma-separated), or 'all' to delete all"
        )
        reply = reply.strip().lower()

        if reply == "all":
            return f"Deleted all files: {file_list}"

        keep = {f.strip() for f in reply.split(",")} if reply else set()
        deleted = [f for f in filenames if f not in keep]
        kept = [f for f in filenames if f in keep]

        parts = []
        if deleted:
            parts.append(f"Deleted: {', '.join(deleted)}")
        if kept:
            parts.append(f"Kept: {', '.join(kept)}")
        return ". ".join(parts) if parts else "No files deleted."

    @agent.on("steering.interrupt")
    async def on_interrupt(event: Event):
        prompt = event.payload.get("prompt", "")
        print("\n  ╔══ CONFIRMATION REQUIRED ═══════════════════════════")
        for line in prompt.split("\n"):
            print(f"  ║ {line}")
        print("  ╚════════════════════════════════════════════════════\n")
        user_reply = await async_input("  Your answer > ")
        print()
        await agent.resume(user_reply.strip())

    result = await agent.run("Delete these files: report.docx, temp.csv, notes.txt, cache.tmp")
    print(f"  Result: {result.content}")


async def main():
    configure(
        api_key="test-key-1",
        base_url="https://sap-ai-proxy-delightful-oribi-lx.cfapps.eu12.hana.ondemand.com/v1",
    )

    print("=== Interrupt Example ===")
    print("(Tools will pause and ask for your input when needed.)\n")
    await demo_transfer()
    await demo_delete_files()

    print("\n\n=== All interrupt demos complete ===")


if __name__ == "__main__":
    asyncio.run(main())
