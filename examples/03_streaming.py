"""Example 03: Streaming output with KAgent via SAP AI Core.

This example demonstrates:
- Using agent.stream() for real-time token-by-token output
- Handling StreamChunk objects
- Streaming across multiple model providers

Credentials are read automatically from AICORE_* environment variables.

Usage:
    # 1. Copy .env.example to .env and fill in your AICORE_* credentials
    # 2. Run:
    python examples/03_streaming.py
"""

import asyncio

from kagent import KAgent, configure

MODELS = [
    "openai:gpt-4o",
    "anthropic:anthropic--claude-4.5-sonnet",
    "gemini:gemini-2.5-pro",
]


async def run_with_model(model: str) -> None:
    """Run a streaming query with the given model."""
    try:
        agent = KAgent(
            model=model,
            system_prompt="You are a creative storyteller.",
        )

        print(f"\n[{model}]")
        print("-" * 40)

        async for chunk in agent.stream("Tell me a very short story about a robot."):
            if chunk.content:
                print(chunk.content, end="", flush=True)

        print()
        print("-" * 40)
    except Exception as e:
        print(f"\n[{model}] Error: {e}")


async def main():
    configure()  # backend="aicore" by default; reads AICORE_* from .env

    print("=== Streaming — Multi-Model ===")
    for model in MODELS:
        await run_with_model(model)


if __name__ == "__main__":
    asyncio.run(main())
