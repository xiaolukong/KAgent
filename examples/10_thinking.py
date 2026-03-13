"""Example 10: Thinking/Reasoning streaming with KAgent.

This example demonstrates:
- Streaming thinking/reasoning tokens from reasoning models
- Non-streaming thinking access via response metadata
- Handling both THINKING_DELTA and TEXT_DELTA chunk types

Models that produce thinking content:
- Anthropic: Claude with extended thinking (claude-opus-4-5-20251101, etc.)
- OpenAI: o1, o3 series (reasoning_content)
- DeepSeek: DeepSeek-R1 (reasoning_content via OpenAI-compatible API)

Usage:
    export KAGENT_API_KEY=sk-xxx
    python examples/10_thinking.py
"""

import asyncio

from kagent import KAgent, configure
from kagent.domain.enums import StreamChunkType


async def streaming_thinking() -> None:
    """Demonstrate streaming thinking/reasoning tokens."""
    agent = KAgent(
        model="anthropic:claude-opus-4-5-20251101",
        system_prompt="You are a careful reasoning assistant.",
    )

    print("=== Streaming Thinking ===\n")

    thinking_parts: list[str] = []

    async for chunk in agent.stream("What is 127 * 83? Think step by step."):
        if chunk.chunk_type == StreamChunkType.THINKING_DELTA and chunk.thinking:
            # Thinking tokens arrive before the final answer
            thinking_parts.append(chunk.thinking)
            print(f"\033[90m{chunk.thinking}\033[0m", end="", flush=True)
        elif chunk.chunk_type == StreamChunkType.TEXT_DELTA and chunk.content:
            print(chunk.content, end="", flush=True)

    print("\n")
    if thinking_parts:
        print(f"[Thinking length: {len(''.join(thinking_parts))} chars]")


async def non_streaming_thinking() -> None:
    """Demonstrate non-streaming thinking access via metadata."""
    agent = KAgent(
        model="anthropic:claude-opus-4-5-20251101",
        system_prompt="You are a careful reasoning assistant.",
    )

    print("\n=== Non-Streaming Thinking ===\n")

    result = await agent.run("What is 127 * 83? Think step by step.")
    print(f"Answer: {result.content}")

    thinking = result.metadata.get("thinking")
    if thinking:
        print(f"\nThinking ({len(thinking)} chars):")
        print(f"\033[90m{thinking[:200]}...\033[0m")
    else:
        print("\n(No thinking content — model may not support extended thinking)")


async def main():
    configure(
        api_key="test-key-1",
        base_url="https://sap-ai-proxy-delightful-oribi-lx.cfapps.eu12.hana.ondemand.com/v1",
    )

    try:
        await streaming_thinking()
    except Exception as e:
        print(f"Streaming error: {e}")

    try:
        await non_streaming_thinking()
    except Exception as e:
        print(f"Non-streaming error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
