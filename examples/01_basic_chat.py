"""Example 01: Basic chat with KAgent via SAP AI Core.

This example demonstrates the simplest usage of KAgent:
- Configure the AI Core backend (default)
- Create an agent with different model providers available on AI Core
- Run a non-streaming query across OpenAI, Anthropic, and Gemini

The model string ("openai:gpt-4o" etc.) stays exactly the same as when using
AI Proxy — only configure() selects the backend.

Credentials are read automatically from AICORE_* environment variables:
    AICORE_AUTH_URL, AICORE_CLIENT_ID, AICORE_CLIENT_SECRET,
    AICORE_BASE_URL, AICORE_RESOURCE_GROUP

Usage:
    # 1. Copy .env.example to .env and fill in your AICORE_* credentials
    # 2. Run:
    python examples/01_basic_chat.py
"""

import asyncio

from kagent import KAgent, configure

MODELS = [
    "openai:gpt-4o",
    "anthropic:anthropic--claude-4.5-sonnet",
    "gemini:gemini-2.5-pro",
]


async def run_with_model(model: str) -> None:
    """Run a basic chat query with the given model."""
    try:
        agent = KAgent(
            model=model,
            system_prompt="You are a helpful assistant. Keep answers concise.",
        )
        result = await agent.run("What are the three laws of robotics?")
        print(f"\n[{model}]")
        print(f"  Response: {result.content}")
        if result.usage:
            print(f"  Tokens used: {result.usage.total_tokens}")
    except Exception as e:
        print(f"\n[{model}] Error: {e}")


async def main():
    configure()  # backend="aicore" by default; reads AICORE_* from .env

    print("=== Basic Chat — Multi-Model ===")
    for model in MODELS:
        await run_with_model(model)


if __name__ == "__main__":
    asyncio.run(main())
