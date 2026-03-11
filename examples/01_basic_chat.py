"""Example 01: Basic chat with KAgent.

This example demonstrates the simplest usage of KAgent:
- Configure global API credentials
- Create an agent with different model providers
- Run a non-streaming query across OpenAI, Anthropic, and Gemini

Usage:
    export KAGENT_API_KEY=sk-xxx
    python examples/01_basic_chat.py
"""

import asyncio

from kagent import KAgent, configure

MODELS = [
    "openai:gpt-5",
    "anthropic:claude-opus-4-5-20251101",
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
    configure(
        api_key="test-key-1",
        base_url="https://sap-ai-proxy-delightful-oribi-lx.cfapps.eu12.hana.ondemand.com/v1",
    )

    print("=== Basic Chat — Multi-Model ===")
    for model in MODELS:
        await run_with_model(model)


if __name__ == "__main__":
    asyncio.run(main())
