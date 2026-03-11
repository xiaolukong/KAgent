"""Example 02: Tool usage with KAgent.

This example demonstrates:
- Registering tools with the @agent.tool decorator
- The agent automatically calls tools when needed
- Tool results are fed back to the model
- Running the same tool-equipped agent across multiple providers

Usage:
    export KAGENT_API_KEY=sk-xxx
    python examples/02_tool_usage.py
"""

import asyncio
import random

from kagent import KAgent, configure

MODELS = [
    "openai:gpt-5",
    "anthropic:claude-opus-4-5-20251101",
    "gemini:gemini-2.5-pro",
]


async def run_with_model(model: str) -> None:
    """Run a tool-usage query with the given model."""
    try:
        agent = KAgent(
            model=model,
            system_prompt="You are a helpful assistant with access to tools.",
            max_turns=5,
        )

        @agent.tool
        async def get_weather(city: str, unit: str = "celsius") -> dict:
            """Get the current weather for a city."""
            temp = random.randint(15, 35) if unit == "celsius" else random.randint(59, 95)
            return {
                "city": city,
                "temperature": temp,
                "unit": unit,
                "condition": random.choice(["sunny", "cloudy", "rainy"]),
            }

        @agent.tool
        async def calculate(expression: str) -> float:
            """Evaluate a mathematical expression."""
            allowed = set("0123456789+-*/.() ")
            if all(c in allowed for c in expression):
                return float(eval(expression))
            return 0.0

        result = await agent.run("What's the weather in Berlin and Paris?")
        print(f"\n[{model}]")
        print(f"  Response: {result.content}")
    except Exception as e:
        print(f"\n[{model}] Error: {e}")


async def main():
    configure(
        api_key="test-key-1",
        base_url="https://sap-ai-proxy-delightful-oribi-lx.cfapps.eu12.hana.ondemand.com/v1",
    )

    print("=== Tool Usage — Multi-Model ===")
    for model in MODELS:
        await run_with_model(model)


if __name__ == "__main__":
    asyncio.run(main())
