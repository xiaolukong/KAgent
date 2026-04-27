"""Example 02: Tool usage with KAgent via SAP AI Core.

This example demonstrates:
- Registering tools with the @agent.tool decorator
- The agent automatically calls tools when needed
- Tool results are fed back to the model
- Running the same tool-equipped agent across multiple providers

Credentials are read automatically from AICORE_* environment variables.

Usage:
    # 1. Copy .env.example to .env and fill in your AICORE_* credentials
    # 2. Run:
    python examples/02_tool_usage.py
"""

import asyncio
import random

from kagent import KAgent, configure

MODELS = [
    "openai:gpt-4o",
    "anthropic:anthropic--claude-4.5-sonnet",
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
    configure()  # backend="aicore" by default; reads AICORE_* from .env

    print("=== Tool Usage — Multi-Model ===")
    for model in MODELS:
        await run_with_model(model)


if __name__ == "__main__":
    asyncio.run(main())
