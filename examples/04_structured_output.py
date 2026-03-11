"""Example 04: Structured output with response_model.

This example demonstrates:
- Using a Pydantic model as response_model for structured output
- The framework automatically injects JSON Schema into the request
- Parsing the LLM response into a typed Pydantic instance
- Structured output across multiple model providers

Usage:
    export KAGENT_API_KEY=sk-xxx
    python examples/04_structured_output.py
"""

import asyncio

from pydantic import BaseModel, Field

from kagent import KAgent, configure

MODELS = [
    "openai:gpt-5",
    "anthropic:claude-opus-4-5-20251101",
    "gemini:gemini-2.5-pro",
]


class MovieReview(BaseModel):
    """Structured movie review output."""

    title: str = Field(..., description="The movie title.")
    rating: int = Field(..., description="Rating from 1 to 10.")
    summary: str = Field(..., description="A one-sentence summary of the review.")
    pros: list[str] = Field(..., description="List of positive aspects.")
    cons: list[str] = Field(..., description="List of negative aspects.")


async def run_with_model(model: str) -> None:
    """Run a structured output query with the given model."""
    try:
        agent = KAgent(
            model=model,
            system_prompt="You are a film critic. Always respond with structured data.",
        )

        result = await agent.run(
            "Review the movie 'Inception' by Christopher Nolan.",
            response_model=MovieReview,
        )

        print(f"\n[{model}]")
        if result.parsed:
            review: MovieReview = result.parsed
            print(f"  Title:   {review.title}")
            print(f"  Rating:  {review.rating}/10")
            print(f"  Summary: {review.summary}")
            print(f"  Pros:    {review.pros}")
            print(f"  Cons:    {review.cons}")
        else:
            print(f"  Raw: {result.content}")

        if result.usage:
            print(f"  Tokens used: {result.usage.total_tokens}")
    except Exception as e:
        print(f"\n[{model}] Error: {e}")


async def main():
    configure(
        api_key="test-key-1",
        base_url="https://sap-ai-proxy-delightful-oribi-lx.cfapps.eu12.hana.ondemand.com/v1",
    )

    print("=== Structured Output — Multi-Model ===")
    for model in MODELS:
        await run_with_model(model)


if __name__ == "__main__":
    asyncio.run(main())
