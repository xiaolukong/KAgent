"""Built-in 'think' tool for structured reasoning."""

from kagent.tools.decorator import tool


@tool(
    name="think",
    description=(
        "Use this tool to think step-by-step before taking action. "
        "The input is your internal reasoning and the output is always empty. "
        "This helps organize complex decision-making."
    ),
)
async def think(thought: str) -> str:
    """Record a thought. The thought is logged but produces no side-effects."""
    return ""
