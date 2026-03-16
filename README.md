# KAgent

A clean-architecture async Python agent framework with event-driven and interceptor patterns.

## Features

- **Multi-provider LLM support** — OpenAI, Anthropic, Gemini via unified `"provider:model"` syntax
- **Tool calling** — `@agent.tool` decorator with auto JSON Schema generation
- **Streaming** — `agent.stream()` with thinking/reasoning token separation
- **Pub/Sub event system** — `@agent.on(pattern)` for lifecycle observability
- **Interceptor pipeline** — `@agent.intercept(hook)` for mutable request/response modification
- **Context transforms** — `@agent.transform` to filter/inject messages before LLM sees them
- **Steering** — `steer()`, `abort()`, `interrupt()`/`resume()` for runtime control
- **Structured output** — Pydantic `response_model` for typed JSON responses
- **Tool self-correction** — validation errors fed back to LLM for auto-retry
- **Circuit breaker** — configurable `max_tool_retries` to prevent infinite tool loops

## Quick Start

```bash
pip install -e .
cp .env.example .env   # then fill in your API key
```

```python
import asyncio
from kagent import KAgent, configure

configure()  # reads KAGENT_API_KEY and KAGENT_BASE_URL from .env
agent = KAgent(model="openai:gpt-4o", system_prompt="You are helpful.")

@agent.tool
async def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 22°C in {city}"

async def main():
    result = await agent.run("What's the weather in Berlin?")
    print(result.content)

asyncio.run(main())
```

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     User Application                         │
├──────────────────────────────────────────────────────────────┤
│  Interface Layer          KAgent (Facade)                    │
│                           KAgentBuilder                      │
├──────────────────────────────────────────────────────────────┤
│  Application Layer        AgentLoop ← Interceptor Pipeline   │
│                           SteeringController                 │
│                           ContextTransformer                 │
├──────────────────────────────────────────────────────────────┤
│  Domain Layer             Message, ToolCall, ModelResponse   │
│                           EventType, StreamChunk, Role       │
├──────────────────────────────────────────────────────────────┤
│  Infrastructure Layer     OpenAI / Anthropic / Gemini        │
│                           EventBus, ToolExecutor             │
└──────────────────────────────────────────────────────────────┘
```

## Examples

| # | File | Description |
|---|------|-------------|
| 01 | `basic_chat.py` | Minimal agent with multi-provider support |
| 02 | `tool_usage.py` | Register tools with `@agent.tool` |
| 03 | `streaming.py` | Real-time token streaming with `agent.stream()` |
| 04 | `structured_output.py` | Typed output via Pydantic `response_model` |
| 05 | `event_hooks.py` | Lifecycle observability with `@agent.on()` |
| 06 | `custom_state.py` | State management, snapshot/restore |
| 07 | `steering.py` | Mid-turn redirect and abort |
| 08 | `interrupt.py` | Human-in-the-loop confirmation inside tools |
| 09 | `interceptors.py` | Mutable request/response hooks |
| 10 | `thinking.py` | Streaming thinking/reasoning from reasoning models |
| 11 | `context_transform.py` | Filter/inject messages before LLM |
| 12 | `self_correction.py` | Tool error recovery and circuit breaker |

## Documentation

- [Usage Guide](docs/usage.md) — detailed API walkthrough
- [Design Document](docs/design.md) — architecture decisions
- [AI Manifest](docs/AI_MANIFEST.json) — intent-to-code mapping for AI agents
