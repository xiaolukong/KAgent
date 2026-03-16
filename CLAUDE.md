# KAgent — AI Coding Agent Guidelines

## Identity

KAgent is a **clean-architecture async Python agent framework** — a single-agent
runtime engine built on event-driven and interceptor patterns.

**KAgent IS**: an async agent loop, tool calling, streaming, interceptors,
context transforms, steering, structured output, thinking/reasoning support.

**KAgent is NOT**: a multi-agent framework, a LangGraph alternative, a UI layer,
a monitoring platform, or a synchronous framework.

## Quick Import

```python
from kagent import (
    KAgent, configure, tool,
    Message, Role, ModelResponse,
    StreamChunk, StreamChunkType, EventType,
    ToolCall, ToolResult,
)
```

## Core API Cheatsheet

### Initialize

```python
# Credentials are loaded automatically from .env file:
#   KAGENT_API_KEY=sk-...
#   KAGENT_BASE_URL=https://...   (optional, for proxies)
configure()  # reads from .env / env vars
agent = KAgent(model="openai:gpt-4o", system_prompt="You are helpful.")
# Models: "openai:<model>", "anthropic:<model>", "gemini:<model>"
```

### Register Tool

```python
@agent.tool
async def search(query: str) -> str:
    """Search the web."""
    return await do_search(query)
```

### Event Hook (read-only Pub/Sub)

```python
@agent.on("tool.call.completed")
async def on_tool(event):
    print(event.payload)
```

### Interceptor (mutable pipeline)

```python
@agent.intercept("before_llm_request")
async def add_header(request):
    request.metadata["trace_id"] = "abc"
    return request
# Hooks: before_prompt_build, before_llm_request, after_llm_response,
#        before_tool_call, after_tool_call, after_tool_round, before_return
```

### Context Transform

```python
@agent.transform
async def inject_time(messages: list[Message]) -> list[Message]:
    time_msg = Message(role=Role.SYSTEM, content=f"Time: {now()}")
    return [messages[0], time_msg] + messages[1:]
```

### Run (non-streaming)

```python
result = await agent.run("Hello!")
print(result.content)  # str
```

### Stream

```python
async for chunk in agent.stream("Hello!"):
    if chunk.chunk_type == StreamChunkType.THINKING_DELTA:
        print(chunk.thinking, end="")
    elif chunk.content:
        print(chunk.content, end="", flush=True)
```

### Structured Output

```python
from pydantic import BaseModel
class Review(BaseModel):
    score: int
    summary: str
result = await agent.run("Review this code", response_model=Review)
print(result.parsed.score)  # typed Pydantic instance
```

### Steering (redirect / abort / interrupt)

```python
await agent.steer("Now focus on security.")   # redirect mid-turn
await agent.abort("User cancelled")           # stop the loop

# Human-in-the-loop (inside a tool):
reply = await agent.interrupt("Approve transfer of $5000?")
```

## NEVER DO

- **No sync blocking** — never use `time.sleep()`, `requests.get()`, or any
  synchronous I/O. KAgent is fully async (`asyncio`).
- **No private imports** — never `import kagent.agent.loop`, `kagent.models.*`,
  `kagent.tools.executor`, `kagent.context.*`, or `kagent.events.*`.
  Only import from `kagent` (top-level) or `kagent.domain.*` if needed.
- **No private attribute access** — never touch `agent._agent`, `agent._context`,
  `_messages`, `_pipeline`, or any underscore-prefixed attribute.
- **No manual ModelRequest** — use `agent.run()` / `agent.stream()`, not raw
  provider calls.
- **No multi-agent inside KAgent** — orchestrate multiple agents from your
  *application* layer, not inside KAgent internals.

## Key Types

| Type | Fields | Notes |
|------|--------|-------|
| `Message` | `role, content, tool_calls, metadata` | App-layer message |
| `Role` | `USER, ASSISTANT, SYSTEM, TOOL` | Enum |
| `ModelResponse` | `content, tool_calls, usage, parsed, metadata` | LLM reply |
| `StreamChunk` | `chunk_type, content, thinking, tool_call` | Stream token |
| `StreamChunkType` | `TEXT_DELTA, THINKING_DELTA, TOOL_CALL_START/DELTA/END, METADATA` | Enum |
| `EventType` | `AGENT_*, LLM_*, TOOL_*, STEERING_*` | 19 event types |
| `ToolCall` | `id, name, arguments` | Model's tool invocation |
| `ToolResult` | `tool_call_id, tool_name, result, error, status` | Tool output |

## Intent Index

When unsure which file to look at, consult `docs/AI_MANIFEST.json` —
it maps natural-language intents (en + zh) to example files and API surfaces.
