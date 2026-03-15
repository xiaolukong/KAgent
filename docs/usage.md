# KAgent Usage Guide

KAgent 是一个基于 Clean Architecture 的 Python Agent 框架，支持多模型提供商（OpenAI / Anthropic / Gemini）、Tool 调用、流式输出、结构化输出、Pub/Sub 事件系统和运行时 Steering 控制。

---

## 目录

1. [安装与配置](#1-安装与配置)
2. [基础对话](#2-基础对话)
3. [Tool 调用](#3-tool-调用)
4. [流式输出](#4-流式输出)
5. [结构化输出（Pydantic Model）](#5-结构化输出pydantic-model)
6. [Pub/Sub 事件系统](#6-pubsub-事件系统)
7. [Steering（运行时干预）](#7-steering运行时干预)
8. [拦截器管道（Interceptors）](#8-拦截器管道interceptors)
9. [上下文转换管道（Context Transforms）](#9-上下文转换管道context-transforms)
10. [Builder 模式](#10-builder-模式)
11. [状态管理](#11-状态管理)
12. [API 参考速查](#12-api-参考速查)

---

## 1. 安装与配置

### 全局配置

使用 `configure()` 设置全局 API 凭证。所有 Provider 在未单独配置时会自动回退到全局配置。

```python
from kagent import configure

configure(
    api_key="your-api-key",
    base_url="https://your-proxy.example.com/v1",  # 可选，用于 OpenAI 兼容代理
)
```

也可以通过环境变量配置：

```bash
export KAGENT_API_KEY=your-api-key
export KAGENT_BASE_URL=https://your-proxy.example.com/v1
```

优先级：`configure()` 显式调用 > 环境变量。

### 模型标识格式

模型使用 `"provider:model_name"` 格式指定：

```python
"openai:gpt-5"
"anthropic:claude-opus-4-5-20251101"
"gemini:gemini-2.5-pro"
```

支持的 Provider：`openai`、`anthropic`、`gemini`。

### 代理模式

当设置了自定义 `base_url`（非官方 API 地址）时，Anthropic 和 Gemini Provider 会自动切换到 OpenAI 兼容代理模式，使用 `/chat/completions` 协议。这意味着你可以通过一个 OpenAI 兼容代理同时使用所有三家模型。

---

## 2. 基础对话

最简单的用法：创建一个 Agent，发送一条消息，获取回复。

```python
import asyncio
from kagent import KAgent, configure

async def main():
    configure(api_key="your-api-key")

    agent = KAgent(
        model="openai:gpt-5",
        system_prompt="You are a helpful assistant. Keep answers concise.",
    )

    result = await agent.run("What are the three laws of robotics?")

    print(result.content)          # 模型回复文本
    if result.usage:
        print(result.usage.total_tokens)  # Token 用量

asyncio.run(main())
```

### KAgent 构造参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | `str` | `"openai:gpt-4o"` | 模型标识，格式 `"provider:model_name"` |
| `system_prompt` | `str` | `"You are a helpful assistant."` | 系统提示词 |
| `max_turns` | `int` | `10` | 最大循环轮次（每次 tool 调用算一轮） |
| `temperature` | `float \| None` | `None` | 温度参数 |
| `max_tokens` | `int \| None` | `None` | 最大输出 token 数 |
| `max_context_tokens` | `int` | `128_000` | 上下文窗口大小 |
| `api_key` | `str \| None` | `None` | 单独指定 API key（覆盖全局配置） |
| `base_url` | `str \| None` | `None` | 单独指定 base URL（覆盖全局配置） |

### ModelResponse 返回值

`agent.run()` 返回 `ModelResponse` 对象：

```python
result = await agent.run("Hello")

result.content        # str | None — 模型回复文本
result.tool_calls     # list[ToolCall] | None — 模型请求的 tool 调用
result.usage          # TokenUsage | None — token 用量
result.usage.prompt_tokens
result.usage.completion_tokens
result.usage.total_tokens
result.parsed         # Any — 结构化输出解析结果（见第 5 节）
result.has_tool_calls # bool — 是否包含 tool 调用
```

### 多模型切换

同一段代码可以轻松切换不同模型：

```python
for model in ["openai:gpt-5", "anthropic:claude-opus-4-5-20251101", "gemini:gemini-2.5-pro"]:
    agent = KAgent(model=model, system_prompt="You are helpful.")
    result = await agent.run("Explain quantum computing in one sentence.")
    print(f"[{model}] {result.content}")
```

---

## 3. Tool 调用

使用 `@agent.tool` 装饰器注册 Python 函数为 Tool。框架会：
1. 从函数签名和 docstring 自动推断 JSON Schema
2. 将 Schema 传递给模型
3. 模型决定调用哪个 Tool 以及参数
4. 框架自动执行 Tool 并将结果反馈给模型

```python
import asyncio
import random
from kagent import KAgent, configure

async def main():
    configure(api_key="your-api-key")

    agent = KAgent(
        model="openai:gpt-5",
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
    print(result.content)

asyncio.run(main())
```

### Tool 注册方式

```python
# 方式 1：裸装饰器（使用函数名和 docstring）
@agent.tool
async def my_tool(param: str) -> str:
    """Tool description used by the model."""
    return f"result for {param}"

# 方式 2：带参数装饰器（自定义名称和描述）
@agent.tool(name="custom_name", description="Custom description for the model.")
async def my_tool(param: str) -> str:
    return f"result for {param}"

# 方式 3：同步函数也支持
@agent.tool
def sync_tool(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y
```

### Schema 自动推断

框架从 Python type hints 自动生成 JSON Schema。支持的类型：

| Python 类型 | JSON Schema 类型 |
|-------------|-----------------|
| `str` | `"string"` |
| `int` | `"integer"` |
| `float` | `"number"` |
| `bool` | `"boolean"` |
| `list[str]` | `{"type": "array", "items": {"type": "string"}}` |
| `dict[str, int]` | `{"type": "object", ...}` |
| `Literal["a", "b"]` | `{"enum": ["a", "b"]}` |
| `str \| None` | `{"type": "string"}` (optional) |
| Pydantic `BaseModel` | 完整 object schema |

带默认值的参数会标记为 optional，函数的 docstring 会作为 tool 的 `description`。

### Tool 调用流程

```
用户输入 → 模型决策 → 调用 tool → 结果返回模型 → 模型生成最终回复
                 ↑                               |
                 └───── 循环（直到无 tool 调用或达到 max_turns）
```

### 工具自纠正（Self-Correction）

当 LLM 传入的参数不符合 Tool 的类型约束时（例如将 `int` 参数传为字符串），框架**不会崩溃**。它会：

1. 将参数验证错误转化为一条友好的 `Role.TOOL` 错误消息
2. 错误消息包含参数名称、错误原因、以及 LLM 实际传入的值
3. 将该消息塞回 context，让 LLM 看到错误并自行修正

```python
@agent.tool
async def lookup_user(user_id: int) -> dict:
    """Look up user by numeric ID."""
    return {"name": "Alice"}

# 如果 LLM 调用 lookup_user(user_id="abc")，框架会返回：
# "Tool 'lookup_user' parameter validation failed.
#  Validation failed for parameter 'user_id': ...
#  You provided: {"user_id": "abc"}
#  Please fix the parameters and try again."
#
# LLM 看到这条消息后，会在下一轮用正确的参数重试。
```

### 熔断器（Circuit Breaker）

为了防止 LLM 反复调用同一个失败的工具陷入死循环，框架内置了熔断器机制：

- 当**同一个工具**连续失败达到 `max_tool_retries` 次（默认 3 次）后，框架会停止重试
- 框架注入一条提示消息告诉 LLM：「这个工具已经连续失败 N 次，请换一种方法」
- LLM 可以选择调用其他工具或直接给出文本回复
- 不同工具的计数器相互独立
- 某个工具一旦成功执行，其计数器会重置为 0

```python
agent = KAgent(
    model="openai:gpt-5",
    max_tool_retries=3,   # 默认值，同一工具连续失败 3 次后熔断
)

# 也可以设置为更高的值允许更多重试
agent = KAgent(
    model="openai:gpt-5",
    max_tool_retries=5,
)
```

---

## 4. 流式输出

使用 `agent.stream()` 获取逐 token 的实时输出：

```python
import asyncio
from kagent import KAgent, configure

async def main():
    configure(api_key="your-api-key")

    agent = KAgent(
        model="openai:gpt-5",
        system_prompt="You are a creative storyteller.",
    )

    async for chunk in agent.stream("Tell me a short story about a robot."):
        if chunk.content:
            print(chunk.content, end="", flush=True)

    print()  # 换行

asyncio.run(main())
```

### StreamChunk 属性

每个 `chunk` 是一个 `StreamChunk` 对象：

```python
chunk.chunk_type   # StreamChunkType — 类型标识
chunk.content      # str | None — 文本增量（chunk_type == TEXT_DELTA 时有值）
chunk.thinking     # str | None — 思考增量（chunk_type == THINKING_DELTA 时有值）
chunk.tool_call    # ToolCallChunk | None — tool 调用增量
chunk.usage        # TokenUsage | None — token 用量（通常在最后一个 chunk）
```

### StreamChunkType 类型

| 类型 | 说明 |
|------|------|
| `TEXT_DELTA` | 文本增量，`chunk.content` 包含文本片段 |
| `THINKING_DELTA` | 思考/推理增量，`chunk.thinking` 包含思考片段 |
| `TOOL_CALL_START` | tool 调用开始，`chunk.tool_call.name` 包含 tool 名称 |
| `TOOL_CALL_DELTA` | tool 调用参数增量 |
| `TOOL_CALL_END` | tool 调用结束 |
| `METADATA` | 元数据（如 token 用量） |

### 流式 + Tool 调用

流式模式同样支持 tool 调用。框架会在流式传输中收集 tool 调用数据，执行完毕后继续下一轮流式输出：

```python
agent = KAgent(model="openai:gpt-5", system_prompt="You are helpful.")

@agent.tool
async def lookup(topic: str) -> str:
    """Look up information about a topic."""
    return f"Here are some facts about {topic}..."

async for chunk in agent.stream("Look up Python programming."):
    if chunk.content:
        print(chunk.content, end="", flush=True)
```

### 流式 Thinking/Reasoning（思考过程）

现代推理模型（Anthropic Claude Extended Thinking、OpenAI o1/o3、DeepSeek-R1）在响应中包含"思考过程"。KAgent 通过 `THINKING_DELTA` chunk 类型支持流式接收这些思考 token：

```python
from kagent.domain.enums import StreamChunkType

agent = KAgent(model="anthropic:claude-opus-4-5-20251101", system_prompt="You reason carefully.")

async for chunk in agent.stream("What is 127 * 83?"):
    if chunk.chunk_type == StreamChunkType.THINKING_DELTA and chunk.thinking:
        # 思考 token（灰色显示）
        print(f"\033[90m{chunk.thinking}\033[0m", end="", flush=True)
    elif chunk.chunk_type == StreamChunkType.TEXT_DELTA and chunk.content:
        # 最终回复
        print(chunk.content, end="", flush=True)
```

非流式模式下，思考内容通过 `response.metadata["thinking"]` 访问：

```python
result = await agent.run("Solve this step by step: 127 * 83")
if "thinking" in result.metadata:
    print(f"Thinking: {result.metadata['thinking']}")
print(f"Answer: {result.content}")
```

思考内容也会写入对话上下文的 assistant message metadata 中，可用于调试和审计。

---

## 5. 结构化输出（Pydantic Model）

通过传入 `response_model` 参数，让模型返回符合指定 Pydantic Schema 的结构化数据。

```python
import asyncio
from pydantic import BaseModel, Field
from kagent import KAgent, configure

class MovieReview(BaseModel):
    """Structured movie review output."""
    title: str = Field(..., description="The movie title.")
    rating: int = Field(..., description="Rating from 1 to 10.")
    summary: str = Field(..., description="A one-sentence summary of the review.")
    pros: list[str] = Field(..., description="List of positive aspects.")
    cons: list[str] = Field(..., description="List of negative aspects.")

async def main():
    configure(api_key="your-api-key")

    agent = KAgent(
        model="openai:gpt-5",
        system_prompt="You are a film critic. Always respond with structured data.",
    )

    result = await agent.run(
        "Review the movie 'Inception' by Christopher Nolan.",
        response_model=MovieReview,
    )

    if result.parsed:
        review: MovieReview = result.parsed
        print(f"Title:   {review.title}")
        print(f"Rating:  {review.rating}/10")
        print(f"Summary: {review.summary}")
        print(f"Pros:    {review.pros}")
        print(f"Cons:    {review.cons}")

asyncio.run(main())
```

### 各 Provider 的实现机制

不同 Provider 使用不同的底层机制，但对用户透明：

| Provider | 机制 | 说明 |
|----------|------|------|
| **OpenAI** | `chat.completions.parse(response_format=PydanticType)` | SDK 原生支持，端到端处理 |
| **Anthropic** | Tool-use trick（Function Calling） | 定义合成 tool，强制模型调用，从 tool_call.arguments 提取数据 |
| **Gemini (native)** | `generation_config.response_schema` | 原生 JSON Schema 支持 |
| **Gemini (proxy)** | `response_format: json_schema` | 代理层强制执行 |

### Pydantic Model 定义技巧

- 使用 `Field(..., description="...")` 为每个字段添加描述，帮助模型理解期望格式
- 支持嵌套模型、`list[T]`、`Optional[T]` 等复杂类型
- 模型的 docstring 会被包含在 Schema 中

```python
class Address(BaseModel):
    street: str
    city: str
    country: str

class Person(BaseModel):
    """A person with address information."""
    name: str = Field(..., description="Full name")
    age: int = Field(..., description="Age in years")
    addresses: list[Address] = Field(..., description="List of addresses")
```

---

## 6. Pub/Sub 事件系统

KAgent 内部所有模块通过 EventBus 进行通信。你可以通过 `@agent.on()` 订阅事件来实现监控、日志、审计等功能。

### 基本用法

```python
import asyncio
import time
from kagent import KAgent, configure
from kagent.domain.events import Event

async def main():
    configure(api_key="your-api-key")

    agent = KAgent(model="openai:gpt-5", system_prompt="You are helpful.")

    start_time = time.time()

    # 订阅所有 agent 生命周期事件
    @agent.on("agent.*")
    async def on_agent_event(event: Event):
        elapsed = time.time() - start_time
        print(f"[{elapsed:.2f}s] AGENT: {event.event_type.value}")

    # 订阅 LLM 请求事件
    @agent.on("llm.request.sent")
    async def on_llm_request(event: Event):
        msg_count = event.payload.get("message_count", "?")
        print(f"LLM request sent ({msg_count} messages)")

    # 订阅 LLM 响应事件
    @agent.on("llm.response.received")
    async def on_llm_response(event: Event):
        usage = event.payload.get("usage")
        if usage:
            print(f"LLM response (tokens: {usage.get('total_tokens', '?')})")

    # 订阅 tool 事件
    @agent.on("tool.call.started")
    async def on_tool_start(event: Event):
        print(f"TOOL started: {event.payload.get('tool_name')}")

    @agent.on("tool.call.completed")
    async def on_tool_done(event: Event):
        name = event.payload.get("tool_name")
        ms = event.payload.get("duration_ms", 0)
        print(f"TOOL completed: {name} ({ms:.1f}ms)")

    @agent.tool
    async def lookup(topic: str) -> str:
        """Look up information about a topic."""
        return f"Here are some facts about {topic}..."

    result = await agent.run("Look up Python programming.")
    print(f"Response: {result.content}")

asyncio.run(main())
```

### 事件类型一览

#### Agent 生命周期事件

| 事件类型 | 匹配模式 | payload 字段 |
|----------|----------|-------------|
| `agent.started` | `agent.*` | `input`, `mode` ("complete" 或 "stream") |
| `agent.loop.iteration` | `agent.*` | `turn_number` |
| `agent.loop.completed` | `agent.*` | `has_content` 或 `mode` |
| `agent.error` | `agent.*` | `error_type`, `message` |
| `agent.state.changed` | `agent.*` | `key`, `old_value`, `new_value` |

#### LLM 事件

| 事件类型 | 匹配模式 | payload 字段 |
|----------|----------|-------------|
| `llm.request.sent` | `llm.*` | `model`, `message_count` |
| `llm.response.received` | `llm.*` | `has_tool_calls`, `usage` |
| `llm.stream.chunk` | `llm.*` | `chunk_type` |
| `llm.stream.complete` | `llm.*` | `content_length` |
| `llm.error` | `llm.*` | 错误信息 |

#### Tool 事件

| 事件类型 | 匹配模式 | payload 字段 |
|----------|----------|-------------|
| `tool.call.started` | `tool.*` | `tool_name`, `call_id`, `arguments` |
| `tool.call.completed` | `tool.*` | `tool_name`, `call_id`, `duration_ms`, `result` |
| `tool.call.error` | `tool.*` | `tool_name`, `call_id`, `error` |

#### Steering 事件

| 事件类型 | 匹配模式 | payload 字段 |
|----------|----------|-------------|
| `steering.redirect` | `steering.*` | `directive` |
| `steering.abort` | `steering.*` | `reason` |
| `steering.interrupt` | `steering.*` | `prompt` |
| `steering.resume` | `steering.*` | `user_input` |
| `steering.inject_message` | `steering.*` | `message` |

### Glob 匹配模式

`@agent.on()` 支持 glob 通配符：

```python
@agent.on("agent.*")       # 匹配所有 agent 事件
@agent.on("tool.*")        # 匹配所有 tool 事件
@agent.on("llm.*")         # 匹配所有 LLM 事件
@agent.on("steering.*")    # 匹配所有 steering 事件
@agent.on("*")             # 匹配所有事件（catch-all）
```

也可以精确匹配单个事件：

```python
@agent.on("tool.call.completed")   # 仅匹配 tool 调用完成
@agent.on("llm.response.received") # 仅匹配 LLM 响应接收
```

### 非装饰器方式注册

```python
async def my_handler(event: Event):
    print(event.payload)

agent.on("tool.*", my_handler)  # 直接传入函数
```

### Event 对象

所有事件处理函数接收一个 `Event` 对象：

```python
event.event_type      # EventType — 事件类型枚举
event.payload         # dict[str, Any] — 事件数据
event.source          # str | None — 事件来源
event.timestamp       # datetime — UTC 时间戳
event.correlation_id  # str — 关联 ID（用于跟踪）
```

---

## 7. Steering（运行时干预）

Steering 允许在 Agent 运行过程中从外部干预其行为。它设计为**并发使用**：Agent 在一个协程中运行，另一个协程调用 `steer()` / `abort()` / `interrupt()` 来干预。

三种干预方式：

| 方法 | 行为 | 循环是否继续 |
|------|------|-------------|
| `agent.steer(directive)` | 将 directive 作为新的 user message 注入对话上下文 | **继续** — 模型响应新指令 |
| `agent.abort(reason)` | 设置 abort 标志 | **停止** — 循环在下一轮次边界退出 |
| `agent.interrupt(prompt)` | 暂停执行，等待用户输入，返回回复 | **暂停** — 返回用户回复后 Tool 继续 |

### Steer（重定向）

使用 `agent.steer()` 在运行过程中改变 Agent 的方向。Agent 不会停止，而是接收到新的指令并继续执行。

```python
import asyncio
from kagent import KAgent, configure
from kagent.domain.events import Event

async def main():
    configure(api_key="your-api-key")

    agent = KAgent(
        model="openai:gpt-5",
        system_prompt=(
            "You are a research assistant. "
            "Always call the research tool for each topic. "
            "After researching, provide a brief summary."
        ),
        max_turns=10,
    )

    topics_researched: list[str] = []

    @agent.tool
    async def research(topic: str) -> str:
        """Research a topic and return findings."""
        topics_researched.append(topic)
        print(f"[tool] research('{topic}')")
        await asyncio.sleep(0.3)
        return f"Key finding: {topic} is a rapidly evolving field."

    # 监听 redirect 事件
    @agent.on("steering.redirect")
    async def on_redirect(event: Event):
        directive = event.payload.get("directive", "")
        print(f"[steering] redirect: \"{directive}\"")

    # Agent 开始研究主题 A，在第一个 tool 完成后重定向到主题 B
    async def redirect_after_first_tool():
        while len(topics_researched) < 1:
            await asyncio.sleep(0.1)
        print(">>> Redirecting agent to a new topic!")
        await agent.steer(
            "Actually, stop what you're doing. "
            "Research 'artificial intelligence' instead and summarize that."
        )

    # 并发运行 agent 和 redirect 逻辑
    agent_task = asyncio.create_task(
        agent.run("Research 'quantum computing' and 'black holes'.")
    )
    steer_task = asyncio.create_task(redirect_after_first_tool())

    result, _ = await asyncio.gather(agent_task, steer_task, return_exceptions=True)

    print(f"Topics researched: {topics_researched}")
    # 输出类似: ['quantum computing', 'artificial intelligence']
    # Agent 在收到 steer 指令后放弃了 'black holes'，转向了 'artificial intelligence'

asyncio.run(main())
```

**工作流程：**

```
用户: "Research quantum computing and black holes"
  ↓
Agent 调用 research("quantum computing")
  ↓
steer("Research artificial intelligence instead")  ← 外部协程发送
  ↓
Agent 循环注入新的 user message，模型看到重定向指令
  ↓
Agent 调用 research("artificial intelligence")  ← 遵循新指令
  ↓
Agent 返回关于 AI 的总结（忽略了 black holes）
```

### Abort（中止运行）

在 Agent 运行过程中发送 abort 信号。Agent 循环会在下一个轮次边界检测到 abort 标志并优雅退出。

```python
import asyncio
from kagent import KAgent, configure
from kagent.domain.events import Event

async def main():
    configure(api_key="your-api-key")

    agent = KAgent(
        model="openai:gpt-5",
        system_prompt="You are a research assistant. Call the research tool for EACH topic.",
        max_turns=10,
    )

    call_count = 0

    @agent.tool
    async def research(topic: str) -> str:
        """Research a topic and return findings."""
        nonlocal call_count
        call_count += 1
        print(f"[tool] research('{topic}') — call #{call_count}")
        await asyncio.sleep(0.5)  # 模拟耗时操作
        return f"Findings about {topic}: it is very interesting."

    # 监听 steering 事件
    @agent.on("steering.*")
    async def on_steering(event: Event):
        print(f"[steering] {event.event_type.value}: {event.payload}")

    # 在另一个协程中等待条件满足后 abort
    async def abort_after_first_tool():
        while call_count < 1:
            await asyncio.sleep(0.1)
        print(">>> Sending abort!")
        await agent.abort("User decided to stop early")

    # 并发运行 agent 和 abort 逻辑
    agent_task = asyncio.create_task(
        agent.run("Research: quantum computing, black holes, and DNA.")
    )
    abort_task = asyncio.create_task(abort_after_first_tool())

    result, _ = await asyncio.gather(agent_task, abort_task, return_exceptions=True)
    print(f"Tool was called {call_count} time(s) (3 were requested)")

asyncio.run(main())
```

### 流式模式下的 Abort

同样适用于 `agent.stream()`：

```python
async def run_stream():
    async for chunk in agent.stream("Research: neural networks, relativity."):
        if chunk.content:
            print(chunk.content, end="", flush=True)

async def abort_after_delay():
    while call_count < 1:
        await asyncio.sleep(0.1)
    await agent.abort("Enough research for now")

stream_task = asyncio.create_task(run_stream())
abort_task = asyncio.create_task(abort_after_delay())
await asyncio.gather(stream_task, abort_task, return_exceptions=True)
```

### Interrupt（暂停并等待用户输入）

使用 `await agent.interrupt(prompt)` 在 Tool 内部暂停执行，向用户发送一条消息，等待用户通过 `agent.resume(user_input)` 回复，然后**返回用户的回复字符串**。

典型场景：
- Tool 执行前需要用户确认（如转账、删除等危险操作）
- Tool 发现参数异常，需要用户二次确认
- Agent 执行多步任务时需要用户提供额外信息

**核心模式**：在 Tool 内部直接 `await agent.interrupt()`，它会阻塞直到用户回复。在 `steering.interrupt` 事件钩子中收集用户输入并调用 `resume()`。

```python
import asyncio
import sys
from kagent import KAgent, configure
from kagent.domain.events import Event


async def async_input(prompt: str = "") -> str:
    """Read a line from stdin without blocking the event loop."""
    loop = asyncio.get_running_loop()
    if prompt:
        sys.stdout.write(prompt)
        sys.stdout.flush()
    return await loop.run_in_executor(None, sys.stdin.readline)


async def main():
    configure(api_key="your-api-key")

    agent = KAgent(
        model="openai:gpt-5",
        system_prompt=(
            "You are a banking assistant. "
            "When asked to transfer money, call the transfer_money tool."
        ),
        max_turns=10,
    )

    @agent.tool
    async def transfer_money(amount: int, account: str) -> str:
        """Transfer money to an account."""
        if amount > 1000:
            # 直接在 tool 内部触发中断，等待用户确认
            reply = await agent.interrupt(
                f"Large transfer: {amount} to {account}. Approve? (yes/no)"
            )
            if reply.strip().lower() != "yes":
                return f"Transfer of {amount} to {account} was cancelled by user."
        return f"Successfully transferred {amount} to {account}."

    # 事件钩子：收集用户输入并 resume
    @agent.on("steering.interrupt")
    async def on_interrupt(event: Event):
        prompt = event.payload.get("prompt", "")
        print(f"\n[Confirmation required] {prompt}")
        user_reply = await async_input("Your answer > ")
        await agent.resume(user_reply.strip())

    result = await agent.run("Transfer 5000 to Bob's account.")
    print(f"Result: {result.content}")

asyncio.run(main())
```

**工作流程：**

```
用户: "Transfer 5000 to Bob's account"
  ↓
模型调用 transfer_money(amount=5000, account="Bob")
  ↓
Tool 检测到 amount > 1000，调用 await agent.interrupt(...)
  ↓
发布 steering.interrupt 事件 → 事件钩子提示用户
  ↓  ... 等待用户输入 ...
用户输入 "yes" → 事件钩子调用 agent.resume("yes")
  ↓
interrupt() 返回 "yes"，Tool 继续执行转账
  ↓
Tool 返回 "Successfully transferred 5000 to Bob."
  ↓
模型生成最终回复
```

**关键点：**
- `interrupt()` 返回 `str`（用户的回复），可以直接在 Tool 里使用返回值做逻辑判断
- 不需要并行协程 — interrupt 在 Tool 执行过程中自然阻塞
- 事件钩子 `steering.interrupt` 负责展示 prompt 和收集用户输入

### Steering 工作原理

Steering 使用双队列系统：

```
                        ┌──────────────────────────────────────┐
                        │         SteeringController           │
                        │                                      │
  agent.steer() ──────▶ │  steering_queue (高优先)              │
  agent.abort() ──────▶ │   ├─ REDIRECT → 注入 user message，  │
                        │   │    继续循环                       │
                        │   └─ ABORT → 设置标志，退出循环       │
                        │                                      │
  agent.interrupt() ──▶ │  (不经过队列，直接在 Tool 内部阻塞)    │
                        │   └─ await wait_for_resume()         │
  agent.resume() ─────▶ │  _interrupt_event.set() → 解除暂停   │
                        │                                      │
  inject_message ─────▶ │  message_queue (跟随消息)             │ ──▶ tool 执行完成后注入
                        └──────────────────────────────────────┘
```

- **steering_queue**：高优先级指令（redirect / abort），在每个循环轮次**开始时**检查
  - **REDIRECT**（`steer()`）：将 directive 注入为 `Role.USER` 消息，循环**继续**
  - **ABORT**（`abort()`）：设置 `is_aborted` 标志，循环在下一轮次边界**退出**
- **interrupt()**：不经过 steering_queue，而是**直接在 Tool 内部阻塞**。Tool 调用 `await agent.interrupt(prompt)`，发布事件后等待 `resume()`，返回用户回复
- **message_queue**：跟随消息，在 tool 执行**完成后**注入到对话上下文中
- **_interrupt_event**：`asyncio.Event` 信号量，`interrupt()` 时 clear（暂停），`resume()` 时 set（恢复）
- 所有 steering 操作通过 EventBus 传递，完全解耦

---

## 8. 拦截器管道（Interceptors）

与 `@agent.on()` 的 Pub/Sub 模式不同，拦截器（Interceptor）是一个**可变的 handler 管道**：每个 handler 接收数据、可以修改数据、返回数据给下一个 handler。这使得你可以在 AgentLoop 的关键节点**拦截并修改**流经的数据。

| 机制 | 模式 | 能否修改数据 | 用途 |
|------|------|-------------|------|
| `@agent.on()` | Pub/Sub（观察） | ❌ 只读 | 日志、监控、审计 |
| `@agent.intercept()` | Pipeline（拦截） | ✅ 可变 | 过滤 tools、修改请求/响应、阻止操作、脱敏 |

### 基本用法

```python
from kagent import KAgent, configure

async def main():
    configure(api_key="your-api-key")

    agent = KAgent(model="openai:gpt-5", system_prompt="You are helpful.")

    # 装饰器方式注册拦截器
    @agent.intercept("before_llm_request")
    async def force_low_temp(request):
        request.temperature = 0.0
        return request

    # 直接调用方式注册
    async def log_response(response):
        print(f"Tokens: {response.usage.total_tokens if response.usage else '?'}")
        return response

    agent.intercept("after_llm_response", log_response)

    result = await agent.run("Hello!")
```

### 7 个拦截点

AgentLoop 在以下位置调用拦截器管道：

```
用户输入
  │
  ▼
① before_prompt_build   ← 过滤/重排 messages、增删 tool 定义
  │
  ▼
  PromptBuilder.build()
  │
  ▼
② before_llm_request    ← 修改 temperature/max_tokens、改写 model 名
  │
  ▼
  LLM 调用
  │
  ▼
③ after_llm_response    ← 审查/修改模型输出、过滤 tool_calls
  │
  ├─ (无 tool_calls) ─→ ⑦ before_return ─→ 返回结果
  │
  ├─ (有 tool_calls) ──→ 对每个 tool call:
  │     ④ before_tool_call  ← 修改参数、阻止执行
  │     │
  │     ▼
  │     执行 tool
  │     │
  │     ▼
  │     ⑤ after_tool_call   ← 修改/脱敏 tool 结果
  │
  ▼
⑥ after_tool_round       ← 查看本轮所有 tool 结果
  │
  ▼
  继续循环 → ①
```

### 各拦截点的数据类型

| 拦截点 | 接收/返回类型 | 说明 |
|--------|-------------|------|
| `before_prompt_build` | `dict` — `{"messages": list[Message], "tool_definitions": list[ToolDefinition]}` | 可过滤 tools、修改 messages |
| `before_llm_request` | `ModelRequest` | 可修改 model、temperature、max_tokens 等 |
| `after_llm_response` | `ModelResponse` | 可修改 content、过滤 tool_calls |
| `before_tool_call` | `dict` — `{"tool_name": str, "arguments": dict, "call_id": str}` | 可修改参数或阻止执行 |
| `after_tool_call` | `ToolResult` | 可修改 result/error（如脱敏） |
| `after_tool_round` | `list[dict]` — 本轮所有 tool 结果 | 可查看/修改汇总结果 |
| `before_return` | `ModelResponse` | 可添加 metadata、修改最终输出 |

### 动态 Tool 过滤

使用 `before_prompt_build` 拦截器动态控制哪些 tools 对模型可见：

```python
agent = KAgent(model="openai:gpt-5", system_prompt="You are helpful.")

safe_mode = True

@agent.tool
async def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: 22°C, sunny"

@agent.tool
async def delete_file(path: str) -> str:
    """Delete a file."""
    return f"Deleted {path}"

@agent.intercept("before_prompt_build")
async def filter_tools(ctx):
    if safe_mode:
        allowed = {"get_weather"}
        ctx["tool_definitions"] = [t for t in ctx["tool_definitions"] if t.name in allowed]
    return ctx
```

### 阻止危险 Tool 调用

使用 `before_tool_call` + `InterceptResult` 阻止特定 tool 执行。被阻止的 tool 会返回一条错误消息给模型，模型可以据此告知用户：

```python
from kagent.agent.interceptor import InterceptResult

@agent.intercept("before_tool_call")
async def block_dangerous(ctx):
    if ctx["tool_name"] == "delete_file":
        return InterceptResult(
            data=ctx,
            blocked=True,
            reason="Tool 'delete_file' is not allowed.",
        )
    return ctx
```

当 `InterceptResult(blocked=True)` 被返回时：
1. 管道抛出 `InterceptBlockedError`
2. AgentLoop 捕获异常，将 `{"blocked": True, "reason": "..."}` 写入上下文
3. Tool **不会执行**
4. 循环继续 — 模型会看到阻止信息并相应回复

### 修改 Tool 结果（脱敏）

使用 `after_tool_call` 拦截器在 tool 结果写入上下文前进行脱敏：

```python
@agent.intercept("after_tool_call")
async def redact_pii(result):
    if result.result and "ssn" in str(result.result):
        result.result = str(result.result).replace("123-45-6789", "***-**-****")
    return result
```

### 添加审计 Metadata

使用 `before_return` 拦截器为最终响应添加元数据：

```python
@agent.intercept("before_return")
async def add_audit(response):
    response.metadata["audited"] = True
    response.metadata["pii_check"] = "passed"
    return response

result = await agent.run("Look up alice.")
print(result.metadata)  # {"audited": True, "pii_check": "passed"}
```

### 优先级

多个拦截器注册在同一个拦截点时，**priority 越高越先执行**。相同 priority 按注册顺序执行：

```python
@agent.intercept("before_return", priority=10)
async def high_priority(response):
    # 先执行
    return response

@agent.intercept("before_return", priority=0)
async def low_priority(response):
    # 后执行
    return response
```

### 与 EventBus 的共存

拦截器和 EventBus 是两套独立的系统，互不影响：

- **拦截器**（`@agent.intercept`）：在 AgentLoop 内部调用，handler 接收并返回数据，可以修改流程
- **EventBus**（`@agent.on`）：在 AgentLoop 发布事件后调用，handler 只能观察，不影响流程

两者可以同时使用。例如，用拦截器脱敏 tool 结果，用 EventBus 记录审计日志。

---

## 9. 上下文转换管道（Context Transforms）

KAgent 的 ContextManager 保留完整的应用层（App-layer）消息历史。上下文转换管道在每次构建 LLM 请求前，将 App 层消息转换为 LLM 层消息。**转换仅影响发送给 LLM 的内容，不修改 ContextManager 中的原始历史。**

### 设计理念

```
ContextManager (完整 App 历史)
  │
  ▼
ContextTransformer.apply(messages)   ← 过滤/变换
  │
  ▼
PromptBuilder.build(transformed_messages)
  │
  ▼
发送给 LLM
```

### 内置默认转换

Agent 启动时自动注册两个内置转换（低优先级，用户转换优先执行）：

| 转换 | 作用 |
|------|------|
| `filter_internal_messages` | 过滤 `metadata.internal == True` 的消息 |
| `strip_thinking_from_context` | 剥离 `metadata.thinking`，避免思考内容被回送给 LLM |

### 注册自定义转换

使用 `@agent.transform` 装饰器注册转换函数。每个转换函数接收 `list[Message]`，返回 `list[Message]`：

```python
from kagent import KAgent, configure
from kagent.domain.entities import Message
from kagent.domain.enums import Role
from datetime import datetime, UTC

async def main():
    configure(api_key="your-api-key")

    agent = KAgent(model="openai:gpt-5", system_prompt="You are helpful.")

    # 注入动态上下文（如当前时间）
    @agent.transform(priority=10)
    async def inject_time(messages: list[Message]) -> list[Message]:
        time_msg = Message(
            role=Role.SYSTEM,
            content=f"Current time: {datetime.now(UTC).isoformat()}",
        )
        return [messages[0], time_msg] + messages[1:]

    # 过滤调试消息
    @agent.transform
    async def filter_debug(messages: list[Message]) -> list[Message]:
        return [m for m in messages if m.metadata.get("message_type") != "debug"]

    result = await agent.run("What time is it?")
    print(result.content)
```

### 内部消息（Internal Messages）

通过 `metadata.internal = True` 标记的消息会被内置转换自动过滤，不会发送给 LLM，但保留在 ContextManager 中：

```python
from kagent.domain.entities import Message
from kagent.domain.enums import Role

# 在 context 中保存调试信息（不发给 LLM）
agent._agent.context.add_message(
    Message(
        role=Role.USER,
        content="Debug: current state is X",
        metadata={"internal": True},
    )
)

# 正常对话 — LLM 不会看到上面的 debug 消息
result = await agent.run("Continue our discussion.")
```

### 优先级

`priority` 参数控制转换执行顺序 — **数值越高越先执行**。相同优先级按注册顺序执行：

```python
@agent.transform(priority=10)
async def runs_first(messages):
    return messages

@agent.transform(priority=0)
async def runs_second(messages):
    return messages
```

内置转换使用 `priority=-100`，确保用户转换始终优先执行。

### 非装饰器方式注册

```python
async def my_transform(messages: list[Message]) -> list[Message]:
    return [m for m in messages if m.role != Role.SYSTEM or m.content != "outdated"]

agent.transform(my_transform, priority=5)
```

### 与拦截器的关系

| | `@agent.transform` | `@agent.intercept("before_prompt_build")` |
|---|---|---|
| 输入/输出 | `list[Message] → list[Message]` | `dict(messages, tool_definitions) → dict` |
| 语义 | 专注消息过滤/变换 | 通用数据修改（含 tools） |
| 执行顺序 | transformer 先于 interceptor | interceptor 在 transformer 之后 |

两者可以同时使用：transformer 做消息过滤，interceptor 做 tool 列表调整。

---

## 10. Builder 模式

除了直接构造 `KAgent`，还可以使用 `KAgentBuilder` 进行链式配置：

```python
import asyncio
from kagent import KAgentBuilder, configure

async def main():
    configure(api_key="your-api-key")

    agent = (
        KAgentBuilder()
        .model("anthropic:claude-opus-4-5-20251101")
        .system_prompt("You are a concise assistant.")
        .max_turns(20)
        .temperature(0.5)
        .max_tokens(1024)
        .build()
    )

    result = await agent.run("Hello!")
    print(result.content)

asyncio.run(main())
```

### Builder 方法

| 方法 | 参数 | 说明 |
|------|------|------|
| `.model(str)` | 模型标识 | `"provider:model_name"` |
| `.system_prompt(str)` | 系统提示词 | |
| `.max_turns(int)` | 最大轮次 | |
| `.temperature(float)` | 温度参数 | |
| `.max_tokens(int)` | 最大输出 token | |
| `.max_context_tokens(int)` | 上下文窗口大小 | |
| `.api_key(str)` | API key | |
| `.base_url(str)` | Base URL | |
| `.build()` | — | 构建并返回 `KAgent` |

所有方法（除 `.build()`）返回 `self`，支持链式调用。

---

## 11. 状态管理

KAgent 提供 `ContextManager` 和 `StateManager` 用于管理对话上下文和运行时状态。

### StateManager：键值状态 + 事件广播

```python
import asyncio
from kagent.context.state import StateManager
from kagent.context.stores.memory import InMemoryStateStore
from kagent.domain.enums import EventType
from kagent.domain.events import Event
from kagent.events.bus import EventBus

async def main():
    bus = EventBus()
    store = InMemoryStateStore()
    state_mgr = StateManager(store, bus)

    # 监听状态变更事件
    async def on_state_change(event: Event):
        key = event.payload.get("key")
        old = event.payload.get("old_value")
        new = event.payload.get("new_value")
        print(f"State changed: {key} = {old} -> {new}")

    bus.subscribe(EventType.AGENT_STATE_CHANGED.value, on_state_change)

    # CRUD 操作（每次操作都会通过 EventBus 广播事件）
    await state_mgr.set("user_name", "Alice")
    await state_mgr.set("turn_count", 0)
    await state_mgr.update("turn_count", 1)
    await state_mgr.delete("user_name")

    print(await state_mgr.get("turn_count"))   # 1
    print(await state_mgr.get("user_name"))    # None

asyncio.run(main())
```

### ContextManager：快照与恢复

```python
from kagent.context.manager import ContextManager
from kagent.domain.entities import Message
from kagent.domain.enums import Role

ctx = ContextManager(max_tokens=10000)
ctx.add_message(Message(role=Role.SYSTEM, content="You are helpful."))
ctx.add_message(Message(role=Role.USER, content="Hello!"))
ctx.add_message(Message(role=Role.ASSISTANT, content="Hi there!"))
ctx.update_context(session_id="abc123")

# 快照
snapshot = ctx.snapshot()

# 恢复到新的 ContextManager
ctx2 = ContextManager()
ctx2.restore(snapshot)

print(len(ctx2.get_all_messages()))  # 3
print(ctx2.get_context())            # {"session_id": "abc123"}
```

---

## 12. API 参考速查

### 核心导出（`from kagent import ...`）

| 名称 | 类型 | 说明 |
|------|------|------|
| `KAgent` | class | 主要入口 Facade |
| `KAgentBuilder` | class | 链式构建器 |
| `configure()` | function | 设置全局配置 |
| `get_config()` | function | 获取当前全局配置 |
| `tool` | decorator | 独立 tool 装饰器 |

### KAgent 方法速查

```python
agent = KAgent(model="openai:gpt-5")

# 注册 tool
@agent.tool
async def my_tool(x: str) -> str: ...

# 注册事件钩子
@agent.on("event.pattern")
async def handler(event): ...

# 注册拦截器
@agent.intercept("before_llm_request")
async def modify(request):
    return request

# 注册上下文转换
@agent.transform
async def filter_msgs(messages):
    return [m for m in messages if not m.metadata.get("internal")]

# 运行
result = await agent.run("input")                              # 非流式
result = await agent.run("input", response_model=MyModel)      # 结构化输出
async for chunk in agent.stream("input"):                      # 流式
    ...

# Steering
await agent.steer("directive")          # 重定向：注入新指令，循环继续
await agent.abort("reason")             # 中止：循环在下一轮次边界退出
reply = await agent.interrupt("prompt") # 暂停：阻塞直到用户回复，返回 str
await agent.resume("user_input")        # 恢复：提供用户输入，解除暂停
```

### KAgent 构造参数速查

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | `str` | `"openai:gpt-4o"` | 模型标识 `"provider:model_name"` |
| `system_prompt` | `str` | `"You are a helpful assistant."` | 系统提示词 |
| `max_turns` | `int` | `10` | 最大循环轮次 |
| `max_tool_retries` | `int` | `3` | 同一工具连续失败上限（熔断器） |
| `temperature` | `float \| None` | `None` | 温度参数 |
| `max_tokens` | `int \| None` | `None` | 最大输出 token 数 |
| `tool_choice` | `str \| None` | `None` | 工具选择策略 |
| `max_context_tokens` | `int` | `128_000` | 上下文窗口 token 预算 |
| `api_key` | `str \| None` | `None` | API 密钥（覆盖全局配置） |
| `base_url` | `str \| None` | `None` | API 基础 URL（覆盖全局配置） |
```

### 事件模式速查

```python
@agent.on("agent.*")           # 所有 agent 生命周期事件
@agent.on("llm.*")             # 所有 LLM 事件
@agent.on("tool.*")            # 所有 tool 事件
@agent.on("steering.*")        # 所有 steering 事件
@agent.on("*")                 # 所有事件

@agent.on("tool.call.completed")        # 精确匹配
@agent.on("llm.response.received")      # 精确匹配
```

### 拦截器 Hook 速查

```python
@agent.intercept("before_prompt_build")   # 过滤 tools / 修改 messages
@agent.intercept("before_llm_request")    # 修改 LLM 请求参数
@agent.intercept("after_llm_response")    # 审查 / 修改模型输出
@agent.intercept("before_tool_call")      # 修改参数 / 阻止执行
@agent.intercept("after_tool_call")       # 修改 / 脱敏 tool 结果
@agent.intercept("after_tool_round")      # 查看本轮所有 tool 结果
@agent.intercept("before_return")         # 添加 metadata / 修改最终输出
```
