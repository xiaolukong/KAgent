# KAgent v2 升级设计：对标 Pi Agent Core

## 0. 背景与目标

KAgent v1 已经是一个功能完整的 Agent 运行时：multi-provider LLM 抽象、tool calling、streaming、structured output、Pub/Sub 事件系统、steering 控制。但与 Pi Agent Core 相比，在**可扩展性深度**和**底层数据结构**上有明显差距。

本文档针对四个核心痛点给出设计方案。设计原则：

1. **微内核（Microkernel）路线** — 保持 AgentLoop 极简，扩展能力通过注入点暴露
2. **向后兼容** — 现有 `@agent.tool`、`@agent.on()`、`agent.run()` 接口不变
3. **渐进式** — 四个改进可以独立实施，互不依赖
4. **不引入 DAG 引擎** — 这不是 LangGraph，保持纯 Agent 运行时的定位

---

## 1. 拦截器管道（Interceptor Pipeline）

### 1.1 问题

KAgent 的 `@agent.on()` 是 Pub/Sub 模式 — 事件发出后，handler 只能**观察**，无法**修改**流经 AgentLoop 的数据。以下场景无法实现：

- 在 LLM 请求发出前，动态过滤 tools 列表
- 在 LLM 响应返回后、写入 context 前，审查或修改内容
- 在 tool 执行前拦截并阻止危险操作
- 在 tool 执行后修改返回值（如脱敏、格式化）
- 在构建 prompt 前对 message 列表做变换

Pi Agent Core 的 coding-agent 层提供了 30+ 个 hook 点，使用四种拦截模式（blocking、mutation、transform chain、accumulation）。KAgent 需要一个统一的拦截器抽象来实现等价能力。

### 1.2 设计

引入 `InterceptorPipeline`：一个有序的 handler 链，每个 handler **接收数据、可修改、返回数据**，下一个 handler 接收上一个的输出。

#### 拦截点（Hook Points）

在 AgentLoop 的关键节点插入 7 个拦截点：

```
用户输入
  │
  ▼
① before_prompt_build(messages, tool_defs) -> (messages, tool_defs)
  │   可以：过滤/重排 message、增删 tool 定义、注入动态 system prompt
  ▼
  PromptBuilder.build()
  │
  ▼
② before_llm_request(ModelRequest) -> ModelRequest
  │   可以：修改 temperature/max_tokens、改写 model 名、注入 response_format
  ▼
  ModelProvider.complete() / stream()
  │
  ▼
③ after_llm_response(ModelResponse) -> ModelResponse
  │   可以：审查内容、过滤 tool_calls、注入 metadata、内容脱敏
  ▼
  写入 ContextManager
  │
  ▼（如果有 tool_calls）
④ before_tool_call(tool_name, arguments) -> (tool_name, arguments) | BLOCK
  │   可以：修改参数、重命名 tool、返回 BLOCK 阻止执行
  ▼
  ToolExecutor.execute()
  │
  ▼
⑤ after_tool_call(tool_name, ToolResult) -> ToolResult
  │   可以：修改返回值、脱敏、格式化、标记错误
  ▼
  写入 ContextManager（tool message）
  │
  ▼（所有 tool 执行完毕后）
⑥ after_tool_round(tool_results: list) -> list
  │   可以：聚合多个 tool 结果、注入额外 context
  ▼
  继续下一轮循环或结束
  │
  ▼（循环结束时）
⑦ before_return(ModelResponse) -> ModelResponse
  │   可以：后处理最终结果、日志记录
  ▼
  返回给调用者
```

#### 拦截器类型

```python
from typing import TypeVar, Generic, Union

T = TypeVar("T")

class InterceptResult(Generic[T]):
    """拦截器返回值：继续传递或阻止。"""
    data: T
    blocked: bool = False
    block_reason: str | None = None

# 拦截器签名
InterceptorFn = Callable[[T], Awaitable[T | InterceptResult[T]]]
```

三种返回语义：
- **返回修改后的数据**：传递给下一个 handler
- **返回 `InterceptResult(blocked=True)`**：阻止执行（仅 `before_tool_call` 支持）
- **抛出异常**：中断管道，异常传播给调用方

#### Pipeline 实现

```python
class InterceptorPipeline:
    """管理所有拦截点的有序 handler 链。"""

    def __init__(self) -> None:
        self._chains: dict[str, list[tuple[int, InterceptorFn]]] = {}

    def add(self, hook: str, fn: InterceptorFn, *, priority: int = 0) -> str:
        """注册拦截器，返回 ID 用于移除。priority 越大越先执行。"""
        ...

    def remove(self, interceptor_id: str) -> None: ...

    async def run(self, hook: str, data: T) -> T:
        """依次执行 hook 上的所有 handler，串联传递 data。"""
        for _, fn in sorted(self._chains.get(hook, []), reverse=True):
            result = await fn(data)
            if isinstance(result, InterceptResult):
                if result.blocked:
                    raise InterceptBlockedError(result.block_reason)
                data = result.data
            else:
                data = result
        return data
```

#### AgentLoop 集成

以 `before_llm_request` 为例，在 loop.py 中修改：

```python
# 现有代码
request = self._prompt_builder.build(messages, tool_defs)
response = await self._provider.complete(request)

# 改为
request = self._prompt_builder.build(messages, tool_defs)
request = await self._pipeline.run("before_llm_request", request)       # ← 新增
response = await self._provider.complete(request)
response = await self._pipeline.run("after_llm_response", response)     # ← 新增
```

#### 用户侧 API

```python
# 方式 1：装饰器
@agent.intercept("before_llm_request")
async def safe_mode_filter(request: ModelRequest) -> ModelRequest:
    if "安全模式" in str(request.messages):
        request.tools = [t for t in (request.tools or []) if t.name == "safe_tool"]
    return request

# 方式 2：直接注册
agent.intercept("after_tool_call", my_handler, priority=10)

# 方式 3：阻止 tool 执行
@agent.intercept("before_tool_call")
async def block_dangerous(ctx):
    if ctx.tool_name == "rm_rf":
        return InterceptResult(data=ctx, blocked=True, block_reason="Dangerous tool blocked")
    return ctx
```

#### 与现有 EventBus 的关系

**Interceptor 和 EventBus 共存，职责不同：**

| | EventBus (`@agent.on`) | Interceptor (`@agent.intercept`) |
|---|---|---|
| 模式 | Pub/Sub（异步广播） | Pipeline（串行传递） |
| 数据流 | 单向通知 | 双向（接收 → 修改 → 返回） |
| 用途 | 日志、监控、UI 推送、审计 | 流程控制、数据变换、权限检查 |
| 执行时机 | 事件发生后 | 事件发生前/后的关键节点 |
| 失败影响 | handler 异常不影响主流程 | handler 异常中断主流程 |

两者在 AgentLoop 中的位置：

```
EventBus.publish(AGENT_LOOP_ITERATION)     ← 通知（只读）
  ↓
pipeline.run("before_prompt_build", ...)   ← 拦截（可写）
  ↓
PromptBuilder.build()
  ↓
pipeline.run("before_llm_request", ...)    ← 拦截（可写）
  ↓
EventBus.publish(LLM_REQUEST_SENT)         ← 通知（只读）
  ↓
provider.complete()
  ↓
pipeline.run("after_llm_response", ...)    ← 拦截（可写）
  ↓
EventBus.publish(LLM_RESPONSE_RECEIVED)    ← 通知（只读）
```

### 1.3 涉及文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `kagent/agent/interceptor.py` | **新建** | InterceptorPipeline、InterceptResult、InterceptBlockedError |
| `kagent/agent/loop.py` | **修改** | 在 7 个节点调用 pipeline.run() |
| `kagent/agent/agent.py` | **修改** | 持有 pipeline 实例，暴露注册方法 |
| `kagent/interface/kagent.py` | **修改** | 暴露 `@agent.intercept()` 装饰器 |
| `kagent/domain/protocols.py` | **修改** | 可选：定义 IInterceptorPipeline protocol |

---

## 2. 上下文转换管道（Context Transform Pipeline）

### 2.1 问题

KAgent 的 ContextManager 是扁平的：所有 Message 存入同一个列表，PromptBuilder 直接将它们构建成 ModelRequest。没有 App Message 与 LLM Message 的分野。

Pi Agent Core 使用 `AgentMessage`（可以包含任意自定义类型如 `BashExecutionMessage`、`BranchSummaryMessage`）+ `convertToLlm()` 边界函数，在发送给 LLM 前过滤和转换。

KAgent 的问题：
- 如果 tool 执行中间产生了内部调试信息，想保存在 context 中但不发给 LLM，没有机制
- 如果想根据当前 turn 动态调整 system prompt（如注入当前工作目录、当前时间），需要修改 PromptBuilder
- 如果想对 message 做压缩/摘要再发给 LLM，没有干净的注入点

### 2.2 设计

采用两层分离 + 转换管道的架构：

#### Message 标记扩展

在现有 `Message.metadata` 字段中使用约定键：

```python
# 标记内部消息（不发送给 LLM）
Message(role=Role.USER, content="debug info", metadata={"internal": True})

# 标记特殊类型
Message(role=Role.USER, content="...", metadata={"message_type": "compaction_summary"})
```

不需要新建 Message 子类 — 利用现有的 `metadata: dict[str, Any]` 字段即可。这比 TypeScript 的 declaration merging 更 Pythonic。

#### 转换管道

在 PromptBuilder 和 AgentLoop 之间引入 `ContextTransformer`：

```python
class ContextTransformer:
    """将 App 层 Message 列表转换为 LLM 层 Message 列表。"""

    def __init__(self) -> None:
        self._transforms: list[TransformFn] = []

    def add(self, fn: TransformFn, *, priority: int = 0) -> None:
        """注册转换函数。"""
        ...

    async def apply(self, messages: list[Message]) -> list[Message]:
        """依次执行所有转换函数。"""
        result = list(messages)
        for fn in self._transforms:
            result = await fn(result)
        return result

# 转换函数签名
TransformFn = Callable[[list[Message]], Awaitable[list[Message]]]
```

#### 内置默认转换

```python
async def filter_internal_messages(messages: list[Message]) -> list[Message]:
    """默认转换：过滤标记为 internal 的消息。"""
    return [m for m in messages if not m.metadata.get("internal")]
```

这个默认转换始终存在。用户可以添加更多转换：

```python
@agent.transform
async def inject_time_context(messages: list[Message]) -> list[Message]:
    """在 system message 后注入当前时间。"""
    from datetime import datetime
    time_msg = Message(
        role=Role.SYSTEM,
        content=f"Current time: {datetime.now().isoformat()}",
    )
    return [messages[0], time_msg] + messages[1:]
```

#### AgentLoop 集成

```python
# 现有
messages = self._context.get_messages()
request = self._prompt_builder.build(messages, tool_defs)

# 改为
messages = self._context.get_messages()
messages = await self._transformer.apply(messages)           # ← 新增
request = self._prompt_builder.build(messages, tool_defs)
```

**注意**：转换后的 messages 仅用于构建当前 LLM 请求，不会写回 ContextManager。ContextManager 始终保留完整的 App 层历史。

#### 与拦截器的关系

`ContextTransformer` 实际上是 `before_prompt_build` 拦截点的一个特化版本。如果实施了 §1 的拦截器管道，转换管道可以作为 `before_prompt_build` 的默认拦截器实现。但独立设计也有价值：

- 拦截器是通用管道，API 是 `(data) -> data`
- 转换器语义更清晰：`(messages) -> messages`，专门处理消息过滤/变换
- 用户可以同时使用两者

建议实施路径：先独立实现 `ContextTransformer`，后续如果需要可以将其底层迁移到拦截器管道上。

### 2.3 涉及文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `kagent/context/transformer.py` | **新建** | ContextTransformer 类 |
| `kagent/agent/loop.py` | **修改** | 在 build 前调用 transformer.apply() |
| `kagent/agent/agent.py` | **修改** | 持有 transformer，暴露注册方法 |
| `kagent/interface/kagent.py` | **修改** | 暴露 `@agent.transform` 装饰器 |

---

## 3. Reasoning/Thinking 流式拆分

### 3.1 问题

现代推理模型（DeepSeek-R1、Claude Extended Thinking、OpenAI o1/o3）在响应中包含"思考过程"。KAgent 的 `StreamChunkType` 只有 `TEXT_DELTA` 和 `TOOL_CALL_*`，没有区分 thinking 和 text。接入推理模型时，思考过程和最终回复会混为一体。

Pi Agent Core 在流式事件中明确区分 `thinking_start`/`thinking_delta`/`thinking_end` 和 `text_start`/`text_delta`/`text_end`，并在 `AssistantMessage.content` 中用 `ThinkingContent` 和 `TextContent` 两种 block 存储。

### 3.2 设计

#### 新增 StreamChunkType

```python
class StreamChunkType(StrEnum):
    TEXT_DELTA = "text_delta"
    THINKING_DELTA = "thinking_delta"           # ← 新增
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_DELTA = "tool_call_delta"
    TOOL_CALL_END = "tool_call_end"
    METADATA = "metadata"
```

#### StreamChunk 扩展

```python
class StreamChunk(BaseModel):
    chunk_type: StreamChunkType
    content: str | None = None               # TEXT_DELTA 内容
    thinking: str | None = None              # THINKING_DELTA 内容（新增）
    tool_call: ToolCallChunk | None = None
    usage: TokenUsage | None = None
    parsed: Any = None
    metadata: dict[str, Any] | None = None
```

#### Message 扩展

在 `Message.metadata` 中存储 thinking 内容：

```python
# assistant message with thinking
Message(
    role=Role.ASSISTANT,
    content="Final answer text",
    metadata={
        "thinking": "Let me reason through this step by step...",
        "thinking_tokens": 1500,
    },
)
```

不新增 Message 字段，利用已有的 `metadata` 字段。这保持了 Message 模型的简洁性。

#### Provider 层修改

##### Anthropic Provider

Anthropic 的 Extended Thinking 返回 `content_block_start` 类型为 `thinking`：

```python
# anthropic_provider.py _do_stream_native() 中：
if block.type == "thinking":
    yield StreamChunk(
        chunk_type=StreamChunkType.THINKING_DELTA,
        thinking=block.thinking,
    )
elif block.type == "text":
    yield StreamChunk(
        chunk_type=StreamChunkType.TEXT_DELTA,
        content=block.text,
    )
```

##### OpenAI Provider

OpenAI o1/o3 的 reasoning tokens 目前不在 streaming delta 中暴露文本（仅在 usage 中报告 `reasoning_tokens`）。但部分兼容代理可能暴露，预留处理：

```python
# openai_provider.py _do_stream() 中：
if hasattr(delta, "reasoning_content") and delta.reasoning_content:
    yield StreamChunk(
        chunk_type=StreamChunkType.THINKING_DELTA,
        thinking=delta.reasoning_content,
    )
```

#### AgentLoop 收集逻辑

在 `run_stream()` 中新增 thinking 收集：

```python
collected_thinking: list[str] = []

async for chunk in self._provider.stream(request):
    yield chunk
    if chunk.chunk_type == StreamChunkType.TEXT_DELTA and chunk.content:
        collected_content.append(chunk.content)
    elif chunk.chunk_type == StreamChunkType.THINKING_DELTA and chunk.thinking:
        collected_thinking.append(chunk.thinking)
    # ... 现有 tool_call 处理

# 构建 assistant message 时：
assistant_msg = Message(
    role=Role.ASSISTANT,
    content=full_content,
    tool_calls=tool_calls or None,
    metadata={"thinking": "".join(collected_thinking)} if collected_thinking else None,
)
```

在 `run()`（非流式）中，从 `ModelResponse.metadata` 提取 thinking：

```python
response = await self._provider.complete(request)
# 如果 provider 在 response.metadata 中放了 thinking，传递到 message 里
thinking = (response.metadata or {}).get("thinking")
assistant_msg = Message(
    role=Role.ASSISTANT,
    content=response.content,
    tool_calls=response.tool_calls,
    metadata={"thinking": thinking} if thinking else None,
)
```

#### 用户侧使用

```python
# 流式模式：分别处理 thinking 和 text
async for chunk in agent.stream("Solve this math problem"):
    if chunk.thinking:
        print(f"[thinking] {chunk.thinking}", end="")
    elif chunk.content:
        print(chunk.content, end="", flush=True)

# 非流式模式：从 response 中获取 thinking
result = await agent.run("Solve this")
thinking = (result.metadata or {}).get("thinking")
if thinking:
    print(f"Thinking: {thinking}")
print(f"Answer: {result.content}")
```

#### ContextTransformer 中的 Thinking 处理

默认情况下，thinking 内容**不应发送给 LLM**（它是模型的内部推理过程）。在 §2 的 ContextTransformer 中添加默认转换：

```python
async def strip_thinking_from_context(messages: list[Message]) -> list[Message]:
    """剥离 thinking 内容，避免发送给 LLM。"""
    result = []
    for m in messages:
        if m.metadata and "thinking" in m.metadata:
            # 保留 message 但移除 thinking
            cleaned = m.model_copy()
            cleaned.metadata = {k: v for k, v in m.metadata.items() if k != "thinking"}
            result.append(cleaned)
        else:
            result.append(m)
    return result
```

### 3.3 涉及文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `kagent/domain/enums.py` | **修改** | 添加 `THINKING_DELTA` |
| `kagent/domain/model_types.py` | **修改** | StreamChunk 添加 `thinking` 字段 |
| `kagent/models/anthropic_provider.py` | **修改** | native stream 中产出 THINKING_DELTA |
| `kagent/models/openai_provider.py` | **修改** | 预留 reasoning_content 处理 |
| `kagent/models/openai_compat.py` | **修改** | 代理模式中预留 thinking 处理 |
| `kagent/agent/loop.py` | **修改** | 收集 thinking chunks，写入 message metadata |

---

## 4. 树状会话历史（Tree-based Context）

### 4.1 问题

KAgent 的 ContextManager 是线性追加的 `list[Message]`。一旦中间某个 turn 产生了幻觉或无关的 tool 调用，这段历史永远残留。没有回滚（rollback）或分支（branching）能力。

Pi Agent Core 使用 append-only JSONL + parentId 指针构建树状结构，支持 branch、checkout、compaction。

### 4.2 设计

#### 核心数据结构：Turn

一个 Turn 代表一次完整的 agent 循环迭代（user → assistant → tool results）。每个 Turn 有一个唯一 ID 和指向父 Turn 的指针。

```python
@dataclass
class Turn:
    """一次完整的 agent 循环迭代。"""
    id: str                                # 唯一标识（UUID 或递增 ID）
    parent_id: str | None                  # 父 Turn ID（根节点为 None）
    messages: list[Message]                # 本轮的所有 message
    metadata: dict[str, Any] | None = None # 附加信息（如 token 用量、时间戳）
    timestamp: datetime                    # 创建时间
```

#### ContextTree

```python
class ContextTree:
    """树状会话管理器。"""

    def __init__(self) -> None:
        self._turns: dict[str, Turn] = {}  # id -> Turn
        self._head: str | None = None      # 当前活跃分支的最新 Turn ID
        self._root: str | None = None      # 根 Turn ID

    # ── 追加 ──

    def append_turn(self, messages: list[Message], **meta) -> Turn:
        """在当前 head 后追加一个新 Turn。"""
        turn = Turn(
            id=generate_id(),
            parent_id=self._head,
            messages=messages,
            metadata=meta,
            timestamp=utcnow(),
        )
        self._turns[turn.id] = turn
        self._head = turn.id
        if self._root is None:
            self._root = turn.id
        return turn

    # ── 读取 ──

    def get_branch(self, turn_id: str | None = None) -> list[Turn]:
        """从 root 到指定 Turn（默认 head）的线性路径。"""
        target = turn_id or self._head
        path = []
        current = target
        while current is not None:
            turn = self._turns[current]
            path.append(turn)
            current = turn.parent_id
        path.reverse()
        return path

    def get_messages(self, turn_id: str | None = None) -> list[Message]:
        """获取从 root 到指定 Turn 的所有 message（展平）。"""
        return [m for t in self.get_branch(turn_id) for m in t.messages]

    def get_children(self, turn_id: str) -> list[Turn]:
        """获取指定 Turn 的所有子 Turn（分支）。"""
        return [t for t in self._turns.values() if t.parent_id == turn_id]

    # ── 分支 ──

    def branch(self, from_turn_id: str | None = None) -> str:
        """创建分支点：将 head 移到 from_turn_id，后续 append 将创建新分支。
        返回新的 head ID。
        """
        target = from_turn_id or self._head
        if target not in self._turns:
            raise ValueError(f"Turn {target} not found")
        self._head = target
        return target

    def checkout(self, turn_id: str) -> None:
        """切换到指定 Turn（类似 git checkout）。"""
        if turn_id not in self._turns:
            raise ValueError(f"Turn {turn_id} not found")
        self._head = turn_id

    # ── 信息 ──

    @property
    def head(self) -> str | None:
        return self._head

    @property
    def turn_count(self) -> int:
        """当前分支的 turn 数量。"""
        return len(self.get_branch())

    def get_tree(self) -> dict:
        """返回完整树结构（用于调试/可视化）。"""
        ...

    # ── 持久化 ──

    def to_jsonl(self) -> str:
        """序列化为 append-only JSONL 格式。"""
        ...

    @classmethod
    def from_jsonl(cls, data: str) -> ContextTree:
        """从 JSONL 反序列化。"""
        ...

    # ── 快照 ──

    def snapshot(self) -> dict:
        """导出当前状态（用于 ContextManager 兼容）。"""
        ...

    def restore(self, snapshot: dict) -> None:
        """从快照恢复。"""
        ...
```

#### 与 ContextManager 的关系

**不替换 ContextManager，而是作为可选的底层存储。**

现有 ContextManager 的 `_messages: list[Message]` 仍然是默认模式。ContextTree 作为一个独立的高级特性，用户可以选择启用：

```python
# 方式 1：传统线性模式（默认，不变）
agent = KAgent(model="openai:gpt-5")
result = await agent.run("Hello")

# 方式 2：启用树状会话
agent = KAgent(model="openai:gpt-5", context_mode="tree")

result1 = await agent.run("Research quantum computing")
# Turn 1: [user: "Research...", assistant: "...", tool: "..."]

checkpoint = agent.context.head  # 保存当前位置

result2 = await agent.run("Now research black holes")
# Turn 2: [user: "Now research...", assistant: "..."]

# 回滚到 Turn 1，创建新分支
agent.context.branch(checkpoint)
result3 = await agent.run("Instead, research AI ethics")
# Turn 3: 从 Turn 1 分出的新分支
```

#### AgentLoop 集成

AgentLoop 不直接感知 ContextTree — 它只需要 `messages: list[Message]`。ContextManager 在 tree 模式下的 `get_messages()` 返回当前 branch 的展平消息列表。对 AgentLoop 完全透明。

```python
class ContextManager:
    def __init__(self, *, max_tokens: int = 128_000, mode: str = "linear") -> None:
        self._mode = mode
        if mode == "tree":
            self._tree = ContextTree()
        else:
            self._messages: list[Message] = []
        ...

    def get_messages(self) -> list[Message]:
        if self._mode == "tree":
            raw = self._tree.get_messages()
        else:
            raw = list(self._messages)
        return self._window.trim(raw, self._max_tokens)

    def add_message(self, message: Message) -> None:
        if self._mode == "tree":
            # 追加到当前 turn 的 pending messages
            self._pending_messages.append(message)
        else:
            self._messages.append(message)

    def commit_turn(self) -> None:
        """Tree 模式：将 pending messages 提交为一个 Turn。"""
        if self._mode == "tree" and self._pending_messages:
            self._tree.append_turn(self._pending_messages)
            self._pending_messages = []
```

AgentLoop 在每轮结束时调用 `commit_turn()`（线性模式下该方法为空操作）。

#### Compaction（压缩）

长会话的 token 超出预算时，可以对早期 turns 做摘要压缩：

```python
async def compact(self, summarizer: Callable[[list[Message]], Awaitable[str]]) -> None:
    """将早期 turns 压缩为一条摘要消息。"""
    branch = self._tree.get_branch()
    if len(branch) <= 2:
        return  # 太少，不需要压缩

    # 保留最近 N 个 turns
    keep_recent = 3
    to_compact = branch[:-keep_recent]
    messages_to_compact = [m for t in to_compact for m in t.messages]

    # 调用 summarizer（通常是一次 LLM 调用）
    summary = await summarizer(messages_to_compact)

    # 创建 compaction turn（替换被压缩的 turns）
    compaction_msg = Message(
        role=Role.SYSTEM,
        content=f"[Context Summary]\n{summary}",
        metadata={"message_type": "compaction_summary", "compacted_turns": len(to_compact)},
    )
    # ... 重构树结构
```

### 4.3 涉及文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `kagent/context/tree.py` | **新建** | ContextTree、Turn |
| `kagent/context/manager.py` | **修改** | 支持 tree 模式，添加 commit_turn()、branch()、checkout() |
| `kagent/agent/loop.py` | **修改** | 每轮结束调用 commit_turn() |
| `kagent/interface/kagent.py` | **修改** | 暴露 context_mode 参数，暴露 branch/checkout API |

---

## 5. 实施优先级与依赖关系

四个改进的独立性和优先级：

```
优先级 1 ─── §1 拦截器管道 ──────── 基础设施，§2 可以构建在其上
             │
优先级 2 ─── §3 Reasoning 拆分 ──── 独立，与其他模块无依赖
             │
优先级 3 ─── §2 上下文转换 ──────── 可独立实施，也可基于 §1
             │
优先级 4 ─── §4 树状会话 ──────── 最复杂，可独立实施
```

建议实施顺序：**§1 → §3 → §2 → §4**

理由：
- **§1 拦截器管道**是最基础的能力，一旦有了它，很多需求都可以通过拦截器实现，不需要修改核心代码
- **§3 Reasoning** 是当前最急迫的市场需求（o1/o3/R1/Extended Thinking 已是主流），且改动量最小
- **§2 上下文转换**在 §1 的基础上可以非常自然地实现
- **§4 树状会话**最复杂，且是锦上添花而非必需品

### 工作量估计

| 改进 | 新文件 | 修改文件 | 预估复杂度 |
|------|--------|----------|-----------|
| §1 拦截器管道 | 1 | 4 | ★★★ 中等 |
| §2 上下文转换 | 1 | 3 | ★★ 较低 |
| §3 Reasoning 拆分 | 0 | 5-6 | ★★ 较低 |
| §4 树状会话 | 1 | 3 | ★★★★ 较高 |

---

## 6. 未纳入范围的事项

以下是用户提到但**有意不纳入本次设计**的内容，以及理由：

### 6.1 Extension System（扩展系统）

Pi Agent Core 的 Extension System（`ExtensionRunner`、extension discovery、declaration merging）是一个**应用层**概念，不属于 agent core。KAgent 的定位是 agent 运行时引擎，不是应用框架。

有了 §1 的拦截器管道 + 现有的 EventBus + `@agent.tool`，开发者已经拥有了构建扩展系统所需的全部基础设施。如果未来需要，可以在 KAgent 之上构建一个 `ExtensionRunner`，但那应该是另一个包。

### 6.2 Session Persistence（会话持久化）

§4 的 ContextTree 提供了树状内存结构和 JSONL 序列化接口，但**不包含**文件系统持久化管理（session 目录、session 切换、session listing）。这些是应用层关注点。

### 6.3 Compaction 的具体实现

§4 提供了 compaction 的接口设计（`compact(summarizer)` 接受一个回调），但具体的 summarizer 实现（调用 LLM 生成摘要、保留文件操作记录等）留给应用层。

### 6.4 UI/TUI 集成

Pi Agent Core 有 `pi-tui` 和 `pi-web-ui`。KAgent 不涉及 UI，这是构建在 KAgent 之上的应用的责任。

---

## 7. 与 Pi Agent Core 的架构对齐总结

| Pi Agent Core 能力 | KAgent v1 | KAgent v2（本方案） |
|---|---|---|
| 25+ 生命周期 hooks（mutable） | ❌ 只有 Pub/Sub（只读） | ✅ §1 拦截器管道（7 个可变拦截点） |
| `convertToLlm()` 边界转换 | ❌ 扁平 context | ✅ §2 ContextTransformer + internal 标记 |
| `thinking_delta` 流式拆分 | ❌ 全部混入 TEXT_DELTA | ✅ §3 THINKING_DELTA + metadata 存储 |
| 树状 session branching | ❌ 线性追加 | ✅ §4 ContextTree + branch/checkout |
| Extension System | `@agent.on()` + `@agent.tool` | §1 拦截器 + 现有 EventBus（足够构建） |
| Session persistence (JSONL) | snapshot/restore (dict) | §4 JSONL 序列化接口 |
| Dual-loop (steering + follow-up) | steering + interrupt | ✅ 已有 |
| Multi-provider LLM | ✅ OpenAI/Anthropic/Gemini | ✅ 不变 |
| Tool calling + streaming | ✅ | ✅ 不变 |
| Structured output | ✅ | ✅ 不变 |
