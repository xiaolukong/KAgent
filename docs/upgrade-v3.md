# KAgent v3 升级设计：工业化加固

## 0. 背景与评估

KAgent v2 已经完成了四项核心升级中的三项：拦截器管道（§1）、上下文转换管道（§2）、Reasoning/Thinking 流式拆分（§3）。框架具备了完整的 Agent 运行时能力：multi-provider LLM 抽象、tool calling、streaming、structured output、Pub/Sub 事件系统、Interceptor 管道、Context Transform、Steering 控制、Thinking 支持。

在此基础上，本文档评估四个「工业化加固」建议，并给出明确的取舍决策和实施方案。

### 设计原则（延续 v2）

1. **微内核路线** — AgentLoop 保持极简，能力通过注入点暴露
2. **向后兼容** — 现有 `@agent.tool`、`@agent.on()`、`@agent.intercept()`、`@agent.transform` 接口不变
3. **单智能体定位** — KAgent 是 Agent 运行时引擎，不是多智能体框架、监控平台、或 UI 层
4. **不引入重依赖** — 可观测性等外部集成以「示例 + 接口」形式提供，不引入 `opentelemetry-sdk` 等重型依赖到核心包

---

## 1. 工具自纠正与防死循环（Tool Self-Correction & Circuit Breaker） ✅ 已实施

### 评价：**强烈推荐 — 必做**

这是 Agent 框架从「demo 能跑」到「生产可用」的分水岭。当前的 ToolExecutor 在参数校验失败时直接 `raise ValidationError`，该异常会向上传播到 AgentLoop 并终止整个 run。这意味着 LLM 只要传错一次参数，整个 Agent 就崩了。

现实中，大模型传错工具参数是**高频事件**（类型错误、缺少必填参数、枚举值拼写错误）。一个工业级框架应该：

1. 将验证错误转化为友好的 `Role.TOOL` 消息塞回 context，让 LLM 自行纠正
2. 对同一工具的连续失败设置熔断阈值，防止死循环

### 1.1 现状分析

```
当前数据流（参数校验失败时）：

ToolWrapper.execute()
  └→ validate_input() raises ValidationError
      └→ 异常直接抛出，ToolResult 不会创建
          └→ ToolExecutor.execute() 不捕获 ValidationError
              └→ AgentLoop 崩溃（无 try/except 保护）
```

关键问题：
- `ToolWrapper.execute()` 第 93 行：`except ValidationError: raise` — 验证错误被特殊处理为不可恢复
- `ToolExecutor.execute()` 没有 `try/except` 包裹 `wrapper.execute()` 的 `ValidationError`
- `AgentLoop` 的 tool 执行块也没有捕获这个异常

### 1.2 设计

#### 1.2.1 ToolExecutor 捕获验证错误

在 `ToolExecutor.execute()` 中捕获 `ValidationError`，将其转化为带有结构化错误信息的 `ToolResult`：

```python
# kagent/tools/executor.py

async def execute(self, tool_name, arguments, *, call_id=None) -> ToolResult:
    # ... 现有 lookup 逻辑 ...

    try:
        result = await wrapper.execute(arguments)
    except ValidationError as exc:
        # 生成 LLM 可理解的纠错提示
        error_msg = self._format_validation_error(exc, tool_name, arguments)
        result = ToolResult(
            tool_call_id=call_id or "",
            tool_name=tool_name,
            error=error_msg,
            status=ToolCallStatus.ERROR,
        )
        await self._event_bus.publish(
            ToolEvent(
                event_type=EventType.TOOL_CALL_ERROR,
                payload={"tool_name": tool_name, "error": error_msg, "error_type": "validation"},
                source="tool_executor",
            )
        )
        return result

    # ... 现有后续逻辑 ...

def _format_validation_error(self, exc: ValidationError, tool_name: str, arguments: dict) -> str:
    """生成对 LLM 友好的参数校验错误信息。"""
    return (
        f"Tool '{tool_name}' parameter validation failed: {exc}\n"
        f"You provided: {json.dumps(arguments, default=str)}\n"
        f"Please fix the parameters and try again."
    )
```

这样 `ValidationError` 不再中断 AgentLoop，而是作为一条 `Role.TOOL` 的错误消息塞回 context，LLM 可以看到并自行修正。

#### 1.2.2 AgentLoop 熔断器（Circuit Breaker）

在 AgentLoop 中跟踪每个工具的连续失败次数，超过阈值后终止循环：

```python
# kagent/agent/loop.py — run() 和 run_stream() 的 tool 执行块中

# 循环开始前初始化
consecutive_errors: dict[str, int] = {}   # tool_name → 连续失败次数
MAX_TOOL_RETRIES = 3                       # 可通过 AgentConfig 配置

# 每次 tool 执行后
if result.status == ToolCallStatus.ERROR:
    consecutive_errors[tc.name] = consecutive_errors.get(tc.name, 0) + 1
    if consecutive_errors[tc.name] >= MAX_TOOL_RETRIES:
        # 注入最终错误消息
        error_msg = Message(
            role=Role.TOOL,
            content=f"Tool '{tc.name}' has failed {MAX_TOOL_RETRIES} consecutive times. "
                    f"Stopping retries. Please use a different approach.",
            tool_call_id=tc.id,
            metadata={"tool_name": tc.name, "internal": False},
        )
        self._context.add_message(error_msg)
        break  # 跳出当前 tool_calls 循环，让 LLM 重新决策
else:
    # 成功则重置计数
    consecutive_errors[tc.name] = 0
```

**设计决策**：熔断后不是 `abort()`（那会终止整个 Agent），而是 `break` 当前 tool round 并告诉 LLM「这个工具不行了，换个办法」。这给了 LLM 一次使用其他策略的机会。

#### 1.2.3 配置项

在 `AgentConfig` 中新增：

```python
class AgentConfig(BaseModel):
    # ... 现有字段 ...
    max_tool_retries: int = 3    # 同一工具连续失败上限
```

### 1.3 涉及文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `kagent/tools/executor.py` | **修改** | 捕获 `ValidationError`，格式化错误信息 |
| `kagent/agent/loop.py` | **修改** | 添加 `consecutive_errors` 跟踪与熔断逻辑 |
| `kagent/agent/config.py` | **修改** | 新增 `max_tool_retries` 配置 |
| `kagent/tools/decorator.py` | **修改** | `ToolWrapper.execute()` 中不再特殊处理 ValidationError（让它作为普通异常被 ToolExecutor 捕获） |

---

## 2. 可观测性与遥测（Observability & Telemetry）

### 评价：**接口有意义，但内置实现不纳入核心**

可观测性对生产系统极其重要，但 KAgent 的定位是**单智能体运行时引擎**，不是监控平台。引入 `opentelemetry-sdk`、`langfuse` 等重型依赖会：

- 增加安装体积和依赖冲突风险
- 绑定特定的监控后端
- 模糊框架的职责边界

**好消息是：KAgent 已经具备了构建可观测性所需的全部基础设施。** EventBus 的 14 种事件 + Interceptor 的 7 个拦截点，覆盖了 Agent 生命周期的每一个关键节点。用户只需要 20-30 行代码就能接入任何监控后端。

### 2.1 我们该做什么

与其在核心包中内置 `LangfuseInterceptor` 或 `OpenTelemetryInterceptor`，更合理的做法是：

1. **提供一个 `UsageCollector` 内置组件** — 收集每次 run 的 token 用量、耗时、tool 调用统计，挂到现有的 `after_llm_response` 和 `after_tool_call` 拦截点上。这是框架应该提供的基础能力。
2. **提供 Example 级别的 Observability 集成示例** — 展示如何用 `@agent.on()` + `@agent.intercept()` 接入 OpenTelemetry / Langfuse / 自定义日志。

### 2.2 设计：UsageCollector

一个轻量级的内置组件，自动收集每次 Agent run 的遥测数据：

```python
# kagent/agent/usage.py

class RunUsage(BaseModel):
    """单次 agent.run() 的用量统计。"""
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    turns: int = 0
    duration_ms: float = 0
    tool_durations: dict[str, float] = {}   # tool_name → 总耗时 ms

class UsageCollector:
    """通过 Interceptor 自动收集 run 级别的用量数据。"""

    def __init__(self) -> None:
        self._current: RunUsage | None = None
        self._history: list[RunUsage] = []

    def register(self, pipeline: InterceptorPipeline) -> None:
        """注册到拦截器管道。"""
        pipeline.add("after_llm_response", self._on_llm_response, priority=-50)
        pipeline.add("after_tool_call", self._on_tool_call, priority=-50)
        pipeline.add("before_return", self._on_before_return, priority=-50)

    @property
    def last_run(self) -> RunUsage | None:
        return self._history[-1] if self._history else None

    @property
    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self._history)
```

### 2.3 设计：Observability Example

```python
# examples/12_observability.py

# 展示如何用 EventBus 实现自定义监控
@agent.on("llm.response.received")
async def log_llm_usage(event):
    usage = event.payload.get("usage")
    if usage:
        print(f"Tokens: {usage['total_tokens']}")

# 展示如何用 Interceptor 实现请求追踪
@agent.intercept("before_llm_request")
async def trace_request(request):
    request.metadata["trace_id"] = str(uuid.uuid4())
    span_start = time.monotonic()
    return request

# 展示如何接入 OpenTelemetry（用户自行安装 opentelemetry-sdk）
# 展示如何接入 Langfuse（用户自行安装 langfuse）
```

### 2.4 涉及文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `kagent/agent/usage.py` | **新建** | RunUsage、UsageCollector |
| `kagent/agent/agent.py` | **修改** | 初始化并注册 UsageCollector |
| `kagent/interface/kagent.py` | **修改** | 暴露 `agent.usage` 属性 |
| `examples/12_observability.py` | **新建** | 可观测性集成示例 |

---

## 3. 异步取消机制（Graceful Cancellation）

### 评价：**有价值但优先级较低 — 可延后**

当前的 `abort()` 机制是「标志位 + 下一个 turn 检查」：

```python
# steering.py
async def _on_abort(self, event):
    self._aborted = True

# loop.py — 每个 turn 开头
if self._steering.is_aborted:
    break
```

这意味着如果 LLM 正在生成一个 60 秒的长响应（例如 Claude 的 Extended Thinking），或者某个 Tool 正在执行一个耗时操作（例如爬取网页），`abort()` 要等到当前操作完成后才生效。

**理论上确实有改进空间**，但实际上：

1. **LLM API 端**：大多数 LLM SDK 不支持优雅取消正在进行的 HTTP 请求。`httpx` 的流式响应可以通过关闭连接来中断，但各 provider SDK 的行为不一致（Anthropic SDK 使用 httpx、OpenAI SDK 使用 httpx、Gemini SDK 使用 google-auth/grpc），要做到跨 provider 统一的取消语义，需要在每个 provider 中维护 `asyncio.Task` 引用，复杂度高且收益有限。
2. **Tool 执行端**：工具函数由用户编写。如果用户的工具是一个简单的 `await httpx.get()`，可以通过 `task.cancel()` 中断。但如果用户的工具内部有数据库事务、文件写入等操作，强行取消反而会造成数据不一致。
3. **流式模式**：在 `run_stream()` 中，abort 的实际延迟已经比较短了——每个 chunk 处理后都可以检查标志位，最多延迟一个 chunk 的时间。

**结论**：在当前阶段，`abort()` 的「下一个 turn 检查」语义是可以接受的。如果未来有明确的「必须立即中断」场景，可以渐进式改进：

### 3.1 渐进式改进路径（不立即实施）

如果未来需要更细粒度的取消，建议分三步：

**Step 1**（低成本）：在 `run_stream()` 的 chunk 循环中增加 abort 检查

```python
async for chunk in self._provider.stream(request):
    if self._steering.is_aborted:
        break  # 立即停止消费流
    yield chunk
```

**Step 2**（中等成本）：为 Tool 执行引入超时机制

```python
@agent.tool(timeout=30)  # 秒
async def slow_tool(): ...

# ToolExecutor 中：
result = await asyncio.wait_for(wrapper.execute(arguments), timeout=tool_timeout)
```

**Step 3**（高成本）：为 Provider 引入 `cancel()` 方法，需要在每个 provider 中管理 task 引用

### 3.2 本次实施范围

仅实施 **Step 1**（流式 abort 检查），因为成本极低（2 行代码）且效果显著。Step 2 和 Step 3 留给后续版本。

### 3.3 涉及文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `kagent/agent/loop.py` | **修改** | `run_stream()` chunk 循环中添加 abort 检查 |

---

## 4. ContextManager 序列化接口标准化

### 评价：**已经具备基础能力 — 需要小幅加固**

当前 `ContextManager` 已经有 `snapshot()` 和 `restore()` 方法：

```python
def snapshot(self) -> dict[str, Any]:
    return {
        "messages": [m.model_dump() for m in self._messages],
        "context": dict(self._context),
    }

def restore(self, data: dict[str, Any]) -> None:
    self._messages = [Message(**m) for m in data.get("messages", [])]
    self._context = dict(data.get("context", {}))
```

这已经**基本够用**。外部 Supervisor Agent 可以用这两个方法将 Agent 的状态存入 Redis / 数据库，或者从快照中恢复。

但有几个可以改进的点：

### 4.1 缺少的能力

1. **版本标记**：snapshot 没有版本号，如果未来 Message 结构变化，旧 snapshot 无法被正确反序列化
2. **增量导出**：每次 `snapshot()` 都导出全量消息，对于长会话不高效
3. **类型安全**：`snapshot()` 返回裸 `dict`，`restore()` 接受裸 `dict`，没有 schema 校验

### 4.2 设计

#### 4.2.1 带版本的 Snapshot

```python
# kagent/context/manager.py

SNAPSHOT_VERSION = 1

class ContextSnapshot(BaseModel):
    """ContextManager 的序列化快照。"""
    version: int = SNAPSHOT_VERSION
    messages: list[dict[str, Any]]
    context: dict[str, Any]
    created_at: str  # ISO 格式时间戳

class ContextManager:
    def snapshot(self) -> ContextSnapshot:
        return ContextSnapshot(
            version=SNAPSHOT_VERSION,
            messages=[m.model_dump() for m in self._messages],
            context=dict(self._context),
            created_at=datetime.now(UTC).isoformat(),
        )

    def restore(self, data: ContextSnapshot | dict[str, Any]) -> None:
        if isinstance(data, dict):
            data = ContextSnapshot(**data)
        if data.version != SNAPSHOT_VERSION:
            raise ValueError(
                f"Snapshot version mismatch: expected {SNAPSHOT_VERSION}, got {data.version}"
            )
        self._messages = [Message(**m) for m in data.messages]
        self._context = dict(data.context)

    # 新增：JSON 字符串级别的便捷方法
    def export_json(self) -> str:
        """导出为 JSON 字符串，方便存入 Redis/数据库。"""
        return self.snapshot().model_dump_json()

    def load_json(self, json_str: str) -> None:
        """从 JSON 字符串恢复。"""
        self.restore(ContextSnapshot.model_validate_json(json_str))
```

#### 4.2.2 向后兼容

- `snapshot()` 现在返回 `ContextSnapshot`（Pydantic BaseModel），但该对象支持 `.model_dump()` 转为 dict，所以 `snapshot().model_dump()` 与旧的 `snapshot()` 返回值结构兼容
- `restore()` 同时接受 `ContextSnapshot` 和 `dict`，保证旧代码无需修改

### 4.3 涉及文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `kagent/context/manager.py` | **修改** | ContextSnapshot 模型、版本化、export_json/load_json |
| `kagent/context/__init__.py` | **修改** | 导出 ContextSnapshot |

---

## 5. 总结与优先级

### 取舍矩阵

| 建议 | 评估 | 纳入 v3？ | 理由 |
|------|------|-----------|------|
| §1 工具自纠正 & 熔断 | **必做** | ✅ 已完成 | 生产环境第一天就会遇到，不做就是 demo |
| §2 可观测性 | **部分做** | ✅ UsageCollector + Example | 核心包提供基础用量收集；Langfuse/OTel 留给用户 |
| §3 异步取消 | **最小做** | ✅ 仅 Step 1 | 流式 abort 检查成本极低；深度取消留给后续版本 |
| §4 序列化标准化 | **小幅做** | ✅ 版本化 + JSON 便捷方法 | 已有基础能力，加固即可 |

### 实施顺序

```
优先级 1 ─── §1 工具自纠正 & 熔断 ──── 影响面最大，核心 loop 改动
             │
优先级 2 ─── §4 序列化标准化 ──────── 改动最小，与 §1 无依赖
             │
优先级 3 ─── §2 UsageCollector ─────── 依赖 Interceptor（已有），独立模块
             │
优先级 4 ─── §3 流式 abort 检查 ────── 2 行代码，随时可做
```

### 工作量估计

| 改进 | 新文件 | 修改文件 | 预估复杂度 |
|------|--------|----------|-----------|
| §1 工具自纠正 & 熔断 | 0 | 4 | ★★★ 中等 |
| §2 UsageCollector | 1 | 2 (+1 example) | ★★ 较低 |
| §3 流式 abort 检查 | 0 | 1 | ★ 极低 |
| §4 序列化标准化 | 0 | 2 | ★★ 较低 |

---

## 6. 未纳入范围的事项

### 6.1 内置 LangfuseInterceptor / OpenTelemetryInterceptor

**不纳入**。理由：
- KAgent 是 Agent 运行时引擎，不应绑定特定监控后端
- 引入 `opentelemetry-sdk`（及其数十个子包）或 `langfuse` 作为核心依赖，违反轻量级原则
- 现有的 EventBus + Interceptor 已经提供了完整的集成点，用户 20 行代码即可接入
- 正确的做法是提供 Example 和文档指引

### 6.2 深度异步取消（Provider 级别 / Task 级别）

**不纳入**。理由：
- 各 LLM SDK 的取消语义不统一，跨 provider 实现成本高
- 强行取消用户工具可能导致数据不一致
- 当前的 turn 级 abort + 流式 chunk 级 abort 已经覆盖大多数场景
- 如果未来有需求，可以渐进式添加 Tool 超时（Step 2）和 Provider cancel（Step 3）

### 6.3 ContextTree（树状会话历史）

树状会话历史是 v2 的 §4，设计方案已存在于 `docs/upgrade-v2.md`。它与 v3 的四个改进正交，可以独立实施。建议在 v3 完成后作为 v4 的主题。

### 6.4 多智能体编排

KAgent 是**单智能体基座**。多智能体编排（Supervisor、Swarm、Graph）应该构建在 KAgent **之上**，作为独立的包或层。KAgent 提供的 `snapshot/restore` + EventBus + Interceptor 已经为上层编排提供了完备的集成点。
