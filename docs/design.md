# KAgent -- Python Agent 调用框架设计计划

## Context

在 `/Users/I758861/Library/CloudStorage/OneDrive-SAPSE/Documents/Working/BAE/KAgent` 目录下（当前为空 git 仓库），构建一个类似 Pi-Agent-Core 的 Python Agent 框架。框架采用 Clean Architecture，通过 Pub/Sub 事件驱动实现模块松耦合，支持多模型提供商、标准化工具引擎、动态上下文管理和类型安全校验。

**本阶段仅输出设计文档（`docs/design.md`），不生成任何代码。**

---

## Part 1: 抽象系统模块设计

### 1.1 Clean Architecture 四层结构

依赖方向严格从外向内：

```
Interface 层 (最外) → Application 层 / Infrastructure 层 → Domain 层 (最内)
```

**各层职责**:

| 层 | 职责 | 关键特征 |
|---|------|----------|
| **Domain 层** | 定义"是什么"——核心 Entity、Protocol 接口、事件类型 | 零外部依赖，纯数据结构与契约定义 |
| **Application 层** | 定义"做什么"——编排用例和流程逻辑 | 只依赖 Domain 层的 Protocol，不知道具体实现 |
| **Infrastructure 层** | 实现"怎么做"——与外部世界的具体交互 | 实现 Domain 层定义的 Protocol，封装第三方 SDK |
| **Interface 层** | 定义"怎么用"——对外 API 表面，组装所有依赖 | 唯一知道所有模块的层，负责依赖注入 |

**Application 层与 Infrastructure 层的关系**: 两者是平级的，都依赖 Domain 层，但互不直接依赖。Application 层通过 Domain 层的 Protocol 接口"指挥"Infrastructure 层的实现，具体的绑定由 Interface 层在运行时注入完成：

```
AgentLoop (Application) ──调用──→ IModelProvider (Domain Protocol)
                                          ↑ 实现
                                OpenAIProvider (Infrastructure)
                                          ↑ 注入
                                KAgent.__init__ (Interface 层负责组装)
```

### 1.2 八大模块总览

| # | 模块 | 所属层 | 核心职责 |
|---|------|--------|----------|
| 1 | `domain` | Domain 层 | 定义所有 Entity、Value Object、Protocol 接口、事件类型。零外部依赖（仅 pydantic） |
| 2 | `events` | Application 层 | Pub/Sub 事件总线——框架的中枢神经系统，所有模块间通信的唯一通道 |
| 3 | `agent` | Application 层 | Agent 核心类 + 编排循环（agent_loop），协调 model→tool→state 全流程 |
| 4 | `models` | Infrastructure 层 | 模型提供者抽象层，封装 OpenAI / Anthropic / Gemini SDK 差异 |
| 5 | `tools` | Application 层 (注册+执行引擎) | decorator 式 Tool 注册、Pydantic Schema 自动推断、Tool 执行引擎 |
| 6 | `context` | Application 层 | 动态上下文管理（token budget 滑动窗口）+ 运行时状态管理 |
| 7 | `interface` | Interface 层 | 对外 Facade API（`KAgent` 入口类 + Builder 模式 + Hooks） |
| 8 | `common` | 横切关注点 | 统一异常体系、结构化日志、配置加载、通用工具函数 |

### 1.3 模块间依赖关系

```
                       +-----------+
                       | interface |          ← Interface 层 (组装 + API 表面)
                       +-----+-----+
                             |
            +----------------+----------------+
            |                                 |
   ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─    ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
      Application 层            Infrastructure 层
   ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─    ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
            |                         |
   +--------v---------+        +------v------+
   |      agent       |        |   models    |
   | (编排: AgentLoop) |        | (OpenAI等)  |
   +--------+---------+        +------+------+
            |                         |
   +--------v---------+               |
   |      events      |               |
   | (Pub/Sub EventBus)|              |
   +--------+---------+               |
            |                         |
   +--------v---------+               |
   |     context      |               |
   | (状态+上下文管理)  |               |
   +--------+---------+               |
            |                         |
   +--------v---------+        +------v------+
   |      tools       |        | (tool 实现)  |
   | (注册表+执行引擎)  |        |             |
   +--------+---------+        +-------------+
            |                         |
   ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
                             |
                      +------v------+
                      |   domain    |          ← Domain 层 (Protocol + Entity)
                      +------+------+
                             |
                      +------v------+
                      |   common    |          ← 横切关注点
                      +-------------+
```

**要点**: Application 层（agent / events / context / tools 的注册与执行逻辑）和 Infrastructure 层（models / tools 的具体实现）是**平级**的，它们都只依赖 Domain 层。Interface 层负责将两者通过依赖注入组装在一起。

### 1.4 各模块详细设计

---

#### 模块 1: `domain` — 领域核心层

**职责**: 定义框架契约（接口 + 数据结构），所有其他层都依赖此层，但此层不依赖任何其他层。

**核心 Entity (Pydantic Model)**:
- `Message` — 统一消息实体 (role / content / metadata / timestamp)
- `Event` — 事件基类 (event_type / payload / source / timestamp / correlation_id)
- `AgentState` — Agent 状态快照 (messages / context / metadata / turn_count)
- `ToolDefinition` — Tool 元数据 (name / description / parameters JSON Schema / return_type)
- `ToolResult` — Tool 执行结果封装
- `StreamChunk` — LLM 流式输出 chunk (text_delta / tool_call_start / tool_call_delta / tool_call_end / metadata)
- `ModelRequest` / `ModelResponse` — 模型调用请求/响应

**核心 Protocol (typing.Protocol + @runtime_checkable)**:
- `IEventBus` — publish() / subscribe() / unsubscribe()
- `IModelProvider` — complete(response_model?) / stream(response_model?) / get_model_info()
  - `complete()` — 非流式调用，返回完整 `ModelResponse`
  - `stream()` — 流式调用，返回 `AsyncIterator[StreamChunk]`
  - 两者都接受可选的 `response_model: type[BaseModel]` 参数，用于结构化输出（见下方说明）
- `ITool` — execute() / validate_input() / get_definition()
- `IStateStore` — get() / set() / update() / delete()
- `IContextManager` — get_context() / update_context() / snapshot() / restore()

**结构化输出 (Structured Output)**:

当调用方传入 `response_model=SomePydanticModel` 时，Provider 负责：
1. 将 Pydantic Model 的 JSON Schema 注入到 LLM 请求中（OpenAI 用 `response_format`，Anthropic 用 tool_use 模拟，Gemini 用 `response_schema`）
2. 将 LLM 返回的 JSON 字符串解析并校验为该 Pydantic Model 实例
3. 校验失败时抛出 `ValidationError`

`ModelResponse` Entity 中增加 `parsed: BaseModel | None` 字段，存放解析后的结构化对象。

```python
# 使用示例
class WeatherInfo(BaseModel):
    city: str
    temperature: float
    unit: Literal["celsius", "fahrenheit"]

# 非流式 + 结构化
response = await provider.complete(request, response_model=WeatherInfo)
weather = response.parsed  # WeatherInfo 实例，类型安全

# 流式 + 结构化（流结束后拼接并解析）
async for chunk in provider.stream(request, response_model=WeatherInfo):
    ...  # 逐 chunk 输出
# 最后一个 chunk 的 parsed 字段携带完整解析结果
```

---

#### 模块 2: `events` — Pub/Sub 事件系统

**职责**: 实现 EventBus，作为所有模块间通信的唯一中枢。

**关键组件**:
- `EventBus` — IEventBus 实现，支持 async publish/subscribe、glob 通配符路由、优先级排序
- `EventFilter` — 事件类型模式匹配 + payload 条件过滤
- `EventMiddleware` — 事件中间件接口（logging / metrics 等）

**四类事件流详解**:

事件系统采用分层命名（dot notation），便于 glob 通配符订阅（如 `subscribe("tool.*", handler)` 监听所有 tool 事件）。

##### 类别 1: Agent 生命周期事件 — 追踪 Agent 从启动到结束的完整生命周期

| 事件 | 触发时机 | 典型 payload | 为什么需要 |
|------|---------|-------------|-----------|
| `agent.started` | `Agent.run()` 被调用时 | `{input, config}` | 标记一次 Agent 执行的起点；外部可用于初始化日志 session、启动计时器 |
| `agent.loop.iteration` | AgentLoop 每轮迭代开始时 | `{turn_number, state_snapshot}` | 允许外部监控循环进度，检测是否陷入无限循环（配合 max_turns） |
| `agent.loop.completed` | AgentLoop 正常退出时 | `{total_turns, final_response}` | 标记执行完成；用于统计耗时、记录最终结果 |
| `agent.error` | Agent 执行过程中任何未捕获异常 | `{error_type, message, traceback}` | 统一错误通知出口；外部可订阅做告警、降级处理 |
| `agent.state.changed` | AgentState 发生任何变更时 | `{old_state, new_state, changed_fields}` | 状态变更的唯一广播通道；UI、调试器、持久化层都可被动订阅而不侵入核心逻辑 |

##### 类别 2: LLM 原语事件 — 追踪与 LLM 的每一次交互

| 事件 | 触发时机 | 典型 payload | 为什么需要 |
|------|---------|-------------|-----------|
| `llm.request.sent` | ModelProvider.stream() 被调用前 | `{model, messages, tools, temperature}` | 记录发出的完整请求；用于调试 prompt、计算 input token、审计 |
| `llm.stream.chunk` | 收到 LLM 的每个流式 chunk 时 | `{chunk: StreamChunk}` | 实现实时流式输出——UI 订阅此事件逐字显示；也用于 SSE 转发 |
| `llm.stream.complete` | 流式响应的所有 chunk 接收完毕 | `{full_content, tool_calls}` | 标记流结束；触发后续的 response 解析和 tool 执行 |
| `llm.response.received` | 完整 ModelResponse 组装完成后 | `{response: ModelResponse, usage: TokenUsage}` | 提供完整响应数据；用于 token 用量统计、cost 计算、日志记录 |
| `llm.error` | LLM 调用失败时（网络错误、rate limit 等） | `{error, retry_count, will_retry}` | 与 agent.error 分离——LLM 错误可能会被 retry 策略消化，不一定导致 Agent 失败 |

##### 类别 3: Tool 事件 — 追踪 Tool 的注册和每一次执行

| 事件 | 触发时机 | 典型 payload | 为什么需要 |
|------|---------|-------------|-----------|
| `tool.registered` | @tool 注册到 ToolRegistry 时 | `{name, definition: ToolDefinition}` | 允许动态感知可用 tool 变化；用于 UI 展示当前工具列表 |
| `tool.call.started` | ToolExecutor 开始执行某个 tool 前 | `{tool_name, arguments, call_id}` | 标记 tool 开始执行；用于 UI 显示"正在调用..."、权限拦截（订阅后可 cancel） |
| `tool.call.completed` | Tool 执行成功返回后 | `{tool_name, arguments, result, duration_ms}` | 记录成功结果；用于将 ToolResult 注入对话上下文、统计执行耗时 |
| `tool.call.error` | Tool 执行抛异常或超时时 | `{tool_name, arguments, error, duration_ms}` | 与 tool.call.completed 对称；错误结果同样需要注入对话让 LLM 感知 |

##### 类别 4: Steering 事件 — 运行时干预 Agent 的执行流

| 事件 | 触发时机 | 典型 payload | 为什么需要 |
|------|---------|-------------|-----------|
| `steering.redirect` | 外部调用 `agent.steer()` 注入新指令时 | `{directive: str}` | 实现"mid-turn steering"——在 Agent 执行中途改变其方向，无需等待当前轮结束 |
| `steering.inject_message` | 外部向 message_queue 注入消息时 | `{message: Message}` | 允许在 Agent 运行时追加上下文（如实时数据推送），不打断当前 tool 执行 |
| `steering.abort` | 外部请求立即终止 Agent 执行时 | `{reason: str}` | 协作式取消机制——AgentLoop 在每轮迭代检查此事件，收到后优雅退出 |

**为什么需要事件系统而不是直接方法调用？**

核心价值是**解耦**。以 tool 执行为例：
- 不用事件：`ToolExecutor` 需要直接引用 `Logger`、`StateManager`、`UI`，每加一个观察者就修改一次 `ToolExecutor` 代码
- 用事件：`ToolExecutor` 只管 `publish("tool.call.completed")`，Logger / StateManager / UI 各自 `subscribe` 即可，互不知道彼此的存在

---

#### 模块 3: `agent` — Agent 核心编排

**职责**: 实现 Agent 生命周期管理和核心循环。

**关键组件**:
- `Agent` — 核心类，持有 event_bus / model_provider / tool_registry / context_manager
- `AgentConfig` — 配置模型 (model / system_prompt / max_turns / temperature)
- `AgentLoop` — 编排循环：prompt 构建 → model 调用 → response 解析 → tool 执行 → steering 处理
- `SteeringController` — 双队列系统：steering_queue（高优先级指令） + message_queue（follow-up 消息）
- `PromptBuilder` — 组合 system_prompt + context + tool_definitions + history

**核心运行流程**:
```
Agent.run(input, response_model?) →           # 非流式，返回 ModelResponse
Agent.stream(input, response_model?) →        # 流式，返回 AsyncIterator[StreamChunk]
  EventBus.publish("agent.started") →
  Loop:
    PromptBuilder.build() →
    if stream_mode:
      ModelProvider.stream(request, response_model?) →
    else:
      ModelProvider.complete(request, response_model?) →
    Parse response →
      if tool_call: ToolExecutor.execute() →
    SteeringController.process_queues() →
    ContextManager.update() →
  EventBus.publish("agent.loop.completed")
```

---

#### 模块 4: `models` — 模型提供者抽象层

**职责**: 封装各 LLM SDK 差异，统一为 domain 层定义的接口。

**关键组件**:
- `BaseModelProvider` (ABC) — 公共逻辑：retry / timeout / rate limiting / token 计数 / 结构化输出解析
- `OpenAIProvider` — 适配 openai SDK，处理 function_calling 格式
- `AnthropicProvider` — 适配 anthropic SDK，处理 tool_use 格式
- `GeminiProvider` — 适配 google-generativeai SDK
- `ModelProviderFactory` — 根据 `"openai:gpt-4o"` 格式字符串创建 Provider
- `converters` — 各厂商 response → `ModelResponse` 格式转换

**流式 / 非流式双模式**:

每个 Provider 同时实现 `complete()` 和 `stream()` 两个方法：

| 方法 | 返回类型 | 适用场景 |
|------|---------|---------|
| `complete(request, response_model?)` | `ModelResponse` | 需要完整响应后再处理（如结构化解析、批量处理） |
| `stream(request, response_model?)` | `AsyncIterator[StreamChunk]` | 需要实时输出（如 UI 逐字显示、SSE 推送） |

`AgentLoop` 根据配置或调用参数决定使用哪种模式。`Agent.run()` 默认非流式，`Agent.stream()` 使用流式。两种模式发出不同的事件：非流式直接发 `llm.response.received`，流式依次发 `llm.stream.chunk` → `llm.stream.complete` → `llm.response.received`。

**结构化输出**: `BaseModelProvider` 实现公共的结构化解析逻辑——各子类只需在请求中注入 schema（各厂商格式不同），解析和校验在基类统一完成。

---

#### 模块 5: `tools` — Tool 注册与执行引擎

**职责**: decorator 式注册、Schema 自动推断、执行生命周期管理。

**关键组件**:
- `ToolRegistry` — 全局 / per-Agent 注册表 (register / unregister / get / list)
- `@tool` decorator — 从 type hints + docstring 自动推断 ToolDefinition (JSON Schema)
- `ToolExecutor` — 参数校验 (Pydantic) → 权限检查 → async 执行 → 结果封装 → 错误处理
- `schema_gen` — Python type hints → JSON Schema 转换引擎

---

#### 模块 6: `context` — 动态上下文与状态管理

**职责**: 管理对话上下文窗口和 Agent 运行时状态。

**关键组件**:
- `ContextManager` — 维护 message_history / system_context / tool_results，管理 token budget
- `ContextWindow` — 滑动窗口：按 token 数截断，支持 summary 压缩，system 消息不截断
- `StateManager` — 状态 CRUD + EventBus 广播变更 + snapshot / restore
- `InMemoryStateStore` — 默认内存存储

---

#### 模块 7: `interface` — 对外接口层

**职责**: 提供用户友好的 API 表面。

**关键组件**:
- `KAgent` — Facade 入口：`kagent = KAgent(model="openai:gpt-4o")`
- `KAgentBuilder` — Builder 链式配置：`.model().system_prompt().tools().build()`
- `hooks` — 用户级回调：on_start / on_tool_call / on_response / on_error

---

#### 模块 8: `common` — 公共基础设施

**职责**: 跨模块共享的基础设施。

**关键组件**:
- 统一异常：`KAgentError` / `ModelError` / `ToolError` / `ValidationError` / `ConfigError`
- 结构化日志：JSON 格式 + correlation_id
- 全局配置 `KAgentConfig`：提供全局 LLM 连接凭证，所有 Provider 共享
- 通用工具：async 辅助、retry decorator、timer

**全局配置 `KAgentConfig`**:

框架提供一个全局配置入口，用于统一设置所有模型 Provider 的连接信息：

```python
from kagent import configure

# 方式 1: 直接调用 configure()
configure(
    base_url="https://your-proxy.example.com/v1",
    api_key="sk-xxx",
)

# 方式 2: 环境变量 (自动读取)
# KAGENT_BASE_URL=https://your-proxy.example.com/v1
# KAGENT_API_KEY=sk-xxx
```

`KAgentConfig` (Pydantic Settings) 包含两个核心字段：
- `base_url: str | None` — 全局 API 代理地址，设置后所有 Provider 都走此地址（适用于统一网关/代理场景）
- `api_key: str | None` — 全局 API Key，当单个 Provider 未单独配置 key 时使用此值

**优先级**: Provider 级配置 > 全局 `KAgentConfig` > 环境变量。即：如果 `OpenAIProvider` 单独传了 `api_key`，则用它自己的；否则 fallback 到全局配置。

---

## Part 2: 文件/目录结构

```
KAgent/
├── pyproject.toml                              # 项目配置 + 依赖 + build + ruff/mypy/pytest
├── README.md
├── LICENSE
├── .gitignore
│
├── kagent/                                     # 顶层包
│   ├── __init__.py                             # 导出公共 API: KAgent, tool, configure, run, stream
│   ├── py.typed                                # PEP 561 type checking marker
│   ├── _version.py                             # 版本号单一来源
│   │
│   ├── domain/                                 # ─── 领域核心层 ───
│   │   ├── __init__.py
│   │   ├── entities.py                         # Message, AgentState, ToolDefinition, ToolResult, StreamChunk
│   │   ├── model_types.py                      # ModelRequest, ModelResponse, ModelInfo, TokenUsage
│   │   ├── events.py                           # Event 基类 + AgentEvent/LLMEvent/ToolEvent/SteeringEvent 子类 + 事件类型枚举
│   │   ├── protocols.py                        # IEventBus, IModelProvider, ITool, IStateStore, IContextManager
│   │   └── enums.py                            # Role, EventType, ModelProviderType, ToolCallStatus
│   │
│   ├── events/                                 # ─── Pub/Sub 事件系统 ───
│   │   ├── __init__.py
│   │   ├── bus.py                              # EventBus: async publish/subscribe, glob 路由, 优先级队列
│   │   ├── filters.py                          # EventFilter: 模式匹配 + payload 条件
│   │   ├── middleware.py                       # EventMiddleware 基类 + LoggingMiddleware + MetricsMiddleware
│   │   └── types.py                            # Subscription, EventHandler 类型别名
│   │
│   ├── agent/                                  # ─── Agent 核心编排 ───
│   │   ├── __init__.py
│   │   ├── agent.py                            # Agent 主类: run() / stream()
│   │   ├── config.py                           # AgentConfig
│   │   ├── loop.py                             # AgentLoop 编排循环
│   │   ├── steering.py                         # SteeringController 双队列
│   │   └── prompt_builder.py                   # PromptBuilder
│   │
│   ├── models/                                 # ─── 模型提供者抽象层 ───
│   │   ├── __init__.py
│   │   ├── base.py                             # BaseModelProvider(ABC): retry / timeout / rate limiting
│   │   ├── factory.py                          # ModelProviderFactory
│   │   ├── config.py                           # ModelConfig
│   │   ├── openai_provider.py                  # OpenAI 适配器
│   │   ├── anthropic_provider.py               # Anthropic 适配器
│   │   ├── gemini_provider.py                  # Gemini 适配器
│   │   └── converters.py                       # 厂商格式 → domain 类型转换
│   │
│   ├── tools/                                  # ─── Tool 注册与执行引擎 ───
│   │   ├── __init__.py
│   │   ├── registry.py                         # ToolRegistry: register / unregister / get / list
│   │   ├── decorator.py                        # @tool decorator: type hints → JSON Schema
│   │   ├── executor.py                         # ToolExecutor: 校验 → 执行 → 封装
│   │   ├── schema_gen.py                       # Python type hints → JSON Schema 转换
│   │   └── builtin/                            # 内置工具
│   │       ├── __init__.py
│   │       └── think.py                        # 结构化思考 tool
│   │
│   ├── context/                                # ─── 上下文与状态管理 ───
│   │   ├── __init__.py
│   │   ├── manager.py                          # ContextManager: token budget 管理
│   │   ├── window.py                           # ContextWindow: 滑动窗口截断/压缩
│   │   ├── state.py                            # StateManager: CRUD + EventBus 广播 + snapshot
│   │   └── stores/                             # StateStore 实现
│   │       ├── __init__.py
│   │       ├── memory.py                       # InMemoryStateStore (默认)
│   │       └── redis.py                        # RedisStateStore (可选)
│   │
│   ├── interface/                              # ─── 对外接口层 ───
│   │   ├── __init__.py
│   │   ├── kagent.py                           # KAgent Facade 入口类
│   │   ├── builder.py                          # KAgentBuilder 链式构建器
│   │   └── hooks.py                            # Lifecycle Hooks 回调注册
│   │
│   └── common/                                 # ─── 公共基础设施 ───
│       ├── __init__.py
│       ├── errors.py                           # KAgentError + ModelError + ToolError + ValidationError + ConfigError
│       ├── logging.py                          # 结构化日志 (JSON + correlation_id)
│       ├── config.py                           # KAgentConfig: 全局 base_url / api_key (Pydantic Settings, 支持环境变量)
│       └── utils.py                            # async 辅助, retry decorator, timer
│
├── tests/                                      # 测试 (镜像 kagent/ 结构)
│   ├── __init__.py
│   ├── conftest.py                             # 共享 fixtures: mock_event_bus, mock_provider, sample_data
│   ├── domain/
│   │   ├── __init__.py
│   │   ├── test_entities.py
│   │   └── test_events.py
│   ├── events/
│   │   ├── __init__.py
│   │   ├── test_bus.py
│   │   └── test_filters.py
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── test_agent.py
│   │   ├── test_loop.py
│   │   └── test_steering.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── test_factory.py
│   │   ├── test_openai.py
│   │   ├── test_anthropic.py
│   │   └── test_converters.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── test_registry.py
│   │   ├── test_decorator.py
│   │   ├── test_executor.py
│   │   └── test_schema_gen.py
│   ├── context/
│   │   ├── __init__.py
│   │   ├── test_manager.py
│   │   ├── test_window.py
│   │   └── test_state.py
│   └── interface/
│       ├── __init__.py
│       ├── test_kagent.py
│       └── test_builder.py
│
└── examples/                                   # 使用示例
    ├── 01_basic_chat.py                        # 最简对话（多模型）
    ├── 02_tool_usage.py                        # Tool 注册与使用（多模型）
    ├── 03_streaming.py                         # 流式输出（多模型）
    ├── 04_structured_output.py                 # Pydantic 结构化输出（多模型）
    ├── 05_event_hooks.py                       # 事件监听（多模型）
    └── 06_custom_state.py                      # 自定义状态管理
```

---

## 关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 接口机制 | `typing.Protocol` (非 ABC) | 鸭子类型兼容，mock 更简单，不强制继承 |
| Schema 验证 | Pydantic v2 | Entity 自动 JSON Schema、运行时校验、序列化一体化 |
| 事件通信 | EventBus 中枢 | 模块零耦合，可插拔观察者，便于扩展 |
| 双队列 Steering | steering_queue + message_queue 分离 | 高优先级指令不被 tool 执行阻塞（借鉴 Pi-Agent-Core） |
| 流式/非流式 | `complete()` + `stream()` 双方法 | 同一 Provider 同时支持两种模式，用户按场景选择 |
| 结构化输出 | `response_model` 参数 + Pydantic 解析 | 类型安全地约束 LLM 输出格式，流/非流通用 |
| 全局配置 | `KAgentConfig` (base_url + api_key) | 一处配置全局生效，Provider 可单独覆盖 |
| 依赖策略 | 核心仅 pydantic，Provider 为可选依赖 | 最小安装体积，按需引入 |

## 用户使用示例

### 基础配置 + 非流式调用
```python
from kagent import configure, KAgent, tool

# 全局配置 (一次设置，所有 Provider 生效)
configure(
    base_url="https://your-proxy.example.com/v1",
    api_key="sk-xxx",
)

agent = KAgent(model="openai:gpt-4o", system_prompt="You are helpful.")

# 非流式
result = await agent.run("What's the weather in Berlin?")
print(result.content)
```

### 流式调用
```python
# 流式 — 逐 chunk 输出
async for chunk in agent.stream("Tell me about Berlin"):
    print(chunk.content, end="")
```

### 结构化输出
```python
from pydantic import BaseModel
from typing import Literal

class WeatherInfo(BaseModel):
    city: str
    temperature: float
    unit: Literal["celsius", "fahrenheit"]
    summary: str

# 非流式 + 结构化 → 返回类型安全的 Pydantic 实例
result = await agent.run("Berlin weather now", response_model=WeatherInfo)
weather = result.parsed  # WeatherInfo(city="Berlin", temperature=18.5, ...)

# 流式 + 结构化 → 逐 chunk 输出，最终解析为 Pydantic 实例
async for chunk in agent.stream("Berlin weather now", response_model=WeatherInfo):
    print(chunk.content, end="")
# chunk.parsed 在最后一个 chunk 上可用
```

### Tool + 事件监听
```python
@agent.tool
async def get_weather(city: str, unit: str = "celsius") -> dict:
    """Get weather for a city."""
    return {"temp": 22, "unit": unit, "city": city}

@agent.on("tool.call.completed")
async def on_tool(event):
    print(f"Tool {event.payload['tool_name']} done")

result = await agent.run("What's the weather in Berlin?")
```

## 交付物

在项目根目录生成 `docs/design.md`，包含以上全部设计内容。
