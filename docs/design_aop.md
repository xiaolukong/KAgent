# AI-Native Codebase Optimization 设计文档

## 0. 问题陈述

当 Claude Code、Cursor、GitHub Copilot 等 AI Coding Agent 面对 KAgent 代码库时，存在三个核心痛点：

1. **上下文浪费**：AI 需要阅读 `kagent/agent/loop.py`（430+ 行）等底层实现才能理解框架行为，但 90% 的用户场景只需要 `KAgent` 这个 Facade 类的接口知识
2. **意图定位困难**：用户说「写一个需要人工审批的 Agent」，AI 不知道该看哪个文件，只能全局搜索 `interrupt`、`approve`、`confirm` 等关键词，可能搜到大量无关结果
3. **缺乏架构护栏**：AI 不知道 KAgent 的设计哲学边界（单智能体 vs 多智能体、纯异步 vs 同步），可能生成违反框架设计意图的代码

**目标**：让 AI Agent 只需读取 **3-5 个文件**（总计 < 500 行）就能完美理解并使用 KAgent，零幻觉、零底层侵入。

---

## 1. 设计思路

### 1.1 三层信息架构

```
┌──────────────────────────────────────────────────┐
│  Layer 1: AI Entrypoint (CLAUDE.md / .cursorrules)│  ← AI 启动时自动加载
│  "你是谁、你能做什么、绝对不要做什么"               │     ~150 行
├──────────────────────────────────────────────────┤
│  Layer 2: Intent Manifest (docs/AI_MANIFEST.json) │  ← AI 按意图检索
│  "用户想做 X → 去看文件 Y"                         │     ~80 行
├──────────────────────────────────────────────────┤
│  Layer 3: Public API Surface                      │  ← AI 按需读取
│  kagent/__init__.py + 类型签名 + Docstrings        │     按需
└──────────────────────────────────────────────────┘
```

**关键洞察**：AI 不需要理解 KAgent 的内部实现。它只需要知道：
- **Layer 1**：框架的身份、哲学、禁区（一次性加载，全局生效）
- **Layer 2**：「用户意图 → 参考文件」的映射表（按需检索，精准定位）
- **Layer 3**：具体的 API 签名和用法示例（只在写代码时读取）

### 1.2 三个核心设计原则

#### 原则一：上下文经济学（Context Window Economy）

AI 的上下文窗口是稀缺资源。每一行被 AI 读取的代码都有「机会成本」。

**规则**：
- 永远不要让 AI 去读 `kagent/agent/loop.py`、`kagent/models/*.py` 等实现文件
- 所有 AI 需要的知识必须浓缩在 Interface 层（`kagent/__init__.py` + `kagent/interface/kagent.py`）
- 使用 `__all__` 明确告诉 AI：「你只需要关注这些导出，其他都是私有实现」

#### 原则二：意图到代码的映射（Intent-to-Code Mapping）

AI 不应该通过「grep 关键词」来找功能，而应该通过「语义意图」来检索。

**规则**：
- 每个 Example 文件都有明确的意图标签（不仅仅是技术描述，而是用户会说的自然语言）
- 一个 JSON 清单文件把「用户可能的表述」映射到「该看的文件」
- 清单中包含多语言意图描述（中文 + 英文），因为用户可能用任何语言提问

#### 原则三：明确的边界与护栏（Explicit Boundaries & Guardrails）

AI 最容易犯的错是「过度设计」——试图在 KAgent 内部实现它不该做的事情。

**规则**：
- 在 AI Entrypoint 文件中明确声明：KAgent 是什么、不是什么
- 列出「绝对禁止」的模式（Anti-patterns）
- 列出「应该在外部应用层做」的事情

---

## 2. 新添加的文件与修改清单

### 2.1 新增文件

| 文件 | 用途 | 预估行数 | 说明 |
|------|------|----------|------|
| `CLAUDE.md` | Claude Code / Claude Agent 的全局 System Prompt | ~150 | 项目根目录，Claude Code 启动时自动读取 |
| `.cursorrules` | Cursor IDE 的全局规则 | ~150 | 与 CLAUDE.md 内容相同，Cursor 启动时自动读取 |
| `.github/copilot-instructions.md` | GitHub Copilot 的全局指引 | ~150 | 与 CLAUDE.md 内容相同，Copilot 启动时参考 |
| `docs/AI_MANIFEST.json` | 意图到代码的语义映射清单 | ~120 | AI 按用户意图检索的索引文件 |

### 2.2 修改文件

| 文件 | 修改内容 | 说明 |
|------|----------|------|
| `kagent/__init__.py` | 扩充 `__all__` 导出 + 添加模块级文档 | 将常用的域对象（`Message`、`Role`、`StreamChunk` 等）加入公共导出，减少 AI 需要探索子模块的次数 |
| `kagent/interface/kagent.py` | 增强关键方法的 Docstrings | 在 `run()`、`stream()`、`tool()`、`intercept()`、`transform()` 的 docstring 中添加完整的 Usage Example |
| `README.md` | 重写为完整的入门指南 | 添加安装、Quick Start、功能概览、架构图；AI 通常会首先读取 README |

### 2.3 不修改的文件

以下文件**不需要修改**，因为 AI 不应该读取它们：

- `kagent/agent/loop.py` — 内部编排逻辑
- `kagent/agent/steering.py` — 内部转向控制
- `kagent/models/*.py` — Provider 实现
- `kagent/tools/executor.py` — 内部执行引擎
- `kagent/context/transformer.py` — 内部转换管道

---

## 3. 各文件详细设计

### 3.1 CLAUDE.md / .cursorrules / copilot-instructions.md

三个文件内容相同（或高度相似），结构如下：

```
# KAgent — AI Coding Agent 指引

## 身份声明（Identity）
- KAgent 是什么：基于事件和拦截器的纯异步单体 Agent 引擎
- KAgent 不是什么：不是多智能体框架、不是 LangGraph、不是 UI 层

## 核心 API 备忘录（Cheatsheet）
- 初始化
- 注册 Tool
- 注册事件钩子
- 注册拦截器
- 注册上下文转换
- 运行（流式 / 非流式）
- Steering（重定向 / 中止 / 中断 / 恢复）

## 绝对禁止事项（NEVER DO）
- 不要使用同步阻塞 IO（time.sleep, requests.get）
- 不要直接访问 _context._messages 等私有属性
- 不要在 KAgent 内部实现多智能体编排
- 不要手动构建 ModelRequest，使用 agent.run() / agent.stream()
- 不要 import kagent.agent.loop 或 kagent.models.* 等内部模块

## 关键类型速查（Types Quick Reference）
- Message(role, content, metadata)
- ModelResponse(content, tool_calls, usage, metadata)
- StreamChunk(chunk_type, content, thinking, tool_call)
- 以及 Role、EventType、StreamChunkType 等枚举

## 意图索引指引
- 指向 docs/AI_MANIFEST.json
- 告诉 AI：「当你不确定该看哪个文件时，先查阅 AI_MANIFEST.json」
```

**关键设计决策**：

1. **语言选择**：使用**英文**书写。原因：主流 AI Agent（Claude Code、Cursor、Copilot）的内部 System Prompt 和推理链都是英文，英文的 AI Entrypoint 能最大程度减少理解损耗。但意图描述部分包含中文标签，因为用户可能用中文提问。

2. **长度控制**：严格控制在 150 行以内。太长会浪费 AI 的上下文窗口。每一行都必须是「高信息密度」的。

3. **代码 > 散文**：用代码块展示 API 用法，不用段落描述。AI 解析代码块的效率远高于自然语言段落。

### 3.2 docs/AI_MANIFEST.json

```json
{
  "version": "1.0",
  "framework": "KAgent",
  "description": "Intent-to-code mapping for AI coding agents",
  "intents": [
    {
      "id": "basic-chat",
      "intent_en": "Basic conversation with an LLM",
      "intent_zh": "基础对话 / 简单问答",
      "keywords": ["chat", "conversation", "basic", "hello world", "对话"],
      "reference_files": ["examples/01_basic_chat.py"],
      "api_surface": ["KAgent", "configure", "agent.run()"]
    },
    {
      "id": "tool-usage",
      "intent_en": "Register and use tools / function calling",
      "intent_zh": "工具注册 / 函数调用 / Tool Use",
      "keywords": ["tool", "function", "calling", "工具", "函数调用"],
      "reference_files": ["examples/02_tool_usage.py"],
      "api_surface": ["@agent.tool", "agent.run()"]
    },
    {
      "id": "streaming",
      "intent_en": "Stream response tokens in real-time",
      "intent_zh": "流式输出 / 逐 token 输出",
      "keywords": ["stream", "streaming", "real-time", "流式", "实时"],
      "reference_files": ["examples/03_streaming.py"],
      "api_surface": ["agent.stream()", "StreamChunk"]
    },
    ...（每个 Example 对应一个 intent 条目）
  ],
  "composite_intents": [
    {
      "id": "code-review-agent",
      "intent_en": "Build a code review agent with hidden internal state and streamed thinking",
      "intent_zh": "构建代码审查 Agent，隐藏内部状态，流式输出思考过程",
      "keywords": ["code review", "hidden state", "thinking stream", "代码审查"],
      "reference_files": [
        "examples/11_context_transform.py",
        "examples/10_thinking.py",
        "examples/09_interceptors.py"
      ],
      "api_surface": ["@agent.transform", "agent.stream()", "StreamChunkType.THINKING_DELTA"]
    },
    ...（复合意图：组合多个 Example 的场景）
  ]
}
```

**关键设计决策**：

1. **JSON 而非 Markdown**：AI 解析结构化 JSON 比解析 Markdown 表格更精确。JSON 可以被精确地 `jq` 查询，也可以被 AI 直接作为数据结构理解。

2. **双语意图标签**：同时提供 `intent_en` 和 `intent_zh`，因为用户可能用任何语言向 AI 发出指令。

3. **Keywords 数组**：提供多个关键词变体（包括同义词、中文），提高 AI 的意图匹配命中率。

4. **Composite Intents**：除了单一功能的意图，还提供「组合意图」——对应真实业务场景需要同时使用多个功能的情况。这是最有价值的部分，因为用户的真实需求通常是复合的。

5. **api_surface 字段**：告诉 AI 这个意图涉及哪些 API，AI 可以直接去检查这些 API 的签名，不需要阅读完整 Example。

### 3.3 kagent/__init__.py 扩充

当前导出：
```python
__all__ = ["KAgent", "KAgentBuilder", "__version__", "configure", "get_config", "tool"]
```

扩充为：
```python
__all__ = [
    # ── Core ──
    "KAgent",
    "KAgentBuilder",
    "configure",
    "get_config",
    "tool",
    "__version__",
    # ── Domain Types (frequently needed by user code) ──
    "Message",
    "Role",
    "ModelResponse",
    "StreamChunk",
    "StreamChunkType",
    "EventType",
    "ToolCall",
    "ToolResult",
]
```

**设计考量**：

当前用户需要写 `from kagent.domain.entities import Message` 和 `from kagent.domain.enums import Role`——这要求 AI 知道 KAgent 的内部包结构。扩充后，用户只需写 `from kagent import KAgent, Message, Role`，AI 也只需要查看一个 `__init__.py` 就能发现所有可用类型。

但**不导出**内部组件：
- 不导出 `ContextManager`、`ContextTransformer`、`InterceptorPipeline` 等——这些通过 `KAgent` Facade 访问
- 不导出 `EventBus`、`ToolRegistry`、`ToolExecutor`——这些是内部实现
- 不导出 `BaseModelProvider` 及具体 Provider——除非用户要写自定义 Provider

### 3.4 kagent/interface/kagent.py Docstring 增强

以 `transform()` 方法为例，当前 docstring 中的示例已经很好，但需要增强为更完整的「可直接复制」的模板：

```python
def transform(self, func=None, *, priority: int = 0) -> Any:
    """Register a context transform function. Can be used as a decorator.

    Transforms filter or modify messages before they are sent to the LLM.
    The original messages in ContextManager are never modified.

    Args:
        func: The transform function (when used as bare decorator).
        priority: Higher priority runs first. Default 0.
            Built-in transforms (internal message filter, thinking stripper)
            use priority -100, so user transforms always run first.

    Returns:
        The original function (when used as decorator).

    Example::

        @agent.transform
        async def inject_time(messages: list[Message]) -> list[Message]:
            from datetime import datetime, UTC
            time_msg = Message(
                role=Role.SYSTEM,
                content=f"Current time: {datetime.now(UTC).isoformat()}",
            )
            return [messages[0], time_msg] + messages[1:]

        @agent.transform(priority=10)
        async def hide_debug(messages: list[Message]) -> list[Message]:
            return [m for m in messages if m.metadata.get("type") != "debug"]
    """
```

对 `run()`、`stream()`、`tool()`、`on()`、`intercept()` 做类似增强。重点是：
- **Args 节**：每个参数的类型和语义
- **Returns 节**：返回值的类型和含义
- **Example 节**：3-5 行可直接复制的代码，包含 import

### 3.5 README.md 重写

从当前的 3 行扩展为标准开源项目 README：

```markdown
# KAgent

A clean-architecture async Python agent framework.

## Features
- Multi-provider LLM support (OpenAI, Anthropic, Gemini)
- Tool calling with auto JSON Schema
- Streaming with thinking/reasoning separation
- Pub/Sub event system + Mutable interceptor pipeline
- Context transforms (App message ↔ LLM message boundary)
- Runtime steering (redirect, abort, interrupt/resume)
- Structured output via Pydantic models
- Tool self-correction & circuit breaker

## Quick Start
（5 行代码的最小示例）

## Architecture
（ASCII 架构图）

## Examples
（12 个 Example 的一行描述列表）

## Documentation
（指向 docs/ 的链接）
```

---

## 4. 更新后的文件夹结构

```
KAgent/
├── CLAUDE.md                           ← [新增] Claude Code 全局指引
├── .cursorrules                        ← [新增] Cursor IDE 全局规则
├── .github/
│   └── copilot-instructions.md         ← [新增] GitHub Copilot 指引
├── README.md                           ← [修改] 从 3 行扩展为完整 README
├── pyproject.toml
├── requirements.txt
├── .gitignore
│
├── kagent/
│   ├── __init__.py                     ← [修改] 扩充 __all__ 导出
│   ├── _version.py
│   ├── py.typed
│   │
│   ├── interface/                      ← 公共 API 层（AI 应该读的）
│   │   ├── __init__.py
│   │   ├── kagent.py                   ← [修改] 增强 Docstrings
│   │   ├── builder.py
│   │   └── hooks.py
│   │
│   ├── domain/                         ← 域模型（AI 按需读的）
│   │   ├── __init__.py
│   │   ├── entities.py                 ← Message, ToolCall, ToolResult, ...
│   │   ├── enums.py                    ← Role, EventType, StreamChunkType, ...
│   │   ├── model_types.py             ← ModelResponse, StreamChunk, ...
│   │   ├── events.py
│   │   └── protocols.py
│   │
│   ├── agent/                          ← 内部（AI 不应该读的）
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── config.py
│   │   ├── interceptor.py
│   │   ├── loop.py
│   │   ├── prompt_builder.py
│   │   └── steering.py
│   │
│   ├── models/                         ← 内部（AI 不应该读的）
│   │   ├── ... (provider implementations)
│   │
│   ├── tools/                          ← 内部（AI 不应该读的）
│   │   ├── ... (tool infrastructure)
│   │
│   ├── context/                        ← 内部（AI 不应该读的）
│   │   ├── ... (context management)
│   │
│   ├── events/                         ← 内部（AI 不应该读的）
│   │   ├── ... (event bus implementation)
│   │
│   └── common/                         ← 内部（AI 不应该读的）
│       ├── ... (utilities)
│
├── examples/                           ← AI 按 Manifest 索引的参考代码
│   ├── 01_basic_chat.py
│   ├── 02_tool_usage.py
│   ├── 03_streaming.py
│   ├── 04_structured_output.py
│   ├── 05_event_hooks.py
│   ├── 06_custom_state.py
│   ├── 07_steering.py
│   ├── 08_interrupt.py
│   ├── 09_interceptors.py
│   ├── 10_thinking.py
│   ├── 11_context_transform.py
│   └── 12_self_correction.py
│
├── tests/
│   ├── ... (test suites)
│
└── docs/
    ├── AI_MANIFEST.json                ← [新增] 意图到代码的语义索引
    ├── design.md
    ├── usage.md
    ├── upgrade-v2.md
    ├── upgrade-v3.md
    ├── design_aop.md                   ← [本文档]
    └── modification/
```

标注说明：
- `[新增]` = 本次新建的文件
- `[修改]` = 本次需要修改的现有文件
- 其余文件不做修改

---

## 5. AI Agent 工作流程推演

以下模拟一次真实的 AI Coding Agent 与 KAgent 代码库的交互过程。

### 场景：用户对 Claude Code 说

> 「请用 KAgent 帮我写一个代码审查 Agent，要求把内部的 review_score 隐藏掉不让大模型看到，并且把它的思考过程流式打印到终端。」

### Claude Code 的内部执行流程

**Step 1：吸收全局规则** (~150 行)

Claude Code 启动时自动读取 `CLAUDE.md`。瞬间知道：
- KAgent 是纯异步单体 Agent 引擎
- 使用 `from kagent import KAgent, Message, Role, StreamChunk, StreamChunkType`
- 不要 import 内部模块
- 不要用同步 IO

**Step 2：意图检索** (~10 行)

读取 `docs/AI_MANIFEST.json`，进行语义匹配：
- 「隐藏内部状态」→ 命中 `intent_id: "context-transform"`，参考 `examples/11_context_transform.py`
- 「思考过程流式打印」→ 命中 `intent_id: "thinking-stream"`，参考 `examples/10_thinking.py`
- 组合意图也命中 `composite_intents` 中的相关条目

**Step 3：精准读取** (~100 行)

只读取两个 Example 文件。学会了：
- `@agent.transform` 装饰器过滤内部消息
- `agent.stream()` + `StreamChunkType.THINKING_DELTA` 分离思考过程

**Step 4：接口确认** (~30 行)

可选地看一眼 `kagent/__init__.py` 确认导出，可能查阅 `KAgent.stream()` 的 docstring 确认参数。

**Step 5：生成代码**

基于 < 300 行的阅读量，生成完美的业务代码：

```python
from kagent import KAgent, configure, Message, Role, StreamChunkType

configure(api_key="...")
agent = KAgent(model="openai:gpt-5", system_prompt="You are a code reviewer.")

@agent.tool
async def analyze_code(code: str) -> dict:
    score = calculate_review_score(code)  # 内部评分
    agent_context.update_context(review_score=score)  # 保存到内部状态
    return {"issues": [...], "suggestions": [...]}

@agent.transform
async def hide_review_score(messages: list[Message]) -> list[Message]:
    return [m for m in messages if m.metadata.get("type") != "review_score"]

async for chunk in agent.stream("Review this code: ..."):
    if chunk.chunk_type == StreamChunkType.THINKING_DELTA:
        print(f"[thinking] {chunk.thinking}", end="")
    elif chunk.content:
        print(chunk.content, end="", flush=True)
```

**零幻觉。零底层侵入。零多余文件阅读。**

---

## 6. 实施优先级

| 优先级 | 文件 | 理由 |
|--------|------|------|
| P0 | `CLAUDE.md` | 直接影响 Claude Code 的代码生成质量 |
| P0 | `docs/AI_MANIFEST.json` | 意图检索的核心，没有它 AI 还是要 grep |
| P1 | `kagent/__init__.py` | 减少 AI 探索子模块的次数 |
| P1 | `.cursorrules` | 覆盖 Cursor 用户（内容与 CLAUDE.md 相同） |
| P1 | `.github/copilot-instructions.md` | 覆盖 Copilot 用户 |
| P2 | `kagent/interface/kagent.py` | Docstring 增强，锦上添花 |
| P2 | `README.md` | 入门指南，AI 和人类都受益 |

**建议实施顺序**：P0 → P1 → P2，总工作量约 **4 个文件新建 + 3 个文件修改**。

---

## 7. 衡量标准

实施完成后，可以通过以下方式验证效果：

1. **文件读取计数**：让 Claude Code 完成一个中等复杂度的 KAgent 任务（如「写一个带工具调用、流式输出、拦截器的 Agent」），统计它读取了多少个文件。目标：**≤ 5 个**（CLAUDE.md + AI_MANIFEST.json + 2 个 Example + __init__.py）。

2. **代码正确性**：生成的代码应该：
   - 只使用 `__all__` 中暴露的公共 API
   - 不 import 任何 `kagent.agent.*`、`kagent.models.*` 等内部模块
   - 全部使用 async/await，无同步阻塞
   - 符合 KAgent 的单智能体定位

3. **零幻觉率**：AI 不应该捏造不存在的 API（如 `agent.add_tool()`、`agent.set_model()`）。通过 `__all__` 和 Docstring 中的 Example，AI 可以精确知道有哪些可用方法。
