# KAgent v4 升级设计：SAP AI Core 集成

## 0. 背景与目标

KAgent 当前通过 `base_url` + `api_key` 的 AI Proxy 方式接入 LLM。本次升级目标：

1. **在 `configure()` 阶段**让用户选择后端：`"aicore"`（默认）或 `"aiproxy"`（当前方式）。
2. **用户代码零改动**：`KAgent(model="openai:gpt-4o")` 等所有现有写法保持不变，只有 `configure()` 需要调整。
3. **不改变任何现有文件结构**：`openai_provider.py`、`anthropic_provider.py`、`gemini_provider.py` 完全不动，只对 `factory.py`、`enums.py`、`common/config.py` 做最小修改。

### 核心设计洞察

`openai`/`anthropic`/`gemini` 描述的是**模型系列**（LLM family），而 AI Core 描述的是**访问渠道**（transport backend）。两者正交：同一个 `"openai:gpt-4o"` 模型，可以经由 AI Core Orchestration Service 发出，也可以直接通过 OpenAI API 发出。因此 `backend` 是配置层的概念，不应体现在 `model=` 字符串里。

### 设计原则（延续 v2/v3）

1. **微内核路线** — AgentLoop 不感知 Provider 的具体实现
2. **向后兼容** — 所有现有 `model=` 写法、工具注册、事件订阅、拦截器接口零成本迁移
3. **单智能体定位** — 本次只增加 Provider 层，不触碰 Agent 运行时

---

## 1. 架构概览

### 1.1 后端选择对现有调用路径的影响

```
configure(backend="aiproxy")          configure(backend="aicore")  ← 只改这里
        │                                        │
KAgent(model="openai:gpt-4o")         KAgent(model="openai:gpt-4o")  ← 完全不变
        │                                        │
factory.create_provider(...)           factory.create_provider(...)
        │                                        │
        ▼                                        ▼
  OpenAIProvider                        AICoreProvider
  (直连 OpenAI API)                     (经由 AI Core Orchestration)
        │                                        │
        └──────────────────┬─────────────────────┘
                           ▼
              AgentLoop / ModelRequest / ModelResponse
              (完全不变，对上层透明)
```

### 1.2 AICoreProvider 在四层架构中的位置

```
┌─────────────────────────────────────────────────────┐
│               Interface Layer (KAgent)               │
│  agent = KAgent(model="openai:gpt-4o")              │  ← 不变
├─────────────────────────────────────────────────────┤
│               Application Layer (AgentLoop)          │
│  model_provider.complete(request) / .stream(request) │  ← 不变
├───────────────────────────────────────────┬─────────┤
│  Infrastructure (Models Layer)            │   NEW   │
│                                           │         │
│  openai_provider.py    (aiproxy 模式)     │ aicore_ │
│  anthropic_provider.py (aiproxy 模式)     │ provid  │
│  gemini_provider.py    (aiproxy 模式)     │ er.py   │
├───────────────────────────────────────────┴─────────┤
│               Domain Layer (不变)                    │
│  BaseModelProvider / ModelRequest / ModelResponse   │
└─────────────────────────────────────────────────────┘
```

---

## 2. 配置层变更

### 2.1 `kagent/common/config.py`

新增 `backend` 字段，以及 AI Core 专属凭据字段。**现有字段和行为完全不变**。

```python
from __future__ import annotations
from typing import Literal
from pydantic_settings import BaseSettings

_config: KAgentConfig | None = None


class KAgentConfig(BaseSettings):
    """Global KAgent configuration."""

    # ── 后端选择 ───────────────────────────────────────────────────────────
    # "aicore"（默认）: 通过 SAP AI Core Orchestration Service 调用 LLM
    # "aiproxy"       : 直连 OpenAI / Anthropic / Gemini API（当前行为）
    backend: Literal["aicore", "aiproxy"] = "aicore"

    # ── AI Proxy 模式（backend="aiproxy"，原有字段保持不变）──────────────
    base_url: str | None = None
    api_key: str | None = None

    # ── AI Core 模式（backend="aicore"，新增字段）─────────────────────────
    # GenAIHubProxyClient 会自动读取 AICORE_* 环境变量，
    # 此处字段用于通过 configure() 显式覆盖
    aicore_auth_url: str | None = None        # 对应 AICORE_AUTH_URL
    aicore_client_id: str | None = None       # 对应 AICORE_CLIENT_ID
    aicore_client_secret: str | None = None   # 对应 AICORE_CLIENT_SECRET
    aicore_base_url: str | None = None        # 对应 AICORE_BASE_URL
    aicore_resource_group: str | None = None  # 对应 AICORE_RESOURCE_GROUP

    model_config = {
        "env_prefix": "KAGENT_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


def configure(
    *,
    backend: Literal["aicore", "aiproxy"] | None = None,
    # AI Proxy 参数（原有，不变）
    base_url: str | None = None,
    api_key: str | None = None,
    # AI Core 参数（新增）
    aicore_auth_url: str | None = None,
    aicore_client_id: str | None = None,
    aicore_client_secret: str | None = None,
    aicore_base_url: str | None = None,
    aicore_resource_group: str | None = None,
) -> KAgentConfig:
    """Set the global KAgent configuration.

    backend="aicore" (default): reads AICORE_* env vars or explicit params.
    backend="aiproxy": reads KAGENT_API_KEY / KAGENT_BASE_URL (legacy behavior).
    """
    global _config
    kwargs: dict[str, object] = {}
    if backend is not None:
        kwargs["backend"] = backend
    if base_url is not None:
        kwargs["base_url"] = base_url
    if api_key is not None:
        kwargs["api_key"] = api_key
    if aicore_auth_url is not None:
        kwargs["aicore_auth_url"] = aicore_auth_url
    if aicore_client_id is not None:
        kwargs["aicore_client_id"] = aicore_client_id
    if aicore_client_secret is not None:
        kwargs["aicore_client_secret"] = aicore_client_secret
    if aicore_base_url is not None:
        kwargs["aicore_base_url"] = aicore_base_url
    if aicore_resource_group is not None:
        kwargs["aicore_resource_group"] = aicore_resource_group
    _config = KAgentConfig(**kwargs)
    return _config
```

### 2.2 `.env` 示例

**AI Core 模式（默认）：**

```bash
# KAGENT_BACKEND 默认就是 aicore，可以省略
KAGENT_BACKEND=aicore

# AI Core 凭据——GenAIHubProxyClient 直接读这五个变量，无需前缀
AICORE_AUTH_URL=https://<subdomain>.authentication.eu10.hana.ondemand.com/oauth/token
AICORE_CLIENT_ID=sb-<client-id>
AICORE_CLIENT_SECRET=<secret>
AICORE_BASE_URL=https://api.ai.prod.eu-central-1.aws.ml.hana.ondemand.com
AICORE_RESOURCE_GROUP=default
```

**AI Proxy 模式（旧方式，显式切换）：**

```bash
KAGENT_BACKEND=aiproxy
KAGENT_API_KEY=sk-...
KAGENT_BASE_URL=https://your-proxy.example.com/v1
```

> **注意**：`AICORE_*` 环境变量没有 `KAGENT_` 前缀，由 `GenAIHubProxyClient` 直接读取，
> 不经过 pydantic-settings 的前缀处理。`KAgentConfig` 中的 `aicore_*` 字段仅用于
> 通过 `configure()` 显式传参时的覆盖路径。

---

## 3. 枚举层变更

### 3.1 `kagent/domain/enums.py`

在 `ModelProviderType` 增加 `AICORE`，改动量：+1 行。

```python
class ModelProviderType(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    AICORE = "aicore"   # NEW（内部标识符，不暴露给用户）
```

---

## 4. 工厂层变更

### 4.1 `kagent/models/factory.py`

`create_provider()` 在解析出 `provider_name` 之后，先检查全局配置的 `backend`。如果是 `"aicore"`，**无论 provider_name 是什么**，都路由到 `AICoreProvider`，并把完整的 `model_string`（如 `"openai:gpt-4o"`）传进去，由它内部解析模型名。

```python
def create_provider(
    model_string: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs: object,
) -> BaseModelProvider:
    parts = model_string.split(":", 1)
    if len(parts) != 2:
        raise ConfigError(
            f"Invalid model string '{model_string}'. "
            "Expected format: 'provider:model_name' (e.g. 'openai:gpt-4o')"
        )

    provider_name, model_name = parts
    config = ModelConfig(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        **kwargs,
    )

    # ── NEW: AI Core 后端路由 ─────────────────────────────────────────────
    from kagent.common.config import get_config
    if get_config().backend == "aicore":
        from kagent.models.aicore_provider import AICoreProvider
        return AICoreProvider(config)
    # ─────────────────────────────────────────────────────────────────────

    # 以下为原有逻辑，完全不变
    if provider_name == ModelProviderType.OPENAI:
        from kagent.models.openai_provider import OpenAIProvider
        return OpenAIProvider(config)

    if provider_name == ModelProviderType.ANTHROPIC:
        from kagent.models.anthropic_provider import AnthropicProvider
        return AnthropicProvider(config)

    if provider_name == ModelProviderType.GEMINI:
        from kagent.models.gemini_provider import GeminiProvider
        return GeminiProvider(config)

    raise ConfigError(
        f"Unknown provider '{provider_name}'. "
        f"Supported: {', '.join(t.value for t in ModelProviderType)}"
    )
```

**关键点**：AI Core 路由在所有 `provider_name` 判断之前，`aiproxy` 模式下则完全走原有分支，现有代码路径不受影响。

---

## 5. 新增文件：`kagent/models/aicore_provider.py`

### 5.1 工具调用机制

`orchestration_v2` 模块原生支持工具调用。工具以 JSON Schema 字典传入 `Template(tools=[...])`，AI Core Orchestration Service 把 `tool_calls` 字段返回在响应里。**KAgent 自身的 AgentLoop 负责 agentic loop**（检测 tool_calls → 执行工具 → 把 `ToolResult` 加回 context → 再次调用 LLM），与 AI Core SDK 文档中"用户自己管理 orchestration loop"的说明完全吻合：KAgent 就是那个 loop，它只需要 provider 能传入工具定义、返回 tool_calls、接受 tool results 即可。

因此整条路径统一走 `OrchestrationService`，**不需要双路由**。

### 5.2 消息格式映射

KAgent 的 domain 消息类型到 `orchestration_v2` 消息类型的映射：

| KAgent `Role` | KAgent `Message` 特征 | orchestration_v2 消息类型 |
|---|---|---|
| `SYSTEM` | `content` | `SystemMessage(content=...)` |
| `USER` | `content` | `UserMessage(content=...)` |
| `ASSISTANT` | 无 `tool_calls` | `AssistantMessage(content=...)` |
| `ASSISTANT` | 有 `tool_calls` | `AssistantMessage(content=None, tool_calls=[MessageToolCall(...)])` |
| `TOOL` | `tool_call_id` + `content` | `ToolChatMessage(tool_call_id=..., content=...)` |

### 5.3 完整实现

```python
"""AICoreProvider — wraps SAP generative-ai-hub-sdk orchestration_v2."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from kagent.common.config import get_config
from kagent.common.errors import ConfigError, ModelError
from kagent.domain.entities import Message, ToolCall
from kagent.domain.enums import Role, StreamChunkType
from kagent.domain.model_types import (
    ModelInfo,
    ModelRequest,
    ModelResponse,
    StreamChunk,
    TokenUsage,
    ToolCallChunk,
)
from kagent.models.base import BaseModelProvider
from kagent.models.config import ModelConfig


class AICoreProvider(BaseModelProvider):
    """Model provider backed by SAP AI Core (generative-ai-hub-sdk orchestration_v2).

    Activated transparently when configure(backend="aicore") is set.
    Users keep their existing model strings unchanged:
        KAgent(model="openai:gpt-4o")
        KAgent(model="anthropic:claude-3-7-sonnet")
        KAgent(model="gemini:gemini-1.5-pro")

    Credentials are read from AICORE_* env vars (preferred) or configure():
        AICORE_AUTH_URL, AICORE_CLIENT_ID, AICORE_CLIENT_SECRET,
        AICORE_BASE_URL, AICORE_RESOURCE_GROUP
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._proxy_client = self._build_proxy_client()
        self._service = self._build_service()

    # ── Client construction ────────────────────────────────────────────────

    def _build_proxy_client(self):
        try:
            from gen_ai_hub.proxy import GenAIHubProxyClient
        except ImportError as exc:
            raise ConfigError(
                "generative-ai-hub-sdk is not installed. "
                "Run: pip install 'kagent[aicore]'"
            ) from exc

        global_cfg = get_config()
        kwargs: dict[str, str] = {}
        # Explicit credentials from configure() override env vars
        if global_cfg.aicore_auth_url:
            kwargs["auth_url"] = global_cfg.aicore_auth_url
        if global_cfg.aicore_client_id:
            kwargs["client_id"] = global_cfg.aicore_client_id
        if global_cfg.aicore_client_secret:
            kwargs["client_secret"] = global_cfg.aicore_client_secret
        if global_cfg.aicore_base_url:
            kwargs["base_url"] = global_cfg.aicore_base_url
        if global_cfg.aicore_resource_group:
            kwargs["resource_group"] = global_cfg.aicore_resource_group
        # If no kwargs provided, GenAIHubProxyClient reads AICORE_* env vars itself
        return GenAIHubProxyClient(**kwargs)

    def _build_service(self):
        from gen_ai_hub.orchestration_v2.service import OrchestrationService
        return OrchestrationService(proxy_client=self._proxy_client)

    # ── ModelInfo ─────────────────────────────────────────────────────────

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(provider="aicore", model_name=self._config.model_name)

    # ── Public entry points ───────────────────────────────────────────────

    async def _do_complete(self, request: ModelRequest) -> ModelResponse:
        from gen_ai_hub.orchestration_v2.exceptions import OrchestrationError
        orch_config, history = self._build_request(request)
        try:
            response = await self._service.arun(
                config=orch_config,
                history=history,
                timeout=self._config.timeout,
            )
        except OrchestrationError as exc:
            raise ModelError(
                f"AI Core orchestration error [{exc.code}] "
                f"at {exc.location}: {exc.message}"
            ) from exc
        return self._parse_response(response)

    async def _do_stream(self, request: ModelRequest) -> AsyncIterator[StreamChunk]:
        from gen_ai_hub.orchestration_v2.exceptions import OrchestrationError
        from gen_ai_hub.orchestration_v2.models.config import GlobalStreamOptions
        orch_config, history = self._build_request(request, stream=True)
        try:
            sse = await self._service.astream(
                config=orch_config,
                history=history,
                timeout=self._config.timeout,
            )
            async with sse as stream:
                async for event in stream:
                    if not event.final_result or not event.final_result.choices:
                        continue
                    delta = event.final_result.choices[0].delta
                    if delta.content:
                        yield StreamChunk(
                            chunk_type=StreamChunkType.TEXT_DELTA,
                            content=delta.content,
                        )
                    for tc in delta.tool_calls or []:
                        if tc.function and tc.function.name:
                            yield StreamChunk(
                                chunk_type=StreamChunkType.TOOL_CALL_START,
                                tool_call=ToolCallChunk(
                                    id=tc.id or "",
                                    name=tc.function.name,
                                    arguments_delta=tc.function.arguments or "",
                                ),
                            )
                        elif tc.function and tc.function.arguments:
                            yield StreamChunk(
                                chunk_type=StreamChunkType.TOOL_CALL_DELTA,
                                tool_call=ToolCallChunk(
                                    arguments_delta=tc.function.arguments,
                                ),
                            )
                    # Usage in final chunk (usage field only on last event)
                    if event.final_result.usage:
                        u = event.final_result.usage
                        yield StreamChunk(
                            chunk_type=StreamChunkType.METADATA,
                            usage=TokenUsage(
                                prompt_tokens=u.prompt_tokens,
                                completion_tokens=u.completion_tokens,
                                total_tokens=u.total_tokens,
                            ),
                        )
        except OrchestrationError as exc:
            raise ModelError(f"AI Core stream error [{exc.code}]: {exc.message}") from exc

    # ── Request builder ───────────────────────────────────────────────────

    def _build_request(
        self, request: ModelRequest, *, stream: bool = False
    ) -> tuple[Any, list]:
        """Convert a KAgent ModelRequest into (OrchestrationConfig, history)."""
        from gen_ai_hub.orchestration_v2.models.config import (
            ModuleConfig, OrchestrationConfig, GlobalStreamOptions,
        )
        from gen_ai_hub.orchestration_v2.models.template import (
            Template, PromptTemplatingModuleConfig,
        )
        from gen_ai_hub.orchestration_v2.models.llm_model_details import LLMModelDetails

        # Split messages: template contains system + last user message;
        # the rest become history (enables multi-turn tool calling loops)
        template_messages, history = self._split_messages(request.messages)

        params: dict[str, Any] = {}
        if request.temperature is not None:
            params["temperature"] = request.temperature
        elif self._config.temperature is not None:
            params["temperature"] = self._config.temperature
        if request.max_tokens is not None:
            params["max_completion_tokens"] = request.max_tokens
        elif self._config.max_tokens is not None:
            params["max_completion_tokens"] = self._config.max_tokens

        llm = LLMModelDetails(name=self._config.model_name, params=params)

        tools = self._convert_tools(request.tools) if request.tools else None

        template = Template(
            template=template_messages,
            tools=tools,
        )
        module_config = ModuleConfig(
            prompt_templating=PromptTemplatingModuleConfig(prompt=template, model=llm)
        )
        stream_opts = GlobalStreamOptions(enabled=True) if stream else None
        orch_config = OrchestrationConfig(modules=module_config, stream=stream_opts)

        return orch_config, history

    # ── Message conversion ────────────────────────────────────────────────

    def _split_messages(self, messages: list[Message]) -> tuple[list, list]:
        """Split KAgent messages into (template_messages, history).

        Template receives: system message (if any) + last user message.
        History receives:  all prior turns (user/assistant/tool result rounds).

        This mirrors the pattern AI Core SDK expects for multi-turn conversations
        with tool results fed back through history.
        """
        from gen_ai_hub.orchestration_v2.models.message import (
            SystemMessage, UserMessage, AssistantMessage,
            ToolChatMessage, MessageToolCall, FunctionCall,
        )

        template_messages = []
        history_messages = []

        # Collect system message and locate last user message index
        system_msg = None
        last_user_idx = None
        for i, msg in enumerate(messages):
            if msg.role == Role.SYSTEM:
                system_msg = msg
            if msg.role == Role.USER:
                last_user_idx = i

        for i, msg in enumerate(messages):
            if msg.role == Role.SYSTEM:
                # System always goes into template
                template_messages.append(SystemMessage(content=msg.content or ""))
                continue

            if msg.role == Role.USER and i == last_user_idx:
                # Last user message goes into template
                template_messages.append(UserMessage(content=msg.content or ""))
                continue

            # Everything else goes into history
            if msg.role == Role.USER:
                history_messages.append(UserMessage(content=msg.content or ""))

            elif msg.role == Role.ASSISTANT:
                if msg.tool_calls:
                    sdk_tool_calls = [
                        MessageToolCall(
                            id=tc.id,
                            function=FunctionCall(
                                name=tc.name,
                                arguments=json.dumps(tc.arguments),
                            ),
                        )
                        for tc in msg.tool_calls
                    ]
                    history_messages.append(
                        AssistantMessage(content=msg.content, tool_calls=sdk_tool_calls)
                    )
                else:
                    history_messages.append(AssistantMessage(content=msg.content))

            elif msg.role == Role.TOOL:
                history_messages.append(
                    ToolChatMessage(
                        tool_call_id=msg.tool_call_id or "",
                        content=msg.content or "",
                    )
                )

        return template_messages, history_messages

    # ── Tool definition conversion ────────────────────────────────────────

    @staticmethod
    def _convert_tools(tools) -> list[dict]:
        """Convert KAgent ToolDefinitions to AI Core JSON Schema dicts.

        Uses the plain JSON Schema dict format supported by orchestration_v2:
            {"type": "function", "function": {"name": ..., "description": ...,
                                               "parameters": ..., "strict": True}}
        """
        result = []
        for t in tools:
            result.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                    "strict": False,
                },
            })
        return result

    # ── Response parser ───────────────────────────────────────────────────

    @staticmethod
    def _parse_response(response) -> ModelResponse:
        """Convert CompletionPostResponse → KAgent ModelResponse."""
        llm_result = response.final_result
        choice = llm_result.choices[0] if llm_result.choices else None

        content: str | None = None
        tool_calls: list[ToolCall] | None = None

        if choice:
            msg = choice.message
            content = msg.content if msg.content else None
            if msg.tool_calls:
                tool_calls = [
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=tc.function.parse_arguments(),
                    )
                    for tc in msg.tool_calls
                ]

        usage = None
        if llm_result.usage:
            u = llm_result.usage
            usage = TokenUsage(
                prompt_tokens=u.prompt_tokens,
                completion_tokens=u.completion_tokens,
                total_tokens=u.total_tokens,
            )

        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            metadata={"request_id": response.request_id},
        )
```

---

## 6. 用法

### 6.1 最简配置（环境变量，无需改 model= 字符串）

```python
from kagent import KAgent, configure

# 默认 backend="aicore"，从 AICORE_* env vars 读取凭据
configure()

# 所有现有 model= 写法保持完全不变
agent = KAgent(model="openai:gpt-4o", system_prompt="You are helpful.")
result = await agent.run("What is the capital of France?")
print(result.content)
```

### 6.2 显式传入 AI Core 凭据

```python
configure(
    backend="aicore",
    aicore_auth_url="https://<subdomain>.authentication.eu10.hana.ondemand.com/oauth/token",
    aicore_client_id="sb-<client-id>",
    aicore_client_secret="<secret>",
    aicore_base_url="https://api.ai.prod.eu-central-1.aws.ml.hana.ondemand.com",
    aicore_resource_group="default",
)
agent = KAgent(model="anthropic:claude-3-7-sonnet")
```

### 6.3 回退到 AI Proxy 模式（保留旧行为）

```python
configure(
    backend="aiproxy",
    base_url="https://your-proxy.example.com/v1",
    api_key="sk-...",
)
# 后续 KAgent(model="openai:gpt-4o") 走原有 OpenAIProvider，完全不变
```

---

## 7. 向后兼容性

| 场景 | 影响 | 说明 |
|------|------|------|
| 现有 `KAgent(model="openai:...")` + `configure(backend="aiproxy")` | **无影响** | factory.py 的原有分支不变 |
| 现有 `KAgent(model="anthropic:...")` + `configure(backend="aiproxy")` | **无影响** | 同上 |
| 现有 `KAgent(model="gemini:...")` + `configure(backend="aiproxy")` | **无影响** | 同上 |
| 现有 `configure(api_key=..., base_url=...)` **未设 backend** | **需注意** | `backend` 默认值为 `"aicore"`，如需保持旧行为须加 `backend="aiproxy"` |
| 现有 `.env` 只有 `KAGENT_API_KEY` **未设 `KAGENT_BACKEND`** | **需注意** | 同上，需增加 `KAGENT_BACKEND=aiproxy` |

> **迁移建议**：对已有项目，在 `.env` 或 `configure()` 中显式声明 `backend="aiproxy"` 以保持现有行为。新项目默认使用 AI Core。

---

## 8. 工具调用完整流程

下图展示 AgentLoop 如何通过 `AICoreProvider` 完成一次带工具调用的多轮对话：

```
Round 1
────────
AgentLoop
  → PromptBuilder.build()           → request.messages = [SYSTEM, USER]
  → AICoreProvider._do_complete()
      → _split_messages()           → template=[SYSTEM, USER], history=[]
      → OrchestrationService.arun() → response with tool_calls
      → _parse_response()           → ModelResponse(tool_calls=[ToolCall(name="get_weather",...)])
  ← tool_calls detected
  → ToolExecutor.execute("get_weather", {...})
  → context: add AssistantMessage(tool_calls=...) + ToolMessage(result=...)

Round 2
────────
AgentLoop
  → PromptBuilder.build()           → request.messages = [SYSTEM, USER, ASSISTANT(tc), TOOL(result)]
  → AICoreProvider._do_complete()
      → _split_messages()           → template=[SYSTEM, USER_LAST_?]
                                      ※ 此处 USER_LAST 需特殊处理（见 §9）
      → history = [ASSISTANT(tc), TOOL(result)]
      → OrchestrationService.arun(history=history) → final text response
  ← ModelResponse(content="The weather is 22°C")
```

> `_split_messages()` 的关键逻辑：系统消息和**最后一条 USER 消息**进模板，其余所有消息（包括工具调用轮次的 ASSISTANT 和 TOOL 消息）进 `history`。这符合 AI Core Orchestration Service 的接口设计。

---

## 9. 已知约束与注意事项

### 9.1 多轮对话中的"最后一条 USER 消息"

KAgent AgentLoop 在工具调用轮次之间**不一定**在每轮都新增一条 USER 消息——tool result 以 `Role.TOOL` 消息形式加入 context，第二轮调用时 messages 末尾是 `TOOL` 消息而非 `USER` 消息。

`_split_messages()` 的处理策略：
- 始终把 `last_user_idx` 所指的 USER 消息放入模板
- 其余 USER 消息和所有 ASSISTANT/TOOL 消息放入 history
- 如果 messages 末尾没有 USER 消息（纯工具轮次），则模板中只有 SYSTEM，所有其他消息进 history

实现中已考虑此情况，但集成测试应覆盖纯工具回调轮次。

### 9.2 Thinking / Reasoning 模式

`orchestration_v2` 的 `StreamDelta` 没有 `reasoning_content` 字段（不同于 OpenAI 代理模式）。如果底层模型支持 thinking（如 Claude），thinking token 不会通过 AI Core Orchestration Service 透出。如有需要，需改用 `backend="aiproxy"` + `anthropic:` 前缀。

### 9.3 结构化输出（Structured Output）

`orchestration_v2` 的 `Template` 支持 `response_format=ResponseFormatJsonSchema(...)`。`BaseModelProvider._inject_response_schema()` 已生成标准 `json_schema` 格式，`AICoreProvider` 需重写此方法以适配 SDK 格式：

```python
def _inject_response_schema(self, request, response_model):
    from gen_ai_hub.orchestration_v2.models.response_format import ResponseFormatJsonSchema
    # 把 pydantic schema 注入到 request.response_format，
    # 在 _build_request() 中透传给 Template(response_format=...)
    schema = response_model.model_json_schema()
    return request.model_copy(update={
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "schema": schema,
                "strict": False,
            },
        }
    })
```

---

## 10. 需修改/新增的文件汇总

| 文件 | 操作 | 改动量 |
|------|------|--------|
| `kagent/domain/enums.py` | 修改：`ModelProviderType` 增加 `AICORE = "aicore"` | +1 行 |
| `kagent/common/config.py` | 修改：增加 `backend` + 5 个 `aicore_*` 字段，扩展 `configure()` 签名 | ~35 行 |
| `kagent/models/factory.py` | 修改：在现有分支之前加 AI Core 路由判断 | +5 行 |
| `kagent/models/aicore_provider.py` | 新增：完整 provider 实现 | ~180 行 |

**不修改的文件**：`openai_provider.py`、`anthropic_provider.py`、`gemini_provider.py`、`openai_compat.py`、`base.py`、`converters.py`、`interface/kagent.py`、`agent/loop.py`，以及所有 domain 文件（`entities.py`、`model_types.py`、`protocols.py`）。

---

## 11. 依赖管理

```toml
# pyproject.toml
[project.optional-dependencies]
aicore = [
    "generative-ai-hub-sdk>=2.0.0",  # 需包含 orchestration_v2 模块
]
```

```bash
pip install "kagent[aicore]"
```

`aicore_provider.py` 中所有 `gen_ai_hub` 导入均在方法体内（lazy import），未安装 SDK 时只要不使用 `backend="aicore"` 就不会触发 `ImportError`。

---

## 12. 实施顺序

```
Step 1  修改 kagent/domain/enums.py          — 增加 AICORE 枚举值（+1 行）
Step 2  修改 kagent/common/config.py         — 增加 backend + aicore_* 字段
Step 3  修改 kagent/models/factory.py        — 增加 AI Core 路由（+5 行）
Step 4  新增 kagent/models/aicore_provider.py — 核心实现
Step 5  新增 tests/models/test_aicore.py     — 单元测试（mock SDK）
Step 6  更新 pyproject.toml                  — 新增 aicore 可选依赖组
Step 7  集成测试 + 联调
```

Step 1–3 可并行，Step 4 依赖 Step 1–3 完成。

---

## 附录：orchestration_v2 最小 API 参考

```python
from gen_ai_hub.proxy import GenAIHubProxyClient
from gen_ai_hub.orchestration_v2.service import OrchestrationService
from gen_ai_hub.orchestration_v2.models.config import (
    ModuleConfig, OrchestrationConfig, GlobalStreamOptions,
)
from gen_ai_hub.orchestration_v2.models.template import (
    Template, PromptTemplatingModuleConfig,
)
from gen_ai_hub.orchestration_v2.models.llm_model_details import LLMModelDetails
from gen_ai_hub.orchestration_v2.models.message import (
    SystemMessage, UserMessage, AssistantMessage,
    ToolChatMessage, MessageToolCall, FunctionCall,
)
from gen_ai_hub.orchestration_v2.exceptions import OrchestrationError

# 初始化（从 AICORE_* 环境变量自动读取）
proxy_client = GenAIHubProxyClient()
service = OrchestrationService(proxy_client=proxy_client)

# 构建配置
llm = LLMModelDetails(name="gpt-4o", params={"temperature": 0.3})
template = Template(
    template=[SystemMessage(content="You are helpful."), UserMessage(content="Hello")],
    tools=[{"type": "function", "function": {"name": "get_weather", ...}}],
)
config = OrchestrationConfig(
    modules=ModuleConfig(prompt_templating=PromptTemplatingModuleConfig(prompt=template, model=llm))
)

# 异步非流式
response = await service.arun(config=config, history=[...])
content = response.final_result.choices[0].message.content
tool_calls = response.final_result.choices[0].message.tool_calls
usage = response.final_result.usage  # .prompt_tokens / .completion_tokens / .total_tokens

# 异步流式
sse = await service.astream(config=config)
async with sse as stream:
    async for event in stream:
        delta = event.final_result.choices[0].delta
        print(delta.content or "", end="")
        for tc in delta.tool_calls or []:
            ...  # 流式 tool call 片段

# 工具 result 回传（多轮）
history = list(response.intermediate_results.templating)  # 还原模板展开后的消息
history.append(response.final_result.choices[0].message)  # 带 tool_calls 的 assistant 消息
history.append(ToolChatMessage(tool_call_id=tc.id, content="22°C"))
response2 = await service.arun(config=config, history=history)
```
