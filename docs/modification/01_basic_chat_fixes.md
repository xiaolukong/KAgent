# Modification Log: examples/01_basic_chat.py

Date: 2026-03-10

## Summary

Running `examples/01_basic_chat.py` revealed 5 issues across 4 rounds of debugging.
Fixed across 6 files.

---

## Round 1

### Issue 1: Prefixed model string leaked into API request

**Symptom**: The `model` parameter sent to the API was `"openai:claude-sonnet-4-6"` instead of `"claude-sonnet-4-6"`.

**Root Cause**: `AgentConfig.model` stores the full provider-prefixed string (e.g. `"openai:claude-sonnet-4-6"`). This was passed to `PromptBuilder.__init__(model=config.model)`, which injected it into `ModelRequest.model`. In `OpenAIProvider._build_kwargs()`, the logic `request.model or self._config.model_name` picked up the prefixed string instead of falling through to the clean name from `self._config.model_name`.

**Fix**: Removed the `model` parameter from `PromptBuilder`. The provider's `_build_kwargs()` now always uses `self._config.model_name` (the clean model name without provider prefix, as extracted by `ModelProviderFactory`).

**Files changed**:
1. `kagent/agent/prompt_builder.py` — Removed `model` parameter from `__init__()` and from `ModelRequest` construction in `build()`.
2. `kagent/agent/agent.py` — Removed `model=config.model` from `PromptBuilder()` construction.
3. `tests/agent/test_loop.py` — Updated `_make_loop()` helper to match new `PromptBuilder` signature.

```python
# kagent/agent/prompt_builder.py — Before
def __init__(self, system_prompt, model=None, temperature=None, ...):
    self._model = model

def build(self, messages, tools=None):
    return ModelRequest(messages=messages, model=self._model, ...)

# kagent/agent/prompt_builder.py — After
def __init__(self, system_prompt, temperature=None, ...):
    # No model parameter

def build(self, messages, tools=None):
    return ModelRequest(messages=messages, ...)  # model=None, provider decides
```

---

## Round 2

### Issue 2: `AnthropicProvider` cannot work with OpenAI-compatible proxies

**Symptom**: `anthropic.AuthenticationError: Error code: 401 - {'error': 'Unauthorized', 'message': 'Invalid API key'}`

**Root Cause**: The example uses `model="anthropic:claude-opus-4-6"` with an SAP AI Proxy (`base_url="https://sap-ai-proxy-xxx.cfapps.eu12.hana.ondemand.com/v1"`). This proxy is **OpenAI-compatible** — it expects requests at `/chat/completions` with `Authorization: Bearer` headers and OpenAI-format payloads.

However, `AnthropicProvider` unconditionally used the Anthropic native SDK (`anthropic.AsyncAnthropic`), which sends requests to `/v1/messages` with `x-api-key` headers — a completely different protocol that the proxy does not understand, resulting in a 401 authentication error.

This is a **protocol mismatch**, not an API key issue.

**Fix**: Rewrote `AnthropicProvider` with **dual-mode support**:

- **Native mode** (default): When `base_url` is `None` or points to Anthropic's official API (`api.anthropic.com`), uses the Anthropic native SDK as before.
- **Proxy mode** (automatic): When `base_url` is a custom address (e.g. any OpenAI-compatible proxy), automatically uses the OpenAI SDK internally, converting requests to `/chat/completions` format.

Detection logic uses URL hostname comparison against known Anthropic hosts (`api.anthropic.com`). The provider transparently delegates to the correct protocol — users simply write `"anthropic:claude-opus-4-6"` and it works against both the Anthropic API directly and any OpenAI-compatible proxy.

**File changed**: `kagent/models/anthropic_provider.py` — complete rewrite with dual-mode architecture.

Key code paths:
```python
class AnthropicProvider(BaseModelProvider):
    def __init__(self, config):
        self._use_openai_compat = not _is_anthropic_native_url(self._config.base_url)

        if self._use_openai_compat:
            # Proxy mode — OpenAI SDK
            self._openai_client = openai.AsyncOpenAI(api_key=..., base_url=...)
        else:
            # Native mode — Anthropic SDK
            self._client = anthropic.AsyncAnthropic(api_key=..., base_url=...)

    async def _do_complete(self, request):
        if self._use_openai_compat:
            return await self._do_complete_openai(request)   # /chat/completions
        return await self._do_complete_native(request)       # /v1/messages
```

### Issue 3: Debug code left in production files

**Symptom**: Extra `print(kwargs)` statement in `anthropic_provider.py` and hardcoded temperature override in `openai_provider.py`.

**Root Cause**: Debug/development code was not cleaned up before committing.

**Fix**:
- `kagent/models/anthropic_provider.py` — Removed `print(kwargs)` on line 79 (cleaned up as part of the rewrite).
- `kagent/models/openai_provider.py` — Removed hardcoded temperature override (`if kwargs["model"] not in ["gpt-4o", "gpt-4.1"]: kwargs["temperature"] = 1`).

---

## Round 3

### Issue 4: `GeminiProvider` cannot work with OpenAI-compatible proxies

**Symptom**: `google.api_core.exceptions.InvalidArgument: 400 API key not valid. Please pass a valid API key.`

**Root Cause**: Identical problem to Issue 2 but for the Gemini provider. The example uses `model="gemini:gemini-2.5-flash"` with the SAP AI Proxy. `GeminiProvider` unconditionally used the native `google.generativeai` SDK, which connects via **gRPC** to `generativelanguage.googleapis.com` — completely bypassing the custom `base_url`. The proxy is an HTTP-based OpenAI-compatible endpoint, so the gRPC connection goes to Google's server directly with an invalid API key.

This is a **dual protocol mismatch**: wrong transport (gRPC vs HTTP) AND wrong endpoint (Google vs proxy).

**Fix**: Rewrote `GeminiProvider` with the same **dual-mode architecture** as `AnthropicProvider`:

- **Native mode** (default): When `base_url` is `None` or points to Google's official APIs (`generativelanguage.googleapis.com`, `aiplatform.googleapis.com`), uses the native `google.generativeai` SDK.
- **Proxy mode** (automatic): When `base_url` is a custom address, uses the OpenAI SDK internally.

**File changed**: `kagent/models/gemini_provider.py` — complete rewrite with dual-mode architecture.

Key code paths:
```python
class GeminiProvider(BaseModelProvider):
    def __init__(self, config):
        self._use_openai_compat = not _is_gemini_native_url(self._config.base_url)

        if self._use_openai_compat:
            # Proxy mode — OpenAI SDK
            self._openai_client = openai.AsyncOpenAI(api_key=..., base_url=...)
        else:
            # Native mode — Google generativeai SDK (gRPC)
            genai.configure(api_key=...)
            self._model = genai.GenerativeModel(model_name)

    async def _do_complete(self, request):
        if self._use_openai_compat:
            return await self._do_complete_openai(request)   # /chat/completions
        return await self._do_complete_native(request)       # gRPC to googleapis.com
```

---

## Round 4

### Issue 5: Models other than gpt-4o / gpt-4.1 reject non-1 temperature

**Symptom**: `openai:gpt-5` returns an error because it only accepts `temperature=1`. The `ModelConfig` default temperature is `0.7`, which gets sent to the API and rejected.

**Root Cause**: OpenAI's newer models (gpt-5, o-series, etc.) and non-OpenAI models accessed via proxy only accept `temperature=1`. Only `gpt-4o` and `gpt-4.1` support arbitrary temperature values. The framework had no awareness of this constraint and passed whatever temperature was configured directly to the API.

Note: A previous version of `openai_provider.py` had a hardcoded workaround for this (`if kwargs["model"] not in ["gpt-4o", "gpt-4.1"]: kwargs["temperature"] = 1`), which was mistakenly removed as "debug code" in Round 2.

**Fix**: Introduced a `clamp_temperature()` function in `kagent/models/openai_provider.py` with a whitelist approach:

- Models in `_FLEXIBLE_TEMP_MODELS` (`gpt-4o`, `gpt-4.1`) keep their configured temperature.
- All other OpenAI-provider models are automatically clamped to `temperature=1`.

This constraint only applies to `OpenAIProvider`. Anthropic and Gemini models do not have this temperature restriction, so their proxy modes pass temperature through unchanged.

**File changed**: `kagent/models/openai_provider.py` — Added `_FLEXIBLE_TEMP_MODELS` set and `clamp_temperature()` function; updated `_build_kwargs()` to use it.

```python
# kagent/models/openai_provider.py
_FLEXIBLE_TEMP_MODELS = {"gpt-4o", "gpt-4.1"}

def clamp_temperature(model_name: str, temperature: float | None) -> float:
    """Return temperature unchanged for models that accept it, else 1."""
    if model_name in _FLEXIBLE_TEMP_MODELS:
        return temperature if temperature is not None else 1.0
    return 1.0
```

---

## Verification

After all fixes:
- `python examples/01_basic_chat.py` with `model="anthropic:claude-opus-4-6"` runs successfully:
  ```
  Response:
  I'm Claude, an AI assistant made by Anthropic. ...
  Tokens used: 74
  ```
- `python examples/01_basic_chat.py` with `model="gemini:gemini-2.5-flash"` runs successfully:
  ```
  Response:
  I am a large language model, trained by Google.
  Tokens used: 49
  ```
- `python examples/01_basic_chat.py` with `model="openai:gpt-5"` runs successfully (temperature auto-clamped to 1):
  ```
  Response:
  I'm ChatGPT, an AI assistant created by OpenAI. ...
  Tokens used: 253
  ```
- All 141 tests pass (0 failures, 1 warning).

## Design Pattern Summary

All three non-OpenAI providers now share the same dual-mode architecture:

| Provider | Native SDK | Native Protocol | Proxy Trigger | Proxy Fallback |
|----------|-----------|----------------|---------------|----------------|
| `AnthropicProvider` | `anthropic` | HTTP `/v1/messages` | `base_url` not `api.anthropic.com` | `openai` SDK |
| `GeminiProvider` | `google.generativeai` | gRPC | `base_url` not `googleapis.com` | `openai` SDK |
| `OpenAIProvider` | `openai` | HTTP `/chat/completions` | N/A (always OpenAI-compatible) | N/A |

Temperature clamping applies only to `OpenAIProvider` (direct `openai:` prefix):

| Model | Temperature Behaviour |
|-------|----------------------|
| `gpt-4o`, `gpt-4.1` | User-configured value preserved |
| Other OpenAI models (`gpt-5`, etc.) | Auto-clamped to `1` |
| Anthropic / Gemini models (via proxy) | User-configured value preserved (no constraint) |
