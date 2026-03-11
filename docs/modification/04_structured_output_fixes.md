# Modification Log: examples/04_structured_output.py

Date: 2026-03-11

## Summary

Running `examples/04_structured_output.py` with the Anthropic model (`anthropic:claude-opus-4-5-20251101`) failed on structured output parsing. OpenAI and Gemini models worked correctly. The issue was fixed through multiple iterations, with the final solution using **function calling (tool-use trick)** for Anthropic models.

---

## Issue: Anthropic models ignore `response_format: json_schema` via OpenAI proxy

**Symptom**:
```
Failed to parse structured output: 5 validation errors for MovieReview
title Field required [type=missing, ...]
```

The model returned free-form Markdown with its own field names (`strengths` instead of `pros`, nested `rating` object instead of int, etc.), completely ignoring the `response_format` parameter.

**Root Cause**: The OpenAI-compatible proxy silently ignores `response_format` for Anthropic models. The parameter has no effect — the model responds with whatever format it chooses.

---

## Solution iterations

### Attempt 1: System prompt injection (rejected by user)
Injected JSON Schema instruction into the system message. Worked, but user explicitly requested **no prompt injection**.

### Attempt 2: OpenAI SDK `chat.completions.parse` (proxy doesn't enforce)
Used `type_to_response_format_param(response_model)` to send `response_format` via the SDK. The proxy still ignores it for Anthropic models — same failure.

### Attempt 3 (final): Function calling / tool-use trick

**Insight**: While the proxy ignores `response_format`, it **fully supports function calling** (`tools` + `tool_choice`) for Anthropic models. This is the same mechanism Anthropic's native API uses for structured output.

**Fix**: `AnthropicProvider._inject_response_schema()` now uses the tool-use trick in **both** native and proxy modes:

1. Define a synthetic tool whose `parameters` is the Pydantic model's JSON Schema
2. Force the model to call it via `tool_choice`
3. Extract the structured data from `tool_call.arguments`

`BaseModelProvider._extract_structured_output()` was added to handle extraction from either tool call arguments (Anthropic) or text content (OpenAI/Gemini).

**Files changed**:
- `kagent/models/anthropic_provider.py` — unified tool-use trick for both modes
- `kagent/models/base.py` — added `_extract_structured_output()` to `complete()`

```python
# AnthropicProvider._inject_response_schema (both native & proxy)
def _inject_response_schema(self, request, response_model):
    struct_tool = ToolDefinition(
        name=f"structured_output_{response_model.__name__}",
        description=f"Output structured data as {response_model.__name__}",
        parameters=response_model.model_json_schema(),
    )
    tools = list(request.tools or []) + [struct_tool]
    tool_choice = (
        {"type": "function", "function": {"name": struct_tool.name}}  # proxy
        if self._use_openai_compat else
        {"type": "tool", "name": struct_tool.name}                    # native
    )
    return request.model_copy(update={"tools": tools, "tool_choice": tool_choice})

# BaseModelProvider._extract_structured_output
def _extract_structured_output(response, response_model):
    # 1. Check tool calls (tool-use trick)
    if response.tool_calls:
        for tc in response.tool_calls:
            if tc.name.startswith("structured_output_"):
                return response_model.model_validate(tc.arguments)
    # 2. Fall back to text content (response_format: json_schema)
    if response.content:
        return parse_structured_output(response.content, response_model)
```

### Other providers (unchanged approach)

- **OpenAI**: Uses `chat.completions.parse(response_format=PydanticType)` — SDK handles schema injection and parsing end-to-end.
- **Gemini (proxy)**: Uses `response_format: json_schema` via base class — proxy enforces it.
- **Gemini (native)**: Uses `generation_config.response_schema`.

---

## Additional fix: `parse_structured_output` wrapper handling

Added a two-pass parsing strategy in `kagent/models/converters.py`:
1. Direct `model_validate_json()` (fast path)
2. Unwrap single-key wrapper dict (e.g. `{"review": {...}}`)
3. Re-raise original error if both fail

---

## Verification

All three models produce correct structured output:
```
[openai:gpt-5]                        Title: Inception  Rating: 9/10  ✓
[anthropic:claude-opus-4-5-20251101]   Title: Inception  Rating: 9/10  ✓
[gemini:gemini-2.5-pro]               Title: Inception  Rating: 9/10  ✓
```
- All 141 tests pass (0 failures, 1 warning).
