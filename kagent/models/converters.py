"""Converters — translate vendor-specific formats to/from domain types."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from kagent.domain.entities import Message, ToolCall, ToolDefinition
from kagent.domain.enums import Role
from kagent.domain.model_types import ModelResponse, TokenUsage

# ── OpenAI helpers ───────────────────────────────────────────────────────────


def messages_to_openai(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert domain Messages to OpenAI chat format."""
    result: list[dict[str, Any]] = []
    for msg in messages:
        entry: dict[str, Any] = {"role": msg.role.value}
        if msg.content is not None:
            entry["content"] = msg.content
        if msg.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in msg.tool_calls
            ]
        if msg.tool_call_id:
            entry["tool_call_id"] = msg.tool_call_id
        result.append(entry)
    return result


def tools_to_openai(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """Convert domain ToolDefinitions to OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


def openai_response_to_model_response(raw: Any) -> ModelResponse:
    """Convert an OpenAI ChatCompletion response to a domain ModelResponse."""
    choice = raw.choices[0]
    message = choice.message

    tool_calls: list[ToolCall] | None = None
    if message.tool_calls:
        tool_calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments) if tc.function.arguments else {},
            )
            for tc in message.tool_calls
        ]

    usage = None
    if raw.usage:
        usage = TokenUsage(
            prompt_tokens=raw.usage.prompt_tokens,
            completion_tokens=raw.usage.completion_tokens,
            total_tokens=raw.usage.total_tokens,
        )

    return ModelResponse(
        content=message.content,
        tool_calls=tool_calls,
        usage=usage,
        model=raw.model,
    )


# ── Anthropic helpers ────────────────────────────────────────────────────────


def messages_to_anthropic(messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert domain Messages to Anthropic format.

    Returns (system_prompt, messages_list).
    Anthropic separates the system prompt from the message list.
    """
    system_prompt: str | None = None
    result: list[dict[str, Any]] = []

    for msg in messages:
        if msg.role == Role.SYSTEM:
            system_prompt = msg.content
            continue

        if msg.role == Role.TOOL:
            result.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id,
                            "content": msg.content or "",
                        }
                    ],
                }
            )
            continue

        entry: dict[str, Any] = {"role": msg.role.value}
        content_blocks: list[dict[str, Any]] = []

        if msg.content:
            content_blocks.append({"type": "text", "text": msg.content})

        if msg.tool_calls:
            for tc in msg.tool_calls:
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments,
                    }
                )

        entry["content"] = content_blocks if content_blocks else (msg.content or "")
        result.append(entry)

    return system_prompt, result


def tools_to_anthropic(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """Convert domain ToolDefinitions to Anthropic tool format."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters,
        }
        for t in tools
    ]


def anthropic_response_to_model_response(raw: Any) -> ModelResponse:
    """Convert an Anthropic Message response to a domain ModelResponse."""
    content_text: str | None = None
    tool_calls: list[ToolCall] = []
    thinking_parts: list[str] = []

    for block in raw.content:
        if block.type == "thinking":
            thinking_parts.append(block.thinking)
        elif block.type == "text":
            content_text = block.text
        elif block.type == "tool_use":
            tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input))

    usage = None
    if raw.usage:
        usage = TokenUsage(
            prompt_tokens=raw.usage.input_tokens,
            completion_tokens=raw.usage.output_tokens,
            total_tokens=raw.usage.input_tokens + raw.usage.output_tokens,
        )

    metadata: dict[str, Any] = {}
    if thinking_parts:
        metadata["thinking"] = "".join(thinking_parts)

    return ModelResponse(
        content=content_text,
        tool_calls=tool_calls if tool_calls else None,
        usage=usage,
        model=raw.model,
        metadata=metadata,
    )


# ── Gemini helpers ───────────────────────────────────────────────────────────


def messages_to_gemini(messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert domain Messages to Gemini format.

    Returns (system_instruction, contents_list).
    """
    system_instruction: str | None = None
    contents: list[dict[str, Any]] = []

    for msg in messages:
        if msg.role == Role.SYSTEM:
            system_instruction = msg.content
            continue

        role = "user" if msg.role in (Role.USER, Role.TOOL) else "model"
        parts: list[dict[str, Any]] = []

        if msg.content:
            parts.append({"text": msg.content})

        if msg.tool_calls:
            for tc in msg.tool_calls:
                parts.append(
                    {
                        "functionCall": {"name": tc.name, "args": tc.arguments},
                    }
                )

        if msg.role == Role.TOOL and msg.tool_call_id:
            parts = [
                {
                    "functionResponse": {
                        "name": msg.metadata.get("tool_name", ""),
                        "response": {"result": msg.content},
                    }
                }
            ]

        contents.append({"role": role, "parts": parts})

    return system_instruction, contents


def tools_to_gemini(tools: list[ToolDefinition]) -> list[dict[str, Any]]:
    """Convert domain ToolDefinitions to Gemini function declarations."""
    declarations = []
    for t in tools:
        decl: dict[str, Any] = {
            "name": t.name,
            "description": t.description,
        }
        if t.parameters:
            decl["parameters"] = t.parameters
        declarations.append(decl)
    return [{"functionDeclarations": declarations}]


# ── Structured output helpers ────────────────────────────────────────────────


def parse_structured_output(
    content: str | None,
    response_model: type[BaseModel],
) -> BaseModel:
    """Parse raw LLM content into a Pydantic model instance.

    Handles two common LLM quirks:
    1. Markdown code fences wrapping the JSON.
    2. An extra wrapper key (e.g. ``{"review": {...}}``) — some models wrap
       the output in a single-key object instead of returning the flat schema.
       When direct parsing fails and the JSON has exactly one key whose value
       is a dict, we retry with that inner dict.
    """
    if content is None:
        raise ValueError("Cannot parse structured output from empty content")

    # Strip markdown code fences if present
    text = content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # remove opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    # First attempt: direct parse
    try:
        return response_model.model_validate_json(text)
    except Exception:
        pass

    # Second attempt: unwrap single-key wrapper
    try:
        data = json.loads(text)
        if isinstance(data, dict) and len(data) == 1:
            inner = next(iter(data.values()))
            if isinstance(inner, dict):
                return response_model.model_validate(inner)
    except Exception:
        pass

    # If both fail, raise with the original error for clarity
    return response_model.model_validate_json(text)
