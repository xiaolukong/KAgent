"""Generate JSON Schema from Python type hints."""

from __future__ import annotations

import inspect
from typing import Any, get_args, get_origin

from pydantic import BaseModel

# Mapping from Python built-in types to JSON Schema types
_PYTHON_TO_JSON: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def python_type_to_json_schema(annotation: Any) -> dict[str, Any]:
    """Convert a Python type annotation to a JSON Schema fragment."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}

    # Pydantic model — delegate to its own schema generation
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation.model_json_schema()

    origin = get_origin(annotation)
    args = get_args(annotation)

    # list[X]
    if origin is list:
        schema: dict[str, Any] = {"type": "array"}
        if args:
            schema["items"] = python_type_to_json_schema(args[0])
        return schema

    # dict[K, V]
    if origin is dict:
        schema = {"type": "object"}
        if len(args) == 2:
            schema["additionalProperties"] = python_type_to_json_schema(args[1])
        return schema

    # X | None (Optional)
    if origin is type(int | None):  # types.UnionType
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return python_type_to_json_schema(non_none[0])
        return {"anyOf": [python_type_to_json_schema(a) for a in non_none]}

    # Literal["a", "b"]
    from typing import Literal
    from typing import get_args as _get_args

    if get_origin(annotation) is Literal:
        values = _get_args(annotation)
        return {"enum": list(values)}

    # Basic types
    if isinstance(annotation, type) and annotation in _PYTHON_TO_JSON:
        return {"type": _PYTHON_TO_JSON[annotation]}

    return {}


def function_to_json_schema(func: Any) -> dict[str, Any]:
    """Generate a JSON Schema 'parameters' object from a function signature.

    Inspects type hints and default values. Docstring is used for
    the overall description via the caller (decorator).
    """
    sig = inspect.signature(func)
    hints = {}
    try:
        hints = func.__annotations__
    except AttributeError:
        pass

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue

        annotation = hints.get(name, inspect.Parameter.empty)
        prop_schema = python_type_to_json_schema(annotation)

        if param.default is inspect.Parameter.empty:
            required.append(name)
        elif param.default is not None:
            prop_schema["default"] = param.default

        properties[name] = prop_schema

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required

    return schema
