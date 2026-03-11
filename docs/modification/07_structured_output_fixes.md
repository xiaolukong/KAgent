# Modification Log: examples/07_structured_output.py

Date: 2026-03-10

## Summary

Running `examples/07_structured_output.py` revealed 1 issue. Fixed in 1 file.

---

## Issue 1: OpenAI structured output rejects schema missing `additionalProperties: false`

**Symptom**:
```
openai.BadRequestError: Error code: 400 - Invalid schema for response_format 'OutputNum':
In context=(), 'additionalProperties' is required to be supplied and to be false.
```

**Root Cause**: OpenAI's structured output strict mode (`"strict": true`) requires every `type: "object"` node in the JSON Schema to include `"additionalProperties": false`. However, Pydantic's `model_json_schema()` does not emit this field by default.

For example, `OutputNum.model_json_schema()` produces:
```json
{
  "properties": {
    "number": { "description": "...", "title": "Number", "type": "integer" }
  },
  "required": ["number"],
  "title": "OutputNum",
  "type": "object"
}
```

OpenAI expects:
```json
{
  "properties": { ... },
  "required": ["number"],
  "type": "object",
  "additionalProperties": false
}
```

**Fix**: Added a static method `_add_additional_properties_false()` to `BaseModelProvider` in `kagent/models/base.py`. This method recursively walks the entire JSON Schema tree and injects `"additionalProperties": false` into every `type: "object"` node, including:

- Top-level schema
- Nested `properties` values
- Array `items`
- `$defs` / `definitions` (nested Pydantic model references)
- `allOf` / `anyOf` / `oneOf` combinators

This is called in `_inject_response_schema()` right after `model_json_schema()`, before the schema is sent to the API.

**File changed**: `kagent/models/base.py`

```python
@staticmethod
def _add_additional_properties_false(schema: dict[str, Any]) -> None:
    """Recursively add 'additionalProperties': false to all object nodes."""
    if schema.get("type") == "object":
        schema.setdefault("additionalProperties", False)
    # Recurse into properties, items, $defs, allOf/anyOf/oneOf ...

def _inject_response_schema(self, request, response_model):
    schema = response_model.model_json_schema()
    self._add_additional_properties_false(schema)  # <-- new line
    # ... rest unchanged
```

---

## Verification

After fix:
- `python examples/07_structured_output.py` runs successfully:
  ```
  Response:
  {"number":2}
  Tokens used: 85
  ```
- `result.parsed` correctly returns an `OutputNum(number=2)` instance.
- All 141 tests pass (0 failures, 1 warning).
