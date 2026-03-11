"""Tests for schema generation from Python type hints."""

from typing import Literal

from pydantic import BaseModel

from kagent.tools.schema_gen import function_to_json_schema, python_type_to_json_schema


class TestPythonTypeToJsonSchema:
    def test_str(self):
        assert python_type_to_json_schema(str) == {"type": "string"}

    def test_int(self):
        assert python_type_to_json_schema(int) == {"type": "integer"}

    def test_float(self):
        assert python_type_to_json_schema(float) == {"type": "number"}

    def test_bool(self):
        assert python_type_to_json_schema(bool) == {"type": "boolean"}

    def test_list_of_str(self):
        schema = python_type_to_json_schema(list[str])
        assert schema["type"] == "array"
        assert schema["items"] == {"type": "string"}

    def test_dict_str_int(self):
        schema = python_type_to_json_schema(dict[str, int])
        assert schema["type"] == "object"
        assert schema["additionalProperties"] == {"type": "integer"}

    def test_literal(self):
        schema = python_type_to_json_schema(Literal["a", "b", "c"])
        assert schema == {"enum": ["a", "b", "c"]}

    def test_optional(self):
        schema = python_type_to_json_schema(int | None)
        assert schema == {"type": "integer"}

    def test_pydantic_model(self):
        class MyModel(BaseModel):
            name: str
            age: int

        schema = python_type_to_json_schema(MyModel)
        assert "properties" in schema
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]


class TestFunctionToJsonSchema:
    def test_basic_function(self):
        def func(a: int, b: str) -> bool:
            ...

        schema = function_to_json_schema(func)
        assert schema["type"] == "object"
        assert "a" in schema["properties"]
        assert "b" in schema["properties"]
        assert schema["required"] == ["a", "b"]

    def test_with_defaults(self):
        def func(a: int, b: str = "default") -> bool:
            ...

        schema = function_to_json_schema(func)
        assert schema["required"] == ["a"]
        assert schema["properties"]["b"]["default"] == "default"

    def test_skips_self(self):
        def method(self, x: int) -> int:
            ...

        schema = function_to_json_schema(method)
        assert "self" not in schema["properties"]

    def test_no_annotations(self):
        def func(a, b):
            ...

        schema = function_to_json_schema(func)
        assert "a" in schema["properties"]
        assert schema["properties"]["a"] == {}
