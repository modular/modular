# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from typing import Any

import pytest
from max.serve.parser import (
    normalize_message_tool_calls,
    normalize_tool_call_arguments,
)
from max.serve.parser.tool_call_normalization import (
    _normalize_tools_parameters,
    _validate_response_format_schema,
    normalize_response_format_schema,
)


def test_normalize_deserializes_json_string_arguments() -> None:
    tc: dict[str, Any] = {
        "id": "call_abc",
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": '{"location": "NYC", "unit": "F"}',
        },
    }
    [out] = normalize_tool_call_arguments([tc])
    assert out["function"]["arguments"] == {"location": "NYC", "unit": "F"}
    # Source dict is not mutated.
    assert isinstance(tc["function"]["arguments"], str)


def test_normalize_passes_through_dict_arguments() -> None:
    args = {"location": "NYC"}
    tc = {
        "id": "call_abc",
        "type": "function",
        "function": {"name": "get_weather", "arguments": args},
    }
    [out] = normalize_tool_call_arguments([tc])
    assert out["function"]["arguments"] == args


def test_normalize_passes_through_list_arguments() -> None:
    args = [{"location": "NYC"}]
    tc = {
        "id": "call_abc",
        "type": "function",
        "function": {"name": "get_weather", "arguments": args},
    }
    [out] = normalize_tool_call_arguments([tc])
    assert out["function"]["arguments"] == args


def test_normalize_passes_through_malformed_json_arguments() -> None:
    """Malformed JSON is left untouched so we don't swallow client errors."""
    tc = {
        "id": "call_abc",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "not json"},
    }
    [out] = normalize_tool_call_arguments([tc])
    assert out["function"]["arguments"] == "not json"


def test_normalize_empty_string_arguments_become_empty_dict() -> None:
    """vLLM parity: empty string args become ``{}`` so templates render."""
    tc = {
        "id": "call_abc",
        "type": "function",
        "function": {"name": "get_weather", "arguments": ""},
    }
    [out] = normalize_tool_call_arguments([tc])
    assert out["function"]["arguments"] == {}


def test_normalize_none_arguments_become_empty_dict() -> None:
    tc = {
        "id": "call_abc",
        "type": "function",
        "function": {"name": "get_weather", "arguments": None},
    }
    [out] = normalize_tool_call_arguments([tc])
    assert out["function"]["arguments"] == {}


def test_normalize_missing_arguments_become_empty_dict() -> None:
    tc = {
        "id": "call_abc",
        "type": "function",
        "function": {"name": "get_weather"},
    }
    [out] = normalize_tool_call_arguments([tc])
    assert out["function"]["arguments"] == {}


def test_normalize_handles_missing_function() -> None:
    tc = {"id": "call_abc", "type": "function"}
    [out] = normalize_tool_call_arguments([tc])
    assert out == tc


def test_normalize_handles_empty_list() -> None:
    assert normalize_tool_call_arguments([]) == []


def test_normalize_message_tool_calls_skips_non_assistant() -> None:
    msg: dict[str, Any] = {
        "role": "user",
        "content": "hi",
        "tool_calls": [
            {
                "function": {
                    "name": "x",
                    "arguments": '{"a": 1}',
                }
            }
        ],
    }
    out = normalize_message_tool_calls(msg)
    # Non-assistant messages are passed through unchanged.
    assert out is msg


def test_normalize_message_tool_calls_assistant_decodes_args() -> None:
    msg: dict[str, Any] = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "x",
                    "arguments": '{"a": 1}',
                },
            }
        ],
    }
    out = normalize_message_tool_calls(msg)
    assert out["tool_calls"][0]["function"]["arguments"] == {"a": 1}
    # Input dict is not mutated.
    assert msg["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'


def test_normalize_message_tool_calls_empty_list_passthrough() -> None:
    msg: dict[str, Any] = {
        "role": "assistant",
        "content": "ok",
        "tool_calls": [],
    }
    out = normalize_message_tool_calls(msg)
    # Empty list is preserved here; the openai_routes layer is responsible
    # for dropping empty tool_calls from incoming requests.
    assert out is msg


def test_normalize_tools_params_replaces_null_with_empty_dict() -> None:
    tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {"name": "test", "parameters": None},
        }
    ]
    [out] = _normalize_tools_parameters(tools)
    assert out["function"]["parameters"] == {}
    # Source dict not mutated.
    assert tools[0]["function"]["parameters"] is None


def test_normalize_tools_params_passes_through_valid_object() -> None:
    params = {"type": "object", "properties": {"city": {"type": "string"}}}
    tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {"name": "test", "parameters": params},
        }
    ]
    [out] = _normalize_tools_parameters(tools)
    assert out["function"]["parameters"] == params


def test_normalize_tools_params_passes_through_missing_parameters() -> None:
    """Missing parameters means 'empty parameter list' per the OpenAPI spec."""
    tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {"name": "test"},
        }
    ]
    [out] = _normalize_tools_parameters(tools)
    # Treat omission the same as null: normalize to {}.
    assert out["function"]["parameters"] == {}


def test_normalize_tools_params_handles_empty_list() -> None:
    assert _normalize_tools_parameters([]) == []


def test_normalize_tools_params_handles_missing_function() -> None:
    """Tool entry without a function dict is passed through unchanged."""
    tools: list[dict[str, Any]] = [{"type": "function"}]
    out = _normalize_tools_parameters(tools)
    assert out == tools


def test_validate_response_format_schema_accepts_object_root() -> None:
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    # Should not raise.
    _validate_response_format_schema(schema)


def test_validate_response_format_schema_rejects_string_root() -> None:
    """A single non-object scalar root is rejected (callers expect an object)."""
    with pytest.raises(ValueError, match=r"must be\s+'object'"):
        _validate_response_format_schema({"type": "string"})


def test_validate_response_format_schema_rejects_array_root() -> None:
    """A single non-object root (array) is rejected."""
    with pytest.raises(ValueError, match=r"must be\s+'object'"):
        _validate_response_format_schema(
            {"type": "array", "items": {"type": "string"}}
        )


def test_validate_response_format_schema_accepts_missing_type() -> None:
    """A missing root ``type`` means "any" and is valid JSON Schema."""
    _validate_response_format_schema({"properties": {"x": {"type": "string"}}})


def test_validate_response_format_schema_accepts_type_union() -> None:
    """A root ``type`` union is valid JSON Schema and compiles fine."""
    _validate_response_format_schema({"type": ["object", "array", "string"]})


def test_validate_response_format_schema_rejects_unknown_type() -> None:
    """A root ``type`` that is a single non-object value is rejected."""
    with pytest.raises(ValueError, match=r"must be\s+'object'"):
        _validate_response_format_schema({"type": "frobnicate"})


def test_validate_response_format_schema_rejects_bad_type_union() -> None:
    """A root ``type`` list with a non-JSON-Schema member is rejected."""
    with pytest.raises(ValueError, match=r"list must"):
        _validate_response_format_schema({"type": ["object", "frob"]})


def test_validate_response_format_schema_accepts_none_schema() -> None:
    """A None schema is acceptable (no validation requested)."""
    _validate_response_format_schema(None)


def test_validate_response_format_schema_accepts_empty_dict() -> None:
    """An empty schema dict is acceptable (treated as no constraint)."""
    _validate_response_format_schema({})


# ---------------------------------------------------------------------------
# normalize_response_format_schema (runaway-output regression)
# ---------------------------------------------------------------------------


def test_normalize_response_format_schema_injects_object_for_missing_type() -> (
    None
):
    """An untyped schema with ``properties`` is inferred to type ``"object"``.

    Regression for the runaway-output incident: an untyped root with
    object-implying keywords compiles to a grammar that allows a bare
    unbounded top-level value, which lets a looping model run to
    ``max_length``. Inferring ``"object"`` (mirroring xgrammar) forces a
    leading ``{`` and restores a terminating grammar.
    """
    schema: dict[str, Any] = {"properties": {"x": {}}}
    normalized = normalize_response_format_schema(schema)
    assert normalized["type"] == "object"
    # The empty ``{}`` property value is preserved ("any value").
    assert normalized["properties"] == {"x": {}}
    # Input is not mutated.
    assert "type" not in schema


def test_normalize_response_format_schema_infers_from_required() -> None:
    """``required`` alone implies an object type."""
    assert normalize_response_format_schema({"required": ["a"]})["type"] == (
        "object"
    )


def test_normalize_response_format_schema_infers_from_additional_props() -> (
    None
):
    """``additionalProperties`` alone implies an object type."""
    normalized = normalize_response_format_schema(
        {"additionalProperties": False}
    )
    assert normalized["type"] == "object"


def test_normalize_response_format_schema_recurses_into_properties() -> None:
    """A nested untyped object-shaped subschema is inferred recursively."""
    schema: dict[str, Any] = {
        "properties": {"inner": {"properties": {"y": {}}}}
    }
    normalized = normalize_response_format_schema(schema)
    assert normalized["type"] == "object"
    assert normalized["properties"]["inner"]["type"] == "object"
    # The innermost genuinely-empty ``{}`` stays "any value".
    assert normalized["properties"]["inner"]["properties"]["y"] == {}


def test_normalize_response_format_schema_recurses_into_items_and_unions() -> (
    None
):
    """Inference reaches ``items``, ``anyOf``, and ``$defs`` subschemas."""
    schema = {
        "type": "object",
        "properties": {
            "a": {"items": {"properties": {"z": {}}}},
            "b": {"anyOf": [{"properties": {"q": {}}}, {"type": "null"}]},
        },
        "$defs": {"D": {"properties": {"k": {}}}},
    }
    normalized = normalize_response_format_schema(schema)
    assert normalized["properties"]["a"]["items"]["type"] == "object"
    assert normalized["properties"]["b"]["anyOf"][0]["type"] == "object"
    assert normalized["$defs"]["D"]["type"] == "object"


def test_normalize_response_format_schema_leaves_present_type() -> None:
    """A schema with an explicit root ``type`` is returned unchanged."""
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    assert normalize_response_format_schema(schema) is schema


def test_normalize_response_format_schema_leaves_type_union() -> None:
    """A root ``type`` union (an explicit caller choice) is left untouched."""
    schema = {"type": ["object", "string"]}
    assert normalize_response_format_schema(schema) is schema


def test_normalize_response_format_schema_preserves_empty_any() -> None:
    """A genuinely empty ``{}`` (no object keyword) stays "any value"."""
    schema: dict[str, Any] = {}
    assert normalize_response_format_schema(schema) == {}
    # No spurious type injected.
    assert "type" not in normalize_response_format_schema({})
