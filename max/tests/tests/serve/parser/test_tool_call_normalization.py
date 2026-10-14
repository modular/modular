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

from max.serve.parser import normalize_tool_call_arguments
from max.serve.parser.tool_call_normalization import (
    _normalize_tools_parameters,
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
