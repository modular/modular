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

from max.serve.parser import (
    normalize_message_tool_calls,
    normalize_tool_call_arguments,
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
