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

from max.pipelines.architectures.deepseekV3.tokenizer import (
    _normalize_tool_call_arguments,
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
    [out] = _normalize_tool_call_arguments([tc])
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
    [out] = _normalize_tool_call_arguments([tc])
    assert out["function"]["arguments"] == args


def test_normalize_passes_through_malformed_json_arguments() -> None:
    """Malformed JSON is left untouched so we don't swallow client errors."""
    tc = {
        "id": "call_abc",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "not json"},
    }
    [out] = _normalize_tool_call_arguments([tc])
    assert out["function"]["arguments"] == "not json"


def test_normalize_handles_missing_function() -> None:
    tc = {"id": "call_abc", "type": "function"}
    [out] = _normalize_tool_call_arguments([tc])
    assert out == tc


def test_normalize_handles_empty_list() -> None:
    assert _normalize_tool_call_arguments([]) == []
