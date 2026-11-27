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
"""Checks the graders of the AGENT-914 admission cases offline, so a served
response is judged under the draft that gives the schema its meaning."""

import json

import pytest
from scenarios._constrained_stream import classify_served_json
from scenarios._schema_admission_cases import CASES, SERVE
from scenarios.schema_admission_policy import (
    _classify_tool_arguments,
    _validator,
)

# Per case: values of ``v`` the schema admits, then values it rules out.
_EXAMPLES: dict[str, tuple[list[object], list[object]]] = {
    "ref_beside_same_object_sibling": ([2, 3], [4, "a"]),
    "ref_beside_properties_and_required": (
        [{"id": "a", "n": 1}],
        [{"id": "a"}, {"n": 1}],
    ),
    "all_of_with_two_any_ofs": (
        [{"b": 1}, {"a": 1, "c": 2}],
        [{"a": 1}, {"c": 1}],
    ),
    "prefix_items_past_max_items": ([["a", 1]], [["a", 1, True], [1]]),
    "untyped_keywords_of_two_kinds": (
        [5, "hello", True, None],
        [-1, "Hello1"],
    ),
}

_BY_NAME = {case.name: case for case in CASES}


def _content_body(document: object) -> str:
    message = {"content": json.dumps(document)}
    return json.dumps(
        {"choices": [{"message": message, "finish_reason": "stop"}]}
    )


def _tool_body(
    name: str,
    *arguments: str,
    finish: str = "tool_calls",
) -> str:
    calls = [
        {"function": {"name": name, "arguments": argument}}
        for argument in arguments
    ]
    message = {"content": None, "tool_calls": calls}
    return json.dumps(
        {"choices": [{"message": message, "finish_reason": finish}]}
    )


def test_every_example_names_a_served_tool_case() -> None:
    for name in _EXAMPLES:
        case = _BY_NAME[name]
        assert case.expect == SERVE and case.as_tool, name


@pytest.mark.parametrize("name", sorted(_EXAMPLES))
def test_graders_follow_the_case_draft(name: str) -> None:
    case = _BY_NAME[name]
    admitted, ruled_out = _EXAMPLES[name]
    for value in admitted:
        document = {"v": value}
        assert classify_served_json(
            _content_body(document), case.schema, _validator(case)
        ) == ("conformant", None), value
        assert _classify_tool_arguments(
            _tool_body(case.name, json.dumps(document)), case
        ) == (
            "conformant",
            None,
        ), value
    for value in ruled_out:
        document = {"v": value}
        grade, _ = classify_served_json(
            _content_body(document), case.schema, _validator(case)
        )
        assert grade == "invalid", value
        grade, _ = _classify_tool_arguments(
            _tool_body(case.name, json.dumps(document)), case
        )
        assert grade == "invalid", value


def test_a_truncated_tool_call_is_truncated() -> None:
    body = json.dumps(
        {"choices": [{"message": {"content": None}, "finish_reason": "length"}]}
    )
    case = _BY_NAME["all_of_with_two_any_ofs"]
    assert _classify_tool_arguments(body, case) == ("truncated", None)


def test_a_call_naming_another_tool_is_invalid() -> None:
    case = _BY_NAME["ref_beside_same_object_sibling"]
    body = _tool_body("other", json.dumps({"v": 2}))
    grade, detail = _classify_tool_arguments(body, case)
    assert grade == "invalid" and detail and "expected" in detail


def test_every_call_is_graded() -> None:
    case = _BY_NAME["ref_beside_same_object_sibling"]
    body = _tool_body(case.name, json.dumps({"v": 2}), json.dumps({"v": 4}))
    assert _classify_tool_arguments(body, case)[0] == "invalid"
    body = _tool_body(case.name, json.dumps({"v": 2}), json.dumps({"v": 3}))
    assert _classify_tool_arguments(body, case) == ("conformant", None)


def test_a_length_finished_call_that_overran_is_invalid() -> None:
    case = _BY_NAME["ref_beside_same_object_sibling"]
    body = _tool_body(case.name, '{"v":2} trailing text', finish="length")
    assert _classify_tool_arguments(body, case)[0] == "invalid"
    body = _tool_body(case.name, '{"v":', finish="length")
    assert _classify_tool_arguments(body, case) == ("truncated", None)
