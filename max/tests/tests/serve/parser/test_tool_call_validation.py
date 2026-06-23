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
"""Tests for observability-only tool-call schema-conformance checking."""

from __future__ import annotations

import logging
from typing import Any

from max.serve.parser.tool_call_validation import (
    check_tool_call_conformance,
    log_tool_call_conformance,
)

_WEATHER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "location": {"type": "string"},
        "count": {"type": "integer"},
        "unit": {"type": "string", "enum": ["C", "F"]},
    },
    "required": ["location"],
    "additionalProperties": False,
}
_SCHEMAS = {"get_weather": _WEATHER_SCHEMA, "no_args": {"type": "object"}}


def _only(calls: list[tuple[str, object]]) -> Any:
    [result] = check_tool_call_conformance(calls, _SCHEMAS)
    return result


def test_valid_args_are_conforming() -> None:
    r = _only([("get_weather", '{"location": "NYC", "unit": "F"}')])
    assert r.outcome == "valid"
    assert r.errors == []


def test_empty_args_string_is_treated_as_empty_object() -> None:
    # A no-arg tool whose schema requires nothing: empty string == {}.
    assert _only([("no_args", "")]).outcome == "valid"
    assert _only([("no_args", "   ")]).outcome == "valid"


def test_already_decoded_mapping_is_accepted() -> None:
    assert _only([("get_weather", {"location": "NYC"})]).outcome == "valid"


def test_invalid_json_is_flagged() -> None:
    r = _only([("get_weather", '{"location": "NYC"')])  # unterminated
    assert r.outcome == "invalid_json"
    assert r.errors == []


def test_unknown_tool_has_no_schema() -> None:
    assert _only([("nonexistent", "{}")]).outcome == "unknown_tool"


def test_missing_required_is_schema_mismatch() -> None:
    r = _only([("get_weather", '{"unit": "F"}')])
    assert r.outcome == "schema_mismatch"
    assert any(e.startswith("required@") for e in r.errors)


def test_wrong_type_reports_keyword_and_path() -> None:
    r = _only([("get_weather", '{"location": "NYC", "count": "five"}')])
    assert r.outcome == "schema_mismatch"
    assert "type@$.count" in r.errors


def test_enum_violation_reports_keyword() -> None:
    r = _only([("get_weather", '{"location": "NYC", "unit": "K"}')])
    assert r.outcome == "schema_mismatch"
    assert any(e.startswith("enum@") for e in r.errors)


def test_additional_properties_violation() -> None:
    r = _only([("get_weather", '{"location": "NYC", "x": 1}')])
    assert r.outcome == "schema_mismatch"
    assert any(e.startswith("additionalProperties@") for e in r.errors)


def test_errors_never_contain_argument_values() -> None:
    # PII guarantee: only schema-defined names (keyword@json_path) are recorded.
    secret = "user-secret-12345"
    r = _only([("get_weather", f'{{"location": "NYC", "count": "{secret}"}}')])
    assert r.outcome == "schema_mismatch"
    assert all(secret not in e for e in r.errors)


def test_multiple_calls_independently_classified() -> None:
    results = check_tool_call_conformance(
        [
            ("get_weather", '{"location": "NYC"}'),
            ("get_weather", "{}"),
            ("nonexistent", "{}"),
        ],
        _SCHEMAS,
    )
    assert [r.outcome for r in results] == [
        "valid",
        "schema_mismatch",
        "unknown_tool",
    ]


def test_log_is_silent_on_valid_and_never_raises(
    caplog: Any,
) -> None:
    with caplog.at_level(logging.INFO, logger="max.serve"):
        log_tool_call_conformance(
            [("get_weather", '{"location": "NYC"}')],
            _SCHEMAS,
            request_id="req-1",
            streaming=False,
        )
    assert not [r for r in caplog.records if "tool_call_conformance" in r.msg]


def test_log_emits_one_line_per_failing_call(caplog: Any) -> None:
    with caplog.at_level(logging.INFO, logger="max.serve"):
        log_tool_call_conformance(
            [
                ("get_weather", '{"location": "NYC"}'),  # valid -> silent
                ("get_weather", "{}"),  # missing required
            ],
            _SCHEMAS,
            request_id="req-2",
            streaming=True,
        )
    lines = [r for r in caplog.records if "tool_call_conformance" in r.msg]
    assert len(lines) == 1
    rendered = lines[0].getMessage()
    assert "outcome=schema_mismatch" in rendered
    assert "stream=True" in rendered
