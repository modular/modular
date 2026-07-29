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

"""Tests for result JSON flattening and loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from max.benchmark.results_to_csv import (
    JSONObject,
    flatten,
    load_result_rows,
)


def test_flatten_nested_dicts_and_lists() -> None:
    obj: JSONObject = {
        "date": "2026-07-29",
        "request_throughput": 12.5,
        "server_metrics": {"prefill_batch_count": 3, "nested": {"a": 1}},
        "input_lens": [1, 2, 3],
        "max_concurrent_conversations": None,
        "steady_state_detected": True,
    }
    flat = flatten(obj)
    assert flat["date"] == "2026-07-29"
    # Non-string scalars are JSON-encoded.
    assert flat["request_throughput"] == "12.5"
    assert flat["server_metrics.prefill_batch_count"] == "3"
    assert flat["server_metrics.nested.a"] == "1"
    # Lists become a single JSON cell.
    assert flat["input_lens"] == "[1, 2, 3]"
    # None becomes an empty string.
    assert flat["max_concurrent_conversations"] == ""
    assert flat["steady_state_detected"] == "true"


def test_load_result_rows_single_blob(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"date": "d", "request_throughput": 1.0}))
    rows = load_result_rows(path)
    assert rows == [{"date": "d", "request_throughput": "1.0"}]


def test_load_result_rows_array(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    path.write_text(json.dumps([{"a": 1}, {"a": 2}]))
    rows = load_result_rows(path)
    assert rows == [{"a": "1"}, {"a": "2"}]


def test_load_result_rows_result_set_in_context(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps(
            {
                "run_context": {"git_commit": "abc"},
                "results": [
                    {
                        "iteration_config": {"iteration": 1},
                        "result": {"request_throughput": 5.0},
                    }
                ],
            }
        )
    )
    rows = load_result_rows(path)
    assert rows == [
        {
            "run_context.git_commit": "abc",
            "iteration_config.iteration": "1",
            "result.request_throughput": "5.0",
        }
    ]


def test_load_result_rows_bad_toplevel(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    path.write_text(json.dumps("just a string"))
    with pytest.raises(ValueError, match="expected a JSON object or array"):
        load_result_rows(path)
