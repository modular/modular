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
"""Tests for turning a MiniMax Provider Verifier run into uploader results."""

from __future__ import annotations

import json
import pathlib

import provider_verifier_score
import pytest

JUNIT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest">
<testcase classname="m3_text_tests.TestBasic" name="test_01" time="1.5"/>
<testcase classname="m3_text_tests.TestBasic" name="test_02" time="2">
  <failure message="assert 1 == 2">long traceback</failure>
</testcase>
<testcase classname="m3_image_tests.TestUrl" name="test_03[a/b]" time="0">
  <error message="ConnectError"/>
</testcase>
<testcase classname="m3_video_tests.TestBig" name="test_04" time="0">
  <skipped type="pytest.skip" message="slow"/>
</testcase>
<testcase classname="m3_text_tests.TestStream" name="test_05" time="1">
  <skipped type="pytest.xfail" message="Known BUG: no [DONE]"/>
</testcase>
</testsuite></testsuites>
"""


def _result(index: int, **fields: object) -> dict[str, object]:
    return {
        "data_index": index,
        "status": "success",
        "finish_reason": "stop",
        "last_run_at": "2026-11-18T00:00:00",
        "response": {
            "choices": [
                {"message": {"content": "hi", "reasoning_content": "think"}}
            ],
            "usage": {"completion_tokens": 7, "prompt_tokens": 3},
        },
        **fields,
    }


def _write_loop(
    batch: pathlib.Path, loop: int, results: list[dict[str, object]] | None
) -> None:
    """Writes one loop's results; ``None`` leaves the loop without any."""
    module = batch / f"loop_{loop:02d}" / "modular-max"
    module.mkdir(parents=True)
    if results is not None:
        (module / "modular-max_results.jsonl").write_text(
            "\n".join(json.dumps(r) for r in results)
        )


def _write_report(batch: pathlib.Path) -> None:
    (batch / "metrics_report.json").write_text(
        json.dumps({"summary": {"average_metrics": {"1. Query": 1.0}}})
    )


def test_format_rows_record_xfails_and_count_skips(
    tmp_path: pathlib.Path,
) -> None:
    junit = tmp_path / "junit.xml"
    junit.write_text(JUNIT)

    rows, skipped = provider_verifier_score.format_rows(junit)

    assert [(r["test_id"], r["passed"], r["verdict"]) for r in rows] == [
        ("m3_text_tests.TestBasic::test_01", True, None),
        ("m3_text_tests.TestBasic::test_02", False, "failed"),
        ("m3_image_tests.TestUrl::test_03[a/b]", False, "error"),
        ("m3_text_tests.TestStream::test_05", False, "xfailed"),
    ]
    assert skipped == 1
    assert rows[1]["failure"] == "long traceback"
    assert rows[1]["error"] is None
    assert rows[2]["error"] == "ConnectError"
    assert rows[3]["failure"] == "Known BUG: no [DONE]"


@pytest.mark.parametrize(
    "fields, want",
    [
        ({"expected_tool_call": None}, []),
        (
            {
                "expected_tool_call": True,
                "finish_reason": "tool_calls",
                "tool_calls_valid": True,
            },
            [],
        ),
        ({"expected_tool_call": True}, ["ToolCalls-Match"]),
        (
            {
                "expected_tool_call": True,
                "finish_reason": "tool_calls",
                "tool_calls_valid": False,
            },
            ["ToolCalls-Schema"],
        ),
        (
            {"expected_tool_call": False, "finish_reason": "length"},
            ["ToolCalls-Match"],
        ),
        (
            {
                "error_only_reasoning": True,
                "language_following_valid": False,
                "scenario_check_valid": False,
            },
            ["Error-Only-Reasoning", "Language-Following", "Scenario-Check"],
        ),
        ({"language_following_valid": None}, []),
    ],
)
def test_failed_checks(fields: dict[str, object], want: list[str]) -> None:
    assert provider_verifier_score.failed_checks(_result(1, **fields)) == want


def test_batch_rows_index_requests_and_loops_from_zero(
    tmp_path: pathlib.Path,
) -> None:
    _write_loop(
        tmp_path, 1, [_result(1), _result(2, error_only_reasoning=True)]
    )
    _write_loop(
        tmp_path,
        2,
        [
            _result(1, status="failed", response={"error": "timeout"}),
            _result(
                2,
                expected_tool_call=True,
                finish_reason="tool_calls",
                tool_calls_valid=True,
                response={
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [{"id": 1}],
                            }
                        }
                    ]
                },
            ),
        ],
    )

    rows = provider_verifier_score.batch_rows(tmp_path)

    assert [
        (r["prompt_index"], r["repeat_index"], r["passed"], r["verdict"])
        for r in rows
    ] == [
        (0, 0, True, None),
        (1, 0, False, "Error-Only-Reasoning"),
        (0, 1, False, None),
        (1, 1, True, None),
    ]
    assert rows[0]["response"] == "hi"
    assert rows[0]["reasoning"] == "think"
    assert rows[0]["completion_tokens"] == 7
    assert rows[2]["error"] == "timeout"
    assert rows[3]["tool_calls"] == '[{"id": 1}]'


def test_main_writes_one_results_dir_per_part(tmp_path: pathlib.Path) -> None:
    junit = tmp_path / "junit.xml"
    junit.write_text(JUNIT)
    root = tmp_path / "output-dir"
    _write_loop(root / "batch_20261117_000000", 1, [_result(1)])
    newest = root / "batch_20261118_000000"
    _write_loop(newest, 1, [_result(1), _result(2, error_only_reasoning=True)])
    _write_report(newest)
    out = tmp_path / "out"

    provider_verifier_score.main(
        [
            f"--out-dir={out}",
            f"--junit-xml={junit}",
            f"--batch-root={root}",
            "--base-url=https://gw.example",
            "--verifier-commit=abc",
        ]
    )

    fmt = json.loads((out / "format" / "score.json").read_text())
    assert fmt["accuracy"] == 0.25
    assert fmt["verdicts"] == {"error": 1, "failed": 1, "xfailed": 1}
    assert fmt["skipped"] == 1
    assert fmt["failed_cases"] == [
        "m3_text_tests.TestBasic::test_02",
        "m3_image_tests.TestUrl::test_03[a/b]",
        "m3_text_tests.TestStream::test_05",
    ]
    batch = json.loads((out / "batch" / "score.json").read_text())
    assert batch["accuracy"] == 0.5
    assert batch["loops"] == 1
    assert batch["average_metrics"] == {"1. Query": 1.0}
    assert batch["verifier_commit"] == "abc"
    assert len((out / "batch" / "results.jsonl").read_text().splitlines()) == 2
    assert json.loads((out / "batch" / "config.json").read_text()) == {
        "base_url": "https://gw.example"
    }


@pytest.mark.parametrize("report", [True, False])
def test_main_does_not_score_an_unfinished_batch_run(
    tmp_path: pathlib.Path, report: bool
) -> None:
    junit = tmp_path / "junit.xml"
    junit.write_text(JUNIT)
    batch = tmp_path / "output-dir" / "batch_20261118_000000"
    _write_loop(batch, 1, [_result(1)])
    # With a report, a loop without results is what makes the run unfinished.
    _write_loop(batch, 2, None if report else [_result(1)])
    if report:
        _write_report(batch)
    out = tmp_path / "out"

    with pytest.raises(SystemExit, match="did not finish"):
        provider_verifier_score.main(
            [
                f"--out-dir={out}",
                f"--junit-xml={junit}",
                f"--batch-root={batch.parent}",
            ]
        )

    assert (out / "format" / "score.json").is_file()
    assert not (out / "batch").exists()


def test_main_records_then_fails_an_over_budget_batch_run(
    tmp_path: pathlib.Path,
) -> None:
    batch = tmp_path / "output-dir" / "batch_20261118_000000"
    _write_loop(
        batch,
        1,
        [_result(1), _result(2, status="failed", response={"error": "401"})],
    )
    _write_report(batch)
    out = tmp_path / "out"

    with pytest.raises(SystemExit, match="1/2 batch requests errored"):
        provider_verifier_score.main(
            [f"--out-dir={out}", f"--batch-root={batch.parent}"]
        )

    score = json.loads((out / "batch" / "score.json").read_text())
    assert score["errors"] == 1


def test_main_fails_with_nothing_to_score(tmp_path: pathlib.Path) -> None:
    with pytest.raises(SystemExit, match="No verifier results"):
        provider_verifier_score.main([f"--out-dir={tmp_path}"])
