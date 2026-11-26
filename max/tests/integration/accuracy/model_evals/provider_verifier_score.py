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
"""Turns a MiniMax Provider Verifier run into results the uploader reads.

The verifier (github.com/MiniMax-AI/MiniMax-Provider-Verifier) has two parts.
Each one becomes a directory under ``--out-dir`` with the ``results.jsonl``,
``score.json`` and ``config.json`` that ``internal/upload_results.py`` reads:

- ``format/``: a row per case of the pytest format checks, read from
  pytest's JUnit XML. An expected failure (xfail) is a failed row with
  verdict ``xfailed``. A skipped case gets no row; ``score.json`` counts it.
- ``batch/``: a row per ``sample.jsonl`` request per loop of
  ``run_batch_sequential.sh``, read from the newest ``batch_*`` directory
  under ``--batch-root``. A row passes when its request succeeded and every
  check that applies to it passed; ``verdict`` names the checks that failed,
  by the verifier's metric names.

A part with no rows writes nothing. A batch run that did not finish, with no
``metrics_report.json`` or a loop that wrote no results, is not scored, so a
run cut short is never recorded as complete. The run exits non-zero when a
batch run did not finish, when more than ``MAX_ERROR_RATE`` of its requests
errored, or when neither part has rows. An over-budget batch run is written
before the exit, so it is still recorded, as with
``eval_common.enforce_error_budget``.

usage: provider_verifier_score.py --out-dir DIR [--junit-xml FILE]
    [--batch-root DIR] [--base-url URL] [--verifier-commit SHA]
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import sys
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from typing import Any

# The default of eval_common.MAX_ERROR_RATE, which this stdlib-only script
# can't import.
MAX_ERROR_RATE = 0.10


def format_rows(
    junit_xml: pathlib.Path,
) -> tuple[list[dict[str, Any]], int]:
    """Reads a row per run case from pytest's JUnit XML.

    Args:
        junit_xml: The file pytest's ``--junitxml`` wrote.

    Returns:
        The rows, in file order, and how many cases were skipped. ``error``
        holds the message of a case that errored outside its assertions,
        such as in a fixture; ``failure`` holds that of a failed assertion
        or an xfail.
    """
    rows = []
    skipped = 0
    for case in ET.parse(junit_xml).iter("testcase"):
        outcome, message = "passed", None
        for tag, name in (
            ("error", "error"),
            ("failure", "failed"),
            ("skipped", "skipped"),
        ):
            child = case.find(tag)
            if child is not None:
                outcome = name
                # JUnit reports an xfail, marked or called at run time, as a
                # skip; the verifier xfails cases the endpoint gets wrong.
                if child.get("type") == "pytest.xfail":
                    outcome = "xfailed"
                message = child.text or child.get("message")
                break
        if outcome == "skipped":
            skipped += 1
            continue
        rows.append(
            {
                "test_id": f"{case.get('classname')}::{case.get('name')}",
                "passed": outcome == "passed",
                "verdict": None if outcome == "passed" else outcome,
                "error": message if outcome == "error" else None,
                "failure": message
                if outcome in ("failed", "xfailed")
                else None,
                "duration_s": float(case.get("time") or 0),
            }
        )
    return rows, skipped


def failed_checks(result: Mapping[str, Any]) -> list[str]:
    """Names the verifier metrics one successful request failed.

    ToolCalls-Match is judged by the response's finish reason. The verifier's
    own ToolCalls-Match-Rate also counts a miss for each labeled request its
    tool-call validator skipped, such as the scenario-check one, so it can
    read lower than these rows.
    """
    failed = []
    expected = result.get("expected_tool_call")
    finish = result.get("finish_reason")
    if expected is True and finish != "tool_calls":
        failed.append("ToolCalls-Match")
    elif expected is True and not result.get("tool_calls_valid"):
        failed.append("ToolCalls-Schema")
    elif expected is False and finish != "stop":
        failed.append("ToolCalls-Match")
    if result.get("error_only_reasoning"):
        failed.append("Error-Only-Reasoning")
    if result.get("language_following_valid") is False:
        failed.append("Language-Following")
    if result.get("scenario_check_valid") is False:
        failed.append("Scenario-Check")
    return failed


def batch_rows(batch_dir: pathlib.Path) -> list[dict[str, Any]]:
    """Reads a row per request per loop from one batch run's directory.

    ``prompt_index`` is the request's line in ``sample.jsonl`` from 0, and
    ``repeat_index`` its loop from 0.

    Args:
        batch_dir: A ``batch_*`` directory ``run_batch_sequential.sh`` wrote.

    Returns:
        The rows, in loop then file order.
    """
    rows = []
    for path in sorted(batch_dir.glob("loop_*/*/*_results.jsonl")):
        loop = int(path.parent.parent.name.removeprefix("loop_"))
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            result = json.loads(line)
            response = result.get("response") or {}
            succeeded = result.get("status") == "success"
            choice = (response.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            usage = response.get("usage") or {}
            checks = failed_checks(result) if succeeded else []
            rows.append(
                {
                    "prompt_index": result["data_index"] - 1,
                    "repeat_index": loop - 1,
                    "timestamp": result.get("last_run_at"),
                    "passed": succeeded and not checks,
                    "verdict": ",".join(checks) or None,
                    "error": None
                    if succeeded
                    else str(response.get("error") or result.get("status")),
                    "finish_reason": result.get("finish_reason"),
                    "expected_tool_call": result.get("expected_tool_call"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "duration_ms": result.get("duration_ms"),
                    "reasoning": message.get("reasoning_content")
                    or message.get("reasoning"),
                    "response": message.get("content"),
                    # A string: arguments vary in shape across calls, which
                    # one Parquet column can't hold.
                    "tool_calls": json.dumps(message["tool_calls"])
                    if message.get("tool_calls")
                    else None,
                }
            )
    return rows


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    """Counts the passes, the errors, and each verdict among the rows."""
    passed = sum(bool(row["passed"]) for row in rows)
    verdicts: collections.Counter[str] = collections.Counter()
    for row in rows:
        if row["verdict"]:
            verdicts.update(row["verdict"].split(","))
    return {
        "accuracy": passed / len(rows),
        "total": len(rows),
        "passed": passed,
        "errors": sum(row["error"] is not None for row in rows),
        "verdicts": dict(sorted(verdicts.items())),
    }


def write_results(
    out: pathlib.Path,
    rows: Sequence[Mapping[str, Any]],
    score: Mapping[str, object],
    base_url: str | None,
) -> None:
    """Writes one results directory for the uploader."""
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "score.json").write_text(json.dumps(score, indent=2))
    (out / "config.json").write_text(json.dumps({"base_url": base_url}))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--junit-xml", type=pathlib.Path)
    parser.add_argument("--batch-root", type=pathlib.Path)
    parser.add_argument("--base-url")
    parser.add_argument("--verifier-commit")
    args = parser.parse_args(argv)

    wrote = False
    problems = []
    if args.junit_xml and args.junit_xml.is_file():
        rows, skipped = format_rows(args.junit_xml)
        if rows:
            score = summarize(rows)
            score["skipped"] = skipped
            score["failed_cases"] = [r["test_id"] for r in rows if r["verdict"]]
            score["verifier_commit"] = args.verifier_commit
            write_results(args.out_dir / "format", rows, score, args.base_url)
            print(f"format: {json.dumps(score, indent=2)}")
            wrote = True
    batches = sorted(args.batch_root.glob("batch_*")) if args.batch_root else []
    if batches:
        batch_dir = batches[-1]
        report = batch_dir / "metrics_report.json"
        unfinished = [
            loop.name
            for loop in sorted(batch_dir.glob("loop_*"))
            if not any(loop.glob("*/*_results.jsonl"))
        ]
        rows = batch_rows(batch_dir)
        if unfinished or not report.is_file():
            problems.append(
                f"The batch run in {batch_dir} did not finish, so it is not "
                f"scored. Loops without results: {unfinished or 'none'}; "
                f"metrics_report.json present: {report.is_file()}."
            )
        elif rows:
            score = summarize(rows)
            score["loops"] = len({row["repeat_index"] for row in rows})
            score["average_metrics"] = json.loads(report.read_text())[
                "summary"
            ]["average_metrics"]
            score["verifier_commit"] = args.verifier_commit
            write_results(args.out_dir / "batch", rows, score, args.base_url)
            print(f"batch: {json.dumps(score, indent=2)}")
            wrote = True
            errors = sum(row["error"] is not None for row in rows)
            if errors / len(rows) > MAX_ERROR_RATE:
                problems.append(
                    f"{errors}/{len(rows)} batch requests errored, above the "
                    f"{MAX_ERROR_RATE:.0%} budget, so this run's score "
                    f"reflects the endpoint, not the model."
                )
    if not wrote and not problems:
        problems.append("No verifier results to score.")
    if problems:
        prefix = "::error::" if os.environ.get("GITHUB_ACTIONS") else "ERROR: "
        sys.exit("\n".join(prefix + problem for problem in problems))


if __name__ == "__main__":
    main()
