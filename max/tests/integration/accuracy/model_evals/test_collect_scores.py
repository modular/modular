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
"""Tests for the suite-level score collector."""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import collect_scores

AIME_SUMMARY = {
    "accuracy": 0.9062,
    "correct": 435,
    "total": 480,
    "errors": 0,
    "mean_output_tokens": 16552,
    "p50_output_tokens": 12657,
}
SCICODE_SUMMARY = {"score": 0.4545, "total": 338}


def make_tree(root: Path) -> None:
    """Mirrors the real downloaded-artifact layouts.

    The text group uploads multiple directories, so upload-artifact keeps
    them as subdirectories; scicode uploads one directory, so its contents
    (score.json included) sit at the artifact root.
    """
    aime = root / "text-non-agentic-text-eval-results" / "aime25-results"
    aime.mkdir(parents=True)
    (aime / "score.json").write_text(json.dumps(AIME_SUMMARY))
    sci = root / "scicode-scicode-eval-results"
    sci.mkdir(parents=True)
    (sci / "score.json").write_text(json.dumps(SCICODE_SUMMARY))
    (root / "text-non-agentic-text-eval-results" / "junk.txt").write_text(
        "not a score"
    )


def test_collect_rows(tmp_path: Path) -> None:
    make_tree(tmp_path)
    rows = collect_scores.collect(tmp_path, ["text-", "scicode-"])
    assert [(r["benchmark"], r["group"]) for r in rows] == [
        ("scicode", "scicode-scicode-eval-results"),
        ("aime25", "text-non-agentic-text-eval-results"),
    ]
    by_name = {r["benchmark"]: r for r in rows}
    assert by_name["aime25"]["score"] == 0.9062
    assert by_name["aime25"]["errors"] == 0
    assert by_name["scicode"]["score"] == 0.4545
    assert by_name["scicode"]["errors"] is None


def test_root_level_score_without_prefix_strip(tmp_path: Path) -> None:
    make_tree(tmp_path)
    rows = collect_scores.collect(tmp_path)
    names = {r["benchmark"] for r in rows}
    assert "scicode-scicode" in names  # only suffix stripped


def test_cli_end_to_end(tmp_path: Path) -> None:
    make_tree(tmp_path)
    out = tmp_path / "consolidated" / "results.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(Path(collect_scores.__file__)),
            "--artifacts-dir",
            str(tmp_path),
            "--out",
            str(out),
            "--expect",
            "aime25,scicode,gpqa",
            "--prefix",
            "text-",
            "--prefix",
            "scicode-",
            "--meta",
            "model=MiniMaxAI/MiniMax-M3-MXFP8",
            "--meta",
            "backend=max-serve",
        ],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(out.read_text())
    assert result["run"] == {
        "model": "MiniMaxAI/MiniMax-M3-MXFP8",
        "backend": "max-serve",
    }
    assert result["missing"] == ["gpqa"]
    assert {r["benchmark"] for r in result["benchmarks"]} == {
        "aime25",
        "scicode",
    }
    assert "| gpqa | | MISSING |" in proc.stdout


def test_malformed_score_json_is_skipped(tmp_path: Path) -> None:
    bad = tmp_path / "group" / "broken-results"
    bad.mkdir(parents=True)
    (bad / "score.json").write_text("{not json")
    make_tree(tmp_path)
    rows = collect_scores.collect(tmp_path, ["text-", "scicode-"])
    assert {r["benchmark"] for r in rows} == {"aime25", "scicode"}


def test_compare_joins_both_sides() -> None:
    ours = [
        {"benchmark": "aime25", "score": 0.9062},
        {"benchmark": "scicode", "score": 0.4545},
    ]
    baseline: dict[str, Any] = {
        "run": {"backend": "vllm"},
        "benchmarks": [
            {"benchmark": "aime25", "score": 0.9000},
            {"benchmark": "gpqa", "score": 0.9242},
        ],
    }
    joined = collect_scores.compare(ours, baseline)
    by_name = {r["benchmark"]: r for r in joined}
    assert abs(by_name["aime25"]["delta"] - 0.0062) < 1e-9
    assert by_name["scicode"]["baseline_score"] is None
    assert by_name["scicode"]["delta"] is None
    assert by_name["gpqa"]["score"] is None
    table = collect_scores.comparison_table(joined, baseline["run"])
    assert "vs baseline vllm" in table
    assert "| aime25 | 0.9062 | 0.9000 | +0.0062 |" in table


def test_root_level_score_named_by_prefix(tmp_path: Path) -> None:
    """A per-dataset call uploads one directory, so v4 flattens it: the
    score.json sits at the artifact root and only the call's prefix names it.
    """
    art = tmp_path / "aime25-non-agentic-text-eval-results"
    art.mkdir(parents=True)
    (art / "score.json").write_text(json.dumps(AIME_SUMMARY))
    rows = collect_scores.collect(tmp_path, ["aime25-", "gpqa-"])
    assert [(r["benchmark"], r["group"]) for r in rows] == [
        ("aime25", "aime25-non-agentic-text-eval-results"),
    ]
