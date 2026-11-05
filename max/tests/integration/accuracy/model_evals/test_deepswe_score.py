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
"""Tests for the leaderboard-style DeepSWE scoring."""

from __future__ import annotations

import json
import pathlib

import deepswe_score
import pytest


def _write_trial(
    job_dir: pathlib.Path,
    trial: str,
    task: str,
    reward: float | None,
    exception: str | None = None,
    verifier_result: bool = True,
) -> None:
    result: dict[str, object] = {"trial_name": trial, "task_name": task}
    if verifier_result:
        result["verifier_result"] = {"rewards": {"reward": reward}}
    if exception:
        result["exception_info"] = {"exception_type": exception}
    (job_dir / trial).mkdir()
    (job_dir / trial / "result.json").write_text(json.dumps(result))


def test_graded_trials_count_and_pass_at_k_is_per_task(
    tmp_path: pathlib.Path,
) -> None:
    _write_trial(tmp_path, "a__1", "datacurve/a", 1)
    _write_trial(tmp_path, "a__2", "datacurve/a", 0)
    _write_trial(tmp_path, "b__1", "datacurve/b", 0)
    _write_trial(tmp_path, "b__2", "datacurve/b", 0)

    summary, excluded = deepswe_score.score_job(tmp_path, attempts="2")

    assert summary is not None
    assert excluded == []
    assert summary["pass@1"] == 0.25
    assert summary["pass@2"] == 0.5
    assert summary["passed"] == 1
    assert summary["graded_trials"] == 4
    assert summary["tasks"] == 2


def test_agent_timeout_is_graded_not_excluded(tmp_path: pathlib.Path) -> None:
    # The leaderboard grades a timed-out agent's partial work; the exception
    # is recorded, not used to drop the trial.
    _write_trial(
        tmp_path, "a__1", "datacurve/a", 0, exception="AgentTimeoutError"
    )

    summary, excluded = deepswe_score.score_job(tmp_path)

    assert summary is not None
    assert excluded == []
    assert summary["graded_trials"] == 1
    assert summary["graded_with_agent_error"] == {"AgentTimeoutError": 1}


def test_missing_reward_and_crash_sentinel_are_infra_errors(
    tmp_path: pathlib.Path,
) -> None:
    _write_trial(tmp_path, "a__1", "datacurve/a", 1)
    _write_trial(tmp_path, "b__1", "datacurve/b", None, verifier_result=False)
    _write_trial(tmp_path, "c__1", "datacurve/c", -1)
    _write_trial(
        tmp_path, "d__1", "datacurve/d", None, exception="RuntimeError"
    )

    summary, excluded = deepswe_score.score_job(tmp_path)

    assert summary is not None
    assert summary["graded_trials"] == 1
    assert summary["pass@1"] == 1.0
    assert excluded == [
        ("b__1", "no verifier reward"),
        ("c__1", "verifier crashed"),
        ("d__1", "RuntimeError"),
    ]
    assert summary["excluded"] == [
        {"trial": t, "reason": r} for t, r in excluded
    ]


def test_nothing_graded_returns_no_summary(tmp_path: pathlib.Path) -> None:
    _write_trial(tmp_path, "a__1", "datacurve/a", None, verifier_result=False)

    summary, excluded = deepswe_score.score_job(tmp_path)

    assert summary is None
    assert excluded == [("a__1", "no verifier reward")]


def test_unreadable_result_is_an_infra_error(tmp_path: pathlib.Path) -> None:
    _write_trial(tmp_path, "a__1", "datacurve/a", 1)
    (tmp_path / "b__1").mkdir()
    (tmp_path / "b__1" / "result.json").write_text('{"trial_name": "b__1", ')

    summary, excluded = deepswe_score.score_job(tmp_path)

    assert summary is not None
    assert summary["graded_trials"] == 1
    assert [t for t, _ in excluded] == ["b__1"]
    assert excluded[0][1].startswith("unreadable result.json")


def test_main_fails_when_nothing_was_graded(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_dir = tmp_path / "deepswe"
    job_dir.mkdir()
    _write_trial(job_dir, "a__1", "datacurve/a", None, verifier_result=False)
    monkeypatch.setattr("sys.argv", ["deepswe_score.py", str(job_dir)])
    monkeypatch.delenv("GITHUB_ENV", raising=False)

    with pytest.raises(SystemExit) as exit_info:
        deepswe_score.main()

    assert exit_info.value.code != 0
    assert not (tmp_path / "score.json").exists()
