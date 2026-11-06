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
"""Scores a Pier DeepSWE job directory the way the public leaderboard does.

Every graded trial counts, including agent timeouts (they are still graded);
trials with no verifier reward, or the verifier's -1 crash sentinel, are infra
errors and are excluded. Writes ``<job dir>/../score.json``, prints the
summary, and exports ``DEEPSWE_SCORE`` through ``GITHUB_ENV`` when running
under GitHub Actions. Exits non-zero when no trial was graded.

usage: deepswe_score.py <job dir>   (DEEPSWE_ATTEMPTS names the pass@k column)
"""

from __future__ import annotations

import collections
import json
import os
import pathlib
import sys


def score_job(
    job_dir: pathlib.Path, attempts: str = "1"
) -> tuple[dict[str, object] | None, list[tuple[str, str]]]:
    """Scores every ``*/result.json`` under ``job_dir``.

    Args:
        job_dir: Pier's job directory, one trial directory per task attempt.
        attempts: Attempts per task, which names the ``pass@k`` column.

    Returns:
        A ``(summary, excluded)`` tuple. ``summary`` is ``None`` when no trial
        was graded; ``excluded`` lists ``(trial, reason)`` for the trials left
        out as infra errors, which the summary also carries.
    """
    per_task: dict[str, list[bool]] = collections.defaultdict(list)
    agent_errors: collections.Counter[str] = collections.Counter()
    excluded: list[tuple[str, str]] = []
    for path in sorted(job_dir.glob("*/result.json")):
        # A trial killed mid-write leaves a truncated file; that is one more
        # infra error, not a reason to lose every other trial's score.
        try:
            result = json.loads(path.read_text())
        except (OSError, ValueError) as err:
            excluded.append(
                (path.parent.name, f"unreadable result.json: {err}")
            )
            continue
        exc = (result.get("exception_info") or {}).get("exception_type")
        rewards = (result.get("verifier_result") or {}).get("rewards") or {}
        reward = rewards.get("reward")
        if reward is None or reward < 0:
            reason = exc or (
                "verifier crashed"
                if reward is not None
                else "no verifier reward"
            )
            excluded.append(
                (result.get("trial_name", path.parent.name), reason)
            )
            continue
        if exc:
            agent_errors[exc] += 1
        per_task[result["task_name"]].append(reward >= 1)

    outcomes = [ok for oks in per_task.values() for ok in oks]
    if not outcomes:
        return None, excluded
    summary: dict[str, object] = {
        "pass@1": sum(outcomes) / len(outcomes),
        f"pass@{attempts}": sum(any(oks) for oks in per_task.values())
        / len(per_task),
        "passed": sum(outcomes),
        "graded_trials": len(outcomes),
        "tasks": len(per_task),
        "graded_with_agent_error": dict(agent_errors),
        "excluded": [{"trial": t, "reason": r} for t, r in excluded],
    }
    return summary, excluded


def main() -> None:
    job_dir = pathlib.Path(sys.argv[1])
    attempts = os.environ.get("DEEPSWE_ATTEMPTS", "1")
    in_ci = os.environ.get("GITHUB_ACTIONS") == "true"
    warn = "::warning::" if in_ci else "WARNING: "
    summary, excluded = score_job(job_dir, attempts)
    if summary is not None:
        (job_dir.parent / "score.json").write_text(
            json.dumps(summary, indent=2)
        )
        print(json.dumps(summary, indent=2))
        print(
            f"DeepSWE pass@1: {summary['pass@1']:.4f} "
            f"({summary['passed']}/{summary['graded_trials']})"
        )
        env_file = os.environ.get("GITHUB_ENV")
        if env_file:
            with open(env_file, "a") as f:
                f.write(f"DEEPSWE_SCORE={summary['pass@1']:.4f}\n")
    for trial, reason in excluded:
        print(f"{warn}DeepSWE trial {trial} excluded: {reason}")
    if summary is None:
        # A crashed Pier or an all-infra-error run must not pass as green.
        sys.exit(
            f"{'::error::' if in_ci else 'ERROR: '}No graded DeepSWE trials "
            f"under {job_dir} ({len(excluded)} excluded)"
        )


if __name__ == "__main__":
    main()
