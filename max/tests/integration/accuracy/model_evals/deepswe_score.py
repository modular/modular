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
errors and are excluded. Beside the job directory it writes ``score.json``,
and for the results uploader (``internal/upload_results.py``) a
``results.jsonl`` row per trial and a ``config.json``. It prints the summary
and exports ``DEEPSWE_SCORE`` through ``GITHUB_ENV`` when running under
GitHub Actions. Exits non-zero when no trial was graded.

usage: deepswe_score.py <job dir>
  DEEPSWE_ATTEMPTS names the pass@k column; OPENAI_API_BASE and SAMPLE_SIZE,
  when set, go into config.json as the endpoint and the sample size.
"""

from __future__ import annotations

import collections
import json
import os
import pathlib
import sys

PATCH = pathlib.Path("artifacts", "model.patch")


def trial_rows(job_dir: pathlib.Path) -> list[dict[str, object]]:
    """Reads every ``*/result.json`` under ``job_dir`` into one row per trial.

    Rows carry the keys the results uploader reads. ``prompt_index`` is the
    task's place among every task in its task directory, so sampled and full
    runs agree, and ``None`` when that directory is gone; ``repeat_index``
    numbers a task's attempts in the order they started; ``error`` says why
    an infra-error trial is excluded; ``response`` is the agent's patch.

    Args:
        job_dir: Pier's job directory, one trial directory per task attempt.

    Returns:
        The rows, in trial directory order.
    """
    rows: list[dict[str, object]] = []
    task_indexes: dict[pathlib.Path, dict[str, int]] = {}
    for path in sorted(job_dir.glob("*/result.json")):
        # A trial killed mid-write leaves a truncated file; that is one more
        # infra error, not a reason to lose every other trial's row.
        try:
            result = json.loads(path.read_text())
        except (OSError, ValueError) as err:
            rows.append(
                {
                    "trial_name": path.parent.name,
                    "prompt_index": None,
                    "passed": False,
                    "error": f"unreadable result.json: {err}",
                }
            )
            continue
        exc = (result.get("exception_info") or {}).get("exception_type")
        rewards = (result.get("verifier_result") or {}).get("rewards") or {}
        reward = rewards.get("reward")
        error = None
        passed = False
        if reward is None or reward < 0:
            error = exc or (
                "verifier crashed"
                if reward is not None
                else "no verifier reward"
            )
        else:
            passed = reward >= 1
        agent = result.get("agent_result") or {}
        patch = path.parent / PATCH
        rows.append(
            {
                "prompt_index": _task_index(
                    (result.get("task_id") or {}).get("path"), task_indexes
                ),
                "task_name": result.get("task_name"),
                "trial_name": result.get("trial_name", path.parent.name),
                "timestamp": result.get("started_at"),
                "passed": passed,
                "reward": reward,
                "partial": rewards.get("partial"),
                "error": error,
                "agent_error": None if error else exc,
                "completion_tokens": agent.get("n_output_tokens"),
                "prompt_tokens": agent.get("n_input_tokens"),
                "cached_tokens": agent.get("n_cache_tokens"),
                "agent_steps": agent.get("n_agent_steps"),
                "response": patch.read_text(errors="replace")
                if patch.is_file()
                else None,
            }
        )
    by_task: dict[object, list[dict[str, object]]] = collections.defaultdict(
        list
    )
    for row in rows:
        by_task[row.get("task_name") or row["trial_name"]].append(row)
    for attempts in by_task.values():
        attempts.sort(
            key=lambda r: (str(r.get("timestamp") or ""), str(r["trial_name"]))
        )
        for i, row in enumerate(attempts):
            row["repeat_index"] = i
    return rows


def _task_index(
    task_path: str | None, cache: dict[pathlib.Path, dict[str, int]]
) -> int | None:
    if not task_path:
        return None
    task_dir = pathlib.Path(task_path)
    if task_dir.parent not in cache:
        try:
            names = sorted(
                p.name
                for p in task_dir.parent.iterdir()
                if p.is_dir() and not p.name.startswith(".")
            )
        except OSError:
            names = []
        cache[task_dir.parent] = {name: i for i, name in enumerate(names)}
    return cache[task_dir.parent].get(task_dir.name)


def score_job(
    job_dir: pathlib.Path, attempts: str = "1"
) -> tuple[dict[str, object] | None, list[tuple[str, str]]]:
    """Scores every ``*/result.json`` under ``job_dir``.

    Args:
        job_dir: Pier's job directory, one trial directory per task attempt.
        attempts: Attempts per task, which names the ``pass@k`` column.

    Returns:
        What :func:`score_rows` returns for the job's :func:`trial_rows`.
    """
    return score_rows(trial_rows(job_dir), attempts)


def score_rows(
    rows: list[dict[str, object]], attempts: str = "1"
) -> tuple[dict[str, object] | None, list[tuple[str, str]]]:
    """Scores the rows :func:`trial_rows` reads.

    Args:
        rows: One row per trial.
        attempts: Attempts per task, which names the ``pass@k`` column.

    Returns:
        A ``(summary, excluded)`` tuple. ``summary`` is ``None`` when no trial
        was graded; ``excluded`` lists ``(trial, reason)`` for the trials left
        out as infra errors, which the summary also carries.
    """
    per_task: dict[object, list[bool]] = collections.defaultdict(list)
    agent_errors: collections.Counter[str] = collections.Counter()
    excluded: list[tuple[str, str]] = []
    for row in rows:
        if row["error"]:
            excluded.append((str(row["trial_name"]), str(row["error"])))
            continue
        if row["agent_error"]:
            agent_errors[str(row["agent_error"])] += 1
        per_task[row["task_name"]].append(bool(row["passed"]))

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
    rows = trial_rows(job_dir)
    summary, excluded = score_rows(rows, attempts)
    if summary is not None:
        out = job_dir.parent
        (out / "score.json").write_text(json.dumps(summary, indent=2))
        with open(out / "results.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        base_url = os.environ.get("OPENAI_API_BASE", "").rstrip("/")
        sample_size = os.environ.get("SAMPLE_SIZE")
        (out / "config.json").write_text(
            json.dumps(
                {
                    "base_url": base_url.removesuffix("/v1") or None,
                    "sample_size": int(sample_size) if sample_size else None,
                },
                indent=2,
            )
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
