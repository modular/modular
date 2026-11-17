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
"""Writes a TerminalBench run in the AutoExacto pipeline's output shape.

Reads the ``results.jsonl`` and ``meta.json`` that ``run_terminalbench_local.sh``
writes and produces, in ``--out-dir``:

- ``score.json``: ``accuracy`` (the mean over tasks of each task's pass rate)
  and its task-clustered ``stderr``, the keys ``exacto_summary.py`` and
  ``collect_scores.py`` read, plus the run's settings for provenance.
- ``config.json``: the endpoint and sampling settings, with
  ``eval_common.dump_config``'s keys.
- ``trajectories.jsonl``: one row per kept trial, with the columns
  ``accuracy_verdict``'s agentic comparison reads, so two arms compare with
  ``--agentic-candidate-file`` and ``--agentic-baseline-file``.

Only valid trials are written. The runner re-runs invalid ones until each
task has ``k``, so an invalid trial is a harness retry, not a model error.

Stdlib-only, like ``terminalbench_score``: CI runs it with a bare python3.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

from terminalbench_score import Trial, read_jsonl, summarize

DATASET = "terminalbench_2_1"


def trajectory_rows(trials: Sequence[Trial]) -> list[dict[str, object]]:
    """Returns one verdict row per valid trial, numbered per task by start time."""
    rows: list[dict[str, object]] = []
    seen: dict[str, int] = {}
    for t in sorted((t for t in trials if t.valid), key=lambda t: t.started):
        repeat = seen.get(t.task, 0)
        seen[t.task] = repeat + 1
        rows.append(
            {
                "task_id": t.task,
                "repeat_index": repeat,
                "trial": t.trial,
                "reward": t.reward,
                "passed": t.reward == 1,
                "error": None,
                "output_tokens": t.gen,
                "input_tokens": t.in_total,
                "cached_input_tokens": t.in_total - t.in_uncached,
                "turns": t.turns,
                "tool_calls": t.tool_calls,
                "cap_hits": t.capped_steps,
                "unfinished_reasoning": t.unclosed_steps,
                "tool_markup_leaks": t.markup_in_text,
                "loop": int(t.loops > 0),
            }
        )
    return rows


def missing_tasks(
    trials: Sequence[Trial], planned: Sequence[str], k: int
) -> list[str]:
    """Returns the planned tasks with fewer than ``k`` valid trials.

    A task whose every attempt failed in the harness has no valid trial, so
    it never appears among the scored tasks; counting against the plan is
    what catches it.
    """
    have = Counter(t.task for t in trials if t.valid)
    return sorted(t for t in planned if have[t] < k)


def exacto_score(
    trials: Sequence[Trial],
    meta: Mapping[str, object],
    runner_invalid: bool = False,
) -> dict[str, object]:
    """Returns ``score.json``: the scorer's summary under exacto's key names.

    Args:
        trials: The run's trials, valid and not.
        meta: The runner's ``meta.json``.
        runner_invalid: Whether the runner judged the run not a measurement
            (it leaves an ``INVALID`` file).
    """
    k = meta.get("k")
    assert isinstance(k, int)
    summary = summarize(trials, k)
    excluded = meta.get("excluded_tasks") or []
    assert isinstance(excluded, list)
    planned = meta.get("planned_tasks")
    if isinstance(planned, list):
        missing = missing_tasks(trials, planned, k)
        short = len(missing)
    else:
        # A run from before the runner recorded its plan.
        missing = None
        short = summary["tasks_short_of_k"]
    return {
        "accuracy": summary["score"],
        "stderr": summary["score_stderr"],
        "scorer": "terminalbench_task_pass_rate",
        "complete": not runner_invalid and short == 0,
        "total": summary["total"],
        "num_tasks": summary["tasks"],
        "planned_tasks": len(planned) if isinstance(planned, list) else None,
        "num_trials": k,
        "tasks_short_of_k": short,
        "missing_tasks": missing,
        "errors": summary["errors"],
        "invalid_reached_model": summary["invalid_reached_model"],
        "mean_output_tokens": summary["mean_output_tokens"],
        "p50_output_tokens": summary["p50_output_tokens"],
        "p50_turns": summary["p50_turns"],
        "p50_in_total": summary["p50_in_total"],
        "capped_trial_rate": summary["capped_trial_rate"],
        "unclosed_trial_rate": summary["unclosed_trial_rate"],
        "markup_trial_rate": summary["markup_trial_rate"],
        "loop_trials": summary["loop_trials"],
        "harbor": meta.get("harbor"),
        "agent": meta.get("agent"),
        "dataset_name": meta.get("dataset"),
        "excluded_tasks": excluded,
        "task_subset": meta.get("task_subset"),
        "output_cap": meta.get("output_cap"),
        "temperature": meta.get("temperature"),
        "top_p": meta.get("top_p"),
        "thinking": meta.get("thinking"),
        "reasoning_effort": meta.get("reasoning_effort"),
        "model": meta.get("model"),
        "serve_config": meta.get("serve_config"),
    }


def exacto_config(meta: Mapping[str, object]) -> dict[str, object]:
    """Returns ``config.json`` with ``eval_common.dump_config``'s keys."""
    subset = meta.get("task_subset")
    return {
        "base_url": meta.get("endpoint"),
        "temperature": meta.get("temperature"),
        "top_p": meta.get("top_p"),
        "max_tokens": meta.get("output_cap"),
        "reasoning_effort": meta.get("reasoning_effort"),
        "thinking": meta.get("thinking"),
        # None is the full task set; a --task run is a subset.
        "sample_size": len(subset) if isinstance(subset, list) else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="The runner's --out-dir, holding results.jsonl and meta.json.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    trials = read_jsonl(args.run_dir / "results.jsonl")
    meta = json.loads((args.run_dir / "meta.json").read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    score = exacto_score(
        trials, meta, runner_invalid=(args.run_dir / "INVALID").exists()
    )
    (args.out_dir / "score.json").write_text(json.dumps(score, indent=2) + "\n")
    (args.out_dir / "config.json").write_text(
        json.dumps(exacto_config(meta), indent=2) + "\n"
    )
    with (args.out_dir / "trajectories.jsonl").open("w") as f:
        for row in trajectory_rows(trials):
            f.write(json.dumps(row) + "\n")

    accuracy, stderr = score["accuracy"], score["stderr"]
    print(
        f"{DATASET}: "
        + (f"{accuracy:.4f}" if isinstance(accuracy, float) else "none")
        + (f" +/-{stderr:.4f}" if isinstance(stderr, float) else "")
        + f" over {score['num_tasks']} tasks, {score['total']} trials"
        + (
            ""
            if score["complete"]
            else f"; INCOMPLETE: {score['tasks_short_of_k']} tasks short of k"
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
