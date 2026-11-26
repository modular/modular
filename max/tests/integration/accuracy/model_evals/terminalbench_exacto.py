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
- ``results.jsonl``: one row per scored trial with the keys
  ``internal/upload_results.py`` reads, so the run uploads to S3 and BigQuery
  like DeepSWE's. Left out, so the run does not upload, for a ``--task``
  subset or a run whose tasks cannot be indexed (see :func:`uploadable`).
- ``attempts.jsonl``: the same row for every attempt, invalid ones included,
  with the reason each was not scored. S3 keeps it beside the upload.

Only valid trials are scored, at most ``k`` per task. The runner re-runs
invalid ones until each task has ``k``, so an invalid trial is a harness
retry, not a model error, and must not count as a fail where an errored row
would.

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


def _invalid_reason(t: Trial) -> str:
    if t.reward is None:
        return t.exception or "no verifier reward"
    if not t.reached_model:
        return t.exception or "never reached the model"
    return t.exception or "last model step did not finish"


def task_indexes(
    meta: Mapping[str, object], dataset_tasks: Sequence[str] | None = None
) -> dict[str, int]:
    """Returns each task's place among the dataset's tasks, sorted by name.

    The runner records every task in the dataset, so full and subset runs
    agree. A run from before that records only the tasks it planned and
    excluded. A list missing a task, or naming one that is not a task, would
    shift every later index, so such a run is indexed only from
    ``dataset_tasks``, and only when it ran the full set and those tasks are
    exactly the ones it planned and excluded, which for a full run is the
    whole dataset. Otherwise it gets no indexes and so no upload rows (see
    :func:`main`).

    Args:
        meta: The runner's ``meta.json``.
        dataset_tasks: The dataset's task names at the revision the run used,
            for a run whose ``meta.json`` does not record them.
    """
    recorded = meta.get("dataset_tasks")
    if isinstance(recorded, list):
        tasks = {str(task) for task in recorded}
    elif dataset_tasks and not meta.get("task_subset"):
        planned = meta.get("planned_tasks")
        excluded = meta.get("excluded_tasks") or []
        if not isinstance(planned, list) or not isinstance(excluded, list):
            return {}
        tasks = set(dataset_tasks)
        if tasks != {str(task) for task in (*planned, *excluded)}:
            return {}
    else:
        return {}
    return {task: i for i, task in enumerate(sorted(tasks))}


def uploadable(
    meta: Mapping[str, object], dataset_tasks: Sequence[str] | None = None
) -> bool:
    """Returns whether the run's rows can go to BigQuery.

    A ``--task`` subset run is a smoke check, not a measurement, and the
    ``runs`` table cannot tell two subsets of one size apart, so their scores
    would merge. A run whose tasks cannot be indexed would be scored per
    trial rather than per task.
    """
    return not meta.get("task_subset") and bool(
        task_indexes(meta, dataset_tasks)
    )


def _row(t: Trial, prompt_index: int | None, repeat: int) -> dict[str, object]:
    return {
        "prompt_index": prompt_index,
        "repeat_index": repeat,
        "task_name": t.task,
        "trial_name": t.trial,
        "timestamp": t.started or None,
        "passed": t.valid and t.reward == 1,
        "reward": t.reward,
        "error": None if t.valid else _invalid_reason(t),
        "agent_error": (t.exception or None) if t.valid else None,
        "completion_tokens": t.gen,
        "prompt_tokens": t.in_total,
        "cached_tokens": t.in_total - t.in_uncached,
        "agent_turns": t.turns,
        "max_turn_completion_tokens": t.max_step_gen,
        "length_capped_turns": t.capped_steps,
        "tool_calls": t.tool_calls,
        "tool_errors": t.tool_errors,
        "loops": t.loops,
    }


def _rows(
    trials: Sequence[Trial],
    meta: Mapping[str, object],
    dataset_tasks: Sequence[str] | None,
    scored_only: bool,
) -> list[dict[str, object]]:
    index = task_indexes(meta, dataset_tasks)
    rows: list[dict[str, object]] = []
    seen: dict[str, int] = {}
    for t in sorted(trials, key=lambda t: (t.started, t.trial)):
        if scored_only and not t.valid:
            continue
        repeat = seen.get(t.task, 0)
        seen[t.task] = repeat + 1
        rows.append(_row(t, index.get(t.task), repeat))
    return rows


def upload_rows(
    trials: Sequence[Trial],
    meta: Mapping[str, object],
    dataset_tasks: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    """Returns one ``results.jsonl`` row per scored trial for the uploader.

    These are the trials the scorer counts: valid ones, at most ``k`` per
    task. ``prompt_index`` comes from :func:`task_indexes`; ``repeat_index``
    numbers a task's scored trials in the order they started.
    """
    return _rows(trials, meta, dataset_tasks, scored_only=True)


def attempt_rows(
    trials: Sequence[Trial],
    meta: Mapping[str, object],
    dataset_tasks: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    """Returns the :func:`upload_rows` row for every attempt, invalid ones too.

    ``repeat_index`` numbers all of a task's attempts here, and an invalid
    attempt's ``error`` says why it was not scored.
    """
    return _rows(trials, meta, dataset_tasks, scored_only=False)


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
    parser.add_argument(
        "--dataset-tasks-dir",
        type=Path,
        help="The dataset's tasks directory, at the revision the run used, for"
        " a run whose meta.json does not list the dataset's tasks. Its task"
        " names index the upload rows.",
    )
    args = parser.parse_args(argv)
    # Both directories hold a results.jsonl: the runner's trials and the
    # uploader's rows.
    if args.out_dir.resolve() == args.run_dir.resolve():
        parser.error("--out-dir must differ from --run-dir")
    dataset_tasks = None
    if args.dataset_tasks_dir is not None:
        dataset_tasks = sorted(
            p.parent.name for p in args.dataset_tasks_dir.glob("*/task.toml")
        )
        if not dataset_tasks:
            parser.error(f"{args.dataset_tasks_dir} holds no tasks")

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
    # The metrics weigh tasks equally by grouping rows on their task index,
    # so rows without one would be scored per trial. The uploader skips a
    # run that has no results.jsonl.
    if uploadable(meta, dataset_tasks):
        with (args.out_dir / "results.jsonl").open("w") as f:
            for row in upload_rows(trials, meta, dataset_tasks):
                f.write(json.dumps(row) + "\n")
    else:
        # An earlier conversion into this directory may have left rows that
        # the uploader would take for this run's.
        (args.out_dir / "results.jsonl").unlink(missing_ok=True)
        print(
            "results.jsonl not written, so this run does not upload: it is a"
            " --task subset, or its tasks cannot be indexed (meta.json lists"
            " no dataset tasks, and --dataset-tasks-dir was not given or does"
            " not match the tasks it planned and excluded)."
        )
    with (args.out_dir / "attempts.jsonl").open("w") as f:
        for row in attempt_rows(trials, meta, dataset_tasks):
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
