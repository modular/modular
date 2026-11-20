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
import datetime
import json
import os
import pathlib
import sys
from collections.abc import Mapping, Sequence
from typing import Any

PATCH = pathlib.Path("artifacts", "model.patch")
TRAJECTORY = pathlib.Path("agent", "mini-swe-agent.trajectory.json")


def trial_rows(job_dir: pathlib.Path) -> list[dict[str, object]]:
    """Reads every ``*/result.json`` under ``job_dir`` into one row per trial.

    Rows carry the keys the results uploader reads. ``prompt_index`` is the
    task's place among every task in its task directory, so sampled and full
    runs agree, and ``None`` when that directory is gone; ``repeat_index``
    numbers a task's attempts in the order they started; ``error`` says why
    an infra-error trial is excluded; ``response`` is the agent's patch.
    ``opening``, ``turns`` and their count ``agent_turns`` come from the
    agent's trajectory (see :func:`conversation_turns`), and are ``None``
    when it left no readable one.

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
                **_trajectory_fields(path.parent / TRAJECTORY),
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


def _trajectory_fields(path: pathlib.Path) -> dict[str, object]:
    # A trial that died before or while the agent wrote its trajectory still
    # gets its row and its score, just without turns.
    try:
        messages = json.loads(path.read_text())["messages"]
        opening, turns = conversation_turns(messages)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as err:
        print(
            f"WARNING: no turns for trial {path.parent.parent.name}: {err!r}",
            file=sys.stderr,
        )
        return {"agent_turns": None, "opening": None, "turns": None}
    return {"agent_turns": len(turns), "opening": opening, "turns": turns}


def conversation_turns(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, object]], list[dict[str, Any]]]:
    """Splits a mini-swe-agent conversation into its opening and its turns.

    A turn is one model call, in the form the results uploader writes to
    ``turns.parquet``: what the model returned, its token usage, and what
    each command it ran printed. The messages before the first call, the
    system prompt and the task, are the opening. A call that fails
    mini-swe-agent's format check, such as one cut off before its tool call,
    is logged as a user message holding the completion in
    ``extra.response``; it is a turn too, and the format error sent back is
    its reply. Any other later message is the reply of the turn before it.

    Args:
        messages: The trajectory's ``messages``.

    Returns:
        The opening's ``role``/``content`` messages, and the turns.
    """
    opening: list[dict[str, object]] = []
    turns: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        extra = message.get("extra") or {}
        response = extra.get("response") or {}
        if role == "assistant":
            turns.append(_turn(response, message))
        elif role == "user" and response:
            choice = (response.get("choices") or [{}])[0]
            turns.append(_turn(response, choice.get("message") or {}))
            turns[-1]["reply"] = message.get("content")
        elif role == "tool":
            content = message.get("content")
            raw = extra.get("raw_output")
            _add_tool_output(
                opening,
                turns,
                message.get("tool_call_id"),
                {
                    "output": content,
                    "full_output": raw
                    if raw and _shown_output(content) != raw
                    else None,
                    "returncode": extra.get("returncode"),
                    "error": extra.get("exception_info") or None,
                },
            )
        elif role != "exit":
            _add_message(opening, turns, role, message.get("content"))
    return opening, turns


def _turn(
    response: Mapping[str, Any], generated: Mapping[str, Any]
) -> dict[str, Any]:
    """Builds a turn from a chat completion and the message it generated."""
    usage = response.get("usage") or {}
    created = response.get("created")
    return {
        "timestamp": datetime.datetime.fromtimestamp(
            created, datetime.timezone.utc
        ).isoformat()
        if isinstance(created, (int, float))
        else None,
        "finish_reason": (response.get("choices") or [{}])[0].get(
            "finish_reason"
        ),
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get(
            "cached_tokens"
        ),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
            "reasoning_tokens"
        ),
        "reasoning": generated.get("reasoning_content"),
        "response": generated.get("content"),
        "tool_calls": [
            {
                "id": call.get("id"),
                "name": (call.get("function") or {}).get("name"),
                "arguments": _json_text(
                    (call.get("function") or {}).get("arguments")
                ),
            }
            for call in generated.get("tool_calls") or []
        ],
        "reply": None,
    }


def _shown_output(content: object) -> object:
    """Returns the whole command output a tool message showed the model.

    mini-swe-agent sends a long output as ``output_head`` and
    ``output_tail`` instead of ``output``, so this is ``None`` for it.
    """
    if not isinstance(content, str):
        return None
    try:
        shown = json.loads(content)
    except ValueError:
        return None
    return shown.get("output") if isinstance(shown, dict) else None


def _json_text(value: object) -> object:
    return (
        value if value is None or isinstance(value, str) else json.dumps(value)
    )


def _add_tool_output(
    opening: list[dict[str, object]],
    turns: list[dict[str, Any]],
    call_id: object,
    fields: dict[str, object],
) -> None:
    if not turns:
        opening.append({"role": "tool", "content": fields["output"]})
        return
    calls = turns[-1]["tool_calls"]
    call = next((c for c in calls if c["id"] == call_id), None)
    if call is None:
        # The model saw this output, so it is kept even with no call to match.
        call = {"id": call_id}
        calls.append(call)
    call.update(fields)


def _add_message(
    opening: list[dict[str, object]],
    turns: list[dict[str, Any]],
    role: object,
    content: object,
) -> None:
    if not turns:
        opening.append({"role": role, "content": content})
        return
    previous = turns[-1]["reply"]
    turns[-1]["reply"] = (
        content if previous is None else f"{previous}\n{content or ''}"
    )


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
