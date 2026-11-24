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
from typing import Any

import deepswe_score
import pytest


def _write_trial(
    job_dir: pathlib.Path,
    trial: str,
    task: str,
    reward: float | None,
    exception: str | None = None,
    verifier_result: bool = True,
    **extra: object,
) -> None:
    result: dict[str, object] = {
        "trial_name": trial,
        "task_name": task,
        **extra,
    }
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


def test_rows_carry_the_uploader_fields(tmp_path: pathlib.Path) -> None:
    tasks = tmp_path / "tasks"
    for name in ("alpha", "beta", "gamma"):
        (tasks / name).mkdir(parents=True)
    (tasks / "README.md").write_text("not a task")
    job_dir = tmp_path / "deepswe"
    job_dir.mkdir()
    # beta's second attempt started first, so it is repeat 0.
    _write_trial(
        job_dir,
        "beta__1",
        "datacurve/beta",
        1,
        task_id={"path": str(tasks / "beta")},
        started_at="2026-11-06T05:20:00Z",
        agent_result={"n_output_tokens": 50, "n_agent_steps": 7},
    )
    _write_trial(
        job_dir,
        "beta__2",
        "datacurve/beta",
        0,
        exception="AgentTimeoutError",
        task_id={"path": str(tasks / "beta")},
        started_at="2026-11-06T05:10:00Z",
    )
    _write_trial(
        job_dir,
        "gamma__1",
        "datacurve/gamma",
        -1,
        exception="VerifierTimeoutError",
        task_id={"path": str(tasks / "gamma")},
    )
    (job_dir / "beta__1" / "artifacts").mkdir()
    (job_dir / "beta__1" / "artifacts" / "model.patch").write_text("diff\n")

    rows = {r["trial_name"]: r for r in deepswe_score.trial_rows(job_dir)}

    assert rows["beta__1"] == {
        "prompt_index": 1,
        "repeat_index": 1,
        "task_name": "datacurve/beta",
        "trial_name": "beta__1",
        "timestamp": "2026-11-06T05:20:00Z",
        "passed": True,
        "reward": 1,
        "partial": None,
        "error": None,
        "agent_error": None,
        "completion_tokens": 50,
        "prompt_tokens": None,
        "cached_tokens": None,
        "agent_steps": 7,
        "response": "diff\n",
        "agent_turns": None,
        "opening": None,
        "turns": None,
    }
    assert rows["beta__2"]["repeat_index"] == 0
    assert rows["beta__2"]["passed"] is False
    assert rows["beta__2"]["error"] is None
    assert rows["beta__2"]["agent_error"] == "AgentTimeoutError"
    assert rows["beta__2"]["response"] is None
    assert rows["gamma__1"]["prompt_index"] == 2
    assert rows["gamma__1"]["passed"] is False
    assert rows["gamma__1"]["error"] == "VerifierTimeoutError"
    assert rows["gamma__1"]["agent_error"] is None


def test_rows_without_their_task_directory_have_no_prompt_index(
    tmp_path: pathlib.Path,
) -> None:
    _write_trial(
        tmp_path, "a__1", "datacurve/a", 1, task_id={"path": "/gone/tasks/a"}
    )
    (tmp_path / "b__1").mkdir()
    (tmp_path / "b__1" / "result.json").write_text("{")

    rows = deepswe_score.trial_rows(tmp_path)

    assert [r["prompt_index"] for r in rows] == [None, None]
    assert [r["repeat_index"] for r in rows] == [0, 0]
    assert str(rows[1]["error"]).startswith("unreadable result.json")


@pytest.mark.parametrize(
    "env, config",
    [
        (
            {"OPENAI_API_BASE": "https://gw.example/v1/", "SAMPLE_SIZE": "3"},
            {"base_url": "https://gw.example", "sample_size": 3},
        ),
        ({}, {"base_url": None, "sample_size": None}),
    ],
)
def test_main_writes_results_and_config_for_the_uploader(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    env: dict[str, str],
    config: dict[str, object],
) -> None:
    job_dir = tmp_path / "deepswe"
    job_dir.mkdir()
    _write_trial(job_dir, "a__1", "datacurve/a", 1)
    _write_trial(job_dir, "b__1", "datacurve/b", None, verifier_result=False)
    monkeypatch.setattr("sys.argv", ["deepswe_score.py", str(job_dir)])
    for name in ("GITHUB_ENV", "OPENAI_API_BASE", "SAMPLE_SIZE"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    deepswe_score.main()

    lines = (tmp_path / "results.jsonl").read_text().splitlines()
    assert [json.loads(line)["trial_name"] for line in lines] == [
        "a__1",
        "b__1",
    ]
    assert json.loads((tmp_path / "config.json").read_text()) == config
    assert json.loads((tmp_path / "score.json").read_text())["passed"] == 1


def _tool_message(
    call_id: str, content: dict[str, object], **extra: object
) -> dict[str, object]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps(content),
        "extra": extra,
    }


def _bash_call(call_id: str, command: str) -> dict[str, object]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "bash",
            "arguments": json.dumps({"command": command}),
        },
    }


SYSTEM = {"role": "system", "content": "You can use a computer."}
TASK = {"role": "user", "content": "Please solve this issue: fix the cache."}
# mini-swe-agent shows the model only the head and tail of a long output.
LONG_OUTPUT = "head" + "-" * 20_000 + "tail"
TRUNCATED = {
    "returncode": 1,
    "output_head": "head",
    "output_tail": "tail",
    "elided_chars": 20_000,
    "warning": "Output too long.",
}
NOT_RUN = {
    "returncode": -1,
    "output": "",
    "exception_info": "action was not executed",
}
TRAJECTORY_MESSAGES: list[dict[str, Any]] = [
    SYSTEM,
    TASK,
    {
        "role": "assistant",
        "content": "Let me look.",
        "reasoning_content": "Start with the tree.",
        "tool_calls": [_bash_call("c1", "ls"), _bash_call("c2", "cat big.log")],
        "extra": {
            "response": {
                "created": 1794328293,
                "choices": [{"finish_reason": "tool_calls"}],
                "usage": {
                    "prompt_tokens": 900,
                    "completion_tokens": 40,
                    "prompt_tokens_details": {"cached_tokens": 512},
                    "completion_tokens_details": {"reasoning_tokens": 12},
                },
            }
        },
    },
    _tool_message(
        "c1", {"returncode": 0, "output": "a.py\n"}, raw_output="a.py\n"
    ),
    _tool_message("c2", TRUNCATED, raw_output=LONG_OUTPUT, returncode=1),
    # mini-swe-agent logs a call that fails its format check this way.
    {
        "role": "user",
        "content": "Tool call error: No tool calls found in the response.",
        "extra": {
            "interrupt_type": "FormatError",
            "response": {
                "created": 1794328297,
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {
                            "content": "I will now edit",
                            "reasoning_content": "Next, the fix.",
                            "tool_calls": None,
                        },
                    }
                ],
                "usage": {"prompt_tokens": 950, "completion_tokens": 64},
            },
        },
    },
    {
        "role": "assistant",
        "content": "Done.",
        "tool_calls": [
            _bash_call("c3", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")
        ],
        "extra": {
            "response": {
                "created": 1794328300,
                "choices": [{"finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 5},
            }
        },
    },
    _tool_message(
        "c3",
        NOT_RUN,
        raw_output="",
        returncode=-1,
        exception_info="action was not executed",
    ),
    {"role": "exit", "content": "", "extra": {"exit_status": "Submitted"}},
]


def _write_trajectory(trial_dir: pathlib.Path, text: str) -> None:
    (trial_dir / "agent").mkdir()
    (trial_dir / "agent" / "mini-swe-agent.trajectory.json").write_text(text)


def test_rows_carry_the_trials_turns(tmp_path: pathlib.Path) -> None:
    _write_trial(tmp_path, "a__1", "datacurve/a", 1)
    _write_trajectory(
        tmp_path / "a__1", json.dumps({"messages": TRAJECTORY_MESSAGES})
    )

    (row,) = deepswe_score.trial_rows(tmp_path)

    assert row["agent_turns"] == 3
    assert row["opening"] == [SYSTEM, TASK]
    assert row["turns"] == [
        {
            "timestamp": "2026-11-10T16:31:33+00:00",
            "finish_reason": "tool_calls",
            "prompt_tokens": 900,
            "cached_tokens": 512,
            "completion_tokens": 40,
            "reasoning_tokens": 12,
            "reasoning": "Start with the tree.",
            "response": "Let me look.",
            "tool_calls": [
                {
                    "id": "c1",
                    "name": "bash",
                    "arguments": '{"command": "ls"}',
                    "output": '{"returncode": 0, "output": "a.py\\n"}',
                    "full_output": None,
                    "returncode": None,
                    "error": None,
                },
                {
                    "id": "c2",
                    "name": "bash",
                    "arguments": '{"command": "cat big.log"}',
                    "output": json.dumps(TRUNCATED),
                    "full_output": LONG_OUTPUT,
                    "returncode": 1,
                    "error": None,
                },
            ],
            "reply": None,
        },
        {
            "timestamp": "2026-11-10T16:31:37+00:00",
            "finish_reason": "length",
            "prompt_tokens": 950,
            "cached_tokens": None,
            "completion_tokens": 64,
            "reasoning_tokens": None,
            "reasoning": "Next, the fix.",
            "response": "I will now edit",
            "tool_calls": [],
            "reply": "Tool call error: No tool calls found in the response.",
        },
        {
            "timestamp": "2026-11-10T16:31:40+00:00",
            "finish_reason": "tool_calls",
            "prompt_tokens": 1000,
            "cached_tokens": None,
            "completion_tokens": 5,
            "reasoning_tokens": None,
            "reasoning": None,
            "response": "Done.",
            "tool_calls": [
                {
                    "id": "c3",
                    "name": "bash",
                    "arguments": json.dumps(
                        {
                            "command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
                        }
                    ),
                    "output": json.dumps(NOT_RUN),
                    "full_output": None,
                    "returncode": -1,
                    "error": "action was not executed",
                }
            ],
            "reply": None,
        },
    ]


@pytest.mark.parametrize(
    "trajectory",
    ['{"messages": [', "{}", '{"messages": null}', '{"messages": ["x"]}'],
)
def test_trials_without_a_readable_trajectory_have_no_turns(
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
    trajectory: str,
) -> None:
    _write_trial(tmp_path, "a__1", "datacurve/a", 1)
    _write_trial(tmp_path, "b__1", "datacurve/b", 0)
    _write_trajectory(tmp_path / "b__1", trajectory)

    summary, _ = deepswe_score.score_job(tmp_path)
    rows = deepswe_score.trial_rows(tmp_path)

    assert summary is not None
    assert summary["graded_trials"] == 2
    assert [(r["agent_turns"], r["opening"], r["turns"]) for r in rows] == [
        (None, None, None)
    ] * 2
    assert "no turns for trial b__1" in capsys.readouterr().err


def test_later_messages_with_no_completion_are_replies() -> None:
    _, turns = deepswe_score.conversation_turns(
        [
            TRAJECTORY_MESSAGES[2],
            {"role": "user", "content": "Keep going."},
            {"role": "user", "content": "Submit when done."},
        ]
    )

    assert len(turns) == 1
    assert turns[0]["reply"] == "Keep going.\nSubmit when done."


def test_outputs_with_no_matching_call_are_kept() -> None:
    opening, turns = deepswe_score.conversation_turns(
        [
            _tool_message("early", {"output": "before any call"}),
            TRAJECTORY_MESSAGES[2],
            _tool_message("stray", {"output": "x"}, raw_output="x"),
        ]
    )

    assert opening == [
        {"role": "tool", "content": json.dumps({"output": "before any call"})}
    ]
    assert [c["id"] for c in turns[0]["tool_calls"]] == ["c1", "c2", "stray"]
    assert turns[0]["tool_calls"][2]["output"] == json.dumps({"output": "x"})


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
    assert not (tmp_path / "results.jsonl").exists()
