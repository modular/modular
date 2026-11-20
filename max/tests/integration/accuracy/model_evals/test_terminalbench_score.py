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

import json
import math
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest
import terminalbench_score as tb


def _step(
    output: int, reasoning: int, reason: str, inp: int = 100, cache: int = 0
) -> dict[str, object]:
    return {
        "type": "step_finish",
        "part": {
            "reason": reason,
            "tokens": {
                "input": inp,
                "output": output,
                "reasoning": reasoning,
                "cache": {"read": cache, "write": 0},
            },
        },
    }


def _trial(
    jobs: Path,
    name: str,
    events: Sequence[object],
    reward: float | None = 1.0,
    exception: str | None = None,
    started: str = "2026-11-01T00:00:00Z",
    job: str = "main",
) -> Path:
    d = jobs / job / name
    (d / "agent").mkdir(parents=True)
    (d / "result.json").write_text(
        json.dumps(
            {
                "started_at": started,
                "verifier_result": None
                if reward is None
                else {"rewards": {"reward": reward}},
                "exception_info": {"exception_type": exception}
                if exception
                else None,
            }
        )
    )
    log = ["not json: opencode banner"] + [json.dumps(e) for e in events]
    (d / "agent" / "opencode.txt").write_text("\n".join(log) + "\n")
    return d


def test_valid_after_non_zero_exit(tmp_path: Path) -> None:
    # opencode exits non-zero after a normal stop; the trial is still valid.
    d = _trial(
        tmp_path,
        "task-a__1",
        [_step(10, 20, "tool-calls", cache=900), _step(5, 0, "stop")],
        exception="NonZeroAgentExitCodeError",
    )
    t = tb.parse_trial(d)
    assert t.valid and t.task == "task-a"
    assert (t.turns, t.gen, t.in_uncached, t.in_total) == (2, 35, 200, 1100)


def test_never_reached_model_is_invalid(tmp_path: Path) -> None:
    d = _trial(
        tmp_path,
        "task-a__1",
        [],
        reward=0.0,
        exception="NonZeroAgentExitCodeError",
    )
    assert not tb.parse_trial(d).valid


def test_missing_reward_is_invalid(tmp_path: Path) -> None:
    d = _trial(tmp_path, "task-a__1", [_step(1, 1, "stop")], reward=None)
    assert not tb.parse_trial(d).valid


def test_timeout_with_unfinished_response_is_valid(tmp_path: Path) -> None:
    # A runaway first response: reasoning streamed, no step ever finished.
    d = _trial(
        tmp_path,
        "task-a__1",
        [{"type": "reasoning", "part": {"text": "Let me think. " * 50}}],
        reward=0.0,
        exception="AgentTimeoutError",
    )
    t = tb.parse_trial(d)
    assert t.valid and t.reached_model
    assert (t.turns, t.gen, t.in_total) == (0, 0, 0)
    assert not any(t.flag(name) for name in tb.RATES)


def test_unfinished_response_without_timeout_is_invalid(tmp_path: Path) -> None:
    d = _trial(
        tmp_path,
        "task-a__1",
        [{"type": "text", "part": {"text": "partial"}}],
        reward=0.0,
        exception="NetworkConnectionError",
    )
    t = tb.parse_trial(d)
    assert t.reached_model and not t.valid


def test_timeout_is_valid(tmp_path: Path) -> None:
    d = _trial(
        tmp_path,
        "task-a__1",
        [_step(1, 1, "tool-calls")],
        reward=0.0,
        exception="AgentTimeoutError",
    )
    assert tb.parse_trial(d).valid


def test_cap_unclosed_and_markup(tmp_path: Path) -> None:
    events = [
        {
            "type": "text",
            "part": {"text": "<\uff5cDSML\uff5cinvoke name=write>"},
        },
        _step(0, 32768, "length"),
        _step(10, 0, "stop"),
    ]
    t = tb.parse_trial(_trial(tmp_path, "task-a__1", events))
    assert (t.capped_steps, t.unclosed_steps, t.markup_in_text) == (1, 1, 1)
    assert t.flag("capped") and t.flag("unclosed") and t.flag("markup")


def test_repeated_tool_call_is_a_loop(tmp_path: Path) -> None:
    call = {
        "type": "tool_use",
        "part": {
            "tool": "bash",
            "state": {"input": {"cmd": "ls"}, "status": "completed"},
        },
    }
    t = tb.parse_trial(
        _trial(tmp_path, "task-a__1", [call] * 8 + [_step(1, 1, "stop")])
    )
    assert t.loops == 1 and t.tool_calls == 8


def test_keeps_k_earliest_valid_trials(tmp_path: Path) -> None:
    for i, started in enumerate(["03", "01", "02"]):
        _trial(
            tmp_path,
            f"task-a__{i}",
            [_step(1, 1, "stop")],
            started=f"2026-11-01T00:00:{started}Z",
            job=f"j{i}",
        )
    _trial(tmp_path, "task-b__0", [], reward=0.0)
    trials = tb.load_trials(tmp_path, k=2)
    kept = sorted(t.started for t in trials if t.valid)
    assert kept == ["2026-11-01T00:00:01Z", "2026-11-01T00:00:02Z"]
    assert tb.missing(trials, ["task-a", "task-b", "task-c"], k=2) == {
        "task-b": 2,
        "task-c": 2,
    }


def test_summarize(tmp_path: Path) -> None:
    _trial(tmp_path, "task-a__0", [_step(10, 0, "stop")], reward=1.0)
    _trial(tmp_path, "task-a__1", [_step(30, 0, "stop")], reward=0.0)
    _trial(tmp_path, "task-b__0", [_step(0, 32768, "length")], reward=0.0)
    s = tb.summarize(tb.load_trials(tmp_path, k=2), k=2)
    assert s["score"] == pytest.approx(0.25)  # mean of per-task 0.5 and 0.0
    assert s["tasks"] == 2 and s["tasks_short_of_k"] == 1 and s["total"] == 3
    assert s["capped_trial_rate"] == pytest.approx(1 / 3)
    assert s["mean_output_tokens"] == s["mean_gen"]


def test_summarize_reports_invalid_trials(tmp_path: Path) -> None:
    for i, started in enumerate(["01", "02", "03"]):
        _trial(
            tmp_path,
            f"task-a__{i}",
            [_step(1, 1, "stop")],
            started=f"2026-11-01T00:00:{started}Z",
            job=f"j{i}",
        )
    _trial(tmp_path, "task-b__0", [], reward=0.0)
    _trial(
        tmp_path,
        "task-b__1",
        [_step(5, 0, "tool-calls")],
        reward=0.0,
        exception="NetworkConnectionError",
    )
    s = tb.summarize(tb.load_trials(tmp_path, k=2), k=2)
    # The third task-a trial is over k, not invalid.
    assert (s["trials_seen"], s["total"]) == (5, 2)
    assert (s["invalid_trials"], s["invalid_reached_model"]) == (2, 1)
    assert s["errors"] == s["invalid_trials"]


def test_wilcoxon_greater_is_exact_for_small_n() -> None:
    # Ten positive differences: only the all-positive assignment reaches W+.
    assert tb.wilcoxon_greater([float(i) for i in range(1, 11)]) == 1 / 1024
    # Four tied positive differences (e.g. 4 of 80 tasks flip pass -> fail):
    # exact 1/16, where the normal approximation says 0.036.
    assert tb.wilcoxon_greater([1.0] * 4 + [0.0] * 76) == 1 / 16
    # Ties: doubled midranks 3, 3, 7, 7; 10 of 16 sign assignments reach W+ >= 5.
    assert tb.wilcoxon_greater([-1.0, 1.0, -2.0, 2.0]) == 10 / 16
    assert tb.wilcoxon_greater([0.0, 0.0]) == 1.0


def test_wilcoxon_greater_uses_normal_approximation_for_large_n() -> None:
    d = [float(i if i % 3 else -i) for i in range(1, 61)]
    w_plus = sum(i for i in range(1, 61) if i % 3)
    z = (w_plus - 60 * 61 / 4 - 0.5) / math.sqrt(60 * 61 * 121 / 24)
    assert tb.wilcoxon_greater(d) == pytest.approx(
        0.5 * math.erfc(z / math.sqrt(2))
    )
    # At the switch-over the two agree closely, ties included.
    d = [float((i // 2) if i % 3 else -(i // 2)) for i in range(2, 52)]
    assert tb._wilcoxon_exact(d) == pytest.approx(
        tb._wilcoxon_normal(d), abs=0.005
    )


def test_fisher_greater() -> None:
    assert tb.fisher_greater(5, 10, 0, 10) == pytest.approx(252 / 15504)
    assert tb.fisher_greater(0, 10, 5, 10) == pytest.approx(1.0)


def _arm(
    gen_scale: float, tasks: int = 60, capped_every: int = 0, turns: int = 20
) -> list[tb.Trial]:
    out = []
    template = tb.Trial(
        trial="",
        task="",
        started="",
        reward=1.0,
        exception="",
        reached_model=True,
        valid=True,
        turns=20,
        in_uncached=1000,
        in_total=100000,
        out=0,
        reasoning=0,
        gen=0,
        max_step_gen=0,
        capped_steps=0,
        unclosed_steps=0,
        markup_in_text=0,
        loops=0,
        tool_calls=10,
        tool_errors=0,
    )
    for i in range(tasks):
        for r in range(2):
            base = 5000 + 137 * i + 61 * r
            out.append(
                replace(
                    template,
                    trial=f"t{i}__{r}",
                    task=f"t{i}",
                    turns=turns,
                    gen=int(base * gen_scale),
                    capped_steps=int(
                        bool(capped_every) and (i * 2 + r) % capped_every == 0
                    ),
                )
            )
    return out


def test_compare_passes_identical_arms() -> None:
    passed, checks = tb.compare(_arm(1.0), _arm(1.0))
    assert passed and not any(c.failed for c in checks)


def test_compare_fails_large_token_increase() -> None:
    passed, checks = tb.compare(_arm(1.0), _arm(1.3))
    assert not passed
    assert [c.metric.split(" ")[0] for c in checks if c.failed] == ["gen"]


def test_compare_ignores_increase_below_floor() -> None:
    # Consistent but small (+5%): significant, yet under the 1.10 floor.
    passed, _ = tb.compare(_arm(1.0), _arm(1.05))
    assert passed


def test_compare_ratio_is_not_shrunk_for_small_counts() -> None:
    # 10 -> 11 turns on every task is x1.10 and meets the floor; a +1
    # smoothed ratio would measure x1.09 and let it pass.
    passed, checks = tb.compare(_arm(1.0, turns=10), _arm(1.0, turns=11))
    turns = next(c for c in checks if c.metric.startswith("turns"))
    assert turns.effect == pytest.approx(1.10)
    assert turns.failed and not passed


def test_log_ratio_zero_baseline() -> None:
    assert tb._log_ratio(0.0, 0.0) == 0.0
    assert tb._log_ratio(0.0, 3.0) == pytest.approx(math.log(4.0))
    assert tb._log_ratio(3.0, 0.0) == pytest.approx(math.log(0.25))


def test_compare_only_flags_regressions() -> None:
    passed, _ = tb.compare(_arm(1.3), _arm(1.0))
    assert passed


def test_compare_fails_more_capped_trials() -> None:
    passed, checks = tb.compare(_arm(1.0), _arm(1.0, capped_every=4))
    assert not passed
    assert [c.metric.split(" ")[0] for c in checks if c.failed] == ["capped"]


def test_jsonl_round_trip(tmp_path: Path) -> None:
    trials = _arm(1.0, tasks=2)
    tb.write_jsonl(tmp_path / "r.jsonl", trials)
    assert tb.read_jsonl(tmp_path / "r.jsonl") == trials


def _write_arms(
    tmp_path: Path, base: list[tb.Trial], cand: list[tb.Trial]
) -> list[str]:
    tb.write_jsonl(tmp_path / "base.jsonl", base)
    tb.write_jsonl(tmp_path / "cand.jsonl", cand)
    return [
        "compare",
        "--baseline",
        str(tmp_path / "base.jsonl"),
        "--candidate",
        str(tmp_path / "cand.jsonl"),
        "--out",
        str(tmp_path / "verdict.json"),
    ]


def test_compare_needs_enough_paired_tasks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = _write_arms(tmp_path, _arm(1.0, tasks=60), _arm(1.0, tasks=60))
    assert tb.main(args) == 2
    out = capsys.readouterr().out
    assert out.startswith("TerminalBench gate: NO VERDICT (60 paired tasks)")
    assert "PASS" not in out
    verdict = json.loads((tmp_path / "verdict.json").read_text())
    assert verdict["status"] == "no_verdict" and verdict["passed"] is None
    assert tb.main(args + ["--min-tasks", "60"]) == 0
    assert json.loads((tmp_path / "verdict.json").read_text())["status"] == (
        "pass"
    )


def test_compare_refuses_unpaired_tasks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cand = [t for t in _arm(1.0, tasks=60) if t.task != "t7"]
    args = _write_arms(tmp_path, _arm(1.0, tasks=60), cand)
    args += ["--min-tasks", "50"]
    assert tb.main(args) == 2
    assert "only in the baseline" in capsys.readouterr().out
    verdict = json.loads((tmp_path / "verdict.json").read_text())
    assert (
        verdict["baseline_only"] == ["t7"] and verdict["candidate_only"] == []
    )
    assert tb.main(args + ["--exclude", "t7"]) == 0
    verdict = json.loads((tmp_path / "verdict.json").read_text())
    assert verdict["excluded"] == ["t7"] and verdict["paired_tasks"] == 59


def test_compare_reports_invalid_shares(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cand = _arm(1.0, tasks=60)
    cand[0] = replace(cand[0], valid=False, exception="NetworkConnectionError")
    cand.append(
        replace(cand[1], trial="t0__9", valid=False, reached_model=False)
    )
    args = _write_arms(tmp_path, _arm(1.0, tasks=60), cand)
    assert tb.main(args + ["--min-tasks", "60"]) == 0
    out = capsys.readouterr().out
    assert "- baseline: 0/120" in out
    assert "- candidate: 2/121 (1.7%), 1 of them reached the model" in out
    arms = json.loads((tmp_path / "verdict.json").read_text())["arms"]
    assert arms["candidate"] == {
        "trials_seen": 121,
        "invalid_trials": 2,
        "invalid_reached_model": 1,
    }
