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
from pathlib import Path

import pytest
import terminalbench_exacto as tbx
import terminalbench_score as tb

META: dict[str, object] = {
    "endpoint": "https://example.test",
    "model": "m",
    "serve_config": "arm a",
    "dataset": "terminal-bench-2-1@7131e43",
    "harbor": "0.25.0",
    "agent": "opencode@1.19.5",
    "k": 2,
    "output_cap": 65536,
    "temperature": 1.0,
    "top_p": 1.0,
    "thinking": "on",
    "reasoning_effort": None,
    "excluded_tasks": ["qemu-startup"],
    "task_subset": None,
    "planned_tasks": ["a", "b"],
}


def _trial(
    task: str,
    started: str,
    reward: float | None = 1.0,
    valid: bool = True,
    gen: int = 100,
) -> tb.Trial:
    return tb.Trial(
        trial=f"{task}__{started}",
        task=task,
        started=started,
        reward=reward,
        exception="",
        reached_model=True,
        valid=valid,
        turns=4,
        in_uncached=300,
        in_total=1000,
        out=gen // 2,
        reasoning=gen // 2,
        gen=gen,
        max_step_gen=gen,
        capped_steps=1,
        unclosed_steps=0,
        markup_in_text=2,
        loops=1,
        tool_calls=3,
        tool_errors=1,
    )


TRIALS = [
    _trial("a", "2026-11-01T00:00:02Z", reward=0.0),
    _trial("a", "2026-11-01T00:00:01Z", reward=1.0),
    _trial("b", "2026-11-01T00:00:03Z", reward=1.0),
    _trial("b", "2026-11-01T00:00:04Z", reward=1.0),
    _trial("b", "2026-11-01T00:00:00Z", reward=None, valid=False),
]


def test_trajectory_rows_number_repeats_per_task_by_start() -> None:
    rows = tbx.trajectory_rows(TRIALS)
    assert [(r["task_id"], r["repeat_index"], r["reward"]) for r in rows] == [
        ("a", 0, 1.0),
        ("a", 1, 0.0),
        ("b", 0, 1.0),
        ("b", 1, 1.0),
    ]


def test_trajectory_rows_carry_the_verdict_columns() -> None:
    row = tbx.trajectory_rows(TRIALS)[0]
    for column in ("task_id", "repeat_index", "error", "output_tokens"):
        assert column in row
    assert row["error"] is None
    assert row["passed"] is True
    assert row["cached_input_tokens"] == 700
    assert row["cap_hits"] == 1
    assert row["tool_markup_leaks"] == 2
    assert row["loop"] == 1


def test_exacto_score_uses_the_per_task_pass_rate() -> None:
    score = tbx.exacto_score(TRIALS, META)
    # Task a passes half its trials, task b all of them.
    assert score["accuracy"] == pytest.approx(0.75)
    assert score["stderr"] == pytest.approx(
        math.sqrt(((0.5 - 0.75) ** 2 + (1 - 0.75) ** 2) / 1) / math.sqrt(2)
    )
    assert score["total"] == 4
    assert score["num_tasks"] == 2
    assert score["num_trials"] == 2
    assert score["errors"] == 1
    assert score["temperature"] == 1.0
    assert score["thinking"] == "on"


def test_a_complete_run_reports_no_missing_tasks() -> None:
    score = tbx.exacto_score(TRIALS, META)
    assert score["complete"] is True
    assert score["tasks_short_of_k"] == 0
    assert score["missing_tasks"] == []
    assert score["planned_tasks"] == 2


def test_a_task_with_no_valid_trial_counts_as_short() -> None:
    # Task c ran but every attempt failed in the harness, so the scorer never
    # sees it among the scored tasks.
    trials = [
        *TRIALS,
        _trial("c", "2026-11-01T00:00:05Z", reward=None, valid=False),
    ]
    meta = {**META, "planned_tasks": ["a", "b", "c"]}
    score = tbx.exacto_score(trials, meta)
    assert score["complete"] is False
    assert score["tasks_short_of_k"] == 1
    assert score["missing_tasks"] == ["c"]
    # The score covers only what ran; complete says it is not a measurement.
    assert score["num_tasks"] == 2


def test_a_planned_task_that_never_ran_counts_as_short() -> None:
    meta = {**META, "planned_tasks": ["a", "b", "d"]}
    assert tbx.exacto_score(TRIALS, meta)["missing_tasks"] == ["d"]


def test_a_task_below_k_counts_as_short() -> None:
    trials = [t for t in TRIALS if t.started != "2026-11-01T00:00:04Z"]
    score = tbx.exacto_score(trials, META)
    assert score["missing_tasks"] == ["b"]
    assert score["complete"] is False


def test_the_runner_marking_a_run_invalid_makes_it_incomplete() -> None:
    score = tbx.exacto_score(TRIALS, META, runner_invalid=True)
    assert score["complete"] is False
    assert score["tasks_short_of_k"] == 0


def test_a_meta_without_a_plan_falls_back_to_the_scorer() -> None:
    meta = {key: v for key, v in META.items() if key != "planned_tasks"}
    score = tbx.exacto_score(TRIALS, meta)
    assert score["missing_tasks"] is None
    assert score["planned_tasks"] is None
    assert score["complete"] is True


def test_main_reads_the_runners_invalid_marker(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    tb.write_jsonl(run / "results.jsonl", TRIALS)
    (run / "meta.json").write_text(json.dumps(META))
    (run / "INVALID").write_text("valid_trials=4 tasks_short_of_k=0\n")
    out = tmp_path / "out"

    assert tbx.main(["--run-dir", str(run), "--out-dir", str(out)]) == 0

    assert json.loads((out / "score.json").read_text())["complete"] is False


def test_exacto_config_reports_a_subset_size() -> None:
    assert tbx.exacto_config(META)["sample_size"] is None
    subset = {**META, "task_subset": ["a", "b", "c"]}
    config = tbx.exacto_config(subset)
    assert config["sample_size"] == 3
    assert config["max_tokens"] == 65536
    assert config["base_url"] == "https://example.test"


def test_main_writes_score_config_and_trajectories(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    tb.write_jsonl(run / "results.jsonl", TRIALS)
    (run / "meta.json").write_text(json.dumps(META))
    out = tmp_path / "out"

    assert tbx.main(["--run-dir", str(run), "--out-dir", str(out)]) == 0

    score = json.loads((out / "score.json").read_text())
    assert score["accuracy"] == pytest.approx(0.75)
    assert json.loads((out / "config.json").read_text())["top_p"] == 1.0
    lines = (out / "trajectories.jsonl").read_text().splitlines()
    assert len(lines) == 4
