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
"""Exercises the smoke CLI without downloading data or issuing model requests."""

import importlib.util
import json
import os
import signal
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

QUESTION_IDS = [
    10,
    15,
    23,
    24,
    40,
    43,
    44,
    48,
    50,
    55,
    56,
    64,
    66,
    73,
    79,
    83,
    86,
    90,
    91,
    109,
    114,
    115,
    139,
    145,
    151,
    154,
    157,
    163,
    167,
    177,
    180,
    190,
]


@pytest.mark.parametrize("scenario", ["complete", "partial", "cancelled"])
def test_smoke_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, scenario: str
) -> None:
    source = Path(__file__).with_name("exacto_gpqa_smoke.py")
    dataset = [
        SimpleNamespace(id=q, target=f"shuffled-{q}") for q in range(1, 199)
    ]
    task = SimpleNamespace(
        dataset=dataset, config=SimpleNamespace(max_tokens=None), metadata=None
    )
    interrupted = threading.Event()
    timers: list[threading.Timer] = []
    real_timer = threading.Timer

    def memory_dataset(
        samples: list[SimpleNamespace], name: str
    ) -> list[SimpleNamespace]:
        assert name == "mimo_gpqa_smoke_32"
        return samples

    def make_timer(seconds: float, callback: object) -> threading.Timer:
        assert callable(callback)
        timer = real_timer(seconds, callback)
        timers.append(timer)
        return timer

    def interrupt(pid: int, sig: int) -> None:
        assert pid == os.getpid() and sig == signal.SIGINT
        interrupted.set()

    def sample(
        question: int, grade: str, reason: str, answer: str | None
    ) -> SimpleNamespace:
        return SimpleNamespace(
            id=question,
            epoch=1,
            error=None,
            error_retries=[],
            scores={"mcq_scorer": SimpleNamespace(value=grade, answer=answer)},
            output=SimpleNamespace(
                choices=[SimpleNamespace(stop_reason=reason)],
                usage=SimpleNamespace(output_tokens=17),
            ),
        )

    def evaluate(
        selected_task: SimpleNamespace, **kwargs: object
    ) -> list[SimpleNamespace]:
        assert [s.id for s in selected_task.dataset] == QUESTION_IDS
        assert all(s is dataset[s.id - 1] for s in selected_task.dataset)
        assert selected_task.metadata["scheduled_question_ids"] == QUESTION_IDS
        assert kwargs["max_tokens"] == 32768
        assert kwargs["epochs"] == 1 and kwargs["temperature"] == 0.5
        assert kwargs["max_connections"] == 3
        assert kwargs["model_args"] == {"timeout": None, "max_retries": 0}
        assert kwargs["retry_on_error"] == 0
        assert os.environ["LOCAL_BASE_URL"] == "http://unused.invalid/v1"
        if scenario == "complete":
            samples = [sample(q, "C", "stop", "A") for q in QUESTION_IDS]
        else:
            samples = [
                sample(QUESTION_IDS[0], "C", "stop", "A"),
                sample(QUESTION_IDS[1], "I", "max_tokens", None),
                sample(QUESTION_IDS[2], "I", "model_length", "B"),
                SimpleNamespace(
                    id=QUESTION_IDS[3],
                    epoch=1,
                    error="server error",
                    error_retries=["earlier error"],
                    scores=None,
                    output=None,
                ),
                SimpleNamespace(
                    id=QUESTION_IDS[4],
                    epoch=1,
                    error=None,
                    error_retries=[],
                    scores=None,
                    output=None,
                ),
            ]
        if scenario == "cancelled":
            assert interrupted.wait(5), "wall-budget timer did not send SIGINT"
        return [
            SimpleNamespace(
                status="success" if scenario == "complete" else "cancelled",
                samples=samples,
            )
        ]

    # These third-party modules exist only in the separately pinned harness
    # venv. Keep this test offline and exercise the actual runner around them.
    boundaries: dict[str, dict[str, object]] = {
        "exacto_gpqa_task": {"gpqa_diamond": lambda: task},
        "exacto_streaming": {
            "__file__": str(source.with_name("exacto_streaming.py"))
        },
        "inspect_ai": {"eval": evaluate},
        "inspect_ai.dataset": {"MemoryDataset": memory_dataset},
    }
    for name, attributes in boundaries.items():
        module = ModuleType(name)
        for key, value in attributes.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("smoke_under_test", source)
    assert spec is not None and spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(
        runner, "WALL_SECONDS", 0.01 if scenario == "cancelled" else 60
    )
    monkeypatch.setattr(threading, "Timer", make_timer)
    monkeypatch.setattr(os, "kill", interrupt)
    output = tmp_path / "smoke"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(source),
            "--output",
            str(output),
            "--url",
            "http://unused.invalid/v1/",
            "--model",
            "synthetic",
            "--max-connections",
            "3",
        ],
    )
    runner.main()
    assert len(timers) == 1
    timers[0].join(timeout=1)
    assert not timers[0].is_alive()
    report = json.loads((output / "smoke-summary.json").read_text())
    assert report["scheduled"] == 32 and report["question_ids"] == QUESTION_IDS
    assert report["output_token_cap"] == 32768
    assert report["wall_budget_expired"] == (scenario == "cancelled")
    assert report["status"] == (
        "COMPLETE_SMOKE" if scenario == "complete" else "PARTIAL_SMOKE"
    )
    assert set(report["harness_sha256"]) == {
        "exacto_gpqa_task.py",
        "exacto_gpqa_openrouter.py",
        "exacto_mcq.py",
        "exacto_streaming.py",
        "exacto_gpqa_smoke.py",
    }
    if scenario == "complete":
        assert report["scored"] == report["correct"] == 32
        assert (
            report["unresolved"]
            == report["errors"]
            == report["token_capped"]
            == 0
        )
        assert report["possible_correct_fraction_range"] == [1, 1]
    else:
        assert report["scored"] == 3 and report["correct"] == 1
        assert report["incorrect"] == report["token_capped"] == 2
        assert (
            report["unanswered"]
            == report["errors"]
            == report["error_retries"]
            == 1
        )
        assert report["unresolved"] == 29
        assert report["observed_correct_fraction_of_scheduled"] == 1 / 32
        assert report["possible_correct_fraction_range"] == [1 / 32, 30 / 32]
        assert report["samples"][3]["grade"] is None
