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
"""Checks that the GPQA task scores with the harness's MCQ extraction.

inspect_ai and datasets exist only in the harness venv, so they are stubbed
with just the names the task touches.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from collections.abc import Callable, Coroutine
from pathlib import Path
from types import ModuleType, SimpleNamespace

import exacto_gpqa_openrouter as gpqa
import exacto_mcq as mcq
import pytest


def _record(*args: object, **kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(args=args, **kwargs)


Score = Callable[[object, object], Coroutine[object, object, object]]


class _RegisteredScorer:
    def __init__(self, score: Score, name: str | None, metrics: object):
        self.score = score
        self.name = name
        self.metrics = metrics


def _scorer(
    metrics: object, name: str | None = None
) -> Callable[[Callable[[], Score]], Callable[[], _RegisteredScorer]]:
    def register(
        factory: Callable[[], Score],
    ) -> Callable[[], _RegisteredScorer]:
        return lambda: _RegisteredScorer(factory(), name, metrics)

    return register


@pytest.fixture
def task_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    boundaries: dict[str, dict[str, object]] = {
        "datasets": {"load_dataset": None},
        "inspect_ai": {
            "Epochs": lambda n: ("epochs", n),
            "Task": _record,
            "task": lambda f: f,
        },
        "inspect_ai.dataset": {"MemoryDataset": _record, "Sample": _record},
        "inspect_ai.model": {"GenerateConfig": _record},
        "inspect_ai.scorer": {
            "CORRECT": "C",
            "INCORRECT": "I",
            "Score": _record,
            "Scorer": object,
            "Target": object,
            "accuracy": lambda: "accuracy",
            "scorer": _scorer,
            "std": lambda: "std",
            "stderr": lambda: "stderr",
        },
        "inspect_ai.solver": {
            "TaskState": object,
            "generate": lambda: "generate",
            "system_message": lambda text: ("system_message", text),
        },
    }
    for name, attributes in boundaries.items():
        module = ModuleType(name)
        for key, value in attributes.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
    source = Path(__file__).with_name("exacto_gpqa_task.py")
    spec = importlib.util.spec_from_file_location(
        "gpqa_task_under_test", source
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _score(
    scorer: _RegisteredScorer, completion: str, target: str
) -> SimpleNamespace:
    state = SimpleNamespace(output=SimpleNamespace(completion=completion))
    result = asyncio.run(scorer.score(state, SimpleNamespace(text=target)))
    assert isinstance(result, SimpleNamespace)
    return result


def test_scorer_keeps_openbench_name_and_metrics(
    task_module: ModuleType,
) -> None:
    registered = task_module.harness_mcq()
    assert registered.name == "mcq_scorer"
    assert registered.metrics == ["accuracy", "stderr", "std"]


@pytest.mark.parametrize(
    ("completion", "target", "value", "answer"),
    [
        # openbench 0.5.3 reads A from the first two and nothing from the third.
        ("Option A is tempting, but no.\nAnswer: $C", "C", "C", "C"),
        ("(A) is out. Answer: $C", "C", "C", "C"),
        ("答案\uff1a\uff25", "E", "C", "E"),  # Full-width colon and E.
        ("Answer: B", "C", "I", "B"),
        ("", "A", "I", None),
    ],
)
def test_scorer_uses_the_harness_extraction(
    task_module: ModuleType,
    completion: str,
    target: str,
    value: str,
    answer: str | None,
) -> None:
    result = _score(task_module.harness_mcq(), completion, target)
    expected = mcq.score_mcq(completion, target)
    assert (result.value, result.answer) == (value, answer)
    assert result.answer == expected.answer
    assert result.explanation == expected.explanation


def test_task_wires_the_harness_scorer(
    task_module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(task_module, "load_samples", list)
    built = task_module.gpqa_diamond()
    assert built.scorer.name == "mcq_scorer"
    assert _score(built.scorer, "(A) is out. Answer: $D", "D").value == "C"
    assert built.config.temperature == gpqa.TEMPERATURE
    assert built.epochs == ("epochs", gpqa.EPOCHS)
    assert built.solver == [("system_message", gpqa.SYSTEM_MESSAGE), "generate"]
    assert built.metadata["harness_commit"] == gpqa.HARNESS_COMMIT
    assert built.metadata["mcq_extractor_commit"] == mcq.HARNESS_COMMIT
