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
"""Tests for gpqa_eval's per-sample driver."""

import csv
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gpqa_eval
from eval_common import GenParams

SAMPLE = {
    "Question": "What is 2+2?",
    "Correct Answer": "four",
    "Incorrect Answer 1": "three",
    "Incorrect Answer 2": "five",
    "Incorrect Answer 3": "six",
}


class _FakeCompletions:
    def __init__(self, message: SimpleNamespace) -> None:
        self._response = SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=SimpleNamespace(completion_tokens=7),
        )

    def create(self, **_: Any) -> SimpleNamespace:
        return self._response


class _FakeClient:
    """Returns one canned completion for every request."""

    def __init__(self, message: SimpleNamespace) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(message))


def _infer(message: SimpleNamespace) -> dict[str, Any]:
    return gpqa_eval.infer(
        _FakeClient(message),
        "model",
        (0, 3, SAMPLE),
        GenParams(),
        root_preamble="",
        system_prompt="sys",
    )


def test_rows_keep_content_and_reasoning() -> None:
    row = _infer(SimpleNamespace(content="Answer: A", reasoning="hmm"))
    assert row["response"] == "Answer: A"
    assert row["reasoning"] == "hmm"
    assert row["prompt_index"] == 3
    assert row["completion_tokens"] == 7


def test_rows_read_reasoning_content() -> None:
    row = _infer(SimpleNamespace(content="Answer: A", reasoning_content="hmm"))
    assert row["reasoning"] == "hmm"


def test_rows_without_separate_reasoning() -> None:
    row = _infer(SimpleNamespace(content="<think>hmm</think>Answer: A"))
    assert row["response"] == "<think>hmm</think>Answer: A"
    assert row["reasoning"] is None


def test_loads_local_dataset_file(tmp_path: Path) -> None:
    # GPQA questions span lines and contain commas, so the CSV quotes them.
    row = {**SAMPLE, "Question": "Given x, y:\nwhat is x + y?"}
    path = tmp_path / "gpqa_diamond.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row)
        writer.writeheader()
        writer.writerow(row)
    assert gpqa_eval.load_gpqa_dataset(str(path)) == [row]
