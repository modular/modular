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
"""Tests for scicode_eval's per-step rows."""

from types import SimpleNamespace
from typing import Any

import scicode_eval
from eval_common import GenParams


class _FakeClient:
    """Answers every step, except for problems that mention "boom"."""

    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs: Any) -> SimpleNamespace:
        if "boom" in kwargs["messages"][0]["content"]:
            raise RuntimeError("HTTP 500")
        message = SimpleNamespace(
            content="```python\nx = 1\n```", reasoning="r"
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=SimpleNamespace(completion_tokens=3),
        )


def _problem(text: str, *steps: str) -> dict[str, Any]:
    return {
        "problem_description_main": text,
        # No test cases, so no step runs a test subprocess.
        "sub_steps": [{"step_number": s, "test_cases": []} for s in steps],
    }


def test_step_index_encodes_problem_and_step() -> None:
    assert scicode_eval.step_index("77.3") == 7703
    assert scicode_eval.step_index("1.12") == 112


def test_errored_problem_keeps_its_steps_out_of_the_score() -> None:
    rows, summary = scicode_eval.run_eval(
        _FakeClient(),
        [(0, _problem("fine", "1.1", "1.2")), (1, _problem("boom", "2.1"))],
        "model",
        workers=1,
        params=GenParams(),
        test_python="python",
        test_data="",
    )
    by_index = {r["prompt_index"]: r for r in rows}
    assert sorted(by_index) == [101, 102, 201]
    assert by_index[201]["error"] == "HTTP 500"
    assert by_index[201]["passed"] is False
    assert by_index[101]["response"] == "```python\nx = 1\n```"
    assert by_index[101]["reasoning"] == "r"
    assert by_index[101]["completion_tokens"] == 3
    assert by_index[101]["finish_reason"] == "stop"
    assert summary["total"] == 2
    # Each step records when its own request was sent; the errored problem's
    # step, which has none, takes the time the problem started.
    assert by_index[101]["timestamp"] < by_index[102]["timestamp"]
    assert by_index[201]["timestamp"]
