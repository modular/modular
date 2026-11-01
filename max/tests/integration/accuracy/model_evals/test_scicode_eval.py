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
"""Tests for scicode_eval's per-step rows, prompts and test-script assembly."""

from types import SimpleNamespace
from typing import Any

import pytest
import scicode_eval
from eval_common import GenParams
from scicode_reference_steps import REFERENCE_STEP_CODE


class _FakeClient:
    """Answers every step with a one-line function named after the step.

    Problems that mention "boom" raise instead. The prompts it was sent are
    kept so tests can inspect them.
    """

    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=self)
        self.prompts: list[str] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        prompt = kwargs["messages"][0]["content"]
        if "boom" in prompt:
            raise RuntimeError("HTTP 500")
        self.prompts.append(prompt)
        # The next step's header is the last "def" in the prompt.
        header = [ln for ln in prompt.splitlines() if ln.startswith("def ")][-1]
        message = SimpleNamespace(
            content=f"```python\n{header}\n    return 1\n```", reasoning="r"
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=SimpleNamespace(completion_tokens=3),
        )


def _step(number: str, tests: bool = False) -> dict[str, Any]:
    name = "f_" + number.replace(".", "_")
    return {
        "step_number": number,
        "step_description_prompt": f"Describe {number}",
        "step_background": f"Background {number}",
        "function_header": f"def {name}():",
        "return_line": "    return result",
        "test_cases": ["assert True"] if tests else [],
    }


def _problem(text: str, *steps: str, tests: bool = False) -> dict[str, Any]:
    sub_steps = [_step(s, tests) for s in steps]
    for step in sub_steps:
        # The prompt carries step descriptions, not the main description.
        step["step_description_prompt"] += f" of {text}"
    return {
        "problem_description_main": text,
        "required_dependencies": "import numpy as np",
        "sub_steps": sub_steps,
    }


@pytest.fixture
def captured_tests(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Records the solution handed to each step's test script, keyed by step."""
    seen: dict[str, str] = {}

    def fake_run(
        code: str, dependencies: str, step_number: object, *a: Any, **kw: Any
    ) -> bool:
        seen[str(step_number)] = code
        return True

    monkeypatch.setattr(scicode_eval, "run_step_tests", fake_run)
    return seen


def test_step_index_encodes_problem_and_step() -> None:
    assert scicode_eval.step_index("77.3") == 7703
    assert scicode_eval.step_index("1.12") == 112


def test_extract_code_drops_import_lines_like_the_official_harness() -> None:
    response = (
        "Here it is:\n```python\n"
        "import numpy as np\n"
        "from scipy.integrate import simps\n"
        "def f():\n"
        "    import math\n"
        "    return np.pi\n"
        "```"
    )
    assert scicode_eval.extract_code(response) == (
        "def f():\n\n    return np.pi"
    )
    assert scicode_eval.extract_code("import os\nx = 1") == "x = 1"


def test_prompt_follows_the_official_template() -> None:
    steps = [_step("1.1"), _step("1.2")]
    prompt = scicode_eval.build_step_prompt(
        "import numpy as np",
        [(steps[0], "def f_1_1():\n    return 1")],
        steps[1],
    )
    # Earlier steps: description, gold background and the code written for them.
    assert (
        "Describe 1.1\nBackground 1.1\n\ndef f_1_1():\n    return 1" in prompt
    )
    # Next step: description, background, header and return line.
    assert (
        "Describe 1.2\nBackground 1.2\n\ndef f_1_2():\n\n    return result"
        in prompt
    )
    assert "import numpy as np" in prompt
    # The model implements only the next step; the harness supplies the rest.
    assert (
        "should NOT include the dependencies and functions of all previous steps"
        in prompt
    )
    assert "Wrap your complete solution" not in prompt


def test_each_step_is_tested_with_every_earlier_step_prepended(
    captured_tests: dict[str, str],
) -> None:
    client = _FakeClient()
    rows, summary = scicode_eval.run_eval(
        client,
        [(0, _problem("fine", "1.1", "1.2", "1.3", tests=True))],
        "model",
        workers=1,
        params=GenParams(),
        test_python="python",
        test_data="",
    )
    assert summary == {
        "pass_at_1": 1.0,
        "passed": 3,
        "total": 3,
        "mean_output_tokens": 3,
        "p50_output_tokens": 3,
        "mean_problem_output_tokens": 9,
        "p50_problem_output_tokens": 9,
    }
    assert captured_tests["1.1"] == "def f_1_1():\n    return 1"
    assert captured_tests["1.3"] == (
        "def f_1_1():\n    return 1\n"
        "def f_1_2():\n    return 1\n"
        "def f_1_3():\n    return 1"
    )
    # The third prompt shows both earlier steps' code.
    assert "def f_1_1():\n    return 1" in client.prompts[2]
    assert "def f_1_2():\n    return 1" in client.prompts[2]
    assert [r["step_number"] for r in rows] == ["1.1", "1.2", "1.3"]


def test_reference_steps_are_spliced_in_but_not_generated_or_scored(
    captured_tests: dict[str, str],
) -> None:
    client = _FakeClient()
    rows, summary = scicode_eval.run_eval(
        client,
        [(0, _problem("maxwell", "13.5", "13.6", "13.7", tests=True))],
        "model",
        workers=1,
        params=GenParams(),
        test_python="python",
        test_data="",
    )
    assert len(client.prompts) == 2
    assert [r["step_number"] for r in rows] == ["13.5", "13.7"]
    assert summary["total"] == 2
    assert REFERENCE_STEP_CODE["13.6"] in client.prompts[1]
    assert REFERENCE_STEP_CODE["13.6"] in captured_tests["13.7"]
    assert "13.6" not in captured_tests


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
    assert (
        by_index[101]["response"]
        == "```python\ndef f_1_1():\n    return 1\n```"
    )
    assert by_index[101]["reasoning"] == "r"
    assert by_index[101]["completion_tokens"] == 3
    assert by_index[101]["finish_reason"] == "stop"
    assert summary["total"] == 2
    # Each step records when its own request was sent; the errored problem's
    # step, which has none, takes the time the problem started.
    assert by_index[101]["timestamp"] < by_index[102]["timestamp"]
    assert by_index[201]["timestamp"]
