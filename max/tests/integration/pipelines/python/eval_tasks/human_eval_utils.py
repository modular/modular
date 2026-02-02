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
"""Functions for processing and evaluating the HumanEval dataset."""

import json
from typing import Any

from human_eval.execution import Problem
from human_eval.execution import check_correctness as _check_correctness


def doc_to_text(doc: dict[str, Any]) -> str:
    return doc["prompt"]


def doc_to_target(doc: dict[str, Any]) -> str:
    return json.dumps(doc)


def check_correctness(
    references: list[Problem], predictions: list[str], **kwargs: Any
) -> dict[str, float]:
    correct = _check_correctness(
        problem=references,
        completion=filter_code(predictions[0]),
        timeout=3.0,
    )

    return {"pass@1": float(correct["passed"])}


def filter_code(completion: str) -> str:
    # The program tends to overwrite, we only take the first function
    # https://github.com/declare-lab/instruct-eval/blob/main/human_eval/main.py#L35
    completion = completion.lstrip("\n")
    completion = completion.split("\n\n")[0]
    return completion
