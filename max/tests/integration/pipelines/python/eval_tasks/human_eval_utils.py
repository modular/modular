# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Functions for processing and evaluating the HumanEval dataset."""

import json

from human_eval.execution import check_correctness as _check_correctness


def doc_to_text(doc):  # noqa: ANN001
    return doc["prompt"]


def doc_to_target(doc):  # noqa: ANN001
    return json.dumps(doc)


def check_correctness(references, predictions, **kwargs):  # noqa: ANN001
    correct = _check_correctness(
        references,
        filter_code(predictions[0]),
        3.0,
    )

    return {"pass@1": float(correct["passed"])}


def filter_code(completion: str) -> str:
    # The program tends to overwrite, we only take the first function
    # https://github.com/declare-lab/instruct-eval/blob/main/human_eval/main.py#L35
    completion = completion.lstrip("\n")
    completion = completion.split("\n\n")[0]
    return completion
