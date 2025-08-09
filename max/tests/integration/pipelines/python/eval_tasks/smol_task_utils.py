# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Function for verifying that the predictions match the references."""

from typing import Any


def results_match(
    references: list[str], predictions: list[str], **kwargs: Any
) -> int:
    total = 0
    for reference, prediction in zip(references, predictions):
        total += reference in prediction
    return total
