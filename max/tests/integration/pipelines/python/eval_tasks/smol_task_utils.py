# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Function for verifying that the predictions match the references."""


def results_match(references, predictions, **kwargs):
    total = 0
    for reference, prediction in zip(references, predictions):
        total += reference in prediction
    return total
