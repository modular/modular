# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Context utility functions for use in testing infrastructure."""

import numpy as np
from max.pipelines.core import TextContext


def create_text_context(
    tokens: np.ndarray, max_length: int = 1000
) -> TextContext:
    return TextContext(max_length=max_length, tokens=tokens)
