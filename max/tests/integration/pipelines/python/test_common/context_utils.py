# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Context utility functions for use in testing infrastructure."""

import numpy as np
from max.interfaces import InputContext
from max.pipelines.core import TextContext


def create_text_context(
    cache_seq_id: int,
    tokens: np.ndarray,
    max_length: int = 1000,
) -> InputContext:
    ctx = TextContext(
        max_length=max_length,
        tokens=tokens,
    )
    return ctx  # type: ignore
