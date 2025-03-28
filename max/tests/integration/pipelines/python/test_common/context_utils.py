# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Context utility functions for use in testing infrastructure."""

import numpy as np
from max.pipelines.context import InputContext, TextContext


def create_text_context(
    cache_seq_id: int,
    tokens: np.ndarray,
) -> InputContext:
    return TextContext(
        cache_seq_id=cache_seq_id,
        prompt=tokens.tolist(),
        max_length=None,
        tokens=tokens,
    )
