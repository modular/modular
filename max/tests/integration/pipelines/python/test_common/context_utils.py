# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Context utility functions for use in testing infrastructure."""

import numpy as np
from max.pipelines.core import InputContext, TextContext


def create_text_context(
    cache_seq_id: int,
    tokens: np.ndarray,
) -> InputContext:
    ctx = TextContext(
        prompt=tokens.tolist(),
        max_length=1000,
        tokens=tokens,
    )
    ctx.assign_to_cache(cache_seq_id)
    return ctx
