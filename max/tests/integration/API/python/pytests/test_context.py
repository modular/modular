# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import numpy as np
from max.pipelines import TextContext


def test_context__current_length():
    context = TextContext(
        cache_seq_id=0,
        prompt="this is a test prompt",
        max_length=10,
        next_tokens=np.array([0, 1, 2, 3]),
    )

    assert context.current_length == 4

    context.update(4)
    assert context.current_length == 5

    # Currently, there are 5 tokens, we are saying
    # here is the next one, and we've generated 3 tokens
    # including that one, so increment the current length
    # accordingly.
    context.update(5, num_steps=3)
    assert context.current_length == 8


def test_context__seq_len():
    context = TextContext(
        cache_seq_id=0,
        prompt="this is a test prompt",
        max_length=10,
        next_tokens=np.array([0, 1, 2, 3]),
    )

    assert context.seq_len == 4
    context.update(4)
    assert context.seq_len == 1
    context.update(5, num_steps=5)
    assert context.seq_len == 1
