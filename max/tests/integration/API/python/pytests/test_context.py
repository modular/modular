# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import numpy as np
import pytest
from max.pipelines import TextContext


def test_context__current_length():
    context = TextContext(
        cache_seq_id=0,
        prompt="this is a test prompt",
        max_length=10,
        tokens=np.array([0, 1, 2, 3]),
    )

    assert context.current_length == 4

    context.update(4)
    assert context.current_length == 5

    # Currently, there are 5 tokens, we are saying
    # here is the next one, and we've generated 3 tokens
    # including that one, so increment the current length
    # accordingly.
    for i in range(3):
        context.update(5 + i)

    assert context.current_length == 8


def test_context__seq_len():
    context = TextContext(
        cache_seq_id=0,
        prompt="this is a test prompt",
        max_length=10,
        tokens=np.array([0, 1, 2, 3]),
    )

    assert context.seq_len == 4
    context.update(4)
    assert context.seq_len == 1
    for i in range(5):
        context.update(5 + i)
    assert context.seq_len == 1


def test_context__trim_prompt():
    context = TextContext(
        cache_seq_id=0,
        prompt="this is a test prompt",
        max_length=10,
        tokens=np.array([0, 1, 2, 3]),
    )

    # Can't trim more tokens than the context has.
    with pytest.raises(AssertionError):
        context.trim_prompt(999)

    # Trimming 0 tokens does nothing.
    context.trim_prompt(0)
    assert (context.next_tokens == np.array([0, 1, 2, 3])).all()
    assert context.active_length == 4
    assert context.current_length == 4

    # Trimming 2 tokens should remove the first 2 tokens of prompt.
    context.trim_prompt(2)
    assert (context.next_tokens == np.array([2, 3])).all()
    assert context.active_length == 2
    assert context.current_length == 4  # does not change

    # Can't trim prompt to 0 tokens.
    with pytest.raises(AssertionError):
        context.trim_prompt(2)


def test_context__update_beyond_chunk_size():
    # This check evaluates whether we can update this array.
    # However, behaviour with max serve for updating, is slightly
    # different than behaviour off the server, as the text context
    # moves between the api worker & server worker.
    # Before making changes to resize behaviour, ensure you
    # test with the server, not just the `generate` entrypoint.
    context = TextContext(
        cache_seq_id=0,
        prompt="this is a test prompt",
        max_length=10,
        tokens=np.array([0, 1, 2, 3]),
    )

    # 128, is the CHUNK_SIZE defined in context
    for i in range(128):
        context.update(i)
