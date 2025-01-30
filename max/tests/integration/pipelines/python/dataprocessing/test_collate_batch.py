# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import math

import numpy as np
import pytest
from dataprocessing import (
    PaddingDirection,
    batch_padded_tokens_and_mask,
    collate_batch,
)
from hypothesis import assume, given
from hypothesis import strategies as st

# Shared batch size between start_pos and tokens.
batch_size_strategy = st.shared(
    st.integers(min_value=1, max_value=10), key="batch_size"
)

# Define start_pos_strategy using flatmap to get concrete batch_size.
start_pos_strategy = batch_size_strategy.flatmap(
    lambda batch_size: st.lists(
        st.integers(min_value=0, max_value=100),
        min_size=batch_size,
        max_size=batch_size,
    )
)
tokens_strategy = batch_size_strategy.flatmap(
    lambda batch_size: st.lists(
        st.lists(
            st.integers(min_value=0, max_value=1000), min_size=1, max_size=20
        ).map(lambda lst: np.array(lst, dtype=int)),
        min_size=batch_size,
        max_size=batch_size,
    )
)

PAD_MULTIPLES = [2, 1, 128]


@given(arrays=..., pad_value=...)
def test_collate_batch(arrays: list[list[int]], pad_value: int):
    assume(arrays)
    # Need to be able to turn these into numpy values.
    assume(-(2**63) <= pad_value < 2**63)
    assume(all(-(2**63) <= v < 2**63 for a in arrays for v in a))

    result, unpadded_last_token_indices = collate_batch(
        [np.array(a) for a in arrays],
        direction=PaddingDirection.LEFT,
        pad_value=pad_value,
    )
    batch_size, length = result.shape
    assert batch_size == len(arrays)
    assert length == max(len(a) for a in arrays)

    assert len(unpadded_last_token_indices) == batch_size
    # Padding left means the last token index should always just be the overall
    # last index.
    assert all(unpadded_last_token_indices == [-1] * len(arrays))

    for array, padded in zip(arrays, result):
        # Use pad_len rather than len(array) since slicing from -0 doesn't do what you want.
        pad_len = len(padded) - len(array)
        np.testing.assert_array_equal(np.array(array), padded[pad_len:])
        assert np.all(padded[:pad_len] == pad_value)


@given(arrays=..., pad_value=...)
def test_collate_batch__pad_right(arrays: list[list[int]], pad_value: int):
    assume(arrays)
    # Need to be able to turn these into numpy values.
    assume(-(2**63) <= pad_value < 2**63)
    assume(all(-(2**63) <= v < 2**63 for a in arrays for v in a))

    for pad_to_multiple_of in PAD_MULTIPLES:
        result, unpadded_last_token_indices = collate_batch(
            [np.array(a) for a in arrays],
            pad_value=pad_value,
            direction=PaddingDirection.RIGHT,
            pad_to_multiple_of=pad_to_multiple_of,
        )
        batch_size, length = result.shape
        assert batch_size == len(arrays)

        max_length = max(len(a) for a in arrays)
        if max_length == 1:
            padded_length = 1
        else:
            padded_length = (
                math.ceil(max_length / pad_to_multiple_of) * pad_to_multiple_of
            )
        assert length == padded_length

        assert len(unpadded_last_token_indices) == batch_size
        # Padding right means the last index should always just be token length - 1.
        assert all(unpadded_last_token_indices == [len(a) - 1 for a in arrays])

        for array, padded in zip(arrays, result):
            np.testing.assert_array_equal(np.array(array), padded[: len(array)])
            assert np.all(padded[len(array) :] == pad_value)


@given(pad_value=...)
def test_collate_batch__no_items(pad_value: int):
    with pytest.raises(ValueError):
        collate_batch([], pad_value=pad_value)


@given(start_pos=start_pos_strategy, tokens=tokens_strategy)
def test_collate_mask__tokens_and_mask_shapes_match(
    start_pos: list[int], tokens: list[np.ndarray]
) -> None:
    assert len(start_pos) == len(tokens), (
        "start_pos and tokens must have the same length"
    )

    for pad_to_multiple_of in PAD_MULTIPLES:
        batched_tokens, unpadded_last_token_index, attention_mask = (
            batch_padded_tokens_and_mask(
                start_pos, tokens, pad_to_multiple_of=pad_to_multiple_of
            )
        )

        assert batched_tokens.shape[0] == len(tokens), (
            "Batch size of tokens does not match"
        )
        assert len(unpadded_last_token_index) == len(tokens), (
            "Length of unpadded last tokens do not match"
        )
        assert attention_mask.shape[:2] == batched_tokens.shape, (
            "Attention mask shape mismatch"
        )
