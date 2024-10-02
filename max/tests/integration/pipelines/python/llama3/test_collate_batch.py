# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import numpy as np
import pytest
from hypothesis import assume, given

from llama3.collate_batch import PaddingDirection, collate_batch


@given(arrays=..., pad_value=...)
def test_collate_batch(arrays: list[list[int]], pad_value: int):
    assume(arrays)
    # Need to be able to turn these into numpy values.
    assume(-(2**63) <= pad_value < 2**63)
    assume(all(-(2**63) <= v < 2**63 for a in arrays for v in a))

    result = collate_batch([np.array(a) for a in arrays], pad_value=pad_value)
    batch_size, length = result.shape
    assert batch_size == len(arrays)
    assert length == max(len(a) for a in arrays)

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

    result = collate_batch(
        [np.array(a) for a in arrays],
        pad_value=pad_value,
        direction=PaddingDirection.RIGHT,
    )
    batch_size, length = result.shape
    assert batch_size == len(arrays)
    assert length == max(len(a) for a in arrays)

    for array, padded in zip(arrays, result):
        np.testing.assert_array_equal(np.array(array), padded[: len(array)])
        assert np.all(padded[len(array) :] == pad_value)


@given(pad_value=...)
def test_collate_batch__no_items(pad_value: int):
    with pytest.raises(ValueError):
        collate_batch([], pad_value=pad_value)
