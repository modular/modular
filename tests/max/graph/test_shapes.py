# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Hypothesis infrastructure test for shape."""

from conftest import MAX_INT64, shapes
from hypothesis import given
from max.graph import StaticDim


@given(shape=shapes())
def test_shape_product_fits_in_int64(shape):
    cumulative_product = 1
    for dim in shape:
        if isinstance(dim, StaticDim):
            cumulative_product *= dim.dim
        else:
            # Currently ignore symbolic dimensions.
            cumulative_product *= 1
    assert cumulative_product <= MAX_INT64
