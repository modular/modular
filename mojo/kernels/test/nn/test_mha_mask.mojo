# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from nn.mha_mask import CausalMask, TileMaskStatus
from testing import assert_equal, assert_true

from utils.index import Index
from utils.numerics import min_or_neg_inf


def test_causal_mask():
    alias type = DType.int32

    print("test_causal_mask")
    var mask = CausalMask()

    # Check mask value.
    var mask_val = min_or_neg_inf[type]()
    var masked_vec = mask.mask(Index(0, 0, 4, 3), SIMD[type, 4](0, 1, 2, 3))
    assert_equal(masked_vec, SIMD[type, 4](0, 1, mask_val, mask_val))

    masked_vec = mask.mask(Index(0, 0, 4, 0), SIMD[type, 4](0, 1, 2, 3))
    assert_equal(masked_vec, SIMD[type, 4](0, 1, 2, 3))

    masked_vec = mask.mask(Index(0, 0, 1, 6), SIMD[type, 4](0, 1, 2, 3))
    assert_equal(masked_vec, SIMD[type, 4](mask_val))

    # Check tile status.
    assert_true(
        mask.status(Index(4, 4), Index(4, 4)) == TileMaskStatus.PARTIAL_MASK
    )
    assert_true(
        mask.status(Index(0, 2), Index(2, 2)) == TileMaskStatus.PARTIAL_MASK
    )
    assert_true(mask.status(Index(2, 0), Index(2, 2)) == TileMaskStatus.NO_MASK)
    assert_true(
        mask.status(Index(1, 5), Index(2, 2)) == TileMaskStatus.FULL_MASK
    )


def main():
    test_causal_mask()
