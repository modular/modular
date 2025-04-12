# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s
from nn.mha_mask import MASK_VALUE, ChunkedMask, TileMaskStatus
from testing import assert_equal

from utils.index import Index, IndexList
from utils.numerics import min_or_neg_inf


def test_chunked_mask_status():
    var mask = ChunkedMask[local_window_size=4]()

    assert_equal(mask.status(Index(0, 0), Index(4, 4)), TileMaskStatus.NO_MASK)
    assert_equal(mask.status(Index(4, 4), Index(4, 4)), TileMaskStatus.NO_MASK)
    assert_equal(
        mask.status(Index(2, 2), Index(4, 4)), TileMaskStatus.PARTIAL_MASK
    )
    assert_equal(
        mask.status(Index(0, 2), Index(4, 4)), TileMaskStatus.PARTIAL_MASK
    )
    assert_equal(
        mask.status(Index(2, 0), Index(4, 4)), TileMaskStatus.PARTIAL_MASK
    )
    assert_equal(
        mask.status(Index(0, 4), Index(4, 4)), TileMaskStatus.FULL_MASK
    )
    assert_equal(
        mask.status(Index(4, 0), Index(4, 4)), TileMaskStatus.FULL_MASK
    )

    # cases where tile_size >> local_window_size
    assert_equal(
        mask.status(Index(100, 0), Index(128, 128)), TileMaskStatus.PARTIAL_MASK
    )
    assert_equal(
        mask.status(Index(0, 0), Index(100, 100)), TileMaskStatus.PARTIAL_MASK
    )
    assert_equal(
        mask.status(Index(50, 0), Index(100, 100)), TileMaskStatus.PARTIAL_MASK
    )

    var bigger_mask = ChunkedMask[local_window_size=256]()
    assert_equal(
        bigger_mask.status(Index(256, 256), Index(64, 128)),
        TileMaskStatus.NO_MASK,
    )
    assert_equal(
        bigger_mask.status(Index(128, 0), Index(128, 128)),
        TileMaskStatus.NO_MASK,
    )


def test_chunked_mask_apply():
    var mask = ChunkedMask[local_window_size=4]()

    var score_vec = SIMD[DType.float32, 4](0.0)
    score_vec[0] = 1.0
    score_vec[1] = 2.0
    score_vec[2] = 3.0
    score_vec[3] = 4.0

    alias SIMD_T = SIMD[DType.float32, 4]
    var inf_vec = SIMD_T(MASK_VALUE)

    # first two dims should be arbitrary, we pass in junk just to help confirm.
    assert_equal(mask.mask(Index(0, 0, 0, 0), score_vec), score_vec)
    assert_equal(mask.mask(Index(10, 0, 4, 8), score_vec), inf_vec)
    assert_equal(mask.mask(Index(2, 10, 8, 8), score_vec), score_vec)

    assert_equal(
        mask.mask(Index(0, 4, 8, 10), score_vec),
        SIMD_T(1.0, 2.0, MASK_VALUE, MASK_VALUE),
    )

    assert_equal(
        mask.mask(Index(4, 0, 12, 10), score_vec),
        SIMD_T(MASK_VALUE, MASK_VALUE, 3.0, 4.0),
    )


def main():
    test_chunked_mask_status()
    test_chunked_mask_apply()
