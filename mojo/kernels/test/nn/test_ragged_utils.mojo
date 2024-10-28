# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from nn._ragged_utils import get_batch_from_row_offsets
from internal_utils import HostNDBuffer
from testing import assert_equal, assert_true


def test_get_batch_from_row_offsets():
    batch_size = 9
    prefix_sums = HostNDBuffer[DType.uint32, 1]((batch_size + 1,))
    prefix_sums.tensor[0] = 0
    prefix_sums.tensor[1] = 100
    prefix_sums.tensor[2] = 200
    prefix_sums.tensor[3] = 300
    prefix_sums.tensor[4] = 400
    prefix_sums.tensor[5] = 500
    prefix_sums.tensor[6] = 600
    prefix_sums.tensor[7] = 700
    prefix_sums.tensor[8] = 800
    prefix_sums.tensor[9] = 900

    assert_equal(
        get_batch_from_row_offsets(prefix_sums.tensor, 100),
        1,
    )
    assert_equal(
        get_batch_from_row_offsets(prefix_sums.tensor, 0),
        0,
    )
    assert_equal(
        get_batch_from_row_offsets(prefix_sums.tensor, 899),
        8,
    )
    assert_equal(
        get_batch_from_row_offsets(prefix_sums.tensor, 555),
        5,
    )

    _ = prefix_sums^


def main():
    test_get_batch_from_row_offsets()
