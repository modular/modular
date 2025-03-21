# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from buffer import NDBuffer
from buffer.dimlist import DimList
from linalg.matrix_band_part import matrix_band_part as _matrix_band_part
from runtime.asyncrt import DeviceContextPtr
from testing import assert_equal

from utils import IndexList


def matrix_band_part[
    rank: Int,
    type: DType,
](
    input: NDBuffer[type, rank],
    output: NDBuffer[mut=True, type, rank],
    num_lower: Int,
    num_upper: Int,
    exclude: Bool,
):
    alias int_type = DType.index
    alias cond_type = DType.bool

    var num_lower_buf = NDBuffer[
        int_type, 1, MutableAnyOrigin, DimList(1)
    ].stack_allocation()
    var num_upper_buf = NDBuffer[
        int_type, 1, MutableAnyOrigin, DimList(1)
    ].stack_allocation()
    var exclude_buf = NDBuffer[
        cond_type, 1, MutableAnyOrigin, DimList(1)
    ].stack_allocation()

    num_lower_buf[0] = num_lower
    num_upper_buf[0] = num_upper
    exclude_buf[0] = exclude

    @parameter
    fn input_fn[
        width: Int,
        _rank: Int,
    ](coords: IndexList[_rank]) -> SIMD[type, width]:
        return input.load[width=width](rebind[IndexList[rank]](coords))

    _matrix_band_part[
        type,
        int_type,
        cond_type,
        rank,
        input_fn,
        simd_width=1,
        single_thread_blocking_override=True,
    ](
        input.get_shape(),
        num_lower_buf.make_dims_unknown(),
        num_upper_buf.make_dims_unknown(),
        exclude_buf.make_dims_unknown(),
        output,
        DeviceContextPtr(),
    )


def test_matrix_band_part():
    alias rank = 2
    alias shape = DimList(3, 3)
    alias type = DType.float32

    var input = NDBuffer[type, rank, MutableAnyOrigin, shape].stack_allocation()
    var output = NDBuffer[
        type, rank, MutableAnyOrigin, shape
    ].stack_allocation()

    input[0, 0] = 1
    input[0, 1] = 2
    input[0, 2] = 3
    input[1, 0] = 4
    input[1, 1] = 5
    input[1, 2] = 6
    input[2, 0] = 7
    input[2, 1] = 8
    input[2, 2] = 9

    matrix_band_part(
        input.make_dims_unknown(),
        output.make_dims_unknown(),
        num_lower=0,
        num_upper=-1,
        exclude=False,
    )

    assert_equal(output[0, 0], 1)
    assert_equal(output[0, 1], 2)
    assert_equal(output[0, 2], 3)
    assert_equal(output[1, 0], 0)
    assert_equal(output[1, 1], 5)
    assert_equal(output[1, 2], 6)
    assert_equal(output[2, 0], 0)
    assert_equal(output[2, 1], 0)
    assert_equal(output[2, 2], 9)

    matrix_band_part(
        input.make_dims_unknown(),
        output.make_dims_unknown(),
        num_lower=0,
        num_upper=-1,
        exclude=True,
    )

    assert_equal(output[0, 0], 0)
    assert_equal(output[0, 1], 0)
    assert_equal(output[0, 2], 0)
    assert_equal(output[1, 0], 4)
    assert_equal(output[1, 1], 0)
    assert_equal(output[1, 2], 0)
    assert_equal(output[2, 0], 7)
    assert_equal(output[2, 1], 8)
    assert_equal(output[2, 2], 0)


def main():
    test_matrix_band_part()
