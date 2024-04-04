# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s

from sys.info import has_neon, simdwidthof

from algorithm.functional import vectorize
from buffer import Buffer
from memory import stack_allocation
from nn.accumulate import (
    _simd_load_maybe_partial,
    load_register_tile,
    store_register_tile,
    _Accumulator,
)
from testing import *


# TODO: rewrite c-layout comments according to the new struct.
def test_maybe_partial_load():
    alias simd_size = 4
    alias size = simd_size + 1

    var a = stack_allocation[size, DType.float32]()

    for i in range(size):
        a[i] = 1.0

    var vec = _simd_load_maybe_partial[simd_size, False](a, 0)
    assert_equal(vec, SIMD[DType.float32, simd_size](1.0))

    vec = _simd_load_maybe_partial[simd_size, True](a, simd_size, 1)
    assert_equal(vec, SIMD[DType.float32, simd_size](1.0, 0.0, 0.0, 0.0))


def test_accumulate[
    simd_size: Int = 4, num_rows: Int = 2, num_cols: Int = 2, length: Int = 2
]():
    alias type = DType.float32

    # A: [[ 0.0, 0.0 ],
    #     [ 1.0, 1.0 ],
    #     [ 2.0, 2.0 ],
    #     [ 3.0, 3.0 ]]
    var a = stack_allocation[2 * num_rows * length, type]()
    for i in range(2 * num_rows):
        var a_ptr = a + i * length
        a_ptr[0] = Scalar[type](i)
        a_ptr[1] = Scalar[type](i)

    # 4 x 0.0 denotes 0.0, 0.0, 0.0, 0.0
    # B: [[4 x 0.0, 4 x 0.0, 4 x 1.0, 4 x 1.0],
    #     [4 x 2.0, 4 x 2.0, 4 x 3.0, 4 x 3.0]]
    alias b_size = 2 * num_cols * simd_size * length
    alias kernel_width = num_cols * simd_size
    var b = stack_allocation[b_size, type]()

    for i in range(2 * length):
        var b_ptr = b + i * num_cols * simd_size

        @unroll
        for j in range(num_cols):
            (b_ptr + j * simd_size).store(SIMD[type, simd_size](i))

    var acc = _Accumulator[type, num_rows, num_cols, simd_size]()
    acc.accumulate(length, a, length, b, kernel_width)

    # C results:
    # C[0,0]:[0.0, 0.0, 0.0, 0.0]  C[0,1]:[0.0, 0.0, 0.0, 0.0]
    # C[1,0]:[1.0, 1.0, 1.0, 1.0]  C[1,1]:[1.0, 1.0, 1.0, 1.0]
    assert_equal(acc[0, 0], SIMD[DType.float32, simd_size](0.0))
    assert_equal(acc[0, 1], SIMD[DType.float32, simd_size](0.0))
    assert_equal(
        acc[1, 0],
        SIMD[DType.float32, simd_size](1.0),
    )
    assert_equal(
        acc[1, 1],
        SIMD[DType.float32, simd_size](1.0),
    )

    acc.accumulate(length, a, 2 * length, b + kernel_width, kernel_width)

    # C results:
    # C[0,0]:[0.0, 0.0, 0.0, 0.0]  C[0,1]:[0.0, 0.0, 0.0, 0.0]
    # C[1,0]:[7.0, 7.0, 7.0, 7.0]  C[1,1]:[7.0, 7.0, 7.0, 7.0]
    assert_equal(acc[0, 0], SIMD[DType.float32, simd_size](0.0))
    assert_equal(acc[0, 1], SIMD[DType.float32, simd_size](0.0))
    assert_equal(
        acc[1, 0],
        SIMD[DType.float32, simd_size](7.0),
    )
    assert_equal(
        acc[1, 1],
        SIMD[DType.float32, simd_size](7.0),
    )

    acc.accumulate(
        length,
        a + length,
        2 * length,
        b + kernel_width,
        2 * kernel_width,
    )

    # C results:
    # C[0,0]:[4.0, 4.0, 4.0, 4.0]     C[0,1]:[4.0, 4.0, 4.0, 4.0]
    # C[1,0]:[19.0, 19.0, 19.0, 19.0] C[1,1]:[19.0, 19.0, 19.0, 19.0]
    assert_equal(acc[0, 0], SIMD[DType.float32, simd_size](4.0))
    assert_equal(acc[0, 1], SIMD[DType.float32, simd_size](4.0))
    assert_equal(
        acc[1, 0],
        SIMD[DType.float32, simd_size](19.0),
    )
    assert_equal(
        acc[1, 1],
        SIMD[DType.float32, simd_size](19.0),
    )


def test_accumulate_with_offsets[
    simd_size: Int = 4, num_rows: Int = 2, num_cols: Int = 2, length: Int = 2
]():
    alias type = DType.float32

    # A: [[ 0.0, 0.0 ],
    #     [ 1.0, 1.0 ],
    #     [ 2.0, 2.0 ],
    #     [ 3.0, 3.0 ]]
    var a = stack_allocation[2 * num_rows * length, type]()
    for i in range(2 * num_rows):
        var a_ptr = a + i * length
        a_ptr[0] = Scalar[type](i)
        a_ptr[1] = Scalar[type](i)

    # 4 x 0.0 denotes 0.0, 0.0, 0.0, 0.0
    # B: [[4 x 0.0, 4 x 0.0, 4 x 1.0, 4 x 1.0],
    #     [4 x 2.0, 4 x 2.0, 4 x 3.0, 4 x 3.0]]
    alias b_size = 2 * num_cols * simd_size * length
    alias kernel_width = num_cols * simd_size
    var b = stack_allocation[b_size, type]()

    for i in range(2 * length):
        var b_ptr = b + i * num_cols * simd_size

        @unroll
        for j in range(num_cols):
            (b_ptr + j * simd_size).store(SIMD[type, simd_size](i))

    # alias c_size = num_rows * num_cols * simd_size
    # var c = stack_allocation[c_size, type]()

    # @__copy_capture(c)
    # @parameter
    # fn fill_c[widthj: Int](offset: Int):
    #     (c + offset).store(SIMD[type, simd_size](0.0))

    # vectorize[fill_c, simd_size, size=c_size]()

    var a_base_offsets = Buffer[DType.int32, num_rows].stack_allocation()
    a_base_offsets[0] = 0
    a_base_offsets[1] = length

    var acc = _Accumulator[type, num_rows, num_cols, simd_size]()
    acc.accumulate(length, a, a_base_offsets, 0, b, kernel_width)

    # C results:
    # [0.0, 0.0, 0.0, 0.0]
    # [0.0, 0.0, 0.0, 0.0]
    # [1.0, 1.0, 1.0, 1.0]
    # [1.0, 1.0, 1.0, 1.0]
    assert_equal(acc[0, 0], SIMD[DType.float32, simd_size](0.0))
    assert_equal(acc[0, 1], SIMD[DType.float32, simd_size](0.0))
    assert_equal(
        acc[1, 0],
        SIMD[DType.float32, simd_size](1.0),
    )
    assert_equal(
        acc[1, 1],
        SIMD[DType.float32, simd_size](1.0),
    )

    a_base_offsets[0] = 0
    a_base_offsets[1] = 2 * length
    acc.accumulate(length, a, a_base_offsets, 0, b + kernel_width, kernel_width)

    # C results:
    # [0.0, 0.0, 0.0, 0.0]
    # [0.0, 0.0, 0.0, 0.0]
    # [7.0, 7.0, 7.0, 7.0]
    # [7.0, 7.0, 7.0, 7.0]
    assert_equal(acc[0, 0], SIMD[DType.float32, simd_size](0.0))
    assert_equal(acc[0, 1], SIMD[DType.float32, simd_size](0.0))
    assert_equal(
        acc[1, 0],
        SIMD[DType.float32, simd_size](7.0),
    )
    assert_equal(
        acc[1, 1],
        SIMD[DType.float32, simd_size](7.0),
    )

    a_base_offsets[0] = length
    a_base_offsets[1] = 3 * length

    acc.accumulate(
        length,
        a,
        a_base_offsets,
        0,
        b + kernel_width,
        2 * kernel_width,
    )

    # C results:
    # [4.0, 4.0, 4.0, 4.0]
    # [4.0, 4.0, 4.0, 4.0]
    # [19.0, 19.0, 19.0, 19.0]
    # [19.0, 19.0, 19.0, 19.0]
    assert_equal(acc[0, 0], SIMD[DType.float32, simd_size](4.0))
    assert_equal(acc[0, 1], SIMD[DType.float32, simd_size](4.0))
    assert_equal(
        acc[1, 0],
        SIMD[DType.float32, simd_size](19.0),
    )
    assert_equal(
        acc[1, 1],
        SIMD[DType.float32, simd_size](19.0),
    )


def test_load_store_register_tile[
    simd_size: Int = 4, num_rows: Int = 2, num_cols: Int = 2, length: Int = 2
]():
    alias type = DType.float32
    alias size = simd_size + 1
    alias residual = 1
    alias row_size = num_cols * simd_size + residual
    alias one_vec = SIMD[type, simd_size](1.0)
    alias residual_vec = SIMD[type, simd_size](-1.0, 0.0, 0.0, 0.0)

    var a = stack_allocation[num_rows * row_size, type]()

    # A: [[ 4x0.0, 4x1.0, -1.0],
    #     [ 4x1.0, 4x2.0, -1.0],]
    @unroll
    for i in range(num_rows):

        @unroll
        for j in range(num_cols):
            a.store(
                i * row_size + j * simd_size,
                SIMD[type, simd_size](i + j),
            )

        a.store(
            i * row_size + num_cols * simd_size,
            SIMD[type, residual](-1.0),
        )

    var tile0 = stack_allocation[num_rows * num_cols * simd_size, type]()

    load_register_tile[num_rows, num_cols, simd_size](tile0, a, row_size)

    assert_equal(
        tile0.load[width=simd_size](),
        SIMD[type, simd_size](0.0),
    )
    assert_equal(
        tile0.load[width=simd_size](simd_size),
        SIMD[type, simd_size](1.0),
    )
    assert_equal(
        tile0.load[width=simd_size](2 * simd_size),
        SIMD[type, simd_size](1.0),
    )
    assert_equal(
        tile0.load[width=simd_size](3 * simd_size),
        SIMD[type, simd_size](2.0),
    )

    # Update A: [[ 4x1.0, 4x1.0, -1.0],
    #            [ 4x1.0, 4x1.0, -1.0],]
    tile0.store(one_vec)
    tile0.store(3 * simd_size, one_vec)
    store_register_tile[num_rows, num_cols, simd_size](a, row_size, tile0)

    var tile1 = stack_allocation[num_rows * (num_cols + 1) * simd_size, type]()

    load_register_tile[num_rows, num_cols + 1, simd_size, partial_load=True](
        tile1, a, row_size, residual
    )

    assert_equal(tile1.load[width=simd_size](), one_vec)
    assert_equal(tile1.load[width=simd_size](simd_size), one_vec)
    assert_equal(tile1.load[width=simd_size](2 * simd_size), residual_vec)
    assert_equal(tile1.load[width=simd_size](3 * simd_size), one_vec)
    assert_equal(tile1.load[width=simd_size](4 * simd_size), one_vec)
    assert_equal(tile1.load[width=simd_size](5 * simd_size), residual_vec)

    alias residual_vec1 = SIMD[type, residual](-2.0)

    tile1.store(num_cols * simd_size, residual_vec1)
    tile1.store((2 * num_cols + 1) * simd_size, residual_vec1)

    # Update A: [[ 4x1.0, 4x1.0, -2.0],
    #            [ 4x1.0, 4x1.0, -2.0],]
    store_register_tile[num_rows, num_cols + 1, simd_size, partial_store=True](
        a, row_size, tile1, residual
    )

    assert_equal(a.load[width=residual](row_size - residual), residual_vec1)
    assert_equal(a.load[width=residual](2 * row_size - residual), residual_vec1)


def main():
    test_maybe_partial_load()
    test_accumulate()
    test_accumulate_with_offsets()
    test_load_store_register_tile()
