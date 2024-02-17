# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s

from sys.info import has_neon, simdwidthof

from algorithm.functional import vectorize
from memory import stack_allocation
from NN.AccumulateSIMD import (
    _simd_load_maybe_partial,
    accumulate,
    load_register_tile,
    store_register_tile,
)
from testing import *


def test_maybe_partial_load():
    alias simd_size = 4
    alias size = simd_size + 1

    let a = stack_allocation[size, DType.float32]()

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
    let a = stack_allocation[2 * num_rows * length, type]()
    for i in range(2 * num_rows):
        let a_ptr = a + i * length
        a_ptr[0] = SIMD[type, 1](i)
        a_ptr[1] = SIMD[type, 1](i)

    # 4 x 0.0 denotes 0.0, 0.0, 0.0, 0.0
    # B: [[4 x 0.0, 4 x 0.0, 4 x 1.0, 4 x 1.0],
    #     [4 x 2.0, 4 x 2.0, 4 x 3.0, 4 x 3.0]]
    alias b_size = 2 * num_cols * simd_size * length
    alias kernel_width = num_cols * simd_size
    let b = stack_allocation[b_size, type]()

    for i in range(2 * length):
        let b_ptr = b + i * num_cols * simd_size

        @unroll
        for j in range(num_cols):
            (b_ptr + j * simd_size).simd_store(SIMD[type, simd_size](i))

    alias c_size = num_rows * num_cols * simd_size
    let c = stack_allocation[c_size, type]()

    @__copy_capture(c)
    @parameter
    fn fill_c[widthj: Int](offset: Int):
        (c + offset).simd_store(SIMD[type, simd_size](0.0))

    vectorize[fill_c, simd_size, c_size]()

    accumulate[num_rows, num_cols, simd_size](
        length, c, a, length, b, kernel_width
    )

    # C results:
    # [0.0, 0.0, 0.0, 0.0]
    # [0.0, 0.0, 0.0, 0.0]
    # [1.0, 1.0, 1.0, 1.0]
    # [1.0, 1.0, 1.0, 1.0]
    assert_equal(c.simd_load[simd_size](), SIMD[DType.float32, simd_size](0.0))
    assert_equal(
        c.simd_load[simd_size](simd_size), SIMD[DType.float32, simd_size](0.0)
    )
    assert_equal(
        c.simd_load[simd_size](2 * simd_size),
        SIMD[DType.float32, simd_size](1.0),
    )
    assert_equal(
        c.simd_load[simd_size](3 * simd_size),
        SIMD[DType.float32, simd_size](1.0),
    )

    accumulate[num_rows, num_cols, simd_size](
        length, c, a, 2 * length, b + kernel_width, kernel_width
    )

    # C results:
    # [0.0, 0.0, 0.0, 0.0]
    # [0.0, 0.0, 0.0, 0.0]
    # [7.0, 7.0, 7.0, 7.0]
    # [7.0, 7.0, 7.0, 7.0]
    assert_equal(c.simd_load[simd_size](), SIMD[DType.float32, simd_size](0.0))
    assert_equal(
        c.simd_load[simd_size](simd_size), SIMD[DType.float32, simd_size](0.0)
    )
    assert_equal(
        c.simd_load[simd_size](2 * simd_size),
        SIMD[DType.float32, simd_size](7.0),
    )
    assert_equal(
        c.simd_load[simd_size](3 * simd_size),
        SIMD[DType.float32, simd_size](7.0),
    )

    accumulate[num_rows, num_cols, simd_size](
        length,
        c,
        a + length,
        2 * length,
        b + kernel_width,
        2 * kernel_width,
    )

    # C results:
    # [4.0, 4.0, 4.0, 4.0]
    # [4.0, 4.0, 4.0, 4.0]
    # [19.0, 19.0, 19.0, 19.0]
    # [19.0, 19.0, 19.0, 19.0]
    assert_equal(c.simd_load[simd_size](), SIMD[DType.float32, simd_size](4.0))
    assert_equal(
        c.simd_load[simd_size](simd_size), SIMD[DType.float32, simd_size](4.0)
    )
    assert_equal(
        c.simd_load[simd_size](2 * simd_size),
        SIMD[DType.float32, simd_size](19.0),
    )
    assert_equal(
        c.simd_load[simd_size](3 * simd_size),
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

    let a = stack_allocation[num_rows * row_size, type]()

    # A: [[ 4x0.0, 4x1.0, -1.0],
    #     [ 4x1.0, 4x2.0, -1.0],]
    @unroll
    for i in range(num_rows):

        @unroll
        for j in range(num_cols):
            a.simd_store(
                i * row_size + j * simd_size,
                SIMD[type, simd_size](i + j),
            )

        a.simd_store(
            i * row_size + num_cols * simd_size,
            SIMD[type, residual](-1.0),
        )

    let tile0 = stack_allocation[num_rows * num_cols * simd_size, type]()

    load_register_tile[num_rows, num_cols, simd_size](tile0, a, row_size)

    assert_equal(
        tile0.simd_load[simd_size](),
        SIMD[type, simd_size](0.0),
    )
    assert_equal(
        tile0.simd_load[simd_size](simd_size),
        SIMD[type, simd_size](1.0),
    )
    assert_equal(
        tile0.simd_load[simd_size](2 * simd_size),
        SIMD[type, simd_size](1.0),
    )
    assert_equal(
        tile0.simd_load[simd_size](3 * simd_size),
        SIMD[type, simd_size](2.0),
    )

    # Update A: [[ 4x1.0, 4x1.0, -1.0],
    #            [ 4x1.0, 4x1.0, -1.0],]
    tile0.simd_store(one_vec)
    tile0.simd_store(3 * simd_size, one_vec)
    store_register_tile[num_rows, num_cols, simd_size](a, row_size, tile0)

    let tile1 = stack_allocation[num_rows * (num_cols + 1) * simd_size, type]()

    load_register_tile[num_rows, num_cols + 1, simd_size, partial_load=True](
        tile1, a, row_size, residual
    )

    assert_equal(tile1.simd_load[simd_size](), one_vec)
    assert_equal(tile1.simd_load[simd_size](simd_size), one_vec)
    assert_equal(tile1.simd_load[simd_size](2 * simd_size), residual_vec)
    assert_equal(tile1.simd_load[simd_size](3 * simd_size), one_vec)
    assert_equal(tile1.simd_load[simd_size](4 * simd_size), one_vec)
    assert_equal(tile1.simd_load[simd_size](5 * simd_size), residual_vec)

    alias residual_vec1 = SIMD[type, residual](-2.0)

    tile1.simd_store(num_cols * simd_size, residual_vec1)
    tile1.simd_store((2 * num_cols + 1) * simd_size, residual_vec1)

    # Update A: [[ 4x1.0, 4x1.0, -2.0],
    #            [ 4x1.0, 4x1.0, -2.0],]
    store_register_tile[num_rows, num_cols + 1, simd_size, partial_store=True](
        a, row_size, tile1, residual
    )

    assert_equal(a.simd_load[residual](row_size - residual), residual_vec1)
    assert_equal(a.simd_load[residual](2 * row_size - residual), residual_vec1)


def main():
    test_maybe_partial_load()
    test_accumulate()
    test_load_store_register_tile()
