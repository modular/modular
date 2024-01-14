# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s

from algorithm.functional import vectorize
from AccumulateSIMD import (
    _simd_load_maybe_partial,
    accumulate_x86_simd,
)
from sys.info import simdwidthof, has_avx2, has_avx512f
from testing import *
from memory import stack_allocation


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


def test_accumulate_avx2_avx512[
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

    @parameter
    fn fill_c[widthj: Int](offset: Int):
        (c + offset).simd_store(SIMD[type, simd_size](0.0))

    vectorize[simd_size, fill_c](c_size)

    accumulate_x86_simd[num_rows, num_cols, simd_size](
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

    accumulate_x86_simd[num_rows, num_cols, simd_size](
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

    accumulate_x86_simd[num_rows, num_cols, simd_size](
        length, c, a + length, 2 * length, b + kernel_width, 2 * kernel_width
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


def main():
    test_maybe_partial_load()

    @parameter
    if has_avx2() or has_avx512f():
        test_accumulate_avx2_avx512()
