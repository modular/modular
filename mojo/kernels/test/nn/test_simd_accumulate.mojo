# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from sys.info import has_neon, simdwidthof

from algorithm.functional import vectorize
from buffer import NDBuffer
from linalg.accumulate import _Accumulator, _simd_load_maybe_partial
from collections import InlineArray
from testing import *


# TODO: rewrite c-layout comments according to the new struct.
def test_maybe_partial_load():
    alias simd_size = 4
    alias size = simd_size + 1

    var a = InlineArray[Float32, size](unsafe_uninitialized=True)
    for i in range(size):
        a[i] = 1.0

    var vec = _simd_load_maybe_partial[simd_size, False](a.unsafe_ptr(), 0)
    assert_equal(vec, SIMD[DType.float32, simd_size](1.0))

    vec = _simd_load_maybe_partial[simd_size, True](
        a.unsafe_ptr(), simd_size, 1
    )
    assert_equal(vec, SIMD[DType.float32, simd_size](1.0, 0.0, 0.0, 0.0))


def test_accumulate[
    simd_size: Int = 4, num_rows: Int = 2, num_cols: Int = 2, length: Int = 2
]():
    alias type = DType.float32

    # A: [[ 0.0, 0.0 ],
    #     [ 1.0, 1.0 ],
    #     [ 2.0, 2.0 ],
    #     [ 3.0, 3.0 ]]
    var a = InlineArray[Scalar[type], 2 * num_rows * length](
        unsafe_uninitialized=True
    )
    for i in range(2 * num_rows):
        var a_ptr = a.unsafe_ptr() + i * length
        a_ptr[0] = Scalar[type](i)
        a_ptr[1] = Scalar[type](i)

    # 4 x 0.0 denotes 0.0, 0.0, 0.0, 0.0
    # B: [[4 x 0.0, 4 x 0.0, 4 x 1.0, 4 x 1.0],
    #     [4 x 2.0, 4 x 2.0, 4 x 3.0, 4 x 3.0]]
    alias b_size = 2 * num_cols * simd_size * length
    alias kernel_width = num_cols * simd_size
    var b = InlineArray[Scalar[type], b_size](unsafe_uninitialized=True)

    for i in range(2 * length):
        var b_ptr = b.unsafe_ptr() + i * num_cols * simd_size

        @parameter
        for j in range(num_cols):
            (b_ptr + j * simd_size).store(SIMD[type, simd_size](i))

    var acc = _Accumulator[type, num_rows, num_cols, simd_size]()
    acc.init(0)
    acc.accumulate(length, a.unsafe_ptr(), length, b.unsafe_ptr(), kernel_width)

    # C results:
    # C[0,0]:[0.0, 0.0, 0.0, 0.0]  C[0,1]:[0.0, 0.0, 0.0, 0.0]
    # C[1,0]:[1.0, 1.0, 1.0, 1.0]  C[1,1]:[1.0, 1.0, 1.0, 1.0]
    assert_equal(acc[0, 0], SIMD[type, simd_size](0.0))
    assert_equal(acc[0, 1], SIMD[type, simd_size](0.0))
    assert_equal(
        acc[1, 0],
        SIMD[type, simd_size](1.0),
    )
    assert_equal(
        acc[1, 1],
        SIMD[type, simd_size](1.0),
    )

    acc.accumulate(
        length,
        a.unsafe_ptr(),
        2 * length,
        b.unsafe_ptr() + kernel_width,
        kernel_width,
    )

    # C results:
    # C[0,0]:[0.0, 0.0, 0.0, 0.0]  C[0,1]:[0.0, 0.0, 0.0, 0.0]
    # C[1,0]:[7.0, 7.0, 7.0, 7.0]  C[1,1]:[7.0, 7.0, 7.0, 7.0]
    assert_equal(acc[0, 0], SIMD[type, simd_size](0.0))
    assert_equal(acc[0, 1], SIMD[type, simd_size](0.0))
    assert_equal(
        acc[1, 0],
        SIMD[type, simd_size](7.0),
    )
    assert_equal(
        acc[1, 1],
        SIMD[type, simd_size](7.0),
    )

    acc.accumulate(
        length,
        a.unsafe_ptr() + length,
        2 * length,
        b.unsafe_ptr() + kernel_width,
        2 * kernel_width,
    )

    # C results:
    # C[0,0]:[4.0, 4.0, 4.0, 4.0]     C[0,1]:[4.0, 4.0, 4.0, 4.0]
    # C[1,0]:[19.0, 19.0, 19.0, 19.0] C[1,1]:[19.0, 19.0, 19.0, 19.0]
    assert_equal(acc[0, 0], SIMD[type, simd_size](4.0))
    assert_equal(acc[0, 1], SIMD[type, simd_size](4.0))
    assert_equal(
        acc[1, 0],
        SIMD[type, simd_size](19.0),
    )
    assert_equal(
        acc[1, 1],
        SIMD[type, simd_size](19.0),
    )


def test_accumulate_with_offsets[
    simd_size: Int = 4, num_rows: Int = 2, num_cols: Int = 2, length: Int = 2
]():
    alias type = DType.float32

    # A: [[ 0.0, 0.0 ],
    #     [ 1.0, 1.0 ],
    #     [ 2.0, 2.0 ],
    #     [ 3.0, 3.0 ]]
    var a = InlineArray[Scalar[type], 2 * num_rows * length](
        unsafe_uninitialized=True
    )
    for i in range(2 * num_rows):
        var a_ptr = a.unsafe_ptr() + i * length
        a_ptr[0] = Scalar[type](i)
        a_ptr[1] = Scalar[type](i)

    # 4 x 0.0 denotes 0.0, 0.0, 0.0, 0.0
    # B: [[4 x 0.0, 4 x 0.0, 4 x 1.0, 4 x 1.0],
    #     [4 x 2.0, 4 x 2.0, 4 x 3.0, 4 x 3.0]]
    alias b_size = 2 * num_cols * simd_size * length
    alias kernel_width = num_cols * simd_size
    var b = InlineArray[Scalar[type], b_size](unsafe_uninitialized=True)

    for i in range(2 * length):
        var b_ptr = b.unsafe_ptr() + i * num_cols * simd_size

        @parameter
        for j in range(num_cols):
            (b_ptr + j * simd_size).store(SIMD[type, simd_size](i))

    var a_base_stack = InlineArray[Int32, num_rows](unsafe_uninitialized=True)
    var a_base_offsets = NDBuffer[DType.int32, 1, num_rows](
        a_base_stack.unsafe_ptr()
    )
    a_base_offsets[0] = 0
    a_base_offsets[1] = length

    var acc = _Accumulator[type, num_rows, num_cols, simd_size]()
    acc.init(0)
    acc.accumulate(
        length, a.unsafe_ptr(), a_base_offsets, 0, b.unsafe_ptr(), kernel_width
    )

    # C results:
    # [0.0, 0.0, 0.0, 0.0]
    # [0.0, 0.0, 0.0, 0.0]
    # [1.0, 1.0, 1.0, 1.0]
    # [1.0, 1.0, 1.0, 1.0]
    assert_equal(acc[0, 0], SIMD[type, simd_size](0.0))
    assert_equal(acc[0, 1], SIMD[type, simd_size](0.0))
    assert_equal(
        acc[1, 0],
        SIMD[type, simd_size](1.0),
    )
    assert_equal(
        acc[1, 1],
        SIMD[type, simd_size](1.0),
    )

    a_base_offsets[0] = 0
    a_base_offsets[1] = 2 * length
    acc.accumulate(
        length,
        a.unsafe_ptr(),
        a_base_offsets,
        0,
        b.unsafe_ptr() + kernel_width,
        kernel_width,
    )

    # C results:
    # [0.0, 0.0, 0.0, 0.0]
    # [0.0, 0.0, 0.0, 0.0]
    # [7.0, 7.0, 7.0, 7.0]
    # [7.0, 7.0, 7.0, 7.0]
    assert_equal(acc[0, 0], SIMD[type, simd_size](0.0))
    assert_equal(acc[0, 1], SIMD[type, simd_size](0.0))
    assert_equal(
        acc[1, 0],
        SIMD[type, simd_size](7.0),
    )
    assert_equal(
        acc[1, 1],
        SIMD[type, simd_size](7.0),
    )

    a_base_offsets[0] = length
    a_base_offsets[1] = 3 * length

    acc.accumulate(
        length,
        a.unsafe_ptr(),
        a_base_offsets,
        0,
        b.unsafe_ptr() + kernel_width,
        2 * kernel_width,
    )

    # C results:
    # [4.0, 4.0, 4.0, 4.0]
    # [4.0, 4.0, 4.0, 4.0]
    # [19.0, 19.0, 19.0, 19.0]
    # [19.0, 19.0, 19.0, 19.0]
    assert_equal(acc[0, 0], SIMD[type, simd_size](4.0))
    assert_equal(acc[0, 1], SIMD[type, simd_size](4.0))
    assert_equal(
        acc[1, 0],
        SIMD[type, simd_size](19.0),
    )
    assert_equal(
        acc[1, 1],
        SIMD[type, simd_size](19.0),
    )


def test_load_store[
    simd_size: Int = 4, num_rows: Int = 2, num_cols: Int = 2, length: Int = 2
]():
    alias type = DType.float32
    alias size = simd_size + 1
    alias residual = 1
    alias row_size = num_cols * simd_size + residual
    alias one_vec = SIMD[type, simd_size](1.0)
    alias residual_vec = SIMD[type, simd_size](-1.0, 0.0, 0.0, 0.0)

    var a = InlineArray[Scalar[type], num_rows * row_size](
        unsafe_uninitialized=True
    )

    # A: [[ 4x0.0, 4x1.0, -1.0],
    #     [ 4x1.0, 4x2.0, -1.0]]
    @parameter
    for i in range(num_rows):

        @parameter
        for j in range(num_cols):
            a.unsafe_ptr().store(
                i * row_size + j * simd_size,
                SIMD[type, simd_size](i + j),
            )

        a.unsafe_ptr().store(
            i * row_size + num_cols * simd_size,
            SIMD[type, residual](-1.0),
        )

    var tile0 = _Accumulator[type, num_rows, num_cols, simd_size]()
    tile0.load(a.unsafe_ptr(), row_size)

    assert_equal(
        tile0[0, 0],
        SIMD[type, simd_size](0.0),
    )
    assert_equal(
        tile0[0, 1],
        SIMD[type, simd_size](1.0),
    )
    assert_equal(
        tile0[1, 0],
        SIMD[type, simd_size](1.0),
    )
    assert_equal(
        tile0[1, 1],
        SIMD[type, simd_size](2.0),
    )

    # Update A: [[ 4x1.0, 4x1.0, -1.0],
    #            [ 4x1.0, 4x1.0, -1.0]]
    tile0[0, 0] = one_vec
    tile0[1, 1] = one_vec
    tile0.store(a.unsafe_ptr(), row_size)

    var tile1 = _Accumulator[type, num_rows, num_cols + 1, simd_size]()

    tile1.load[partial_load=True](a.unsafe_ptr(), row_size, residual)

    assert_equal(tile1[0, 0], one_vec)
    assert_equal(tile1[0, 1], one_vec)
    assert_equal(tile1[0, 2], residual_vec)
    assert_equal(tile1[1, 0], one_vec)
    assert_equal(tile1[1, 1], one_vec)
    assert_equal(tile1[1, 2], residual_vec)

    var residual_vec1 = SIMD[type, residual](-2.0)

    # TODO: replace the following with simd.mojo:insert (after resolving its issue).
    @always_inline
    fn simd_insert(mut x: SIMD[type, _], y: SIMD[type, _]):
        constrained[x.size >= y.size]()

        @parameter
        for i in range(y.size):
            x[i] = y[i]

    simd_insert(tile1[0, 2], residual_vec1)
    simd_insert(tile1[1, 2], residual_vec1)

    # Update A: [[ 4x1.0, 4x1.0, -2.0],
    #            [ 4x1.0, 4x1.0, -2.0]]
    tile1.store[partial_store=True](a.unsafe_ptr(), row_size, residual)

    assert_equal(
        a.unsafe_ptr().load[width=residual](row_size - residual), residual_vec1
    )
    assert_equal(
        a.unsafe_ptr().load[width=residual](2 * row_size - residual),
        residual_vec1,
    )


def main():
    test_maybe_partial_load()
    test_accumulate()
    test_accumulate_with_offsets()
    test_load_store()
