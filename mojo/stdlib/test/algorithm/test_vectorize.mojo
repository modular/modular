# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo %s | FileCheck %s

from Buffer import Buffer
from DType import DType
from Functional import vectorize, vectorize_unroll
from IO import print
from Memory import memcmp
from Range import range


# CHECK-LABEL: test_vectorize
fn test_vectorize():
    print("== test_vectorize")

    # Create a mem of size 5
    let vector = Buffer[5, DType.f32].stack_allocation()

    vector[0] = 1.0
    vector[1] = 2.0
    vector[2] = 3.0
    vector[3] = 4.0
    vector[4] = 5.0

    @always_inline
    fn add_two[simd_width: Int](idx: Int):
        vector.simd_store[simd_width](
            idx, vector.simd_load[simd_width](idx) + 2
        )

    vectorize[2, add_two](vector.__len__())

    # CHECK: 3.00
    print(vector[0])
    # CHECK: 4.00
    print(vector[1])
    # CHECK: 5.00
    print(vector[2])
    # CHECK: 6.00
    print(vector[3])
    # CHECK: 7.00
    print(vector[4])

    @always_inline
    fn add[simd_width: Int](idx: Int):
        vector.simd_store[simd_width](
            idx,
            vector.simd_load[simd_width](idx)
            + vector.simd_load[simd_width](idx),
        )

    vectorize[2, add](vector.__len__())

    # CHECK: 6.00
    print(vector[0])
    # CHECK: 8.00
    print(vector[1])
    # CHECK: 10.00
    print(vector[2])
    # CHECK: 12.00
    print(vector[3])
    # CHECK: 14.00
    print(vector[4])


# CHECK-LABEL: test_vectorize_unroll
fn test_vectorize_unroll():
    print("== test_vectorize_unroll")

    alias buf_len = 23
    let vec = Buffer[buf_len, DType.f32].stack_allocation()
    let ref = Buffer[buf_len, DType.f32].stack_allocation()

    for i in range(buf_len):
        vec[i] = i
        ref[i] = i

    @always_inline
    fn double_ref[simd_width: Int](idx: Int):
        ref.simd_store[simd_width](
            idx,
            ref.simd_load[simd_width](idx) + ref.simd_load[simd_width](idx),
        )

    @always_inline
    fn double_vec[simd_width: Int](idx: Int):
        vec.simd_store[simd_width](
            idx,
            vec.simd_load[simd_width](idx) + vec.simd_load[simd_width](idx),
        )

    alias simd_width = 4
    alias unroll_factor = 2

    vectorize_unroll[simd_width, unroll_factor, double_vec](vec.__len__())
    vectorize[simd_width, double_ref](ref.__len__())

    let err = memcmp(vec.data, ref.data, ref.bytecount())
    # CHECK: 0
    print(err)


fn main():
    test_vectorize()
    test_vectorize_unroll()
