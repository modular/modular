# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from memory.buffer import Buffer
from algorithm import vectorize, vectorize_unroll
from memory import memcmp


# CHECK-LABEL: test_vectorize
fn test_vectorize():
    print("== test_vectorize")

    # Create a mem of size 5
    let vector = Buffer[5, DType.float32].stack_allocation()

    vector[0] = 1.0
    vector[1] = 2.0
    vector[2] = 3.0
    vector[3] = 4.0
    vector[4] = 5.0

    @always_inline
    @parameter
    fn add_two[simd_width: Int](idx: Int):
        vector.simd_store[simd_width](
            idx, vector.simd_load[simd_width](idx) + 2
        )

    vectorize[2, add_two](vector.__len__())

    # CHECK: 3.0
    print(vector[0])
    # CHECK: 4.0
    print(vector[1])
    # CHECK: 5.0
    print(vector[2])
    # CHECK: 6.0
    print(vector[3])
    # CHECK: 7.0
    print(vector[4])

    @always_inline
    @parameter
    fn add[simd_width: Int](idx: Int):
        vector.simd_store[simd_width](
            idx,
            vector.simd_load[simd_width](idx)
            + vector.simd_load[simd_width](idx),
        )

    vectorize[2, add](vector.__len__())

    # CHECK: 6.0
    print(vector[0])
    # CHECK: 8.0
    print(vector[1])
    # CHECK: 10.0
    print(vector[2])
    # CHECK: 12.0
    print(vector[3])
    # CHECK: 14.0
    print(vector[4])


# CHECK-LABEL: test_vectorize_unroll
fn test_vectorize_unroll():
    print("== test_vectorize_unroll")

    alias buf_len = 23
    let vec = Buffer[buf_len, DType.float32].stack_allocation()
    let buf = Buffer[buf_len, DType.float32].stack_allocation()

    for i in range(buf_len):
        vec[i] = i
        buf[i] = i

    @always_inline
    @parameter
    fn double_buf[simd_width: Int](idx: Int):
        buf.simd_store[simd_width](
            idx,
            buf.simd_load[simd_width](idx) + buf.simd_load[simd_width](idx),
        )

    @parameter
    @always_inline
    fn double_vec[simd_width: Int](idx: Int):
        vec.simd_store[simd_width](
            idx,
            vec.simd_load[simd_width](idx) + vec.simd_load[simd_width](idx),
        )

    alias simd_width = 4
    alias unroll_factor = 2

    vectorize_unroll[simd_width, unroll_factor, double_vec](vec.__len__())
    vectorize[simd_width, double_buf](buf.__len__())

    let err = memcmp(vec.data, buf.data, buf.__len__())
    # CHECK: 0
    print(err)


fn main():
    test_vectorize()
    test_vectorize_unroll()
