# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from math import iota, reduce_bit_count


fn test_reduce_bit_count():
    print("== test_reduce_bit_count")
    let int_0xFFFF = SIMD[DType.int32, 1](0xFFFF)
    let int_iota8 = iota[DType.int32, 8]()

    let bool_true = SIMD[DType.bool, 1].splat(True)
    let bool_false = SIMD[DType.bool, 1].splat(False)
    let bool_true16 = SIMD[DType.bool, 16].splat(True)

    # CHECK: 16
    print(reduce_bit_count(int_0xFFFF))
    # CHECK: 12
    print(reduce_bit_count(int_iota8))
    # CHECK: 1
    print(reduce_bit_count(bool_true))
    # CHECK: 0
    print(reduce_bit_count(bool_false))
    # CHECK: 16
    print(reduce_bit_count(bool_true16))


fn main():
    test_reduce_bit_count()
