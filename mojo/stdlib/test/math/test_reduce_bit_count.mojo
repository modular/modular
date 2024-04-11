# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

from math import iota, reduce_bit_count


fn test_reduce_bit_count():
    print("== test_reduce_bit_count")
    var int_0xFFFF = Int32(0xFFFF)
    var int_iota8 = iota[DType.int32, 8]()

    var bool_true = Scalar[DType.bool].splat(True)
    var bool_false = Scalar[DType.bool].splat(False)
    var bool_true16 = SIMD[DType.bool, 16].splat(True)

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
