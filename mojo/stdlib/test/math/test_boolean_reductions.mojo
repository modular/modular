# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

from math import all_true, any_true, none_true


# CHECK-LABEL: test_boolean_reductions
fn test_boolean_reductions():
    print("== test_boolean_reductions")

    # CHECK: True
    print(all_true(Scalar[DType.bool](True)))
    # CHECK: True
    print(all_true(SIMD[DType.bool, 4](True)))

    # CHECK: True
    print(any_true(Scalar[DType.bool](True)))
    # CHECK: True
    print(any_true(SIMD[DType.bool, 4](True)))

    # CHECK: False
    print(none_true(Scalar[DType.bool](True)))
    # CHECK: False
    print(none_true(SIMD[DType.bool, 4](True)))


fn main():
    test_boolean_reductions()
