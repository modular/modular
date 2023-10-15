# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from math import all_true, any_true, none_true

# CHECK-LABEL: test_boolean_reductions
fn test_boolean_reductions():
    print("== test_boolean_reductions")

    # CHECK: True
    print(all_true(SIMD[DType.bool, 1].splat(True)))
    # CHECK: True
    print(all_true(SIMD[DType.bool, 4].splat(True)))

    # CHECK: True
    print(any_true(SIMD[DType.bool, 1].splat(True)))
    # CHECK: True
    print(any_true(SIMD[DType.bool, 4].splat(True)))

    # CHECK: False
    print(none_true(SIMD[DType.bool, 1].splat(True)))
    # CHECK: False
    print(none_true(SIMD[DType.bool, 4].splat(True)))


fn main():
    test_boolean_reductions()
