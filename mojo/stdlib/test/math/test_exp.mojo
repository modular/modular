# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: lit %s -execute | FileCheck %s

from DType import DType
from Int import Int
from IO import print
from Math import exp
from SIMD import SIMD


# CHECK-LABEL: test_exp
fn test_exp():
    print("== test_exp\n")

    # CHECK: 0.904837
    print(exp[1, DType.f32.value](SIMD[1, DType.f32.value](-0.1)))

    # CHECK: 1.105171
    print(exp[1, DType.f32.value](SIMD[1, DType.f32.value](0.1)))

    # CHECK: 7.389056
    print(exp(SIMD[1, DType.f32.value](2)))


fn main() -> Int:
    test_exp()
    return 0
