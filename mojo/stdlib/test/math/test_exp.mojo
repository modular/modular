# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo %s | FileCheck %s

from DType import DType
from IO import print
from Math import exp
from SIMD import SIMD


# CHECK-LABEL: test_exp
fn test_exp():
    print("== test_exp")

    # CHECK: 0.904837
    print(exp[1, DType.f32](SIMD[DType.f32, 1](-0.1)))

    # CHECK: 1.105171
    print(exp[1, DType.f32](SIMD[DType.f32, 1](0.1)))

    # CHECK: 7.389056
    print(exp(SIMD[DType.f32, 1](2)))


fn main():
    test_exp()
