# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen %s -execute -func='$test_exp::main():index()' | FileCheck %s

from DType import DType
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


@export
fn main() -> __mlir_type.index:
    test_exp()
    return 0
