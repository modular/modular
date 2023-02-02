# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: lit %s -execute | FileCheck %s


from DType import DType
from Int import Int
from IO import print
from Math import erf
from SIMD import SIMD


# CHECK-LABEL: test_erf
fn test_erf():
    print("== test_erf\n")

    # CHECK: 0.000000
    print(erf[1, DType.f32.value](SIMD[1, DType.f32.value](0)))

    # CHECK: 0.995322
    # CHECK: 0.995322
    print(erf[2, DType.f32.value](SIMD[2, DType.f32.value](2)))

    # CHECK: 0.112463
    print(erf[1, DType.f32.value](SIMD[1, DType.f32.value](0.1)))

    # CHECK: -0.112463
    print(erf[1, DType.f32.value](SIMD[1, DType.f32.value](-0.1)))

    # CHECK: -0.842701
    print(erf[1, DType.f32.value](SIMD[1, DType.f32.value](-1)))

    # CHECK: -0.995322
    print(erf[1, DType.f32.value](SIMD[1, DType.f32.value](-2)))


@export
fn main() -> Int:
    test_erf()
    return 0
