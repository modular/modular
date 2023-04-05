# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo %s | FileCheck %s


from DType import DType
from IO import print
from Math import erf
from SIMD import SIMD


# CHECK-LABEL: test_erf
fn test_erf():
    print("== test_erf\n")

    # CHECK: 0.000000
    print(erf[1, DType.f32](SIMD[1, DType.f32](0)))

    # CHECK: 0.995322
    # CHECK: 0.995322
    print(erf[2, DType.f32](SIMD[2, DType.f32](2)))

    # CHECK: 0.112463
    print(erf[1, DType.f32](SIMD[1, DType.f32](0.1)))

    # CHECK: -0.112463
    print(erf[1, DType.f32](SIMD[1, DType.f32](-0.1)))

    # CHECK: -0.842701
    print(erf[1, DType.f32](SIMD[1, DType.f32](-1)))

    # CHECK: -0.995322
    print(erf[1, DType.f32](SIMD[1, DType.f32](-2)))


fn main():
    test_erf()
