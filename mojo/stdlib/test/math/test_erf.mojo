# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo %s | FileCheck %s


from Activations import _erf
from DType import DType
from IO import print
from Math import erf
from SIMD import SIMD


# CHECK-LABEL: test_erf
fn test_erf():
    print("== test_erf")

    # Test MLAS erf.

    # CHECK: 0.000000
    print(erf(SIMD[DType.f32, 1](0)))

    # CHECK: 0.995322
    # CHECK: 0.995322
    print(erf(SIMD[DType.f32, 2](2)))

    # CHECK: 0.112463
    print(erf(SIMD[DType.f32, 1](0.1)))

    # CHECK: -0.112463
    print(erf(SIMD[DType.f32, 1](-0.1)))

    # CHECK: -0.842701
    print(erf(SIMD[DType.f32, 1](-1)))

    # CHECK: -0.995322
    print(erf(SIMD[DType.f32, 1](-2)))

    # Test oneDNN erf.

    # CHECK: 0.000000
    print(_erf(SIMD[DType.f32, 1](0)))

    # CHECK: 0.995322
    # CHECK: 0.995322
    print(_erf(SIMD[DType.f32, 2](2)))

    # CHECK: 0.112463
    print(_erf(SIMD[DType.f32, 1](0.1)))

    # CHECK: -0.112463
    print(_erf(SIMD[DType.f32, 1](-0.1)))

    # CHECK: -0.842701
    print(_erf(SIMD[DType.f32, 1](-1)))

    # CHECK: -0.995322
    print(_erf(SIMD[DType.f32, 1](-2)))


fn main():
    test_erf()
