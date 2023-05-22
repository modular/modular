# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo %s | FileCheck %s


from Activations import _erf
from DType import DType
from IO import print
from Math import erf, round
from SIMD import SIMD


# CHECK-LABEL: test_erf_f32
fn test_erf_f32():
    print("== test_erf_f32")

    # Test MLAS erf.

    # CHECK: 0.0
    print(erf(SIMD[DType.f32, 1](0)))

    # CHECK: 0.995322
    # CHECK: 0.995322
    print(erf(SIMD[DType.f32, 2](2)))

    # CHECK: 0.11246
    print(erf(SIMD[DType.f32, 1](0.1)))

    # CHECK: -0.11246
    print(erf(SIMD[DType.f32, 1](-0.1)))

    # CHECK: -0.8427007
    print(erf(SIMD[DType.f32, 1](-1)))

    # CHECK: -0.995322
    print(erf(SIMD[DType.f32, 1](-2)))

    # Test oneDNN erf.

    # CHECK: 0.0
    print(_erf(SIMD[DType.f32, 1](0)))

    # CHECK: 0.995322
    # CHECK: 0.995322
    print(_erf(SIMD[DType.f32, 2](2)))

    # CHECK: 0.11246
    print(_erf(SIMD[DType.f32, 1](0.1)))

    # CHECK: -0.11246
    print(_erf(SIMD[DType.f32, 1](-0.1)))

    # CHECK: -0.8427006
    print(_erf(SIMD[DType.f32, 1](-1)))

    # CHECK: -0.995322
    print(_erf(SIMD[DType.f32, 1](-2)))


# CHECK-LABEL: test_erf_f64
fn test_erf_f64():
    print("== test_erf_f64")

    # Test MLAS erf.

    # CHECK: 0.0
    print(erf(SIMD[DType.f64, 1](0)))

    # CHECK: 0.995322
    # CHECK: 0.995322
    print(erf(SIMD[DType.f64, 2](2)))

    # CHECK: 0.112462
    print(erf(SIMD[DType.f64, 1](0.1)))

    # CHECK: -0.112462
    print(erf(SIMD[DType.f64, 1](-0.1)))

    # CHECK: -0.8427007
    print(erf(SIMD[DType.f64, 1](-1)))

    # CHECK: -0.995322
    print(erf(SIMD[DType.f64, 1](-2)))

    # Test oneDNN erf.

    # CHECK: 0.0
    print(round(_erf(SIMD[DType.f64, 1](0))))

    # CHECK: 0.995322
    # CHECK: 0.995322
    print(_erf(SIMD[DType.f64, 2](2)))

    # CHECK: 0.112462
    print(_erf(SIMD[DType.f64, 1](0.1)))

    # CHECK: -0.112462
    print(_erf(SIMD[DType.f64, 1](-0.1)))

    # CHECK: -0.8427006
    print(_erf(SIMD[DType.f64, 1](-1)))

    # CHECK: -0.995322
    print(_erf(SIMD[DType.f64, 1](-2)))


fn main():
    test_erf_f32()
    test_erf_f64()
