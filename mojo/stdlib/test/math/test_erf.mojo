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


# CHECK-LABEL: test_erf_f16
fn test_erf_f16():
    print("== test_erf_f16")

    # Test MLAS erf.

    # CHECK: 0.000000
    print(erf(SIMD[DType.f16, 1](0)))

    # CHECK: 0.995117
    # CHECK: 0.995117
    print(erf(SIMD[DType.f16, 2](2)))

    # CHECK: 0.112427
    print(erf(SIMD[DType.f16, 1](0.1)))

    # CHECK: -0.112427
    print(erf(SIMD[DType.f16, 1](-0.1)))

    # CHECK: -0.842773
    print(erf(SIMD[DType.f16, 1](-1)))

    # CHECK: -0.995117
    print(erf(SIMD[DType.f16, 1](-2)))

    # Test oneDNN erf.

    # CHECK: 0.000977
    print(_erf(SIMD[DType.f16, 1](0)))

    # CHECK: 0.995117
    # CHECK: 0.995117
    print(_erf(SIMD[DType.f16, 2](2)))

    # CHECK: 0.113586
    print(_erf(SIMD[DType.f16, 1](0.1)))

    # CHECK: -0.113586
    print(_erf(SIMD[DType.f16, 1](-0.1)))

    # CHECK: -0.842773
    print(_erf(SIMD[DType.f16, 1](-1)))

    # CHECK: -0.995117
    print(_erf(SIMD[DType.f16, 1](-2)))


# CHECK-LABEL: test_erf_f32
fn test_erf_f32():
    print("== test_erf_f32")

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


# CHECK-LABEL: test_erf_f64
fn test_erf_f64():
    print("== test_erf_f64")

    # Test MLAS erf.

    # CHECK: 0.000000
    print(erf(SIMD[DType.f64, 1](0)))

    # CHECK: 0.995322
    # CHECK: 0.995322
    print(erf(SIMD[DType.f64, 2](2)))

    # CHECK: 0.112463
    print(erf(SIMD[DType.f64, 1](0.1)))

    # CHECK: -0.112463
    print(erf(SIMD[DType.f64, 1](-0.1)))

    # CHECK: -0.842701
    print(erf(SIMD[DType.f64, 1](-1)))

    # CHECK: -0.995322
    print(erf(SIMD[DType.f64, 1](-2)))

    # Test oneDNN erf.

    # CHECK: 0.000000
    print(_erf(SIMD[DType.f64, 1](0)))

    # CHECK: 0.995322
    # CHECK: 0.995322
    print(_erf(SIMD[DType.f64, 2](2)))

    # CHECK: 0.112463
    print(_erf(SIMD[DType.f64, 1](0.1)))

    # CHECK: -0.112463
    print(_erf(SIMD[DType.f64, 1](-0.1)))

    # CHECK: -0.842701
    print(_erf(SIMD[DType.f64, 1](-1)))

    # CHECK: -0.995322
    print(_erf(SIMD[DType.f64, 1](-2)))


fn main():
    test_erf_f16()
    test_erf_f32()
    test_erf_f64()
