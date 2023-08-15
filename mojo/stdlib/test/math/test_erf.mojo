# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s


from Activations import _erf
from IO import print
from math import erf, round
from SIMD import SIMD, Float32, Float64


# CHECK-LABEL: test_erf_float32
fn test_erf_float32():
    print("== test_erf_float32")

    # Test MLAS erf.

    # CHECK: 0.0
    print(erf(Float32(0)))

    # CHECK: 0.995322
    # CHECK: 0.995322
    print(erf(SIMD[DType.float32, 2](2)))

    # CHECK: 0.11246
    print(erf(Float32(0.1)))

    # CHECK: -0.11246
    print(erf(Float32(-0.1)))

    # CHECK: -0.8427007
    print(erf(Float32(-1)))

    # CHECK: -0.995322
    print(erf(Float32(-2)))

    # Test oneDNN erf.

    # CHECK: 0.0
    print(_erf(Float32(0)))

    # CHECK: 0.995322
    # CHECK: 0.995322
    print(_erf(SIMD[DType.float32, 2](2)))

    # CHECK: 0.11246
    print(_erf(Float32(0.1)))

    # CHECK: -0.11246
    print(_erf(Float32(-0.1)))

    # CHECK: -0.8427006
    print(_erf(Float32(-1)))

    # CHECK: -0.995322
    print(_erf(Float32(-2)))


# CHECK-LABEL: test_erf_float64
fn test_erf_float64():
    print("== test_erf_float64")

    # Test MLAS erf.

    # CHECK: 0.0
    print(erf(Float64(0)))

    # CHECK: 0.995322
    # CHECK: 0.995322
    print(erf(SIMD[DType.float64, 2](2)))

    # CHECK: 0.112462
    print(erf(Float64(0.1)))

    # CHECK: -0.112462
    print(erf(Float64(-0.1)))

    # CHECK: -0.8427007
    print(erf(Float64(-1)))

    # CHECK: -0.995322
    print(erf(Float64(-2)))

    # Test oneDNN erf.

    # CHECK: 0.0
    print(round(_erf(Float64(0))))

    # CHECK: 0.995322
    # CHECK: 0.995322
    print(_erf(SIMD[DType.float64, 2](2)))

    # CHECK: 0.112462
    print(_erf(Float64(0.1)))

    # CHECK: -0.112462
    print(_erf(Float64(-0.1)))

    # CHECK: -0.8427006
    print(_erf(Float64(-1)))

    # CHECK: -0.995322
    print(_erf(Float64(-2)))


fn main():
    test_erf_float32()
    test_erf_float64()
