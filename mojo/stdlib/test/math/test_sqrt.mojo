# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

from math import iota, rsqrt, sqrt


# CHECK-LABEL: test_int
fn test_int():
    print("== test_int")

    var simd_val = iota[DType.index, 4]() ** 2

    # CHECK: [0, 1, 2, 3]
    print(sqrt(simd_val))

    # CHECK: 8
    print(sqrt(64))

    # CHECK: 7
    print(sqrt(63))


# CHECK-LABEL: test_sqrt
fn test_sqrt_float32():
    print("== test_sqrt")

    var simd_val = 0.5 * iota[DType.float32, 4]()

    # CHECK: [0.0, 0.70710{{[0-9]+}}, 1.0, 1.22474{{[0-9]+}}]
    print(sqrt(simd_val))

    # CHECK: [0.0, 0.5, 0.70710{{[0-9]+}}, 0.86602{{[0-9]+}}]
    print(sqrt(0.5 * simd_val))


# CHECK-LABEL: test_sqrt_float64
fn test_sqrt_float64():
    print("== test_sqrt_float64")

    var simd_val = 0.5 * iota[DType.float64, 4]()

    # CHECK: [0.0, 0.70710{{[0-9]+}}, 1.0, 1.22474{{[0-9]+}}]
    print(sqrt(simd_val))

    # CHECK: [0.0, 0.5, 0.70710{{[0-9]+}}, 0.86602{{[0-9]+}}]
    print(sqrt(0.5 * simd_val))


# CHECK-LABEL: test_rsqrt_float32
fn test_rsqrt_float32():
    print("== test_rsqrt_float32")

    var simd_val = 0.5 * iota[DType.float32, 4]() + 1

    # CHECK: [1.0, 0.81649{{[0-9]+}}, 0.70710{{[0-9]+}}, 0.63245{{[0-9]+}}]
    print(rsqrt(simd_val))

    # CHECK: [1.41421{{[0-9]+}}, 1.15470{{[0-9]+}}, 1.0, 0.89442{{[0-9]+}}]
    print(rsqrt(0.5 * simd_val))


# CHECK-LABEL: test_rsqrt_float64
fn test_rsqrt_float64():
    print("== test_rsqrt_float64")

    var simd_val = 0.5 * iota[DType.float64, 4]() + 1

    # CHECK: [1.0, 0.81649{{[0-9]+}}, 0.70710{{[0-9]+}}, 0.63245{{[0-9]+}}]
    print(rsqrt(simd_val))

    # CHECK: [1.41421{{[0-9]+}}, 1.15470{{[0-9]+}}, 1.0, 0.89442{{[0-9]+}}]
    print(rsqrt(0.5 * simd_val))


fn main():
    test_int()
    test_sqrt_float32()
    test_sqrt_float64()
    test_rsqrt_float32()
    test_rsqrt_float64()
