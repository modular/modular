# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from math import abs, factorial, sin, cos, nan, isnan
from Complex import ComplexFloat32
from Limits import isfinite, isinf, inf, neginf
from IO import print

# CHECK-LABEL: test_inf
fn test_inf():
    print("== test_inf")

    # CHECK: False
    print(isfinite(inf[DType.float32]()))

    # CHECK: False
    print(isfinite(inf[DType.float64]()))

    # CHECK: True
    print(isinf(inf[DType.float32]()))

    # CHECK: True
    print(isinf(inf[DType.float64]()))

    # CHECK: False
    print(isfinite(neginf[DType.float32]()))

    # CHECK: False
    print(isfinite(neginf[DType.float64]()))

    # CHECK: True
    print(isinf(neginf[DType.float32]()))

    # CHECK: True
    print(isinf(neginf[DType.float64]()))

    # CHECK: False
    print(isfinite(nan[DType.float32]()))

    # CHECK: False
    print(isfinite(nan[DType.float64]()))

    # CHECK: True
    print(isfinite(Float32(33)))

    # CHECK: True
    print(isinf(Float32(33) / 0))

    # CHECK: False
    print(isfinite(Float32(33) / 0))


# CHECK-LABEL: test_nan
fn test_nan():
    print("== test_nan")

    # CHECK: False
    print(isnan(inf[DType.float32]()))

    # CHECK: False
    print(isnan(neginf[DType.float32]()))

    # CHECK: True
    print(isnan(nan[DType.float32]()))

    # CHECK: True
    print(isnan(nan[DType.float64]()))

    # CHECK: False
    print(isnan(Float32(33)))

    # CHECK: [False, True, False, False]
    print(
        isnan(
            SIMD[DType.float32, 4](1, 0, 3, -1)
            / SIMD[DType.float32, 4](0, 0, 1, 0)
        )
    )

    # CHECK: [False, True, False, False]
    print(
        isnan(
            SIMD[DType.float64, 4](1, 0, 3, -1)
            / SIMD[DType.float64, 4](0, 0, 1, 0)
        )
    )

    # CHECK: False
    print(isnan(SIMD[DType.float32, 1](1) / SIMD[DType.float32, 1](0)))

    # CHECK: False
    print(isnan(inf[DType.float64]()))


# CHECK-LABEL: test_sin
fn test_sin():
    print("== test_sin")

    # CHECK: 0.841470956802{{[0-9]+}}
    print(sin(Float32(1.0)))


# CHECK-LABEL: test_cos
fn test_cos():
    print("== test_cos")

    # CHECK: 0.540302276611{{[0-9]+}}
    print(cos(Float32(1.0)))


# CHECK-LABEL: test_abs
fn test_abs():
    print("== test_abs")

    # CHECK: 1.0
    print(abs(Float32(1.0)))

    # CHECK: 1.0
    print(abs(Float32(-1.0)))

    # CHECK: 0.0
    print(abs(Float32(0.0)))

    # CHECK: 0.0
    print(abs(ComplexFloat32 {re: 0, im: 0}))

    # CHECK: 1.0
    print(abs(ComplexFloat32 {re: 1, im: 0}))

    # CHECK: 1.0
    print(abs(ComplexFloat32 {re: 0, im: 1}))

    # CHECK: 1.41421
    print(abs(ComplexFloat32 {re: -1, im: -1}))

    # CHECK: 95.801
    print(abs(ComplexFloat32 {re: -93, im: -23}))


# CHECK-LABEL: test_factorial
fn test_factorial():
    print("== test_factorial")

    # CHECK: 1
    print(factorial(0))

    # CHECK: 1
    print(factorial(1))

    # CHECK: 1307674368000
    print(factorial(15))

    # CHECK: 2432902008176640000
    print(factorial(20))


fn main():
    test_inf()
    test_nan()
    test_sin()
    test_cos()
    test_abs()
    test_factorial()
