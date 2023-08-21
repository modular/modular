# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from math import (
    abs,
    cos,
    factorial,
    isnan,
    nan,
    rotate_bits_left,
    rotate_bits_right,
    rotate_left,
    rotate_right,
    sin,
)
from math.limit import inf, isfinite, isinf, neginf

from complex import ComplexFloat32

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


# CHECK-LABEL: test_rotate
fn test_rotate():
    print("== test_rotate")

    alias simd_width = 4
    alias type = DType.uint32

    # CHECK: [0, 1, 0, 1, 1, 0, 1, 0]
    print(
        rotate_right[DType.uint16, 8, 1](
            SIMD[DType.uint16, 8](1, 0, 1, 1, 0, 1, 0, 0)
        )
    )
    # CHECK: [1, 0, 1, 0, 0, 1, 0, 1]
    print(
        rotate_right[DType.uint32, 8, 5](
            SIMD[DType.uint32, 8](1, 0, 1, 1, 0, 1, 0, 0)
        )
    )

    # CHECK: [1, 0, 1, 1]
    print(rotate_left[type, simd_width, 0](SIMD[type, simd_width](1, 0, 1, 1)))
    # CHECK: [0, 1, 1, 1]
    print(rotate_left[type, simd_width, 1](SIMD[type, simd_width](1, 0, 1, 1)))
    # CHECK: [1, 1, 1, 0]
    print(rotate_left[type, simd_width, 2](SIMD[type, simd_width](1, 0, 1, 1)))
    # CHECK: [1, 1, 0, 1]
    print(rotate_left[type, simd_width, 3](SIMD[type, simd_width](1, 0, 1, 1)))
    # CHECK: [1, 1, 0, 1]
    print(rotate_left[type, simd_width, -1](SIMD[type, simd_width](1, 0, 1, 1)))
    # CHECK: [1, 1, 1, 0]
    print(rotate_left[type, simd_width, -2](SIMD[type, simd_width](1, 0, 1, 1)))
    # CHECK: [0, 1, 1, 1]
    print(rotate_left[type, simd_width, -3](SIMD[type, simd_width](1, 0, 1, 1)))
    # CHECK: [1, 0, 1, 1]
    print(rotate_left[type, simd_width, -4](SIMD[type, simd_width](1, 0, 1, 1)))

    # CHECK: [1, 0, 1, 1]
    print(rotate_right[type, simd_width, 0](SIMD[type, simd_width](1, 0, 1, 1)))
    # CHECK: [1, 1, 0, 1]
    print(rotate_right[type, simd_width, 1](SIMD[type, simd_width](1, 0, 1, 1)))
    # CHECK: [1, 1, 1, 0]
    print(rotate_right[type, simd_width, 2](SIMD[type, simd_width](1, 0, 1, 1)))
    # CHECK: [0, 1, 1, 1]
    print(rotate_right[type, simd_width, 3](SIMD[type, simd_width](1, 0, 1, 1)))
    # CHECK: [1, 0, 1, 1]
    print(rotate_right[type, simd_width, 4](SIMD[type, simd_width](1, 0, 1, 1)))
    # CHECK: [0, 1, 1, 1]
    print(
        rotate_right[type, simd_width, -1](SIMD[type, simd_width](1, 0, 1, 1))
    )
    # CHECK: [1, 1, 1, 0]
    print(
        rotate_right[type, simd_width, -2](SIMD[type, simd_width](1, 0, 1, 1))
    )
    # CHECK: [1, 1, 0, 1]
    print(
        rotate_right[type, simd_width, -3](SIMD[type, simd_width](1, 0, 1, 1))
    )


# CHECK-LABEL: test_rotate_bits
fn test_rotate_bits():
    print("== test_rotate_bits")

    alias simd_width = 1
    alias type = DType.uint8

    # CHECK: 104
    print(rotate_bits_left[type, simd_width, 0](SIMD[type, simd_width](104)))
    # CHECK: 161
    print(rotate_bits_left[type, 1, 2](SIMD[type, 1](104)))
    # CHECK: [161, 161]
    print(rotate_bits_left[type, 2, 2](SIMD[type, 2](104)))
    # CHECK: 120
    print(rotate_bits_left[type, 1, 11](SIMD[type, 1](15)))

    # CHECK: 96
    print(rotate_bits_left[type, 1, 0](SIMD[type, 1](96)))
    # CHECK: 192
    print(rotate_bits_left[type, 1, 1](SIMD[type, 1](96)))
    # CHECK: 129
    print(rotate_bits_left[type, 1, 2](SIMD[type, 1](96)))
    # CHECK: 3
    print(rotate_bits_left[type, 1, 3](SIMD[type, 1](96)))
    # CHECK: 6
    print(rotate_bits_left[type, 1, 4](SIMD[type, 1](96)))
    # CHECK: 12
    print(rotate_bits_left[type, 1, 5](SIMD[type, 1](96)))

    # CHECK: 104
    print(rotate_bits_right[type, simd_width, 0](SIMD[type, simd_width](104)))
    # CHECK: 26
    print(rotate_bits_right[type, 1, 2](SIMD[type, 1](104)))
    # CHECK: [26, 26]
    print(rotate_bits_right[type, 2, 2](SIMD[type, 2](104)))
    # CHECK: 225
    print(rotate_bits_right[type, 1, 11](SIMD[type, 1](15)))

    # CHECK: 96
    print(rotate_bits_right[type, 1, 0](SIMD[type, 1](96)))
    # CHECK: 48
    print(rotate_bits_right[type, 1, 1](SIMD[type, 1](96)))
    # CHECK: 24
    print(rotate_bits_right[type, 1, 2](SIMD[type, 1](96)))
    # CHECK: 12
    print(rotate_bits_right[type, 1, 3](SIMD[type, 1](96)))
    # CHECK: 6
    print(rotate_bits_right[type, 1, 4](SIMD[type, 1](96)))
    # CHECK: 3
    print(rotate_bits_right[type, 1, 5](SIMD[type, 1](96)))
    # CHECK: 129
    print(rotate_bits_right[type, 1, 6](SIMD[type, 1](96)))


fn main():
    test_inf()
    test_nan()
    test_sin()
    test_cos()
    test_abs()
    test_factorial()
    test_rotate()
    test_rotate_bits()
