# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s

from math import (
    abs,
    cos,
    factorial,
    isfinite,
    isinf,
    isnan,
    nan,
    rotate_bits_left,
    rotate_bits_right,
    rotate_left,
    rotate_right,
    sin,
)
from math.limit import inf, neginf
from math.math import _boole
from sys.info import has_neon

from complex import ComplexFloat32
from testing import assert_almost_equal, assert_equal, assert_false, assert_true


fn test_inf() raises:
    @parameter
    if not has_neon():
        assert_false(isfinite(inf[DType.bfloat16]()))

    assert_false(isfinite(inf[DType.float32]()))
    assert_false(isfinite(inf[DType.float64]()))
    assert_true(isinf(inf[DType.float32]()))
    assert_true(isinf(inf[DType.float64]()))

    @parameter
    if not has_neon():
        assert_false(isfinite(neginf[DType.bfloat16]()))

    assert_false(isfinite(neginf[DType.float32]()))
    assert_false(isfinite(neginf[DType.float64]()))
    assert_true(isinf(neginf[DType.float32]()))
    assert_true(isinf(neginf[DType.float64]()))

    @parameter
    if not has_neon():
        assert_false(isfinite(nan[DType.bfloat16]()))

    assert_false(isfinite(nan[DType.float32]()))
    assert_false(isfinite(nan[DType.float64]()))
    assert_true(isfinite(Float32(33)))
    assert_true(isinf(Float32(33) / 0))
    assert_false(isfinite(Float32(33) / 0))


fn test_nan() raises:
    @parameter
    if not has_neon():
        assert_false(isnan(inf[DType.bfloat16]()))

    assert_false(isnan(inf[DType.float32]()))
    assert_false(isnan(neginf[DType.float32]()))

    @parameter
    if not has_neon():
        assert_true(isnan(nan[DType.bfloat16]()))

    assert_true(isnan(nan[DType.float32]()))
    assert_true(isnan(nan[DType.float64]()))
    assert_false(isnan(Float32(33)))

    assert_equal(
        isnan(
            SIMD[DType.float32, 4](1, 0, 3, -1)
            / SIMD[DType.float32, 4](0, 0, 1, 0)
        ),
        SIMD[DType.bool, 4](False, True, False, False),
    )

    assert_equal(
        isnan(
            SIMD[DType.float64, 4](1, 0, 3, -1)
            / SIMD[DType.float64, 4](0, 0, 1, 0)
        ),
        SIMD[DType.bool, 4](False, True, False, False),
    )

    assert_false(isnan(Float32(1) / Float32(0)))
    assert_false(isnan(inf[DType.float64]()))


fn test_sin() raises:
    assert_almost_equal(sin(Float32(1.0)), 0.841470956802)


fn test_cos() raises:
    assert_almost_equal(cos(Float32(1.0)), 0.540302276611)


fn test_abs() raises:
    assert_equal(abs(Float32(1.0)), 1)
    assert_equal(abs(Float32(-1.0)), 1)
    assert_equal(abs(Float32(0.0)), 0)
    assert_equal(abs(ComplexFloat32 {re: 0, im: 0}), 0)
    assert_equal(abs(ComplexFloat32 {re: 1, im: 0}), 1)
    assert_equal(abs(ComplexFloat32 {re: 0, im: 1}), 1)
    assert_almost_equal(abs(ComplexFloat32 {re: -1, im: -1}), 1.41421)
    assert_almost_equal(abs(ComplexFloat32 {re: -93, im: -23}), 95.801)


fn test_factorial() raises:
    assert_equal(factorial(0), 1)
    assert_equal(factorial(1), 1)
    assert_equal(factorial(15), 1307674368000)
    assert_equal(factorial(20), 2432902008176640000)


fn test_rotate() raises:
    alias simd_width = 4
    alias type = DType.uint32

    assert_equal(
        rotate_right[1](SIMD[DType.uint16, 8](1, 0, 1, 1, 0, 1, 0, 0)),
        SIMD[DType.uint16, 8](0, 1, 0, 1, 1, 0, 1, 0),
    )
    assert_equal(
        rotate_right[5](SIMD[DType.uint32, 8](1, 0, 1, 1, 0, 1, 0, 0)),
        SIMD[DType.uint32, 8](1, 0, 1, 0, 0, 1, 0, 1),
    )
    assert_equal(rotate_left[2](104), 416)
    assert_equal(rotate_right[2](104), 26)
    assert_equal(rotate_left[-2](104), 26)
    assert_equal(rotate_right[-2](104), 416)

    assert_equal(
        rotate_left[0](SIMD[type, simd_width](1, 0, 1, 1)),
        SIMD[type, simd_width](1, 0, 1, 1),
    )
    assert_equal(
        rotate_left[1](SIMD[type, simd_width](1, 0, 1, 1)),
        SIMD[type, simd_width](0, 1, 1, 1),
    )
    assert_equal(
        rotate_left[2](SIMD[type, simd_width](1, 0, 1, 1)),
        SIMD[type, simd_width](1, 1, 1, 0),
    )
    assert_equal(
        rotate_left[3](SIMD[type, simd_width](1, 0, 1, 1)),
        SIMD[type, simd_width](1, 1, 0, 1),
    )
    assert_equal(
        rotate_left[-1](SIMD[type, simd_width](1, 0, 1, 1)),
        SIMD[type, simd_width](1, 1, 0, 1),
    )
    assert_equal(
        rotate_left[-2](SIMD[type, simd_width](1, 0, 1, 1)),
        SIMD[type, simd_width](1, 1, 1, 0),
    )
    assert_equal(
        rotate_left[-3](SIMD[type, simd_width](1, 0, 1, 1)),
        SIMD[type, simd_width](0, 1, 1, 1),
    )
    assert_equal(
        rotate_left[-4](SIMD[type, simd_width](1, 0, 1, 1)),
        SIMD[type, simd_width](1, 0, 1, 1),
    )
    assert_equal(
        rotate_right[0](SIMD[type, simd_width](1, 0, 1, 1)),
        SIMD[type, simd_width](1, 0, 1, 1),
    )
    assert_equal(
        rotate_right[1](SIMD[type, simd_width](1, 0, 1, 1)),
        SIMD[type, simd_width](1, 1, 0, 1),
    )
    assert_equal(
        rotate_right[2](SIMD[type, simd_width](1, 0, 1, 1)),
        SIMD[type, simd_width](1, 1, 1, 0),
    )
    assert_equal(
        rotate_right[3](SIMD[type, simd_width](1, 0, 1, 1)),
        SIMD[type, simd_width](0, 1, 1, 1),
    )
    assert_equal(
        rotate_right[4](SIMD[type, simd_width](1, 0, 1, 1)),
        SIMD[type, simd_width](1, 0, 1, 1),
    )
    assert_equal(
        rotate_right[-1](SIMD[type, simd_width](1, 0, 1, 1)),
        SIMD[type, simd_width](0, 1, 1, 1),
    )
    assert_equal(
        rotate_right[-2](SIMD[type, simd_width](1, 0, 1, 1)),
        SIMD[type, simd_width](1, 1, 1, 0),
    )
    assert_equal(
        rotate_right[-3](SIMD[type, simd_width](1, 0, 1, 1)),
        SIMD[type, simd_width](1, 1, 0, 1),
    )


fn test_rotate_bits() raises:
    alias simd_width = 1
    alias type = DType.uint8

    assert_equal(rotate_bits_left[0](104), 104)

    assert_equal(rotate_bits_left[0](UInt64(104)), 104)

    assert_equal(rotate_bits_left[0](SIMD[type, simd_width](104)), 104)

    assert_equal(rotate_bits_left[2](104), 416)

    assert_equal(rotate_bits_left[-2](104), 26)

    assert_equal(rotate_bits_left[2](Scalar[type](104)), 161)
    assert_equal(
        rotate_bits_left[2](SIMD[type, 2](104)), SIMD[type, 2](161, 161)
    )
    assert_equal(rotate_bits_left[11](Scalar[type](15)), 120)

    assert_equal(rotate_bits_left[0](Scalar[type](96)), 96)
    assert_equal(rotate_bits_left[1](Scalar[type](96)), 192)
    assert_equal(rotate_bits_left[2](Scalar[type](96)), 129)
    assert_equal(rotate_bits_left[3](Scalar[type](96)), 3)
    assert_equal(rotate_bits_left[4](Scalar[type](96)), 6)
    assert_equal(rotate_bits_left[5](Scalar[type](96)), 12)

    assert_equal(rotate_bits_right[0](104), 104)

    assert_equal(rotate_bits_right[0](UInt64(104)), 104)

    assert_equal(rotate_bits_right[0](SIMD[type, simd_width](104)), 104)

    assert_equal(rotate_bits_right[2](104), 26)

    assert_equal(rotate_bits_right[-2](104), 416)

    assert_equal(rotate_bits_right[2](Scalar[type](104)), 26)
    assert_equal(
        rotate_bits_right[2](SIMD[type, 2](104)), SIMD[type, 2](26, 26)
    )
    assert_equal(rotate_bits_right[11](Scalar[type](15)), 225)

    assert_equal(rotate_bits_right[0](Scalar[type](96)), 96)
    assert_equal(rotate_bits_right[1](Scalar[type](96)), 48)
    assert_equal(rotate_bits_right[2](Scalar[type](96)), 24)
    assert_equal(rotate_bits_right[3](Scalar[type](96)), 12)
    assert_equal(rotate_bits_right[4](Scalar[type](96)), 6)
    assert_equal(rotate_bits_right[5](Scalar[type](96)), 3)
    assert_equal(rotate_bits_right[6](Scalar[type](96)), 129)


def test_boole():
    assert_equal(_boole(True), 1)
    assert_equal(_boole(False), 0)


def test_copysign():
    var x = Int32(2)
    assert_equal(2, math.copysign(x, x))
    assert_equal(-2, math.copysign(x, -x))
    assert_equal(2, math.copysign(-x, x))
    assert_equal(-2, math.copysign(-x, -x))

    assert_equal(1, math.copysign(Float32(1), Float32(2)))
    assert_equal(-1, math.copysign(Float32(1), Float32(-2)))
    assert_equal(
        neginf[DType.float32](), math.copysign(inf[DType.float32](), -2.0)
    )
    assert_equal(
        -nan[DType.float32](), math.copysign(nan[DType.float32](), -2.0)
    )

    # Test some cases with 0 and signed zero
    assert_equal(1, math.copysign(Float32(1.0), Float32(0.0)))
    assert_equal(0, math.copysign(Float32(0.0), Float32(1.0)))
    assert_equal(0, math.copysign(Float32(-0.0), Float32(1.0)))
    assert_equal(-0, math.copysign(Float32(0.0), Float32(-1.0)))

    # TODO: Add some test cases for SIMD vector with width > 1


def main():
    test_inf()
    test_nan()
    test_sin()
    test_cos()
    test_abs()
    test_factorial()
    test_rotate()
    test_rotate_bits()
    test_boole()
    test_copysign()
