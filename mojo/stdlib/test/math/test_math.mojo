# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s

from math import (
    ceil,
    cos,
    factorial,
    floor,
    isfinite,
    isinf,
    isnan,
    isclose,
    nan,
    sin,
    trunc,
)
from math.limit import inf, neginf
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


fn test_factorial() raises:
    assert_equal(factorial(0), 1)
    assert_equal(factorial(1), 1)
    assert_equal(factorial(15), 1307674368000)
    assert_equal(factorial(20), 2432902008176640000)


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


def test_isclose():
    assert_true(isclose(Int64(2), Int64(2)))
    assert_false(isclose(Int64(2), Int64(3)))

    assert_true(isclose(Float32(2), Float32(2)))
    assert_true(isclose(Float32(2), Float32(2), rtol=1e-9))
    assert_true(isclose(Float32(2), Float32(2.00001), rtol=1e-3))
    assert_true(
        isclose(nan[DType.float32](), nan[DType.float32](), equal_nan=True)
    )

    assert_true(
        isclose(
            SIMD[DType.float32, 4](1, 2, 3, nan[DType.float32]()),
            SIMD[DType.float32, 4](1, 2, 3, nan[DType.float32]()),
            equal_nan=True,
        )
    )

    assert_false(
        isclose(
            SIMD[DType.float32, 4](1, 2, nan[DType.float32](), 3),
            SIMD[DType.float32, 4](1, 2, nan[DType.float32](), 4),
            equal_nan=True,
        )
    )


def test_ceil():
    # We just test that the `ceil` function resolves correctly for a few common
    # types. Types should test their own `__ceil__` implementation explicitly.
    assert_equal(ceil(0), 0)
    assert_equal(ceil(Int(5)), 5)
    assert_equal(ceil(1.5), 2.0)
    assert_equal(ceil(Float32(1.4)), 2.0)
    assert_equal(ceil(Float64(-3.6)), -3.0)


def test_floor():
    # We just test that the `floor` function resolves correctly for a few common
    # types. Types should test their own `__floor__` implementation explicitly.
    assert_equal(floor(0), 0)
    assert_equal(floor(Int(5)), 5)
    assert_equal(floor(1.5), 1.0)
    assert_equal(floor(Float32(1.6)), 1.0)
    assert_equal(floor(Float64(-3.4)), -4.0)


def test_trunc():
    # We just test that the `trunc` function resolves correctly for a few common
    # types. Types should test their own `__floor__` implementation explicitly.
    assert_equal(trunc(0), 0)
    assert_equal(trunc(Int(5)), 5)
    assert_equal(trunc(1.5), 1.0)
    assert_equal(trunc(Float32(1.6)), 1.0)
    assert_equal(trunc(Float64(-3.4)), -3.0)


def main():
    test_inf()
    test_nan()
    test_sin()
    test_cos()
    test_factorial()
    test_copysign()
    test_isclose()
    test_ceil()
    test_floor()
    test_trunc()
