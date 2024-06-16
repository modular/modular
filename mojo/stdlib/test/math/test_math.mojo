# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s

from collections import List
from math import (
    ceil,
    cos,
    exp2,
    factorial,
    floor,
    frexp,
    iota,
    isclose,
    log,
    log2,
    rsqrt,
    sin,
    sqrt,
    trunc,
    copysign,
)
from sys.info import has_neon

from testing import assert_almost_equal, assert_equal, assert_false, assert_true

from utils.numerics import inf, isinf, nan, neg_inf


fn test_sin() raises:
    assert_almost_equal(sin(Float32(1.0)), 0.841470956802)


fn test_cos() raises:
    assert_almost_equal(cos(Float32(1.0)), 0.540302276611)

    # TODO(KERN-228): support BF16 on neon systems.
    @parameter
    if not has_neon():
        assert_equal(cos(BFloat16(2.0)), -0.416015625)


fn test_factorial() raises:
    assert_equal(factorial(0), 1)
    assert_equal(factorial(1), 1)
    assert_equal(factorial(15), 1307674368000)
    assert_equal(factorial(20), 2432902008176640000)


def test_copysign():
    var x = Int32(2)
    assert_equal(2, copysign(x, x))
    assert_equal(-2, copysign(x, -x))
    assert_equal(2, copysign(-x, x))
    assert_equal(-2, copysign(-x, -x))

    assert_equal(1, copysign(Float32(1), Float32(2)))
    assert_equal(-1, copysign(Float32(1), Float32(-2)))
    assert_equal(neg_inf[DType.float32](), copysign(inf[DType.float32](), -2.0))
    assert_equal(-nan[DType.float32](), copysign(nan[DType.float32](), -2.0))

    # Test some cases with 0 and signed zero
    assert_equal(1, copysign(Float32(1.0), Float32(0.0)))
    assert_equal(0, copysign(Float32(0.0), Float32(1.0)))
    assert_equal(0, copysign(Float32(-0.0), Float32(1.0)))
    assert_equal(-0, copysign(Float32(0.0), Float32(-1.0)))

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
        ).reduce_and()
    )

    assert_false(
        isclose(
            SIMD[DType.float32, 4](1, 2, nan[DType.float32](), 3),
            SIMD[DType.float32, 4](1, 2, nan[DType.float32](), 4),
            equal_nan=True,
        ).reduce_and()
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
    # types. Types should test their own `__trunc__` implementation explicitly.
    assert_equal(trunc(0), 0)
    assert_equal(trunc(Int(5)), 5)
    assert_equal(trunc(1.5), 1.0)
    assert_equal(trunc(Float32(1.6)), 1.0)
    assert_equal(trunc(Float64(-3.4)), -3.0)


def test_exp2():
    assert_equal(exp2(Float32(1)), 2.0)
    assert_almost_equal(exp2(Float32(0.2)), 1.148696)
    assert_equal(exp2(Float32(0)), 1.0)
    assert_equal(exp2(Float32(-1)), 0.5)
    assert_equal(exp2(Float32(2)), 4.0)


def test_iota():
    alias length = 103
    var offset = 2

    var vector = List[Int32]()
    vector.resize(length, 0)

    var buff = rebind[DTypePointer[DType.int32]](vector.data)
    iota(buff, length, offset)

    for i in range(length):
        assert_equal(vector[i], offset + i)

    iota(vector, offset)

    for i in range(length):
        assert_equal(vector[i], offset + i)

    var vector2 = List[Int]()
    vector2.resize(length, 0)
    iota(vector2, offset)

    for i in range(length):
        assert_equal(vector2[i], offset + i)


alias F32x4 = SIMD[DType.float32, 4]
alias F64x4 = SIMD[DType.float64, 4]


def test_sqrt():
    var i = SIMD[DType.index, 4](0, 1, 2, 3)
    assert_equal(sqrt(i**2), i)
    assert_equal(sqrt(64), 8)
    assert_equal(sqrt(63), 7)

    var f32x4 = 0.5 * F32x4(0.0, 1.0, 2.0, 3.0)

    var s1_f32 = sqrt(f32x4)
    assert_equal(s1_f32[0], 0.0)
    assert_almost_equal(s1_f32[1], 0.70710)
    assert_equal(s1_f32[2], 1.0)
    assert_almost_equal(s1_f32[3], 1.22474)

    var s2_f32 = sqrt(0.5 * f32x4)
    assert_equal(s2_f32[0], 0.0)
    assert_equal(s2_f32[1], 0.5)
    assert_almost_equal(s2_f32[2], 0.70710)
    assert_almost_equal(s2_f32[3], 0.86602)

    var f64x4 = 0.5 * F64x4(0.0, 1.0, 2.0, 3.0)

    var s1_f64 = sqrt(f64x4)
    assert_equal(s1_f64[0], 0.0)
    assert_almost_equal(s1_f64[1], 0.70710)
    assert_equal(s1_f64[2], 1.0)
    assert_almost_equal(s1_f64[3], 1.22474)

    var s2_f64 = sqrt(0.5 * f64x4)
    assert_equal(s2_f64[0], 0.0)
    assert_equal(s2_f64[1], 0.5)
    assert_almost_equal(s2_f64[2], 0.70710)
    assert_almost_equal(s2_f64[3], 0.86602)


def test_rsqrt():
    var f32x4 = 0.5 * F32x4(0.0, 1.0, 2.0, 3.0) + 1

    var s1_f32 = rsqrt(f32x4)
    assert_equal(s1_f32[0], 1.0)
    assert_almost_equal(s1_f32[1], 0.81649)
    assert_almost_equal(s1_f32[2], 0.70710)
    assert_almost_equal(s1_f32[3], 0.63245)

    var s2_f32 = rsqrt(0.5 * f32x4)
    assert_almost_equal(s2_f32[0], 1.41421)
    assert_almost_equal(s2_f32[1], 1.15470)
    assert_equal(s2_f32[2], 1.0)
    assert_almost_equal(s2_f32[3], 0.89442)

    var f64x4 = 0.5 * F64x4(0.0, 1.0, 2.0, 3.0) + 1

    var s1_f64 = rsqrt(f64x4)
    assert_equal(s1_f64[0], 1.0)
    assert_almost_equal(s1_f64[1], 0.81649)
    assert_almost_equal(s1_f64[2], 0.70710)
    assert_almost_equal(s1_f64[3], 0.63245)

    var s2_f64 = rsqrt(0.5 * f64x4)
    assert_almost_equal(s2_f64[0], 1.41421)
    assert_almost_equal(s2_f64[1], 1.15470)
    assert_equal(s2_f64[2], 1.0)
    assert_almost_equal(s2_f64[3], 0.89442)


def _test_frexp_impl[type: DType](*, atol: Float32, rtol: Float32):
    var res0 = frexp(Scalar[type](123.45))
    assert_true(
        isclose(res0[0].cast[DType.float32](), 0.964453, atol=atol, rtol=rtol)
    )
    assert_true(
        isclose(res0[1].cast[DType.float32](), 7.0, atol=atol, rtol=rtol)
    )

    var res1 = frexp(Scalar[type](0.1))
    assert_true(
        isclose(res1[0].cast[DType.float32](), 0.8, atol=atol, rtol=rtol)
    )
    assert_true(
        isclose(res1[1].cast[DType.float32](), -3.0, atol=atol, rtol=rtol)
    )

    var res2 = frexp(Scalar[type](-0.1))
    assert_true(
        isclose(res2[0].cast[DType.float32](), -0.8, atol=atol, rtol=rtol)
    )
    assert_true(
        isclose(res2[1].cast[DType.float32](), -3.0, atol=atol, rtol=rtol)
    )

    var res3 = frexp(SIMD[type, 4](0, 2, 4, 5))
    assert_true(
        isclose(
            res3[0].cast[DType.float32](),
            SIMD[DType.float32, 4](0.0, 0.5, 0.5, 0.625),
            atol=atol,
            rtol=rtol,
        ).reduce_and()
    )
    assert_true(
        isclose(
            res3[1].cast[DType.float32](),
            SIMD[DType.float32, 4](-0.0, 2.0, 3.0, 3.0),
            atol=atol,
            rtol=rtol,
        ).reduce_and()
    )


def _test_log_impl[type: DType](*, atol: Float32, rtol: Float32):
    var res0 = log(Scalar[type](123.45))
    assert_true(
        isclose(res0.cast[DType.float32](), 4.8158, atol=atol, rtol=rtol)
    )

    var res1 = log(Scalar[type](0.1))
    assert_true(
        isclose(res1.cast[DType.float32](), -2.3025, atol=atol, rtol=rtol)
    )

    var res2 = log(SIMD[type, 4](1, 2, 4, 5))
    assert_true(
        isclose(
            res2.cast[DType.float32](),
            SIMD[DType.float32, 4](0.0, 0.693147, 1.38629, 1.6094),
            atol=atol,
            rtol=rtol,
        ).reduce_and()
    )

    var res3 = log(Scalar[type](2.7182818284590452353602874713526624977572))
    assert_true(isclose(res3.cast[DType.float32](), 1.0, atol=atol, rtol=rtol))

    var res4 = isinf(log(SIMD[type, 4](0, 1, 0, 0)))
    assert_equal(res4, SIMD[DType.bool, 4](True, False, True, True))


def _test_log2_impl[type: DType](*, atol: Float32, rtol: Float32):
    var res0 = log2(Scalar[type](123.45))
    assert_true(
        isclose(
            res0.cast[DType.float32](), 6.9477, atol=atol, rtol=rtol
        ).reduce_and()
    )

    var res1 = log2(Scalar[type](0.1))
    assert_true(
        isclose(res1.cast[DType.float32](), -3.3219, atol=atol, rtol=rtol)
    )

    var res2 = log2(SIMD[type, 4](1, 2, 4, 5))
    assert_true(
        isclose(
            res2.cast[DType.float32](),
            SIMD[DType.float32, 4](0.0, 1.0, 2.0, 2.3219),
            atol=atol,
            rtol=rtol,
        ).reduce_and()
    )


def test_frexp():
    _test_frexp_impl[DType.float32](atol=1e-4, rtol=1e-5)
    _test_frexp_impl[DType.float16](atol=1e-2, rtol=1e-5)

    # TODO(KERN-228): support BF16 on neon systems.
    @parameter
    if not has_neon():
        _test_frexp_impl[DType.bfloat16](atol=1e-1, rtol=1e-5)


def test_log():
    _test_log_impl[DType.float32](atol=1e-4, rtol=1e-5)
    _test_log_impl[DType.float16](atol=1e-2, rtol=1e-5)

    # TODO(KERN-228): support BF16 on neon systems.
    @parameter
    if not has_neon():
        _test_log_impl[DType.bfloat16](atol=1e-1, rtol=1e-5)


def test_log2():
    _test_log2_impl[DType.float32](atol=1e-4, rtol=1e-5)
    _test_log2_impl[DType.float16](atol=1e-2, rtol=1e-5)

    # TODO(KERN-228): support BF16 on neon systems.
    @parameter
    if not has_neon():
        _test_log2_impl[DType.bfloat16](atol=1e-1, rtol=1e-5)


def main():
    test_sin()
    test_cos()
    test_factorial()
    test_copysign()
    test_isclose()
    test_ceil()
    test_floor()
    test_trunc()
    test_exp2()
    test_iota()
    test_sqrt()
    test_rsqrt()
    test_frexp()
    test_log()
    test_log2()
