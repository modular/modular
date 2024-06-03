# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s

from math import ceil, cos, factorial, floor, isclose, sin, trunc, exp2, iota
from utils.numerics import inf, neg_inf
from sys.info import has_neon
from collections import List

from testing import assert_almost_equal, assert_equal, assert_false, assert_true

from utils.numerics import nan


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
    assert_equal(2, math.copysign(x, x))
    assert_equal(-2, math.copysign(x, -x))
    assert_equal(2, math.copysign(-x, x))
    assert_equal(-2, math.copysign(-x, -x))

    assert_equal(1, math.copysign(Float32(1), Float32(2)))
    assert_equal(-1, math.copysign(Float32(1), Float32(-2)))
    assert_equal(
        neg_inf[DType.float32](), math.copysign(inf[DType.float32](), -2.0)
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
