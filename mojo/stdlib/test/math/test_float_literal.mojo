# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s

from math import round

from testing import *

alias nan = FloatLiteral.nan
alias neg_zero = FloatLiteral.negative_zero


def test_division():
    # TODO: https://github.com/modularml/mojo/issues/1787
    # allow this at compile time
    assert_equal(FloatLiteral(4.4) / 0.5, 8.8)
    assert_equal(FloatLiteral(4.4) // 0.5, 8.0)
    assert_equal(FloatLiteral(-4.4) // 0.5, -9.0)
    assert_equal(FloatLiteral(4.4) // -0.5, -9.0)
    assert_equal(FloatLiteral(-4.4) // -0.5, 8.0)


fn round10(x: Float64) -> Float64:
    return (round(Float64(x * 10)) / 10).value


def test_round10():
    assert_equal(round10(FloatLiteral(4.4) % 0.5), 0.4)
    assert_equal(round10(FloatLiteral(-4.4) % 0.5), 0.1)
    assert_equal(round10(FloatLiteral(4.4) % -0.5), -0.1)
    assert_equal(round10(FloatLiteral(-4.4) % -0.5), -0.4)
    assert_equal(round10(3.1 % 1.0), 0.1)


def test_power():
    assert_almost_equal(FloatLiteral(4.5) ** 2.5, 42.95673695)
    assert_almost_equal(FloatLiteral(4.5) ** -2.5, 0.023279235)
    # TODO (https://github.com/modularml/modular/issues/33045): Float64/SIMD has
    # issues with negative numbers raised to fractional powers.
    # assert_almost_equal(FloatLiteral(-4.5) ** 2.5, -42.95673695)
    # assert_almost_equal(FloatLiteral(-4.5) ** -2.5, -0.023279235)


def test_int_conversion():
    assert_equal(int(FloatLiteral(-4.0)), -4)
    assert_equal(int(FloatLiteral(-4.5)), -4)
    assert_equal(int(FloatLiteral(-4.3)), -4)
    assert_equal(int(FloatLiteral(4.5)), 4)
    assert_equal(int(FloatLiteral(4.0)), 4)


def test_boolean_comparable():
    var f1 = FloatLiteral(0.0)
    assert_false(f1)

    var f2 = FloatLiteral(2.0)
    assert_true(f2)

    var f3 = FloatLiteral(1.0)
    assert_true(f2)


def test_equality():
    var f1 = FloatLiteral(4.4)
    var f2 = FloatLiteral(4.4)
    var f3 = FloatLiteral(42.0)
    assert_equal(f1, f2)
    assert_not_equal(f1, f3)


def test_is_special_value():
    assert_true(nan.is_nan())
    assert_false(neg_zero.is_nan())
    assert_true(neg_zero.is_neg_zero())
    assert_false(nan.is_neg_zero())


def main():
    test_division()
    test_round10()
    test_power()
    test_int_conversion()
    test_boolean_comparable()
    test_equality()
    test_is_special_value()
