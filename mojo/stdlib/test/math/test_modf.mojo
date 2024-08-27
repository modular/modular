# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s

from math import modf

from test_utils import libm_call
from testing import assert_almost_equal, assert_equal


def test_modf():
    var i32 = modf(Int32(123))
    assert_equal(i32[0], 123)
    assert_equal(i32[1], 0)

    var f32 = modf(Float32(123.5))
    assert_almost_equal(f32[0], 123)
    assert_almost_equal(f32[1], 0.5)

    var f64 = modf(Float64(123.5))
    assert_almost_equal(f64[0], 123)
    assert_almost_equal(f64[1], 0.5)

    f64 = modf(Float64(0))
    assert_almost_equal(f64[0], 0)
    assert_almost_equal(f64[1], 0)

    f64 = modf(Float64(0.5))
    assert_almost_equal(f64[0], 0)
    assert_almost_equal(f64[1], 0.5)

    f64 = modf(Float64(-0.5))
    assert_almost_equal(f64[0], -0)
    assert_almost_equal(f64[1], -0.5)

    f64 = modf(Float64(-1.5))
    assert_almost_equal(f64[0], -1)
    assert_almost_equal(f64[1], -0.5)


def main():
    test_modf()
