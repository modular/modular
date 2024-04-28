# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s


from complex import ComplexFloat32, abs

from testing import assert_equal, assert_almost_equal


def test_abs():
    assert_equal(abs(ComplexFloat32 {re: 0, im: 0}), 0)
    assert_equal(abs(ComplexFloat32 {re: 1, im: 0}), 1)
    assert_equal(abs(ComplexFloat32 {re: 0, im: 1}), 1)
    assert_almost_equal(abs(ComplexFloat32 {re: -1, im: -1}), 1.41421)
    assert_almost_equal(abs(ComplexFloat32 {re: -93, im: -23}), 95.801)


def main():
    test_abs()
