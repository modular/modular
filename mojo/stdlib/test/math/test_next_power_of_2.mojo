# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from testing import assert_equal
from math import next_power_of_two


def test_next_power_of_two():
    assert_equal(1, next_power_of_two(1))
    assert_equal(2, next_power_of_two(2))
    assert_equal(4, next_power_of_two(3))
    assert_equal(4, next_power_of_two(4))
    assert_equal(8, next_power_of_two(5))
    assert_equal(8, next_power_of_two(6))
    assert_equal(8, next_power_of_two(7))
    assert_equal(8, next_power_of_two(8))
    assert_equal(16, next_power_of_two(9))
    assert_equal(16, next_power_of_two(10))
    assert_equal(16, next_power_of_two(11))
    assert_equal(16, next_power_of_two(12))
    assert_equal(16, next_power_of_two(13))
    assert_equal(16, next_power_of_two(14))
    assert_equal(16, next_power_of_two(15))
    assert_equal(16, next_power_of_two(16))
    assert_equal(32, next_power_of_two(17))


def main():
    test_next_power_of_two()
