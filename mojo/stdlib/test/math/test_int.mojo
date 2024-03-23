# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s

from math import divmod, max, min

from testing import *


def test_int():
    var a = 0
    var b = a + Int(1)
    assert_equal(a, min(a, b))
    assert_equal(b, max(a, b))


def test_divmod():
    assert_equal(StaticIntTuple[2](-2, 1), divmod(-3, 2))


def main():
    test_int()
    test_divmod()
