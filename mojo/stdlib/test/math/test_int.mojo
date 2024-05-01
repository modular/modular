# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s

from math import divmod

from testing import *


def test_divmod():
    assert_equal(StaticIntTuple[2](-2, 1), divmod(-3, 2))


def main():
    test_divmod()
