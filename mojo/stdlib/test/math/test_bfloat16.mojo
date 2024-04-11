# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s

from math import ceil, cos, exp, floor

from sys.info import has_neon

from testing import *


def test_math():
    assert_equal(exp(BFloat16(2.0)), 7.375)
    assert_equal(cos(BFloat16(2.0)), -0.416015625)

    assert_equal(floor(BFloat16(2.0)), 2.0)
    assert_equal(ceil(BFloat16(2.0)), 2.0)


def main():
    # TODO re-enable this test when we sort out BF16 support for graviton3 #30525
    @parameter
    if not has_neon():
        test_math()
