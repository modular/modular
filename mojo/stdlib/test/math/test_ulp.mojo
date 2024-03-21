# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s

from math import isinf, isnan, nan, ulp
from math.limit import inf, max_finite

from testing import *


def test_ulp():
    assert_true(isnan(ulp(nan[DType.float32]())))

    assert_true(isinf(ulp(inf[DType.float32]())))

    assert_true(isinf(ulp(-inf[DType.float32]())))

    assert_almost_equal(ulp(Float64(0)), 5e-324)

    assert_equal(ulp(max_finite[DType.float64]()), 1.99584030953472e292)

    assert_equal(ulp(Float64(5)), 8.881784197001252e-16)

    assert_equal(ulp(Float64(-5)), 8.881784197001252e-16)


def main():
    test_ulp()
