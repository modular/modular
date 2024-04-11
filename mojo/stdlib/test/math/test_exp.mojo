# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s

from math import exp
from random import seed

from tensor import Tensor, TensorShape, randn
from test_utils import libm_call
from testing import *


# CHECK-LABEL: test_exp_float16
def test_exp_float16():
    print("== test_exp_float16")
    assert_almost_equal(exp(Float16(-0.1)), 0.9047)
    assert_almost_equal(exp(Float16(0.1)), 1.105)
    assert_almost_equal(exp(Float16(2)), 7.389)
    assert_equal(str(exp(Float16(89))), "inf")
    assert_equal(str(exp(Float16(108.5230))), "inf")


# CHECK-LABEL: test_exp_float32
def test_exp_float32():
    print("== test_exp_float32")
    assert_almost_equal(exp(Float32(-0.1)), 0.90483)
    assert_almost_equal(exp(Float32(0.1)), 1.10517)
    assert_almost_equal(exp(Float32(2)), 7.38905)
    assert_equal(str(exp(Float32(89))), "inf")
    assert_equal(str(exp(Float32(108.5230))), "inf")


# CHECK-LABEL: test_exp_float64
def test_exp_float64():
    print("== test_exp_float64")
    assert_almost_equal(exp(Float64(-0.1)), 0.90483)
    assert_almost_equal(exp(Float64(0.1)), 1.10517)
    assert_almost_equal(exp(Float64(2)), 7.38905)
    assert_equal(str(exp(Float64(89))), 4.4896128193366053e38)
    assert_equal(str(exp(Float64(108.5230))), 1.3518859659123633e47)


@always_inline
def exp_libm[
    type: DType, simd_width: Int
](arg: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    var eval = libm_call[type, simd_width, "expf", "exp"](arg)
    return eval


def test_exp_libm[type: DType]():
    seed(0)
    alias N = 8192
    var x = randn[type](N, 0, 9.0)

    for i in range(N):
        assert_almost_equal(
            exp(x[i]), exp_libm(x[i]), msg="for the input " + str(x[i])
        )


def main():
    test_exp_float16()
    test_exp_float32()
    test_exp_float64()
    test_exp_libm[DType.float32]()
    test_exp_libm[DType.float64]()
