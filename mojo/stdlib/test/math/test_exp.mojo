# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from math import exp
from random import randn_float64, seed
from sys import has_neon

from test_utils import libm_call
from testing import assert_almost_equal, assert_equal


def test_exp_bfloat16():
    # TODO(KERN-228): support BF16 on neon systems.
    @parameter
    if not has_neon():
        assert_equal(exp(BFloat16(2.0)), 7.375)


def test_exp_float16():
    assert_almost_equal(exp(Float16(-0.1)), 0.9047)
    assert_almost_equal(exp(Float16(0.1)), 1.105)
    assert_almost_equal(exp(Float16(2)), 7.389)
    assert_equal(String(exp(Float16(89))), "inf")
    assert_equal(String(exp(Float16(108.5230))), "inf")


def test_exp_float32():
    assert_almost_equal(exp(Float32(-0.1)), 0.90483)
    assert_almost_equal(exp(Float32(0.1)), 1.10517)
    assert_almost_equal(exp(Float32(2)), 7.38905)
    assert_equal(String(exp(Float32(89))), "inf")
    assert_equal(String(exp(Float32(108.5230))), "inf")


def test_exp_float64():
    assert_almost_equal(exp(Float64(-0.1)), 0.90483)
    assert_almost_equal(exp(Float64(0.1)), 1.10517)
    assert_almost_equal(exp(Float64(2)), 7.38905)
    # FIXME (40568) should remove str
    assert_equal(String(exp(Float64(89))), String(4.4896128193366053e38))
    assert_equal(String(exp(Float64(108.5230))), String(1.3518859659123633e47))


@always_inline
def exp_libm[
    type: DType, simd_width: Int
](arg: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    var eval = libm_call[type, simd_width, "expf", "exp"](arg)
    return eval


def test_exp_libm[type: DType]():
    seed(0)
    alias N = 8192
    for _i in range(N):
        var x = randn_float64(0, 9.0).cast[type]()
        assert_almost_equal(
            exp(x), exp_libm(x), msg=String("for the input ", x)
        )


def main():
    test_exp_bfloat16()
    test_exp_float16()
    test_exp_float32()
    test_exp_float64()
    test_exp_libm[DType.float32]()
    test_exp_libm[DType.float64]()
