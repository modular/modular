# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s

from testing import *
from utils import StaticTuple, unroll


alias dtype = DType.float32
alias simd_width = 1
alias coefficients_len = 7
alias coefficients = StaticTuple[SIMD[dtype, simd_width], coefficients_len](
    4.89352455891786e-03,
    6.37261928875436e-04,
    1.48572235717979e-05,
    5.12229709037114e-08,
    -8.60467152213735e-11,
    2.00018790482477e-13,
    -2.76076847742355e-16,
)


@always_inline
fn eval1(x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    var c_last = coefficients[coefficients_len - 1]
    var c_second_from_last = coefficients[coefficients_len - 2]

    var result = x.fma(c_last, c_second_from_last)

    @unroll
    for idx in range(coefficients_len - 2):
        var c = coefficients[coefficients_len - 3 - idx]
        result = x.fma(result, c)

    return result


@always_inline
fn eval2(x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    var c_last = coefficients[coefficients_len - 1]
    var c_second_from_last = coefficients[coefficients_len - 2]

    var result = x.fma(c_last, c_second_from_last)

    for idx in range(coefficients_len - 2):
        var c = coefficients[coefficients_len - 3 - idx]
        result = x.fma(result, c)

    return result


def main():
    var x = 6.0
    var x2 = x * x
    var result1 = eval1(x2)
    var result2 = eval2(x2)

    assert_equal(result1, result2)
