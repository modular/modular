# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: linux
# RUN: %mojo -debug-level full -I%S/../.. %s | FileCheck %s


from math import erf
from random import seed

from tensor import Tensor, TensorShape, randn
from test_utils import compare, get_minmax, libm_call


# CHECK-LABEL: test_erf_float32
fn test_erf_float32():
    print("== test_erf_float32")

    # Test MLAS erf.

    # CHECK: 0.0
    print(erf(Float32(0)))

    # CHECK: 0.995322
    # CHECK: 0.995322
    print(erf(SIMD[DType.float32, 2](2)))

    # CHECK: 0.11246
    print(erf(Float32(0.1)))

    # CHECK: -0.11246
    print(erf(Float32(-0.1)))

    # CHECK: -0.8427007
    print(erf(Float32(-1)))

    # CHECK: -0.995322
    print(erf(Float32(-2)))


# CHECK-LABEL: test_erf_float64
fn test_erf_float64():
    print("== test_erf_float64")

    # Test MLAS erf.

    # CHECK: 0.0
    print(erf(Float64(0)))

    # CHECK: 0.995322
    # CHECK: 0.995322
    print(erf(SIMD[DType.float64, 2](2)))

    # CHECK: 0.112462
    print(erf(Float64(0.1)))

    # CHECK: -0.112462
    print(erf(Float64(-0.1)))

    # CHECK: -0.8427007
    print(erf(Float64(-1)))

    # CHECK: -0.995322
    print(erf(Float64(-2)))


# CHECK-LABEL: test_erf_libm
fn test_erf_libm():
    print("== test_erf_libm")
    seed(0)
    alias N = 8192
    # generate input values and write them to file
    var x32 = randn[DType.float32](N, 0, 9.0)
    print("For N=" + String(N) + " randomly generated vals; mean=0.0, var=9.0")

    ####################
    # math.erf result
    ####################
    var y32 = Tensor[DType.float32](TensorShape(N))
    for i in range(N):
        y32[i] = erf(x32[i])  # math.erf

    ####################
    ## libm erf result
    ####################
    @always_inline
    fn erf_libm[
        type: DType, simd_width: Int
    ](arg: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
        var eval = libm_call[type, simd_width, "erff", "err"](arg)
        return eval

    var libm_out = Tensor[DType.float32](TensorShape(N))
    for i in range(N):
        libm_out[i] = erf_libm(x32[i])

    # CHECK: Compare Mojo math.erf vs. LibM
    # CHECK: AbsErr-Min/Max 0.0 5.9604644775390625e-08
    # CHECK: RelErr-Min/Max 0.0 1.172195140952681e-07
    compare[DType.float32, N](y32, libm_out, "Compare Mojo math.erf vs. LibM")


fn main():
    test_erf_float32()
    test_erf_float64()
    test_erf_libm()
