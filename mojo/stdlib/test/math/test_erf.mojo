# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: linux
# RUN: %mojo %s


from math import erf
from random import randn, seed
from testing import assert_almost_equal

from internal_utils import compare
from test_utils import libm_call

from utils import InlineArray

alias alignment = 64


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
def test_erf_libm():
    print("== test_erf_libm")
    seed(0)
    var N = 8192
    alias test_dtype = DType.float32

    # generate input values and write them to file
    var x32 = DTypePointer[test_dtype].alloc(N)
    randn[test_dtype](x32, N, 0, 9.0)
    print("For N=" + str(N) + " randomly generated vals; mean=0.0, var=9.0")

    ####################
    # math.erf result
    ####################
    var y32 = DTypePointer[test_dtype].alloc(N)
    for i in range(N):
        y32[i] = erf(x32[i])  # math.erf

    ####################
    ## libm erf result
    ####################
    @always_inline
    fn erf_libm[
        type: DType, simd_width: Int
    ](arg: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
        return libm_call[type, simd_width, "erff", "err"](arg)

    var libm_out = DTypePointer[test_dtype].alloc(N)
    for i in range(N):
        libm_out[i] = erf_libm(x32[i])

    # CHECK: Compare Mojo math.erf vs. LibM
    # CHECK: AbsErr-Min/Max 0.0 5.9604644775390625e-08
    # CHECK: RelErr-Min/Max 0.0 1.172195140952681e-07

    # abs_rel_err = (abs_min, abs_max, rel_min, rel_max)
    var abs_rel_err = SIMD[test_dtype, 4](
        0.0, 5.9604644775390625e-08, 0.0, 1.172195140952681e-07
    )

    var err = compare[test_dtype](
        y32, libm_out, N, msg="Compare Mojo math.erf vs. LibM"
    )

    assert_almost_equal(err, abs_rel_err)

    x32.free()
    y32.free()
    libm_out.free()


def main():
    test_erf_float32()
    test_erf_float64()
    test_erf_libm()
