# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: linux
# RUN: %mojo  -I%S/../ %s | FileCheck %s

from math import tanh
from random import seed, randn
from buffer import Buffer
from test_utils import libm_call
from utils import InlineArray

alias alignment = 64


fn get_minmax[
    dtype: DType
](x: DTypePointer[dtype], N: Int) -> InlineArray[Scalar[dtype], 2]:
    var max_val = x[0]
    var min_val = x[0]
    for i in range(1, N):
        if x[i] > max_val:
            max_val = x[i]
        if x[i] < min_val:
            min_val = x[i]
    return InlineArray[Scalar[dtype], 2](min_val, max_val)


fn compare[
    dtype: DType, N: Int
](x: DTypePointer[dtype], y: DTypePointer[dtype], label: String):
    var atol = DTypePointer[dtype].alloc(N, alignment=alignment)
    var rtol = DTypePointer[dtype].alloc(N, alignment=alignment)

    for i in range(N):
        var xx = x[i].cast[dtype]()
        var yy = y[i].cast[dtype]()

        var d = abs(xx - yy)
        var e = abs(d / yy)
        atol[i] = d
        rtol[i] = e

    print(label)
    var atol_minmax = get_minmax[dtype](atol, N)
    var rtol_minmax = get_minmax[dtype](rtol, N)
    print("AbsErr-Min/Max", atol_minmax[0], atol_minmax[1])
    print("RelErr-Min/Max", rtol_minmax[0], rtol_minmax[1])
    print("==========================================================")
    DTypePointer[dtype].free(atol)
    DTypePointer[dtype].free(rtol)


fn tanh_libm[
    type: DType, simd_width: Int
](arg: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    return libm_call[type, simd_width, "tanhf", "tanh"](arg)


# CHECK-LABEL: test_tanh_tfvals_fp32
fn test_tanh_tfvals_fp32() raises:
    print("== test_tanh_tfvals_fp32")
    alias dtype = DType.float32

    # The following input values for x are taken from
    # https://github.com/modularml/modular/issues/28981#issuecomment-1890182667
    var x = Buffer[dtype, 4].stack_allocation()
    x.store[width=4](
        0,
        SIMD[dtype, 4](
            -1.2583316564559937,
            -8.081921577453613,
            -8.626264572143555,
            -0.7127348184585571,
        ),
    )

    var y = Buffer[dtype, 4].stack_allocation()
    for i in range(4):
        y[i] = tanh(x[i])

    #################################################
    # TF results
    # use `tf.print(tf.math.tanh(numpy.float32(x)))`
    var tfvals_fp32 = Buffer[dtype, 4].stack_allocation()
    tfvals_fp32.store[width=4](
        0, SIMD[dtype, 4](-0.850603521, -1, -1, -0.612388909)
    )

    # CHECK: AbsErr-Min/Max 0.0 1.1920928955078125e-07
    # CHECK: RelErr-Min/Max 0.0 1.1920928955078125e-07
    compare[dtype, 4](
        y.data, tfvals_fp32.data, "Compare Mojo vs. Tensorflow FP32"
    )


# CHECK-LABEL: test_tanh_tfvals_fp64
fn test_tanh_tfvals_fp64() raises:
    print("== test_tanh_tfvals_fp64")
    alias dtype = DType.float64

    # The following input values for x are taken from
    # https://github.com/modularml/modular/issues/28981#issuecomment-1890182667
    var x = Buffer[dtype, 4].stack_allocation()
    x.store[width=4](
        0,
        SIMD[dtype, 4](
            -1.2583316564559937,
            -8.081921577453613,
            -8.626264572143555,
            -0.7127348184585571,
        ),
    )

    var y = Buffer[dtype, 4].stack_allocation()
    for i in range(4):
        y[i] = tanh(x[i])

    #################################################
    # TF results
    # use `tf.print(tf.math.tanh(numpy.float64(x)))`
    var tfvals_fp64 = Buffer[dtype, 4].stack_allocation()
    tfvals_fp64.store[width=4](
        0,
        SIMD[dtype, 4](
            -0.85060351067231821,
            -0.99999980894339091,
            -0.99999993567914991,
            -0.61238890225714893,
        ),
    )

    # CHECK: AbsErr-Min/Max 7.2062200651146213e-09 1.2149700800989649e-08
    # CHECK: RelErr-Min/Max 8.3577847290501252e-09 1.4283624095774667e-08
    compare[dtype, 4](
        y.data, tfvals_fp64.data, "Compare Mojo vs. Tensorflow FP64"
    )


# CHECK-LABEL: test_tanh_libm
# CHECK: For N=8192 randomly generated vals; mean=0.0, var=9.0
fn test_tanh_libm[N: Int = 8192]() raises:
    print("== test_tanh_libm")
    seed(0)
    alias test_dtype = DType.float32
    var x32 = DTypePointer[test_dtype].alloc(N, alignment=alignment)
    randn[test_dtype](x32, N, 0, 9.0)
    print("For N=" + String(N) + " randomly generated vals; mean=0.0, var=9.0")

    ####################
    # mojo tanh result
    ####################
    var y32 = DTypePointer[test_dtype].alloc(N, alignment=alignment)
    for i in range(N):
        y32[i] = tanh(x32[i])

    ####################
    ## libm tanh result
    ####################
    var libm_out = DTypePointer[test_dtype].alloc(N, alignment=alignment)
    for i in range(N):
        libm_out[i] = tanh_libm(x32[i])

    # CHECK: Compare Mojo vs. LibM
    # CHECK: AbsErr-Min/Max 0.0 2.384185791015625e-07
    # CHECK: RelErr-Min/Max 0.0 2.5438197326366208e-07
    compare[test_dtype, N](y32, libm_out, "Compare Mojo vs. LibM")
    DTypePointer[test_dtype].free(x32)
    DTypePointer[test_dtype].free(y32)
    DTypePointer[test_dtype].free(libm_out)


fn main() raises:
    test_tanh_tfvals_fp32()
    test_tanh_tfvals_fp64()
    test_tanh_libm()
