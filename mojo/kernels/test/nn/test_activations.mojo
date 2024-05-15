# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: linux
# RUN: %mojo  -I%S/.. %s | FileCheck %s

from math import iota
from random import seed

from nn.activations import elu, gelu, gelu_approximate, relu, relu_n1
from tensor import Tensor, TensorShape, randn
from test_utils import libm_call


fn get_minmax[dtype: DType](x: Tensor[dtype], N: Int) -> Tensor[dtype]:
    var max_val = x[0]
    var min_val = x[0]
    for i in range(1, N):
        if x[i] > max_val:
            max_val = x[i]
        if x[i] < min_val:
            min_val = x[i]
    return Tensor[dtype](TensorShape(2), min_val, max_val)


fn compare[_dtype: DType, N: Int](x: Tensor, y: Tensor, label: String):
    var atol = Tensor[_dtype](TensorShape(N))
    var rtol = Tensor[_dtype](TensorShape(N))

    for i in range(N):
        var xx = x[i].cast[_dtype]()
        var yy = y[i].cast[_dtype]()

        var d = abs(xx - yy)
        var e = abs(d / yy)
        atol[i] = d
        rtol[i] = e

    print(label)
    var atol_minmax = get_minmax[_dtype](atol, N)
    var rtol_minmax = get_minmax[_dtype](rtol, N)
    print("AbsErr-Min/Max", atol_minmax[0], atol_minmax[1])
    print("RelErr-Min/Max", rtol_minmax[0], rtol_minmax[1])
    print("==========================================================")


# CHECK-LABEL: test_elu
fn test_elu():
    print("== test_elu")

    var simd_val = iota[DType.float32, 4]()

    # CHECK: [0.0, 1.0, 2.0, 3.0]
    print(elu(simd_val))

    # CHECK: [-0.86466{{[0-9]+}}, -0.63212{{[0-9]+}}, 0.0, 1.0]
    print(elu(simd_val - 2))

    # CHECK: [0.0, 0.5, 1.0, 1.5]
    print(elu(0.5 * simd_val))


# CHECK-LABEL: test_relu
fn test_relu():
    print("== test_relu")

    var simd_val = iota[DType.float32, 4]()

    # CHECK: [0.0, 1.0, 2.0, 3.0]
    print(relu(simd_val))

    # CHECK: [0.0, 0.0, 0.0, 1.0]
    print(relu(simd_val - 2))

    # CHECK: [0.0, 0.5, 1.0, 1.5]
    print(relu(0.5 * simd_val))


# CHECK-LABEL: test_relu_n1
fn test_relu_n1():
    print("== test_relu_n1")

    var simd_val = iota[DType.float32, 4]()

    # CHECK: [0.0, 1.0, 1.0, 1.0]
    print(relu_n1(simd_val))

    # CHECK: [-1.0, -1.0, 0.0, 1.0]
    print(relu_n1(simd_val - 2))

    # CHECK: [0.0, 0.5, 1.0, 1.0]
    print(relu_n1(0.5 * simd_val))


# CHECK-LABEL: test_gelu_float32
fn test_gelu_float32():
    print("== test_gelu_float32")

    var simd_val = 2 - 0.5 * iota[DType.float32, 4]()

    # There is no difference in the results from MLAS and oneDNN gelu.
    # CHECK: [1.95449{{[0-9]+}}, 1.39978{{[0-9]+}}, 0.84134{{[0-9]+}}, 0.34573{{[0-9]+}}]
    print(gelu(simd_val))

    # The results from MLAS gelu is [0.841345, 0.580029, 0.345731, 0.149677].
    # CHECK: [0.84134{{[0-9]+}}, 0.580029{{[0-9]+}}, 0.34573{{[0-9]+}}, 0.14967{{[0-9]+}}]
    print(gelu(0.5 * simd_val))

    # CHECK: [1.95459{{[0-9]+}}, 1.39957{{[0-9]+}}, 0.84119{{[0-9]+}}, 0.34571{{[0-9]+}}]
    print(gelu_approximate(simd_val))

    # CHECK: [0.84119{{[0-9]+}}, 0.57996{{[0-9]+}}, 0.34571{{[0-9]+}}, 0.14967{{[0-9]+}}]
    print(gelu_approximate(0.5 * simd_val))

    # CHECK: 108.523
    print(gelu_approximate(Float32(108.5230)))

    # CHECK: 107.523
    print(gelu_approximate(Float32(107.5230)))


# CHECK-LABEL: test_gelu_float64
fn test_gelu_float64():
    print("== test_gelu_float64")

    var simd_val = 2 - 0.5 * iota[DType.float64, 4]()

    # There is no difference in the results from MLAS and oneDNN gelu.
    # CHECK: [1.95449{{[0-9]+}}, 1.39978{{[0-9]+}}, 0.84134{{[0-9]+}}, 0.34573{{[0-9]+}}]
    print(gelu(simd_val))

    # The results from MLAS gelu is [0.841345, 0.580029, 0.345731, 0.149677].
    # CHECK: [0.84134{{[0-9]+}}, 0.580029{{[0-9]+}}, 0.34573{{[0-9]+}}, 0.14967{{[0-9]+}}]
    print(gelu(0.5 * simd_val))

    # CHECK: [1.95459{{[0-9]+}}, 1.39957{{[0-9]+}}, 0.84119{{[0-9]+}}, 0.34571{{[0-9]+}}]
    print(gelu_approximate(simd_val))

    # CHECK: [0.84119{{[0-9]+}}, 0.57996{{[0-9]+}}, 0.34571{{[0-9]+}}, 0.14967{{[0-9]+}}]
    print(gelu_approximate(0.5 * simd_val))

    # CHECK: 108.5229
    print(gelu_approximate(Float64(108.5230)))

    # CHECK: 107.5229
    print(gelu_approximate(Float64(107.5230)))


@always_inline
fn erf_libm[
    type: DType, simd_width: Int
](arg: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    var eval = libm_call[type, simd_width, "erff", "err"](arg)
    return eval


@always_inline
fn gelu_libm[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Compute the GELU Op using the equation
    $0.5 * x * (1 + erf_libm(x / sqrt(2)))$.

    Parameters:
        type: DType used for the computation.
        simd_width: SIMD width used for the computation.

    Args:
        x: The value to compute the GELU operation on.

    Returns:
        SIMD[type, size]: The result of the GELU operation.

    Constraints:
        Type must be a floating point type.
    """
    alias inv_SQRT_2 = 0.70710678118654752440
    constrained[
        type.is_floating_point(),
        "dtype must be a floating point type",
    ]()
    # 0.5 * x * (1 + erf(x / SQRT_2))
    # x_half + x_half * erf_res
    var x_half = 0.5 * x
    var erf_res = erf_libm(x * inv_SQRT_2)
    return x_half.fma(erf_res, x_half)


# CHECK-LABEL: test_gelu_libm
fn test_gelu_libm():
    print("== test_gelu_libm")
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
        y32[i] = gelu(x32[i])  # gelu using math.erf

    ####################
    ## libm erf result
    ####################
    var libm_out = Tensor[DType.float32](TensorShape(N))
    for i in range(N):
        libm_out[i] = gelu_libm(x32[i])

    # CHECK: Compare Mojo activations.gelu vs. LibM
    # CHECK: AbsErr-Min/Max 0.0 4.76837158203125e-07
    # CHECK: RelErr-Min/Max 0.0 0.035714227706193924
    compare[DType.float32, N](
        y32, libm_out, "Compare Mojo activations.gelu vs. LibM"
    )


fn main():
    test_elu()
    test_relu()
    test_relu_n1()
    test_gelu_float32()
    test_gelu_float64()
    test_gelu_libm()
