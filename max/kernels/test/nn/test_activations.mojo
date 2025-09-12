# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from math import iota
from random import randn, seed
from sys.info import CompilationTarget

from nn.activations import (
    elu,
    gelu,
    gelu_approximate,
    leaky_relu,
    relu,
    relu_n1,
)
from test_utils import compare, libm_call
from testing import assert_almost_equal


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


# CHECK-LABEL: test_leaky_relu
fn test_leaky_relu():
    print("== test_leaky_relu")

    var simd_val = iota[DType.float32, 4]()

    # Test with negative slope of 0.01
    var slope_001 = SIMD[DType.float32, 1](0.01)

    # CHECK: [0.0, 1.0, 2.0, 3.0]
    print(leaky_relu(simd_val, slope_001))

    # For negative values: [-2, -1, 0, 1] with slope 0.01
    # Expected: [-0.02, -0.01, 0.0, 1.0]
    # CHECK: [-0.02, -0.01, 0.0, 1.0]
    print(leaky_relu(simd_val - 2, slope_001))

    # Test with different slope (0.1)
    var slope_01 = SIMD[DType.float32, 1](0.1)

    # For negative values: [-2, -1, 0, 1] with slope 0.1
    # Expected: [-0.2, -0.1, 0.0, 1.0]
    # CHECK: [-0.2, -0.1, 0.0, 1.0]
    print(leaky_relu(simd_val - 2, slope_01))


def test_gelu_bfloat16():
    # Ground truth values from torch 2.5.1+cu124.
    assert_almost_equal(gelu(BFloat16(-2.6094)), BFloat16(-1.1841e-02))
    assert_almost_equal(
        gelu_approximate(BFloat16(-2.6094)), BFloat16(-1.1353e-02)
    )


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
    dtype: DType, simd_width: Int
](arg: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    return libm_call["erff", "err"](arg)


@always_inline
fn gelu_libm[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    """Compute the GELU Op using the equation
    $0.5 * x * (1 + erf_libm(x / sqrt(2)))$.

    Parameters:
        dtype: DType used for the computation.
        simd_width: SIMD width used for the computation.

    Args:
        x: The value to compute the GELU operation on.

    Returns:
        SIMD[dtype, size]: The result of the GELU operation.

    Constraints:
        Type must be a floating point type.
    """
    alias inv_SQRT_2 = 0.70710678118654752440
    constrained[
        dtype.is_floating_point(),
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
    alias dtype = DType.float32
    alias alignment = 64
    # generate input values and write them to file
    var x32 = UnsafePointer[Scalar[dtype], alignment2=alignment].alloc(N)
    randn[dtype](x32, N, 0, 9.0)
    print("For N=", N, " randomly generated vals; mean=0.0, var=9.0")

    ####################
    # math.erf result
    ####################
    var y32 = UnsafePointer[Scalar[dtype], alignment2=alignment].alloc(N)
    for i in range(N):
        y32[i] = gelu(x32[i])  # gelu using math.erf

    ####################
    ## libm erf result
    ####################
    var libm_out = UnsafePointer[Scalar[dtype], alignment2=alignment].alloc(N)
    for i in range(N):
        libm_out[i] = gelu_libm(x32[i])

    # CHECK: Compare Mojo activations.gelu vs. LibM
    # CHECK: AbsErr-Min/Max 0.0 4.7683716e-07
    # CHECK: RelErr-Min/Max 0.0 0.035714228
    _ = compare(y32, libm_out, N, msg="Compare Mojo activations.gelu vs. LibM")

    x32.free()
    y32.free()
    libm_out.free()


def main():
    test_elu()
    test_relu()
    test_relu_n1()
    test_leaky_relu()
    test_gelu_float32()
    test_gelu_float64()
    test_gelu_libm()

    @parameter
    if not CompilationTarget.has_neon():
        test_gelu_bfloat16()
