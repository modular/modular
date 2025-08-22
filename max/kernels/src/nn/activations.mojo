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

"""The module contains implementations of activation functions."""

import math

from utils.numerics import get_accum_type

# ===----------------------------------------------------------------------=== #
# sign
# ===----------------------------------------------------------------------=== #


@always_inline("nodebug")
fn _is_neg[
    dtype: DType, simd_width: Int
](val: SIMD[dtype, simd_width]) -> SIMD[DType.bool, simd_width]:
    """Returns True if the input value is negative.

    The value is computed separately for each element in the SIMD vector. For
    unsigned dtypes the result is always a SIMD vector filled with False.

    Parameters:
        dtype: dtype used for the computation.
        simd_width: SIMD width used for the computation.

    Args:
        val: The value to check.

    Returns:
        A SIMD value where the element at position `i` is True if the value is
        negative at position `i` and False otherwise.
    """

    @parameter
    if dtype.is_unsigned():
        return SIMD[DType.bool, simd_width](fill=False)
    return val.lt(0)


@always_inline
fn sign[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    """Compute the sign (0, 1) of the input value.

    Parameters:
        dtype: DType used for the computation.
        simd_width: SIMD width used for the computation.

    Args:
        x : The value to compute the sign operation on.

    Returns:
        The result of the sign operation.
    """
    var is_neg_mask = _is_neg(x)
    var is_zero_mask = x.eq(0)
    return is_neg_mask.select[dtype](-1, is_zero_mask.select[dtype](0, 1))


# ===----------------------------------------------------------------------=== #
# elu
# ===----------------------------------------------------------------------=== #


@always_inline
fn elu[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    """Compute the Elu Op using the equation $z if z >= 0 else alpha*(e^z -1)$.

    Parameters:
        dtype: DType used for the computation.
        simd_width: SIMD width used for the computation.

    Args:
        x: The value to compute the ELU operation on.

    Returns:
        The result of the ELU operation.
    """
    return x.ge(0).select(x, math.expm1(x))


# ===----------------------------------------------------------------------=== #
# relu
# ===----------------------------------------------------------------------=== #


@always_inline
fn relu[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    """Compute the Relu Op using the equation $max(0, x)$.

    Parameters:
        dtype: DType used for the computation.
        simd_width: SIMD width used for the computation.

    Args:
        x : The value to compute the RELU operation on.

    Returns:
        The result of the RELU operation.
    """
    return max(x, 0)


# ===----------------------------------------------------------------------=== #
# relu-n1
# ===----------------------------------------------------------------------=== #


@always_inline
fn relu_n1[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    """Compute the Relu N1 Op using the equation $max(min(x,1),-1)$.

    Parameters:
        dtype: DType used for the computation.
        simd_width: SIMD width used for the computation.

    Args:
        x : The value to compute the RELU N1 operation on.

    Returns:
        The result of the RELU N1 operation.
    """
    return x.clamp(-1, 1)


# ===----------------------------------------------------------------------=== #
# gelu
# ===----------------------------------------------------------------------=== #


@always_inline
fn gelu[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    """Compute the GELU Op using the equation
    $0.5 * x * (1 + erf(x / sqrt(2)))$.

    Parameters:
        dtype: DType used for the computation.
        simd_width: SIMD width used for the computation.

    Args:
        x: The value to compute the GELU operation on.

    Returns:
        The result of the GELU operation.

    Constraints:
        Type must be a floating point dtype.
    """
    # Perform the intermediate computation in `accum_type` to match
    # torch.nn.functional.gelu:
    # https://github.com/pytorch/pytorch/blob/3054aae493a5347cf8187b5ce611b9a38aace202/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L21-L42
    alias accum_type = get_accum_type[dtype]()
    alias inv_SQRT_2 = 0.70710678118654752440
    constrained[
        dtype.is_floating_point(),
        "dtype must be a floating point dtype",
    ]()

    var val = x.cast[accum_type]()
    var val_half = 0.5 * val
    var erf_res = math.erf(val * inv_SQRT_2)
    return val_half.fma(erf_res, val_half).cast[dtype]()


# ===----------------------------------------------------------------------=== #
# gelu_approximate
# ===----------------------------------------------------------------------=== #


@always_inline
fn gelu_approximate[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    """Compute the approximate GELU Op using the equation
    $0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))$.

    Parameters:
        dtype: The `DType` used for the computation.
        simd_width: SIMD width used for the computation.

    Args:
        x: The value to compute the GELU operation on.

    Constraints:
        Type must be a floating point dtype.

    Returns:
        The result of the approximate GELU operation.
    """
    # Perform the intermediate computation in `accum_type` to match
    # torch.nn.functional.gelu:
    # https://github.com/pytorch/pytorch/blob/3054aae493a5347cf8187b5ce611b9a38aace202/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L21-L42
    alias accum_type = get_accum_type[dtype]()
    alias SQRT_TWO_OVER_PI = 0.797884560802865
    constrained[
        dtype.is_floating_point(),
        "dtype must be a floating point dtype",
    ]()

    var val = x.cast[accum_type]()

    var val3 = val * val * val
    return (
        0.5 * val * (1 + math.tanh(SQRT_TWO_OVER_PI * (val + 0.044715 * val3)))
    ).cast[dtype]()


# ===----------------------------------------------------------------------=== #
# leaky_relu
# ===----------------------------------------------------------------------=== #


@always_inline
fn leaky_relu[
    dtype: DType, simd_width: Int
](x: SIMD[dtype, simd_width], negative_slope: SIMD[dtype, 1]) -> SIMD[
    dtype, simd_width
]:
    """Compute the Leaky ReLU using the equation
    $max(0, x) + negative_slope * min(0, x)$.

    Parameters:
        dtype: DType used for the computation.
        simd_width: SIMD width used for the computation.

    Args:
        x: The value to compute the Leaky ReLU operation on.
        negative_slope: The slope for negative values.

    Constraints:
        Type must be a floating point Dtype.

    Returns:
        The result of the Leaky ReLU operation.
    """
    constrained[
        dtype.is_floating_point(),
        "dtype must be a floating point dtype",
    ]()
    return x.ge(0).select(x, negative_slope * x)
