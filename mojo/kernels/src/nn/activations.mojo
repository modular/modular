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

from register import register_internal

from utils.numerics import get_accum_type

# ===----------------------------------------------------------------------=== #
# sign
# ===----------------------------------------------------------------------=== #


@always_inline("nodebug")
fn _is_neg[
    type: DType, simd_width: Int
](val: SIMD[type, simd_width]) -> SIMD[DType.bool, simd_width]:
    """Returns True if the input value is negative.

    The value is computed separately for each element in the SIMD vector. For
    unsigned types the result is always a SIMD vector filled with False.

    Parameters:
        type: dtype used for the computation.
        simd_width: SIMD width used for the computation.

    Args:
        val: The value to check.

    Returns:
        A SIMD value where the element at position `i` is True if the value is
        negative at position `i` and False otherwise.
    """

    @parameter
    if type.is_unsigned():
        return False
    return val < 0


@always_inline
fn sign[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Compute the sign (0, 1) of the input value.

    Parameters:
        type: DType used for the computation.
        simd_width: SIMD width used for the computation.

    Args:
        x : The value to compute the sign operation on.

    Returns:
        The result of the sign operation.
    """
    var is_neg_mask = _is_neg(x)
    var is_zero_mask = x == 0
    return is_neg_mask.select[type](-1, is_zero_mask.select[type](0, 1))


# ===----------------------------------------------------------------------=== #
# elu
# ===----------------------------------------------------------------------=== #


@always_inline
fn elu[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Compute the Elu Op using the equation $z if z >= 0 else alpha*(e^z -1)$.

    Parameters:
        type: DType used for the computation.
        simd_width: SIMD width used for the computation.

    Args:
        x: The value to compute the ELU operation on.

    Returns:
        The result of the ELU operation.
    """
    return (x >= 0).select(x, math.expm1(x))


# ===----------------------------------------------------------------------=== #
# relu
# ===----------------------------------------------------------------------=== #


@register_internal("mo.relu")
@always_inline
fn relu[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Compute the Relu Op using the equation $max(0, x)$.

    Parameters:
        type: DType used for the computation.
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
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Compute the Relu N1 Op using the equation $max(min(x,1),-1)$.

    Parameters:
        type: DType used for the computation.
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


@register_internal("mo.gelu")
@always_inline
fn gelu[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Compute the GELU Op using the equation
    $0.5 * x * (1 + erf(x / sqrt(2)))$.

    Parameters:
        type: DType used for the computation.
        simd_width: SIMD width used for the computation.

    Args:
        x: The value to compute the GELU operation on.

    Returns:
        The result of the GELU operation.

    Constraints:
        Type must be a floating point type.
    """
    # Perform the intermediate computation in `accum_type` to match
    # torch.nn.functional.gelu:
    # https://github.com/pytorch/pytorch/blob/3054aae493a5347cf8187b5ce611b9a38aace202/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L21-L42
    alias accum_type = get_accum_type[type]()
    alias inv_SQRT_2 = 0.70710678118654752440
    constrained[
        type.is_floating_point(),
        "dtype must be a floating point type",
    ]()

    var val = x.cast[accum_type]()
    var val_half = 0.5 * val
    var erf_res = math.erf(val * inv_SQRT_2)
    return val_half.fma(erf_res, val_half).cast[type]()


# ===----------------------------------------------------------------------=== #
# gelu_approximate
# ===----------------------------------------------------------------------=== #


@always_inline
fn gelu_approximate[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Compute the approximate GELU Op using the equation
    $0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))$.

    Parameters:
        type: The `DType` used for the computation.
        simd_width: SIMD width used for the computation.

    Args:
        x: The value to compute the GELU operation on.

    Constraints:
        Type must be a floating point type.

    Returns:
        The result of the approximate GELU operation.
    """
    # Perform the intermediate computation in `accum_type` to match
    # torch.nn.functional.gelu:
    # https://github.com/pytorch/pytorch/blob/3054aae493a5347cf8187b5ce611b9a38aace202/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L21-L42
    alias accum_type = get_accum_type[type]()
    alias SQRT_TWO_OVER_PI = 0.797884560802865
    constrained[
        type.is_floating_point(),
        "dtype must be a floating point type",
    ]()

    var val = x.cast[accum_type]()

    var val3 = val * val * val
    return (
        0.5 * val * (1 + math.tanh(SQRT_TWO_OVER_PI * (val + 0.044715 * val3)))
    ).cast[type]()
