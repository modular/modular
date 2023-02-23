# ===----------------------------------------------------------------------===#
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------===#
"""The module contains implementations of activation functions."""

from Assert import assert_param_bool_msg
from DType import DType
from Math import erf, exp, tanh
from SIMD import SIMD

# ===----------------------------------------------------------------------===#
# relu
# ===----------------------------------------------------------------------===#


fn relu[
    simd_width: __mlir_type.index, type: __mlir_type.`!kgen.dtype`
](x: SIMD[simd_width, type]) -> SIMD[simd_width, type]:
    """Compute the Relu Op using the equation $max(0, x)$.

    Parameters:
        simd_width: SIMD width used for the computation.
        type: dtype used for the computation.

    Args:
        x (SIMD[simd_width, type]): The value to compute the RELU operation on.

    Returns:
        SIMD[simd_width, type]: The result of the RELU operation.
    """
    return x.max(0)


# ===----------------------------------------------------------------------===#
# prelu
# ===----------------------------------------------------------------------===#


fn prelu[
    simd_width: __mlir_type.index, type: __mlir_type.`!kgen.dtype`
](x: SIMD[simd_width, type], alpha: SIMD[1, type]) -> SIMD[simd_width, type]:
    """Compute the Prelu Op using the equation $max(x,0) + alpha * min(x,0)$.

    Parameters:
        simd_width: SIMD width used for the computation.
        type: dtype used for the computation.

    Args:
        x (SIMD[simd_width, type]): The value to compute the PRELU operation on.

    Returns:
        SIMD[simd_width, type]: The result of the PRELU operation.
    """
    return x.max(0) + SIMD[simd_width, type].splat(alpha) * x.min(0)


# ===----------------------------------------------------------------------===#
# relu-n1
# ===----------------------------------------------------------------------===#


fn relu_n1[
    simd_width: __mlir_type.index, type: __mlir_type.`!kgen.dtype`
](x: SIMD[simd_width, type]) -> SIMD[simd_width, type]:
    """Compute the Relu N1 Op using the equation $max(min(x,1),-1)$.

    Parameters:
        simd_width: SIMD width used for the computation.
        type: dtype used for the computation.

    Args:
        x (SIMD[simd_width, type]): The value to compute the RELU N1 operation on.

    Returns:
        SIMD[simd_width, type]: The result of the RELU N1 operation.
    """
    return x.min(1).max(-1)


# ===----------------------------------------------------------------------===#
# gelu
# ===----------------------------------------------------------------------===#


fn gelu[
    simd_width: __mlir_type.index, type: __mlir_type.`!kgen.dtype`
](x: SIMD[simd_width, type]) -> SIMD[simd_width, type]:
    """Compute the GELU Op using the equation
    $0.5 * x * (1 + erf(x / sqrt(2)))$.

    Parameters:
        simd_width: SIMD width used for the computation.
        type: dtype used for the computation.

    Args:
        x (SIMD[size, type]): The value to compute the GELU operation on.

    Returns:
        SIMD[size, type]: The result of the GELU operation.

    Constraints:
        type must be a floating point type.
    """
    alias SQRT_2 = 1.4142135623730950488
    assert_param_bool_msg[
        DType(type).is_floating_point(),
        "dtype must be a floating point type",
    ]()
    return 0.5 * x * (1 + erf(x / SQRT_2))


# ===----------------------------------------------------------------------===#
# gelu_approximate
# ===----------------------------------------------------------------------===#


fn gelu_approximate[
    simd_width: __mlir_type.index, type: __mlir_type.`!kgen.dtype`
](x: SIMD[simd_width, type]) -> SIMD[simd_width, type]:
    """Compute the approximate GELU Op using the equation
    $0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))$.

    Parameters:
        simd_width: SIMD width used for the computation.
        type: dtype used for the computation.

    Args:
        x (SIMD[size, type]): The value to compute the GELU operation on.

    Returns:
        SIMD[size, type]: The result of the approximate GELU operation.

    Constraints:
        type must be a floating point type.
    """
    alias SQRT_TWO_OVER_PI = 0.797884560802865
    assert_param_bool_msg[
        DType(type).is_floating_point(),
        "dtype must be a floating point type",
    ]()
    let x3 = x * x * x
    return (
        0.5
        * x
        * (1 + tanh[simd_width, type](SQRT_TWO_OVER_PI * (x + 0.044715 * x3)))
    )


# ===----------------------------------------------------------------------===#
# sigmoid
# ===----------------------------------------------------------------------===#


fn sigmoid[
    simd_width: __mlir_type.index, type: __mlir_type.`!kgen.dtype`
](x: SIMD[simd_width, type]) -> SIMD[simd_width, type]:
    """Compute the Sigmoid Op using the equation $e^x / (e^x + 1)$.

    Parameters:
        simd_width: SIMD width used for the computation.
        type: dtype used for the computation.

    Args:
        x (SIMD[size, type]): The value to compute the sigmoid operation on.

    Returns:
        SIMD[size, type]: The result of the approximate sigmoid operation.
    """

    let ex = exp[simd_width, type](x)
    return ex / (ex + 1)
