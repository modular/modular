# ===----------------------------------------------------------------------===#
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------===#

from Assert import assert_param
from SIMD import SIMD
from TypeTraits import is_floating_point
from Math import erf, tanh

# ===----------------------------------------------------------------------===#
# relu
# ===----------------------------------------------------------------------===#


fn relu[
    simd_width: __mlir_type.index, type: __mlir_type.`!kgen.dtype`
](x: SIMD[simd_width, type]) -> SIMD[simd_width, type]:
    """Compute the Relu Op using the equation
    $max(0, x)$.

    Args:
        x (SIMD[simd_width, type]): The value to compute the RELU operation on.

    Returns:
        SIMD[simd_width, type]: The result of the RELU operation.
    """
    return x.max(0)


# ===----------------------------------------------------------------------===#
# gelu
# ===----------------------------------------------------------------------===#


fn gelu[
    simd_width: __mlir_type.index, type: __mlir_type.`!kgen.dtype`
](x: SIMD[simd_width, type]) -> SIMD[simd_width, type]:
    """Compute the GELU Op using the equation
    $0.5 * x * (1 + erf(x / sqrt(2)))$.

    Args:
        x (SIMD[size, type]): The value to compute the GELU operation on.

    Returns:
        SIMD[size, type]: The result of the GELU operation.
    """
    alias SQRT_2 = 1.4142135623730950488
    assert_param[is_floating_point[type]()]()
    return 0.5 * x * (1 + erf[simd_width, type](x / SQRT_2))


# ===----------------------------------------------------------------------===#
# gelu_approximate
# ===----------------------------------------------------------------------===#


fn gelu_approximate[
    simd_width: __mlir_type.index, type: __mlir_type.`!kgen.dtype`
](x: SIMD[simd_width, type]) -> SIMD[simd_width, type]:
    """Compute the approximate GELU Op using the equation
    $0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))$.

    Args:
        x (SIMD[size, type]): The value to compute the GELU operation on.

    Returns:
        SIMD[size, type]: The result of the approximate GELU operation.
    """
    alias SQRT_TWO_OVER_PI = 0.797884560802865
    assert_param[is_floating_point[type]()]()
    let x3 = x * x * x
    return (
        0.5
        * x
        * (1 + tanh[simd_width, type](SQRT_TWO_OVER_PI * (x + 0.044715 * x3)))
    )
