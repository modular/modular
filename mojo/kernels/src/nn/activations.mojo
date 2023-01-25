# ===----------------------------------------------------------------------===#
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------===#

from Assert import assert_param
from List import create_kgen_list
from Math import erf, exp, tanh, clamp
from Polynomial import polynomial_evaluate
from SIMD import SIMD
from TypeTraits import is_floating_point

# ===----------------------------------------------------------------------===#
# relu
# ===----------------------------------------------------------------------===#


fn relu[
    simd_width: __mlir_type.index, type: __mlir_type.`!kgen.dtype`
](x: SIMD[simd_width, type]) -> SIMD[simd_width, type]:
    """Compute the Relu Op using the equation $max(0, x)$.

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


# ===----------------------------------------------------------------------===#
# sigmoid
# ===----------------------------------------------------------------------===#


fn sigmoid[
    simd_width: __mlir_type.index, type: __mlir_type.`!kgen.dtype`
](x0: SIMD[simd_width, type]) -> SIMD[simd_width, type]:
    """Compute the Sigmoid Op using the equation $e^x / (e^x + 1)$

    We implement the sigmoid in terms of the approximation:

    $$
    \frac{0.248288 x + 0.00851377 x^3 + 0.0000608575 x^5 + 1.15627\times10^{-7} x^7 +
    4.37031\times10^{-11} x^9}{0.993152 + 0.116818 x^2 + 0.00170199 x^4 + 6.29107\times10^{-6} x^6 +
    5.76102\times10^{-9} x^8 + 6.10247*10^{-13} x^{10}} + 0.5
    $$

    Args:
        x (SIMD[size, type]): The value to compute the sigmoid operation on.

    Returns:
        SIMD[size, type]: The result of the approximate sigmoid operation.
    """

    let x = clamp[simd_width, type](x0, -18, 18)
    let x2 = x * x

    let numerator = x * polynomial_evaluate[
        __mlir_type.f64,
        type,
        simd_width,
        5,
        create_kgen_list[__mlir_type.f64](
            2.48287947061529e-01,
            8.51377133304701e-03,
            6.08574864600143e-05,
            1.15627324459942e-07,
            4.37031012579801e-11,
        ),
    ](x2)

    let denominator = polynomial_evaluate[
        __mlir_type.f64,
        type,
        simd_width,
        6,
        create_kgen_list[__mlir_type.f64](
            9.93151921023180e-01,
            1.16817656904453e-01,
            1.70198817374094e-03,
            6.29106785017040e-06,
            5.76102136993427e-09,
            6.10247389755681e-13,
        ),
    ](x2)

    return numerator / denominator + 0.5
