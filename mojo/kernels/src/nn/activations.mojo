# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""The module contains implementations of activation functions."""

from Assert import assert_param
from bit import _is_neg
from DType import DType
from List import VariadicList
from LLCL import OutputChainPtr
from math import (
    abs,
    copysign,
    erf,
    exp,
    expm1,
    clamp,
    max,
    min,
    identity,
    tanh,
    fma,
)
from polynomial import polynomial_evaluate
from SIMD import SIMD


@value
@register_passable("trivial")
struct ActivationType:
    var value: Int
    alias IDENTITY = ActivationType(0)
    alias GELU = ActivationType(1)
    alias RELU = ActivationType(2)

    @always_inline("nodebug")
    fn __eq__(self, rhs: ActivationType) -> Bool:
        return self.value == rhs.value

    @always_inline("nodebug")
    fn __ne__(self, rhs: ActivationType) -> Bool:
        return self.value != rhs.value

    @always_inline
    fn dispatch[
        func: fn[act: ActivationType] () capturing -> None
    ](self, out_chain: OutputChainPtr):
        if self == ActivationType.IDENTITY:
            func[ActivationType.IDENTITY]()
        elif self == ActivationType.RELU:
            func[ActivationType.RELU]()
        elif self == ActivationType.GELU:
            func[ActivationType.GELU]()
        else:
            out_chain.mark_error("Unsupported activation function.")


@always_inline
fn dispatch_activation_fn[
    activation: ActivationType, type: DType, simd_width: Int
](val: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    @parameter
    if activation == ActivationType.IDENTITY:
        return identity(val)
    elif activation == ActivationType.RELU:
        return relu(val)
    elif activation == ActivationType.GELU:
        return gelu(val)
    else:
        assert_param[False, "unsupported activation"]()

    return val


# ===----------------------------------------------------------------------===#
# _tanh
# ===----------------------------------------------------------------------===#


fn _tanh[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    return tanh(x)


# ===----------------------------------------------------------------------===#
# sign
# ===----------------------------------------------------------------------===#


@always_inline
fn sign[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Compute the sign (0, 1) of the input value.

    Parameters:
        simd_width: SIMD width used for the computation.
        type: dtype used for the computation.

    Args:
        x : The value to compute the sign operation on.

    Returns:
        SIMD[type, simd_width]: The result of the sign operation.
    """
    let is_neg_mask = _is_neg(x)
    let is_zero_mask = x == 0
    return is_neg_mask.select[type](-1, is_zero_mask.select[type](0, 1))


# ===----------------------------------------------------------------------===#
# elu
# ===----------------------------------------------------------------------===#


@always_inline
fn elu[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Compute the Elu Op using the equation $z if z >= 0 else alpha*(e^z -1)$.

    Parameters:
        simd_width: SIMD width used for the computation.
        type: dtype used for the computation.

    Args:
        x : The value to compute the ELU operation on.

    Returns:
        SIMD[type, simd_width]: The result of the ELU operation.
    """
    return (x >= 0).select(x, expm1(x))


# ===----------------------------------------------------------------------===#
# relu
# ===----------------------------------------------------------------------===#


@always_inline
fn relu[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Compute the Relu Op using the equation $max(0, x)$.

    Parameters:
        simd_width: SIMD width used for the computation.
        type: dtype used for the computation.

    Args:
        x : The value to compute the RELU operation on.

    Returns:
        SIMD[type, simd_width]: The result of the RELU operation.
    """
    return max(x, 0)


# ===----------------------------------------------------------------------===#
# relu6
# ===----------------------------------------------------------------------===#


@always_inline
fn relu6[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Compute the Relu6 Op using the equation $min(max(0,x),6)$.

    Parameters:
        simd_width: SIMD width used for the computation.
        type: dtype used for the computation.

    Args:
        x : The value to compute the RELU6 operation on.

    Returns:
        SIMD[type, simd_width]: The result of the RELU6 operation.
    """
    return clamp(x, 0, 6)


# ===----------------------------------------------------------------------===#
# prelu
# ===----------------------------------------------------------------------===#


@always_inline
fn prelu[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width], alpha: SIMD[type, 1]) -> SIMD[type, simd_width]:
    """Compute the Prelu Op using the equation $max(x,0) + alpha * min(x,0)$.

    Parameters:
        simd_width: SIMD width used for the computation.
        type: dtype used for the computation.

    Args:
        x : The value to compute the PRELU operation on.

    Returns:
        SIMD[type, simd_width]: The result of the PRELU operation.
    """
    return max(x, 0) + alpha * min(x, 0)


# ===----------------------------------------------------------------------===#
# relu-n1
# ===----------------------------------------------------------------------===#


@always_inline
fn relu_n1[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Compute the Relu N1 Op using the equation $max(min(x,1),-1)$.

    Parameters:
        simd_width: SIMD width used for the computation.
        type: dtype used for the computation.

    Args:
        x : The value to compute the RELU N1 operation on.

    Returns:
        SIMD[type, simd_width]: The result of the RELU N1 operation.
    """
    return clamp(x, -1, 1)


# ===----------------------------------------------------------------------===#
# gelu
# ===----------------------------------------------------------------------===#


@always_inline
fn _erf[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Compute the erf function. This uses the formula 7.1.26 on page 299 from
    `Abramowitz and Stegun from "Handbook of Mathematical Functions"`.
    This version is used in oneDNN.
    """
    assert_param[type.is_floating_point(), "must be a floating point value"]()
    let x_abs = abs(x)
    # t = 1 / (1 + p * abs(x))
    let t = 1 / x_abs.fma(0.3275911, 1)
    # auxiliary value =  t * exp(-x*x)
    let val_aux = t * exp(-x_abs * x_abs)
    # r = 1 - polynomial * t * exp(-x*x)
    let polynomial = polynomial_evaluate[
        simd_width,
        type,
        VariadicList[SIMD[type, simd_width]](
            0.254829592,
            -0.284496736,
            1.421413741,
            -1.453152027,
            1.061405429,
        ),
    ](t)
    let r = polynomial.fma(-val_aux, 1)
    return copysign(r, x)


@always_inline
fn gelu[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Compute the GELU Op using the equation
    $0.5 * x * (1 + erf(x / sqrt(2)))$.

    Parameters:
        simd_width: SIMD width used for the computation.
        type: dtype used for the computation.

    Args:
        x : The value to compute the GELU operation on.

    Returns:
        SIMD[type, size]: The result of the GELU operation.

    Constraints:
        type must be a floating point type.
    """
    alias inv_SQRT_2 = 0.70710678118654752440
    assert_param[
        type.is_floating_point(),
        "dtype must be a floating point type",
    ]()
    # 0.5 * x * (1 + erf(x / SQRT_2))
    # x_half + x_half * erf_res
    let x_half = 0.5 * x
    let erf_res = _erf(x * inv_SQRT_2)
    return x_half.fma(erf_res, x_half)


# ===----------------------------------------------------------------------===#
# gelu_approximate
# ===----------------------------------------------------------------------===#


@always_inline
fn gelu_approximate[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Compute the approximate GELU Op using the equation
    $0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))$.

    Parameters:
        type: The `DType` used for the computation.
        simd_width: SIMD width used for the computation.

    Constraints:
        type must be a floating point type.

    Args:
        x : The value to compute the GELU operation on.

    Returns:
        SIMD[type, size]: The result of the approximate GELU operation.
    """
    alias SQRT_TWO_OVER_PI = 0.797884560802865
    assert_param[
        type.is_floating_point(),
        "dtype must be a floating point type",
    ]()
    let x3 = x * x * x
    return 0.5 * x * (1 + tanh(SQRT_TWO_OVER_PI * (x + 0.044715 * x3)))


@always_inline
fn gelu_approximate_sigmoid[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Compute the approximate GELU Op using the equation
    $x*sigmoid(1.702x)$.

    Parameters:
        simd_width: SIMD width used for the computation.
        type: dtype used for the computation.

    Args:
        x : The value to compute the GELU operation on.

    Returns:
        SIMD[type, size]: The result of the approximate GELU operation.

    Constraints:
        type must be a floating point type.
    """
    assert_param[
        type.is_floating_point(), "dtype must be a floating point type"
    ]()
    return x * sigmoid(x * 1.702)


# ===----------------------------------------------------------------------===#
# sigmoid
# ===----------------------------------------------------------------------===#


@always_inline
fn sigmoid[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Compute the Sigmoid Op using the equation $e^x / (e^x + 1)$.

    Parameters:
        type: The `dtype` used for the computation.
        simd_width: SIMD width used for the computation.

    Args:
        x : The value to compute the sigmoid operation on.

    Returns:
        SIMD[type, size]: The result of the sigmoid operation.
    """
    return 1 / (1 + exp(-x))


# ===----------------------------------------------------------------------===#
# sigmoid_grad
# ===----------------------------------------------------------------------===#


@always_inline
fn sigmoid_grad[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Compute the Sigmoid Grad Op using the equation
    $(1-sigmoid(x))*sigmoid(x)$.

    Parameters:
        type: The `dtype` used for the computation.
        simd_width: SIMD width used for the computation.

    Args:
        x : The value to compute the sigmoid grad operation on.

    Returns:
        The result of the sigmoid grad operation.
    """

    let s = sigmoid(x)
    return (1 - s) * s
