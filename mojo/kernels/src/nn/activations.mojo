# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""The module contains implementations of activation functions."""

from Assert import assert_param_msg, debug_assert
from Bit import _is_neg
from DType import DType
from Math import erf, exp, clamp, max, min, identity, tanh
from SIMD import SIMD


@register_passable("trivial")
struct ActivationType:
    var value: Int
    alias IDENTITY = ActivationType(0)
    alias GELU = ActivationType(1)
    alias GELU_APPROX = ActivationType(2)
    alias RELU = ActivationType(3)
    alias RELU_N1 = ActivationType(4)
    alias RELU6 = ActivationType(5)
    alias SIGMOID = ActivationType(6)
    alias SIGN = ActivationType(7)
    alias TANH = ActivationType(8)

    @always_inline("nodebug")
    fn __init__(value: Int) -> ActivationType:
        return ActivationType {value: value}

    @always_inline("nodebug")
    fn __eq__(self, rhs: ActivationType) -> Bool:
        return self.value == rhs.value

    @always_inline("nodebug")
    fn __ne__(self, rhs: ActivationType) -> Bool:
        return self.value != rhs.value

    @always_inline
    fn dispatch[
        func: __mlir_type[
            `!kgen.signature<<`, ActivationType, `>() -> `, NoneType, `>`
        ]
    ](self):
        if self == ActivationType.IDENTITY:
            func[ActivationType.IDENTITY]()
        elif self == ActivationType.RELU:
            func[ActivationType.RELU]()
        elif self == ActivationType.RELU6:
            func[ActivationType.RELU6]()
        elif self == ActivationType.RELU_N1:
            func[ActivationType.RELU_N1]()
        elif self == ActivationType.GELU:
            func[ActivationType.GELU]()
        elif self == ActivationType.GELU_APPROX:
            func[ActivationType.GELU_APPROX]()
        elif self == ActivationType.SIGMOID:
            func[ActivationType.SIGMOID]()
        elif self == ActivationType.SIGN:
            func[ActivationType.SIGN]()
        elif self == ActivationType.TANH:
            func[ActivationType.TANH]()
        else:
            debug_assert(True, "unsupported activation")


@always_inline
fn dispatch_activation_fn[
    activation: ActivationType, simd_width: Int, type: DType
](val: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    @parameter
    if activation == ActivationType.IDENTITY:
        return identity(val)
    elif activation == ActivationType.RELU:
        return relu(val)
    elif activation == ActivationType.RELU6:
        return relu6(val)
    elif activation == ActivationType.RELU_N1:
        return relu_n1(val)
    elif activation == ActivationType.GELU:
        return gelu(val)
    elif activation == ActivationType.GELU_APPROX:
        return gelu_approximate(val)
    elif activation == ActivationType.SIGMOID:
        return sigmoid(val)
    elif activation == ActivationType.SIGN:
        return sign(val)
    elif activation == ActivationType.TANH:
        return _tanh(val)
    else:
        assert_param_msg[False, "unsupported activation"]()

    return val


# ===----------------------------------------------------------------------===#
# _tanh
# ===----------------------------------------------------------------------===#


fn _tanh[
    simd_width: Int, type: DType
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    return tanh(x)


# ===----------------------------------------------------------------------===#
# sign
# ===----------------------------------------------------------------------===#


@always_inline
fn sign[
    simd_width: Int, type: DType
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Compute the sign (0, 1) of the input value.

    Parameters:
        simd_width: SIMD width used for the computation.
        type: dtype used for the computation.

    Args:
        x (SIMD[type, simd_width]): The value to compute the sign operation on.

    Returns:
        SIMD[type, simd_width]: The result of the sign operation.
    """
    let is_neg_mask = _is_neg(x)
    let is_zero_mask = x == 0
    return is_neg_mask.select[type](-1, is_zero_mask.select[type](0, 1))


# ===----------------------------------------------------------------------===#
# relu
# ===----------------------------------------------------------------------===#


@always_inline
fn relu[
    simd_width: Int, type: DType
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Compute the Relu Op using the equation $max(0, x)$.

    Parameters:
        simd_width: SIMD width used for the computation.
        type: dtype used for the computation.

    Args:
        x (SIMD[type, simd_width]): The value to compute the RELU operation on.

    Returns:
        SIMD[type, simd_width]: The result of the RELU operation.
    """
    return max(x, 0)


# ===----------------------------------------------------------------------===#
# relu6
# ===----------------------------------------------------------------------===#


@always_inline
fn relu6[
    simd_width: Int, type: DType
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Compute the Relu6 Op using the equation $min(max(0,x),6)$.

    Parameters:
        simd_width: SIMD width used for the computation.
        type: dtype used for the computation.

    Args:
        x (SIMD[type, simd_width]): The value to compute the RELU6 operation on.

    Returns:
        SIMD[type, simd_width]: The result of the RELU6 operation.
    """
    return clamp(x, 0, 6)


# ===----------------------------------------------------------------------===#
# prelu
# ===----------------------------------------------------------------------===#


@always_inline
fn prelu[
    simd_width: Int, type: DType
](x: SIMD[type, simd_width], alpha: SIMD[type, 1]) -> SIMD[type, simd_width]:
    """Compute the Prelu Op using the equation $max(x,0) + alpha * min(x,0)$.

    Parameters:
        simd_width: SIMD width used for the computation.
        type: dtype used for the computation.

    Args:
        x (SIMD[type, simd_width]): The value to compute the PRELU operation on.

    Returns:
        SIMD[type, simd_width]: The result of the PRELU operation.
    """
    return max(x, 0) + alpha * min(x, 0)


# ===----------------------------------------------------------------------===#
# relu-n1
# ===----------------------------------------------------------------------===#


@always_inline
fn relu_n1[
    simd_width: Int, type: DType
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Compute the Relu N1 Op using the equation $max(min(x,1),-1)$.

    Parameters:
        simd_width: SIMD width used for the computation.
        type: dtype used for the computation.

    Args:
        x (SIMD[type, simd_width]): The value to compute the RELU N1 operation on.

    Returns:
        SIMD[type, simd_width]: The result of the RELU N1 operation.
    """
    return clamp(x, -1, 1)


# ===----------------------------------------------------------------------===#
# gelu
# ===----------------------------------------------------------------------===#


@always_inline
fn gelu[
    simd_width: Int, type: DType
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Compute the GELU Op using the equation
    $0.5 * x * (1 + erf(x / sqrt(2)))$.

    Parameters:
        simd_width: SIMD width used for the computation.
        type: dtype used for the computation.

    Args:
        x (SIMD[type, size]): The value to compute the GELU operation on.

    Returns:
        SIMD[type, size]: The result of the GELU operation.

    Constraints:
        type must be a floating point type.
    """
    alias SQRT_2 = 1.4142135623730950488
    assert_param_msg[
        type.is_floating_point(),
        "dtype must be a floating point type",
    ]()
    return 0.5 * x * (1 + erf(x / SQRT_2))


# ===----------------------------------------------------------------------===#
# gelu_approximate
# ===----------------------------------------------------------------------===#


@always_inline
fn gelu_approximate[
    simd_width: Int, type: DType
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Compute the approximate GELU Op using the equation
    $0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))$.

    Parameters:
        simd_width: SIMD width used for the computation.
        type: dtype used for the computation.

    Args:
        x (SIMD[type, size]): The value to compute the GELU operation on.

    Returns:
        SIMD[type, size]: The result of the approximate GELU operation.

    Constraints:
        type must be a floating point type.
    """
    alias SQRT_TWO_OVER_PI = 0.797884560802865
    assert_param_msg[
        type.is_floating_point(),
        "dtype must be a floating point type",
    ]()
    let x3 = x * x * x
    return 0.5 * x * (1 + tanh(SQRT_TWO_OVER_PI * (x + 0.044715 * x3)))


# ===----------------------------------------------------------------------===#
# sigmoid
# ===----------------------------------------------------------------------===#


@always_inline
fn sigmoid[
    simd_width: Int, type: DType
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Compute the Sigmoid Op using the equation $e^x / (e^x + 1)$.

    Parameters:
        simd_width: SIMD width used for the computation.
        type: dtype used for the computation.

    Args:
        x (SIMD[type, size]): The value to compute the sigmoid operation on.

    Returns:
        SIMD[type, size]: The result of the approximate sigmoid operation.
    """

    let ex = exp(x)
    return ex / (ex + 1)
