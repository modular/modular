# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
"""Defines math utilities.

You can import these APIs from the `math` package. For example:

```mojo
from std.math import floor
```
"""

from std.sys import (
    CompilationTarget,
    bit_width_of,
    is_gpu,
    llvm_intrinsic,
    simd_width_of,
    size_of,
)
from std.ffi import _external_call_const
from std.sys.info import is_32bit

from std.algorithm import vectorize
from std.bit import count_leading_zeros, count_trailing_zeros
from std.builtin.dtype import _integral_type_of
from std.simd import _modf, _simd_apply
from std.collections import Span
from . import pi, inf, isfinite, isinf, isnan, nan, nextafter

from std.utils.numerics import FPUtils, isnan, nan
from std.memory import bitcast
from std.utils.static_tuple import StaticTuple

from std._plugin import CurrentPlugin

from .constants import log2e
from .polynomial import polynomial_evaluate

# ===----------------------------------------------------------------------=== #
# floor
# ===----------------------------------------------------------------------=== #


@inline(.always)
def floor[T: Floorable, //](value: T) -> T:
    """Get the floor value of the given object.

    Parameters:
        T: The type conforming to `Floorable`.

    Args:
        value: The object to get the floor value of.

    Returns:
        The floor value of the object.
    """
    return value.__floor__()


# ===----------------------------------------------------------------------=== #
# ceil
# ===----------------------------------------------------------------------=== #


@inline(.always)
def ceil[T: Ceilable, //](value: T) -> T:
    """Get the ceiling value of the given object.

    Parameters:
        T: The type conforming to `Ceilable`.

    Args:
        value: The object to get the ceiling value of.

    Returns:
        The ceiling value of the object.
    """
    return value.__ceil__()


# ===----------------------------------------------------------------------=== #
# ceildiv
# ===----------------------------------------------------------------------=== #


@inline(.always)
def ceildiv[T: CeilDivable, //](numerator: T, denominator: T) -> T:
    """Return the rounded-up result of dividing numerator by denominator.

    Parameters:
        T: A type that support floor division.

    Args:
        numerator: The numerator.
        denominator: The denominator.

    Returns:
        The ceiling of dividing numerator by denominator.
    """
    return numerator.__ceildiv__(denominator)


@inline(.always)
def ceildiv[
    T: CeilDivableRaising, //
](numerator: T, denominator: T) raises -> T:
    """Return the rounded-up result of dividing numerator by denominator, potentially raising.

    Parameters:
        T: A type that support floor division.

    Args:
        numerator: The numerator.
        denominator: The denominator.

    Returns:
        The ceiling of dividing numerator by denominator.

    Raises:
        If the operation fails.
    """
    return numerator.__ceildiv__(denominator)


# NOTE: this overload is needed because IntLiteral promotes to a runtime type
# before overload resolution.
@always_inline("builtin")
def ceildiv(
    numerator: IntLiteral, denominator: IntLiteral
) -> type_of(numerator.__ceildiv__(denominator)):
    """Return the rounded-up result of dividing numerator by denominator.

    Args:
        numerator: The numerator.
        denominator: The denominator.

    Returns:
        The ceiling of dividing numerator by denominator.
    """
    return {}


# ===----------------------------------------------------------------------=== #
# trunc
# ===----------------------------------------------------------------------=== #


@inline(.always)
def trunc[T: Truncable, //](value: T) -> T:
    """Get the truncated value of the given object.

    Parameters:
        T: The type conforming to Truncable.

    Args:
        value: The object to get the truncated value of.

    Returns:
        The truncated value of the object.
    """
    return value.__trunc__()


# ===----------------------------------------------------------------------=== #
# sqrt
# ===----------------------------------------------------------------------=== #


@inline(.always)
def _sqrt_int(x: Int) -> Int:
    if x < 0:
        return 0

    var r = 0
    var r2 = 0

    comptime for p in reversed(range(bit_width_of[Int]() // 2)):
        var dr2 = (r << (p + 1)) + (1 << (p + p))
        if r2 <= x - dr2:
            r2 += dr2
            r |= 1 << p

    return r


comptime _UnaryMathPluginHookFnType = def[dtype: DType, width: SIMDLength, //](
    SIMD[dtype, width]
) thin -> SIMD[dtype, width]
"""Plugin-hook signature for `PluginHooks.unary_math_fn`."""

comptime _BinaryMathPluginHookFnType = def[dtype: DType, width: SIMDLength, //](
    SIMD[dtype, width], SIMD[dtype, width]
) thin -> SIMD[dtype, width]
"""Plugin-hook signature for `PluginHooks.binary_math_fn`."""


__enum _UnaryMathOp(ImplicitlyCopyable):
    """A `std.math` function that `PluginHooks.unary_math_fn` can lower."""

    case exp
    case exp2
    case log
    case log2
    case sqrt
    case rsqrt
    case recip
    case sin
    case cos
    case tanh


__enum _BinaryMathOp(ImplicitlyCopyable):
    """A two-argument function that `PluginHooks.binary_math_fn` can lower."""

    case pow


@inline(.always)
def sqrt[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Performs elementwise square root on the elements of a SIMD vector.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: SIMD vector to perform square root on.

    Returns:
        The elementwise square root of x.
    """
    comptime assert (
        dtype.is_numeric() or dtype == .bool
    ), "type must be arithmetic or boolean"

    comptime if dtype == .bool:
        return x
    elif dtype.is_integral():
        var res = SIMD[dtype, width]()

        comptime for i in range(width):
            res[i] = Scalar[dtype](_sqrt_int(Int(x[i])))
        return res

    if __is_run_in_comptime_interpreter:
        return _llvm_unary_fn["llvm.sqrt"](x)

    comptime hook = CurrentPlugin.unary_math_fn[.sqrt, dtype, width]
    comptime if hook:
        return comptime (hook.value())(x)

    return _llvm_unary_fn["llvm.sqrt"](x)


# ===----------------------------------------------------------------------=== #
# rsqrt
# ===----------------------------------------------------------------------=== #


@inline(.always)
def rsqrt[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> type_of(x):
    """Performs elementwise reciprocal square root on a SIMD vector.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: SIMD vector to perform reciprocal square root on.

    Returns:
        The elementwise reciprocal square root of x.
    """
    comptime assert (
        dtype.is_floating_point()
    ), "rsqrt requires floating point type"

    if __is_run_in_comptime_interpreter:
        return 1.0 / sqrt(x)

    comptime hook = CurrentPlugin.unary_math_fn[.rsqrt, dtype, width]
    comptime if hook:
        return comptime (hook.value())(x)

    return 1.0 / sqrt(x)


# ===----------------------------------------------------------------------=== #
# recip
# ===----------------------------------------------------------------------=== #


@inline(.always)
def recip[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> type_of(x):
    """Performs elementwise reciprocal on a SIMD vector.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: SIMD vector to perform reciprocal on.

    Returns:
        The elementwise reciprocal of x.
    """
    comptime assert (
        dtype.is_floating_point()
    ), "recip requires floating point type"

    if __is_run_in_comptime_interpreter:
        return 1.0 / x

    comptime hook = CurrentPlugin.unary_math_fn[.recip, dtype, width]
    comptime if hook:
        return comptime (hook.value())(x)

    return 1.0 / x


# ===----------------------------------------------------------------------=== #
# exp2
# ===----------------------------------------------------------------------=== #


@inline(.always)
def exp2[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    """Computes elementwise 2 raised to the power of n, where n is an element
    of the input SIMD vector.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: SIMD vector to perform exp2 on.

    Returns:
        Vector containing $2^n$ computed elementwise, where n is an element in
        the input SIMD vector.
    """

    if __is_run_in_comptime_interpreter:
        return _exp2_impl(x)

    comptime hook = CurrentPlugin.unary_math_fn[.exp2, dtype, width]
    comptime if hook:
        return comptime (hook.value())(x)

    return _exp2_impl(x)


@inline(.always)
def _exp2_impl[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    comptime if dtype == .float32:
        return _exp2_float32(x._refine[.float32]())._refine[dtype]()
    elif dtype == .float64:
        return SIMD[dtype, width](2.0) ** x
    else:
        return exp2(x.cast[.float32]()).cast[dtype]()


@inline(.always)
def _exp2_float32(x: SIMD[.float32, _]) -> type_of(x):
    comptime u32 = DType.uint32
    var xc = x.clamp(-126, 126)
    var m = xc.cast[.int32]()
    xc -= m.cast[x.dtype]()

    var r = polynomial_evaluate[
        [
            Float32(1.0),
            0.693144857883,
            0.2401793301105,
            5.551834031939e-2,
            9.810352697968e-3,
            1.33336498402e-3,
        ],
    ](xc)
    return type_of(x)(
        from_bits=r.to_bits[u32]()
        + (
            m.cast[u32]()
            << SIMD[.uint32, x.length](FPUtils[.float32].mantissa_width())
        )
    )


# ===----------------------------------------------------------------------=== #
# ldexp
# ===----------------------------------------------------------------------=== #


@inline(.always)
def _ldexp_impl[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width], exp: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Computes elementwise ldexp function.

    The ldexp function multiplies a floating point value x by the number 2
    raised to the exp power. I.e. $ldexp(x,exp)$ calculate the value of $x *
    2^{exp}$ and is used within the $erf$ function.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: SIMD vector of floating point values.
        exp: SIMD vector containing the exponents.

    Returns:
        Vector containing elementwise result of ldexp on x and exp.
    """

    comptime hardware_width = simd_width_of[dtype]()

    comptime if (
        CompilationTarget.has_avx512f()
        and dtype == .float32
        and width >= hardware_width
    ):
        var res: SIMD[dtype, width] = 0
        var zero: SIMD[dtype, hardware_width] = 0

        comptime for idx in range(width // hardware_width):
            comptime i = idx * hardware_width
            # On AVX512, we can use the scalef intrinsic to compute the ldexp
            # function.
            var part = llvm_intrinsic[
                "llvm.x86.avx512.mask.scalef.ps.512",
                SIMD[dtype, hardware_width],
                has_side_effect=False,
            ](
                x.slice[hardware_width, offset=i](),
                exp.slice[hardware_width, offset=i](),
                zero,
                Int16(-1),
                Int32(4),
            )
            res = res.insert[offset=i](part)

        return res

    comptime integral_type = FPUtils[dtype].integral_type
    var m = exp.cast[integral_type]() + SIMD[integral_type, width](
        FPUtils[dtype].exponent_bias()
    )

    return x * type_of(x)(
        from_bits=m
        << SIMD[integral_type, width](FPUtils[dtype].mantissa_width())
    )


@inline(.always)
def ldexp[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width], exp: SIMD[.int32, width]) -> SIMD[
    dtype, width
] where dtype.is_floating_point():
    """Computes elementwise ldexp function.

    The ldexp function multiplies a floating point value x by the number 2
    raised to the exp power. I.e. $ldexp(x,exp)$ calculate the value of $x *
    2^{exp}$ and is used within the $erf$ function.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: SIMD vector of floating point values.
        exp: SIMD vector containing the exponents.

    Returns:
        Vector containing elementwise result of ldexp on x and exp.
    """
    return _ldexp_impl(x, exp.cast[dtype]())


# ===----------------------------------------------------------------------=== #
# exp
# ===----------------------------------------------------------------------=== #


trait _Expable:
    """Trait for types that support the exp function."""

    def __exp__(self) -> Self:
        """Computes the exponential of the input value.

        Returns:
            The exponential of the input value.
        """
        ...


@inline(.always)
def _exp_taylor[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    comptime coefficients = List(
        [
            Scalar[dtype](1.0),
            1.0,
            0.5,
            0.16666666666666666667,
            0.041666666666666666667,
            0.0083333333333333333333,
            0.0013888888888888888889,
            0.00019841269841269841270,
            0.000024801587301587301587,
            2.7557319223985890653e-6,
            2.7557319223985890653e-7,
            2.5052108385441718775e-8,
            2.0876756987868098979e-9,
        ]
    )
    return polynomial_evaluate[
        coefficients[:] if dtype == .float64 else coefficients[:8],
    ](x)


@inline(.always)
def exp[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    """Calculates elementwise exponential of the input vector.

    Given an input vector $X$ and an output vector $Y$, sets $Y_i = e^{X_i}$ for
    each position $i$ in the input vector (where $e$ is the mathematical constant
    $e$).

    Constraints:
        The input must be a floating-point type.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: The input SIMD vector.

    Returns:
        A SIMD vector containing $e$ raised to the power $X_i$ where $X_i$ is an
        element in the input SIMD vector.
    """

    if __is_run_in_comptime_interpreter:
        return _exp_impl(x)

    comptime hook = CurrentPlugin.unary_math_fn[.exp, dtype, width]
    comptime if hook:
        return comptime (hook.value())(x)

    comptime if is_gpu():
        comptime if dtype in (DType.float16, DType.float32):
            return exp2(x * log2e)

    return _exp_impl(x)


@inline(.always)
def _exp_impl[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    comptime neg_ln2 = -0.69314718055966295651160180568695068359375

    comptime if dtype not in (DType.float32, DType.float64):
        return exp(x.cast[.float32]()).cast[dtype]()

    var min_val: SIMD[dtype, width]
    var max_val: SIMD[dtype, width]

    comptime if dtype == .float64:
        min_val = -709.436139303
        max_val = 709.437
    else:
        min_val = -88.3762626647949
        max_val = 88.3762626647950

    var xc = x.clamp(min_val, max_val)
    var k = floor(xc.fma(log2e, 0.5))
    var r = k.fma(neg_ln2, xc)
    return max(_ldexp_impl(_exp_taylor(r), k), xc)


@inline(.always)
def exp[T: _Expable](x: T) -> T:
    """Computes the exponential of the input value.

    Parameters:
        T: The type of the input value.

    Args:
        x: The input value.

    Returns:
        The exponential of the input value.
    """
    return x.__exp__()


@inline(.always)
def _exp2_approx_f32[W: Int](x: SIMD[.float32, W]) -> SIMD[.float32, W]:
    """Computes a fast approximation of 2^x using a fused analytic (FA-4)
        exponential method, using Polynomial (Horner form):
        p(r) = c0 + r*(c1 + r*(c2 + r*c3))

    Approximation strategy:
        We compute 2^x by range reduction x = n + r where r ∈ [−0.5, 0.5]
        then evaluate 2^r with a fixed-degree Horner polynomial and scale
        by 2^n via the float exponent-bias (±2^23) / ldexp trick. e^x is
        then 2^(x·log2(e)).

        This function splits the input `x` into integer and fractional components,
        clamps it to avoid underflow, and reconstructs 2^x as `ldexp(p(r), n)`,
        where `p(r)` is a cubic polynomial approximation computed via
        `polynomial_computation`. The method uses floating-point biasing and fused
        multiply-add operations for efficient SIMD execution with high numerical
        stability.

    Expected accuracy:
        Order-of-magnitude relative error is ~1e-4,1e-3 for 2^r on the target
        interval, which is adequate for softmax once values are stabilized by
        subtracting max(x).

    Constraints:
        The input must be a SIMD vector of 32-bit floating-point values.

    Parameters:
        W: The width of the SIMD vector.

    Args:
        x: The input SIMD vector representing the exponent.

    Returns:
        A SIMD vector containing the fast approximate result of 2^x.
    """

    # --- Constants ------------------------------------------------------------

    # Rounding bias for IEEE-754 float32 via the “add/subtract big constant”
    # trick.
    # We use 1.5 * 2^23 (i.e., 2^23 + 2^22) so it works cleanly with
    # round-to-nearest-even across positive/negative inputs in this range.
    # Decomposed as (1.5 * 2) * 2^(23 -1) to avoid floating point arithmetic.
    comptime ROUND_BIAS_F32 = 3 * (
        1 << (FPUtils[.float32].mantissa_width() - 1)
    )
    comptime NEG_ROUND_BIAS_F32 = -ROUND_BIAS_F32

    # Lower clamp for exp2 range reduction:
    # The float32 exponent bias is 127. Clamping at −127 keeps n from becoming
    # too negative (extreme subnormals/FTZ) and maintains accuracy of the cubic.
    # If you require strictly normal outputs, use −126.0 instead.
    comptime EXP2_MIN_INPUT = -FPUtils[.float32].exponent_bias()
    # --- Kernel ---------------------------------------------------------------

    # 1) clamp in float
    var x_min = max(x, SIMD[.float32, W](EXP2_MIN_INPUT))

    # 2) bias trick: vi = round(x_min) in float via +bias then −bias
    # (works for |x| < 2^23; we use 1.5*2^23 to behave well around 0 and negatives)
    var vb = x_min + SIMD[.float32, W](ROUND_BIAS_F32)
    var vi = vb + SIMD[.float32, W](NEG_ROUND_BIAS_F32)

    # 3) fractional part in [−0.5, 0.5] without extra clamp
    var r = x_min - vi

    # 4) cubic (FA-4) poly approximation Degree-3 coefficients for 2^r
    #  A cubic gives the best throughput/accuracy trade-off for softmax on GPU:
    #  only ~3 FMAs in the hot path, vectorizes cleanly (SIMD W=1/2), and yields
    #  low relative error that remains stable once we subtract max(x) before
    #  exponentiation. Going to degree-4/5 reduces error a bit but costs extra
    #  FMAs, registers, and latency with minimal end-to-end benefit.
    #  The coefficients below are a minimax fit for 2^r over the centered reduced
    #  interval (usually r ∈ [−0.5, 0.5]). They look close to the Taylor series
    #  at r=0: 2^r ≈ 1 + (ln 2) r + (ln 2)^2 r^2 / 2 + (ln 2)^3 r^3 / 6,
    #  but are tweaked (via Remez) to minimize the maximum relative error
    #  across the interval, which improves worst-case behavior vs plain Taylor.
    var p = polynomial_evaluate[
        [
            Float32(1.0000000000),
            0.6951461434,
            0.2275643945,
            0.0771190897,
        ],
    ](r)

    # 5) exponent as int lanes (no extra clamp needed due to early float clamp)
    var n = SIMD[.int32, W](vi)

    # result: 2^x ≈ 2^n * p
    return ldexp(p, n)


# ---------- e^x helpers ----------
@inline(.always)
def exp_approx_f32[W: Int](x: SIMD[.float32, W]) -> SIMD[.float32, W]:
    """Computes a fast approximate e^x for SIMD vectors of 32-bit floats
    using the base-2 approximation as a backend.

    This function converts the natural exponential input `z` to base-2 space
    using the identity e^z = 2^(z * log2(e)), then calls the internal
    `_exp2_approx_f32` function to evaluate the FA-4 polynomial approximation
    of 2^x. It is optimized for small SIMD widths and is fully inlined for
    high performance.

    Constraints:
        The input must be a SIMD vector of 32-bit floating-point values.

    Parameters:
        W: The width of the SIMD vector.

    Args:
        x: The input SIMD vector representing the exponent.

    Returns:
        A SIMD vector containing the approximate value of e^x.
    """
    return _exp2_approx_f32[W](x * SIMD[.float32, W](log2e))


# ===----------------------------------------------------------------------=== #
# frexp
# ===----------------------------------------------------------------------=== #


@inline(.always)
def _frexp_mask1[
    dtype: DType, width: Int
]() -> SIMD[_integral_type_of[dtype](), width]:
    comptime match dtype:
        case .float16:
            return 0x7C00
        case .bfloat16:
            return 0x7F80
        case .float32:
            return 0x7F800000
        case _:
            comptime assert dtype == .float64, "unhandled fp type"
            return 0x7FF0000000000000


@inline(.always)
def _frexp_mask2[
    dtype: DType, width: Int
]() -> SIMD[_integral_type_of[dtype](), width]:
    comptime match dtype:
        case .float16:
            return 0x3800
        case .bfloat16:
            return 0x3F00
        case .float32:
            return 0x3F000000
        case _:
            comptime assert dtype == .float64, "unhandled fp type"
            return 0x3FE0000000000000


def frexp[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> StaticTuple[
    SIMD[dtype, width], 2
] where dtype.is_floating_point():
    """Breaks floating point values into a fractional part and an exponent part.
    This follows C and Python in increasing the exponent by 1 and normalizing the
    fraction from 0.5 to 1.0 instead of 1.0 to 2.0.

    Constraints:
        The input must be a floating-point type.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: The input values.

    Returns:
        A tuple of two SIMD vectors containing the fractional and exponent parts
        of the input floating point values.
    """

    # Based on the implementation in boost/simd/arch/common/simd/function/ifrexp.hpp

    comptime T = SIMD[dtype, width]
    comptime TInt = SIMD[_integral_type_of[dtype](), width]
    comptime zero = T(0)
    # Add one to the resulting exponent up by subtracting 1 from the bias
    comptime exponent_bias = FPUtils[dtype].exponent_bias() - 1
    comptime mantissa_width = FPUtils[dtype].mantissa_width()
    comptime subnorm_scale = T(
        from_bits=TInt(
            (mantissa_width + FPUtils[dtype].exponent_bias()) << mantissa_width
        )
    )

    var mask1 = _frexp_mask1[dtype, width]()
    var mask2 = _frexp_mask2[dtype, width]()

    var subnorm_test = (x._to_bits_signed() & mask1).eq(0) & x.ne(0)
    var t = subnorm_test.select(TInt(mantissa_width), TInt(0))
    var x0 = subnorm_test.select(subnorm_scale * x, x)

    var x_int = x0._to_bits_signed()
    var selector = x0.ne(zero)
    var r1 = mask1 & x_int
    var mb = x_int & ~mask1
    r1 = (r1 >> TInt(mantissa_width)) - TInt(exponent_bias)

    var r0 = mb | mask2

    var test1 = r1.gt(TInt(FPUtils[dtype].max_exponent()))
    r1 = (~test1 & selector).select(r1, TInt(0))

    r1 -= t
    return StaticTuple[size=2](
        selector.select(
            bitcast[dtype, width](r0) + test1.select(x0, zero), zero
        ),
        T(r1),
    )


# ===----------------------------------------------------------------------=== #
# log
# ===----------------------------------------------------------------------=== #


@inline(.always)
def _log_base[
    dtype: DType, width: SIMDLength, //, base: Int
](x: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    """Performs elementwise log of a SIMD vector with a specific base.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.
        base: The logarithm base.

    Args:
        x: Vector to perform logarithm operation on.

    Returns:
        Vector containing result of performing logarithm on x.
    """
    # Based on the Cephes approximation.
    comptime sqrt2_div_2 = 0.70710678118654752440

    comptime assert base == 2 or base == 27, "input base must be either 2 or 27"

    var frexp_result = frexp(x)
    var x1 = frexp_result[0]
    var exp = frexp_result[1]
    exp = x1.lt(sqrt2_div_2).select(exp - 1, exp)
    x1 = x1.lt(sqrt2_div_2).select(x1 + x1, x1) - 1

    var x2 = x1 * x1
    var x3 = x2 * x1

    var y = (
        polynomial_evaluate[
            [
                Scalar[dtype](3.3333331174e-1),
                -2.4999993993e-1,
                2.0000714765e-1,
                -1.6668057665e-1,
                1.4249322787e-1,
                -1.2420140846e-1,
                1.1676998740e-1,
                -1.1514610310e-1,
                7.0376836292e-2,
            ],
        ](x1)
        * x3
    )
    y = x1 + x2.fma(-0.5, y)

    # TODO: fix this hack
    comptime if base == 27:  # Natural log
        comptime ln2 = 0.69314718055994530942
        y = exp.fma(ln2, y)
    else:
        y = y.fma(log2e, exp)
    y = x.eq(inf[dtype]()).select(x, y)
    return x.eq(0).select(Scalar[dtype].MIN, x.gt(0).select(y, nan[dtype]()))


@inline(.always)
def log[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> type_of(x) where dtype.is_floating_point():
    """Performs elementwise natural log (base E) of a SIMD vector.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: Vector to perform logarithm operation on.

    Returns:
        Vector containing result of performing natural log base E on x.
    """

    _assert_float64_supported[dtype, "log"]()

    if not __is_run_in_comptime_interpreter:
        comptime hook = CurrentPlugin.unary_math_fn[.log, dtype, width]
        comptime if hook:
            return comptime (hook.value())(x)

    comptime if dtype == .float64:
        var r = _log_accurate(x)
        r = x.eq(inf[dtype]()).select(x, r)
        return x.eq(0).select(-inf[dtype](), x.gt(0).select(r, nan[dtype]()))
    elif size_of[dtype]() < size_of[DType.float32]():
        return log(x.cast[.float32]()).cast[dtype]()

    if __is_run_in_comptime_interpreter:
        return _log_base[27](x)

    comptime if CurrentPlugin.flushes_subnormals[dtype]:
        # Subnormals read as zero in comparisons and `frexp`, so reduce from
        # the bits instead.
        var r = _log_accurate(x)
        r = x.eq(inf[dtype]()).select(x, r)
        var positive = x.gt(0) | (
            _is_subnormal_bits(x) & x._to_bits_signed().gt(0)
        )
        return _is_zero_bits(x).select(
            -inf[dtype](), positive.select(r, nan[dtype]())
        )

    return _log_base[27](x)


# ===----------------------------------------------------------------------=== #
# log2
# ===----------------------------------------------------------------------=== #


@inline(.always)
def log2[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> type_of(x) where dtype.is_floating_point():
    """Performs elementwise log (base 2) of a SIMD vector.

    Args:
        x: Vector to perform logarithm operation on.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Returns:
        Vector containing result of performing log base 2 on x.
    """

    if __is_run_in_comptime_interpreter:
        comptime if size_of[dtype]() < size_of[DType.float32]():
            return log2(x.cast[.float32]()).cast[dtype]()

        return _log_base[2](x)

    comptime hook = CurrentPlugin.unary_math_fn[.log2, dtype, width]
    comptime if hook:
        return comptime (hook.value())(x)

    comptime if size_of[dtype]() < size_of[DType.float32]():
        return log2(x.cast[.float32]()).cast[dtype]()

    return _log_base[2](x)


# ===----------------------------------------------------------------------=== #
# copysign
# ===----------------------------------------------------------------------=== #


@inline(.always)
def copysign[
    dtype: DType, width: SIMDLength, //
](magnitude: SIMD[dtype, width], sign: SIMD[dtype, width]) -> SIMD[
    dtype, width
]:
    """Returns a value with the magnitude of the first operand and the sign of
    the second operand.

    Constraints:
        The type of the input must be numeric.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        magnitude: The magnitude to use.
        sign: The sign to copy.

    Returns:
        Copies the sign from sign to magnitude.
    """
    comptime assert dtype.is_numeric(), "operands must be a numeric type"

    comptime if dtype.is_unsigned():
        return magnitude
    elif dtype.is_integral():
        var mag_abs = abs(magnitude)
        return sign.lt(0).select(-mag_abs, mag_abs)
    return llvm_intrinsic[
        "llvm.copysign", SIMD[dtype, width], has_side_effect=False
    ](magnitude, sign)


# ===----------------------------------------------------------------------=== #
# erf
# ===----------------------------------------------------------------------=== #


@inline(.always)
def erf[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    """Performs the elementwise Erf on a SIMD vector.

    Constraints:
        The type must be a floating-point type.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: SIMD vector to perform elementwise Erf on.

    Returns:
        The result of the elementwise Erf operation.
    """

    _assert_float64_supported[dtype, "erf"]()

    comptime if dtype == .float64:
        return _erf_f64(x._refine[.float64]())._refine[dtype]()

    var x_abs = abs(x)

    var r_large = polynomial_evaluate[
        [
            Scalar[dtype](1.28717512e-1),
            6.34846687e-1,
            1.06777847e-1,
            -2.42545605e-2,
            3.88393435e-3,
            -3.83208680e-4,
            1.72948930e-5,
        ],
    ](min(x_abs, 3.925))

    r_large = r_large.fma(x_abs, x_abs)
    r_large = copysign(1.0 - exp(-r_large), x)

    var r_small = polynomial_evaluate[
        [
            Scalar[dtype](1.28379151e-1),
            -3.76124859e-1,
            1.12818025e-1,
            -2.67667342e-2,
            4.99339588e-3,
            -5.99104969e-4,
        ],
    ](x_abs * x_abs).fma(x, x)

    return x_abs.gt(0.921875).select[dtype](r_large, r_small)


# ===----------------------------------------------------------------------=== #
# tanh
# ===----------------------------------------------------------------------=== #


@inline(.always)
def tanh[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    """Performs elementwise evaluation of the tanh function.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: The vector to perform the elementwise tanh on.

    Returns:
        The result of the elementwise tanh operation.
    """

    if __is_run_in_comptime_interpreter:
        return _tanh_impl(x)

    comptime hook = CurrentPlugin.unary_math_fn[.tanh, dtype, width]
    comptime if hook:
        return comptime (hook.value())(x)

    return _tanh_impl(x)


@inline(.always)
def _tanh_impl[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    comptime if dtype == .float64 and not is_gpu():
        # For float64 on CPU, use the expm1-based identity for full accuracy:
        #   tanh(x) = -expm1(-2|x|) / (expm1(-2|x|) + 2)
        # expm1 computes e^y - 1 accurately even for small |y|, avoiding the
        # catastrophic cancellation that occurs in (e^y - 1) computed naively.
        # This matches the accuracy of FreeBSD/Cephes libm tanh for float64.
        # Clamp at ±20: tanh(±20) rounds to exactly ±1 in float64.
        var xc64 = x.clamp(-20, 20)
        var x_abs = abs(xc64)
        var e = expm1((-2) * x_abs)
        return copysign((-e) / (e + 2), xc64)

    var xc = x.clamp(-9, 9)
    var x_squared = xc * xc

    var numerator = xc * polynomial_evaluate[
        [
            Scalar[dtype](4.89352455891786e-03),
            6.37261928875436e-04,
            1.48572235717979e-05,
            5.12229709037114e-08,
            -8.60467152213735e-11,
            2.00018790482477e-13,
            -2.76076847742355e-16,
        ],
    ](x_squared)

    var denominator = polynomial_evaluate[
        [
            Scalar[dtype](4.89352518554385e-03),
            2.26843463243900e-03,
            1.18534705686654e-04,
            1.19825839466702e-06,
        ],
    ](x_squared)

    return numerator / denominator


# ===----------------------------------------------------------------------=== #
# isclose
# ===----------------------------------------------------------------------=== #


@inline(.always)
def isclose[
    dtype: DType,
    width: SIMDLength,
    *,
    symmetrical: Bool = True,
](
    a: SIMD[dtype, width],
    b: SIMD[dtype, width],
    *,
    atol: Float64 = 1e-08,
    rtol: Float64 = 1e-05,
    equal_nan: Bool = False,
) -> SIMD[.bool, width]:
    """Returns a boolean SIMD vector indicating which element pairs of `a` and
    `b` are equal within a given tolerance.

    For floating-point dtypes, the following criteria apply:

    - Symmetric (Python `math.isclose` style), when `symmetrical` is true:
        ```text
        |a - b| ≤ max(atol, rtol * max(|a|, |b|))
        ```
    - Asymmetric (NumPy style), when `symmetrical` is false:
        ```text
        |a - b| ≤ atol + rtol * |b|
        ```

    NaN values are considered equal only if `equal_nan` is true.

    Parameters:
        dtype: Element type of the input and output vectors.
        width: Number of lanes in each SIMD vector.
        symmetrical: If true, use the symmetric comparison formula (default: true).

    Args:
        a: First input vector.
        b: Second input vector.
        atol: Absolute tolerance.
        rtol: Relative tolerance.
        equal_nan: If true, treat NaNs as equal (default: false).

    Returns:
        A boolean vector where `a` and `b` are equal within the given tolerance.
    """
    comptime assert (
        a.dtype.is_floating_point()
    ), "isclose only supports floating-point types"

    comptime T = type_of(a)

    var check_nan = isnan(a) & isnan(b)
    var check_fin: T._Mask
    var in_range: T._Mask

    comptime if symmetrical:
        check_fin = isfinite(a) & isfinite(b)
        in_range = abs(a - b).le(max(T(atol), T(rtol) * max(abs(a), abs(b))))
    else:
        check_fin = isfinite(b)
        in_range = abs(a - b).le(T(atol) + T(rtol) * abs(b))
    return (
        a.eq(b) | (check_nan & T._Mask(fill=equal_nan)) | (check_fin & in_range)
    )


# ===----------------------------------------------------------------------=== #
# iota
# ===----------------------------------------------------------------------=== #


@inline(.always)
def iota[
    dtype: DType, width: Int
](offset: Scalar[dtype] = Scalar[dtype](0)) -> SIMD[dtype, width]:
    """Creates a SIMD vector containing an increasing sequence, starting from
    offset.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        offset: The value to start the sequence at. Default is zero.

    Returns:
        An increasing sequence of values, starting from offset.
    """

    comptime if width == 1:
        return offset

    comptime step_dtype = dtype if dtype.is_integral() else DType.int
    var step: SIMD[step_dtype, width]
    if __is_run_in_comptime_interpreter:
        step = 0

        comptime for i in range(width):
            step[i] = Scalar[step_dtype](i)
    else:
        step = llvm_intrinsic[
            "llvm.stepvector", SIMD[step_dtype, width], has_side_effect=False
        ]()
    return step.cast[dtype]() + offset


def iota[
    dtype: DType, //
](
    buff: MutPointer[Scalar[dtype], _, address_space=_],
    len: Int,
    offset: Int = 0,
):
    """Fill the buffer with numbers ranging from offset to offset + len - 1,
    spaced by 1.

    The function doesn't return anything, the buffer is updated inplace.

    Parameters:
        dtype: DType of the underlying data.

    Args:
        buff: The buffer to fill.
        len: The length of the buffer to fill.
        offset: The value to fill at index 0.
    """

    @inline(.always)
    def fill[width: Int](i: Int) {var offset, var buff}:
        buff.unsafe_store(i, iota[dtype, width](Scalar[dtype](offset + i)))

    vectorize[simd_width_of[dtype]()](len, fill)


def iota[dtype: DType, //](span: MutSpan[Scalar[dtype], _], offset: Int = 0):
    """Fill a Span with consecutive numbers starting from the specified offset.

    Parameters:
        dtype: DType of the underlying data.

    Args:
        span: The Span to fill with numbers.
        offset: The starting value to fill at index 0.
    """
    iota(span.unsafe_ptr(), len(span), offset)


# ===----------------------------------------------------------------------=== #
# fma
# ===----------------------------------------------------------------------=== #


@inline(.nodebug)
def fma[
    dtype: DType, width: SIMDLength, //
](
    a: SIMD[dtype, width],
    b: SIMD[dtype, width],
    c: SIMD[dtype, width],
) -> SIMD[
    dtype, width
]:
    """Performs elementwise `fma` (fused multiply-add) on the inputs.

    Each element in the result SIMD vector is $(A_i * B_i) + C_i$, where $A_i$,
    $B_i$ and $C_i$ are elements at index $i$ in a, b, and c respectively.

    Parameters:
        dtype: The `dtype` of the input SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        a: The first vector of inputs.
        b: The second vector of inputs.
        c: The third vector of inputs.

    Returns:
        Elementwise `fma` of a, b and c.
    """
    return a.fma(b, c)


# ===----------------------------------------------------------------------=== #
# align_down
# ===----------------------------------------------------------------------=== #


@inline(.always)
def align_down[
    dtype: DType, width: SIMDLength, //
](value: SIMD[dtype, width], alignment: SIMD[dtype, width]) -> SIMD[
    dtype, width
] where dtype.is_integral():
    """Returns the closest multiple of alignment that is less than or equal to
    value, elementwise.

    Parameters:
        dtype: The `dtype` of the input SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        value: The value to align.
        alignment: Value to align to.

    Returns:
        Closest multiple of the alignment that is less than or equal to the
        input value. In other words, floor(value / alignment) * alignment.
    """
    assert alignment != 0, "zero alignment"
    return (value // alignment) * alignment


# ===----------------------------------------------------------------------=== #
# align_up
# ===----------------------------------------------------------------------=== #


@inline(.always)
def align_up[
    dtype: DType, width: SIMDLength, //
](value: SIMD[dtype, width], alignment: SIMD[dtype, width]) -> SIMD[
    dtype, width
] where dtype.is_integral():
    """Returns the closest multiple of alignment that is greater than or equal
    to value, elementwise.

    Parameters:
        dtype: The `dtype` of the input SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        value: The value to align.
        alignment: Value to align to.

    Returns:
        Closest multiple of the alignment that is greater than or equal to the
        input value. In other words, ceiling(value / alignment) * alignment.
    """
    assert alignment != 0, "zero alignment"
    return ceildiv(value, alignment) * alignment


# ===-----------------------------------------------------------------------===#
# mulhi
# ===-----------------------------------------------------------------------===#


@inline(.always)
def mulhi(a: UInt32, b: UInt32) -> UInt32:
    """Calculates the most significant 32 bits of the product of two 32-bit
    unsigned integers.

    Multiplies two 32-bit unsigned integers and returns the high 32 bits
    of their product. Useful for fixed-point arithmetic and overflow detection.

    Args:
        a: First 32-bit unsigned integer operand.
        b: Second 32-bit unsigned integer operand.

    Returns:
        The high 32 bits of the product a * b

    Note:
        The product is computed in 64-bit arithmetic, which NVIDIA GPUs
        compile to a single `mul.hi.u32` instruction.
    """

    var au64 = a.cast[.uint64]()
    var bu64 = b.cast[.uint64]()
    return ((au64 * bu64) >> 32).cast[.uint32]()


@inline(.always)
def mulhi(a: Int32, b: Int32) -> Int32:
    """Calculates the most significant 32 bits of the product of two 32-bit
    signed integers.

    Multiplies two 32-bit signed integers and returns the high 32 bits
    of their product. Useful for fixed-point arithmetic and overflow detection.

    Args:
        a: First 32-bit signed integer operand.
        b: Second 32-bit signed integer operand.

    Returns:
        The high 32 bits of the product a * b

    Note:
        The product is computed in 64-bit arithmetic, which NVIDIA GPUs
        compile to a single `mul.hi.s32` instruction.
    """

    var ai64 = a.cast[.int64]()
    var bi64 = b.cast[.int64]()
    return ((ai64 * bi64) >> 32).cast[.int32]()


@inline(.always)
def mulhi(a: UInt64, b: UInt64) -> UInt64:
    """Calculates the most significant 64 bits of the product of two 64-bit
    unsigned integers.

    Multiplies two 64-bit unsigned integers and returns the high 64 bits
    of their product. Useful for fixed-point arithmetic and overflow detection.

    Args:
        a: First 64-bit unsigned integer operand.
        b: Second 64-bit unsigned integer operand.

    Returns:
        The high 64 bits of the product a * b.

    Note:
        The product is computed in 128-bit arithmetic, which NVIDIA GPUs
        compile to a single `mul.hi.u64` instruction.
    """

    var au128 = a.cast[.uint128]()
    var bu128 = b.cast[.uint128]()
    return ((au128 * bu128) >> 64).cast[.uint64]()


@inline(.always)
def mulhi(a: Int64, b: Int64) -> Int64:
    """Calculates the most significant 64 bits of the product of two 64-bit
    signed integers.

    Multiplies two 64-bit signed integers and returns the high 64 bits
    of their product. Useful for fixed-point arithmetic and overflow detection.

    Args:
        a: First 64-bit signed integer operand.
        b: Second 64-bit signed integer operand.

    Returns:
        The high 64 bits of the product a * b.

    Note:
        The product is computed in 128-bit arithmetic, which NVIDIA GPUs
        compile to a single `mul.hi.s64` instruction.
    """

    var ai128 = a.cast[.int128]()
    var bi128 = b.cast[.int128]()
    return ((ai128 * bi128) >> 64).cast[.int64]()


# ===-----------------------------------------------------------------------===#
# mulwide
# ===-----------------------------------------------------------------------===#


@inline(.always)
def mulwide(a: UInt32, b: UInt32) -> UInt64:
    """Performs a wide multiplication of two 32-bit unsigned integers.

    Multiplies two 32-bit unsigned integers and returns the full 64-bit result.
    Useful when the product may exceed 32 bits.

    Args:
        a: First 32-bit unsigned integer operand.
        b: Second 32-bit unsigned integer operand.

    Returns:
        The full 64-bit product of a * b

    Note:
        The product is computed in 64-bit arithmetic, which NVIDIA GPUs
        compile to a single `mul.wide.u32` instruction.
    """

    var au64 = a.cast[.uint64]()
    var bu64 = b.cast[.uint64]()
    return au64 * bu64


@inline(.always)
def mulwide(a: Int32, b: Int32) -> Int64:
    """Performs a wide multiplication of two 32-bit signed integers.

    Multiplies two 32-bit signed integers and returns the full 64-bit result.
    Useful when the product may exceed 32 bits or be negative.

    Args:
        a: First 32-bit signed integer operand.
        b: Second 32-bit signed integer operand.

    Returns:
        The full 64-bit signed product of a * b

    Note:
        The product is computed in 64-bit arithmetic, which NVIDIA GPUs
        compile to a single `mul.wide.s32` instruction.
    """

    var ai64 = a.cast[.int64]()
    var bi64 = b.cast[.int64]()
    return ai64 * bi64


# ===----------------------------------------------------------------------=== #
# acos
# ===----------------------------------------------------------------------=== #


def acos[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> type_of(x) where dtype.is_floating_point():
    """Computes the `acos` of the inputs.

    Constraints:
        The input must be a floating-point type.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: The input argument.

    Returns:
        The `acos` of the input.
    """

    comptime if size_of[dtype]() < size_of[DType.float32]():
        return acos(x.cast[.float32]()).cast[dtype]()
    elif dtype == .float64:
        return _llvm_unary_fn["llvm.acos"](x)

    # For F32 types, use the Remez approximation found in Sleef with range
    # splitting to improve accuracy.

    # Determine which approximation method to use based on domain.
    var x_abs = clamp(abs(x), 0.0, 1.0)
    var directed_polynomial_mask = x_abs.lt(0.5)

    # Compute x² for polynomial evaluation
    # Small domain: x² = x²
    # Large domain: x² = (1 - |x|) / 2 for identity transformation
    var x_squared = directed_polynomial_mask.select(x * x, (1.0 - x_abs) * 0.5)

    # Compute d for evaluation
    # Small domain: d = |x|
    # Large domain: d = sqrt((1-|x|)/2) using the identity
    var d = directed_polynomial_mask.select(abs(x), sqrt(x_squared))

    # Special case: handle |x| = 1 to avoid numerical instability
    d = x_abs.eq(1.0).select(type_of(x)(0.0), d)

    # Evaluate Remez polynomial using Horner's method
    # Coefficients derived to minimize maximum absolute error
    var poly = polynomial_evaluate[
        [
            Scalar[x.dtype](0.1666677296e0),
            0.7495029271e-1,
            0.4547423869e-1,
            0.2424046025e-1,
            0.4197454825e-1,
        ]
    ](x_squared)

    # Final polynomial term: poly * x² * d
    poly *= x_squared * d

    # Small domain: compute π/2 - asin(x) where asin(x) = d + poly with sign.
    var y = (pi / 2.0) - (copysign(d, x) + copysign(poly, x))

    # Large domain: compute 2 * asin(sqrt((1-|x|)/2)) = 2 * (d + poly)
    var d_plus_poly = d + poly

    # Select result based on domain
    var result = directed_polynomial_mask.select(y, 2.0 * d_plus_poly)

    # Large domain with negative x: apply π - result transformation.
    return (~directed_polynomial_mask & x.lt(0)).select(pi - result, result)


# ===----------------------------------------------------------------------=== #
# asin
# ===----------------------------------------------------------------------=== #


def asin[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> type_of(x) where dtype.is_floating_point():
    """Computes the `asin` of the inputs.

    Constraints:
        The input must be a floating-point type.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: The input argument.

    Returns:
        The `asin` of the input.
    """

    comptime if size_of[dtype]() < size_of[DType.float32]():
        return asin(x.cast[.float32]()).cast[dtype]()
    elif dtype == .float64:
        return _llvm_unary_fn["llvm.asin"](x)

    # For F32 types, use the Remez approximation found in Sleef with range
    # splitting to improve accuracy.

    # Domain split to use a different approximation outside the domain of
    # -0.5 <= x <= 0.5
    var x_abs = abs(x)
    var directed_polynomial_mask = x_abs.lt(0.5)

    # Compute d² for polynomial evaluation:
    #  - For |x| < 0.5: d² = x²
    #  - For |x| >= 0.5: d² = (1 - |x|) / 2  (for identity transformation)
    var d2 = directed_polynomial_mask.select(x * x, (1.0 - x_abs) * 0.5)

    # Compute d for evaluation:
    # - For |x| < 0.5: d = |x|
    # - For |x| >= 0.5: d = sqrt((1-|x|)/2)
    #   (using identity: asin(d) = π/2 - 2*asin(sqrt((1-d)/2)))
    var d = directed_polynomial_mask.select(x_abs, sqrt(d2))

    # Evaluate Remez polynomial approximation using Horner's method
    # This approximates the series: asin(x)/x ≈ 1 + x²/6 + 3x⁴/40 + ...
    var poly = polynomial_evaluate[
        [
            Scalar[x.dtype](0.1666677296e0),
            0.7495029271e-1,
            0.4547423869e-1,
            0.2424046025e-1,
            0.4197454825e-1,
        ]
    ](d2)

    # Final polynomial evaluation: poly*x*x² + x = x*(poly*x² + 1)
    poly = poly.fma(d * d2, d)

    # Compute final result based on domain:
    # - For |x| < 0.5: result = poly  (direct approximation)
    # - For |x| >= 0.5: result = π/2 - 2*poly  (using identity)
    var result = directed_polynomial_mask.select(poly, pi / 2.0 - 2.0 * poly)

    return copysign(result, x)


# ===----------------------------------------------------------------------=== #
# atan
# ===----------------------------------------------------------------------=== #


def atan[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> type_of(x) where dtype.is_floating_point():
    """Computes the `atan` of the inputs.

    Constraints:
        The input must be a floating-point type.
        `float64` is not supported on Apple GPUs.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: The input argument.

    Returns:
        The `atan` of the input.
    """

    _assert_float64_supported[dtype, "atan"]()

    comptime if bit_width_of[dtype]() < 32:
        return _narrow_from_f32[dtype](_atan_f32(x.cast[.float32]()))
    elif dtype == .float32:
        return _atan_f32(x._refine[.float32]())._refine[dtype]()
    return _atan_f64(x._refine[.float64]())._refine[dtype]()


# ===----------------------------------------------------------------------=== #
# atan2
# ===----------------------------------------------------------------------=== #


def atan2[
    dtype: DType, width: SIMDLength, //
](y: SIMD[dtype, width], x: SIMD[dtype, width]) -> SIMD[
    dtype, width
] where dtype.is_floating_point():
    """Computes the `atan2` of the inputs.

    Constraints:
        The input must be a floating-point type.
        `float64` is not supported on Apple GPUs.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        y: The first input argument.
        x: The second input argument.

    Returns:
        The `atan2` of the inputs.
    """

    _assert_float64_supported[dtype, "atan2"]()

    comptime if bit_width_of[dtype]() < 32:
        return _narrow_from_f32[dtype](
            _atan2_impl(y.cast[.float32](), x.cast[.float32]())
        )
    return _atan2_impl(y, x)


# ===----------------------------------------------------------------------=== #
# cos
# ===----------------------------------------------------------------------=== #


def cos[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    """Computes the `cos` of the inputs.

    Constraints:
        The input must be a floating-point type.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: The input argument.

    Returns:
        The `cos` of the input.
    """

    if __is_run_in_comptime_interpreter:
        return _llvm_unary_fn["llvm.cos"](x)

    comptime hook = CurrentPlugin.unary_math_fn[.cos, dtype, width]
    comptime if hook:
        return comptime (hook.value())(x)

    comptime if size_of[dtype]() < size_of[DType.float32]():
        return cos(x.cast[.float32]()).cast[dtype]()

    return _llvm_unary_fn["llvm.cos"](x)


# ===----------------------------------------------------------------------=== #
# sin
# ===----------------------------------------------------------------------=== #


def sin[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    """Computes the `sin` of the inputs.

    Constraints:
        The input must be a floating-point type.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: The input argument.

    Returns:
        The `sin` of the input.
    """

    if __is_run_in_comptime_interpreter:
        return _llvm_unary_fn["llvm.sin"](x)

    comptime hook = CurrentPlugin.unary_math_fn[.sin, dtype, width]
    comptime if hook:
        return comptime (hook.value())(x)

    comptime if size_of[dtype]() < size_of[DType.float32]():
        return sin(x.cast[.float32]()).cast[dtype]()

    return _llvm_unary_fn["llvm.sin"](x)


# ===----------------------------------------------------------------------=== #
# tan
# ===----------------------------------------------------------------------=== #


def tan[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> type_of(x) where dtype.is_floating_point():
    """Computes the `tan` of the inputs.

    Constraints:
        The input must be a floating-point type.
        `float64` is not supported on Apple GPUs.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: The input argument.

    Returns:
        The `tan` of the input.
    """

    _assert_float64_supported[dtype, "tan"]()

    comptime if bit_width_of[dtype]() < 32:
        return _narrow_from_f32[dtype](_tan_impl(x.cast[.float32]()))
    return _tan_impl(x)


# ===----------------------------------------------------------------------=== #
# acosh
# ===----------------------------------------------------------------------=== #


def acosh[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> type_of(x) where dtype.is_floating_point():
    """Computes the `acosh` of the inputs.

    Constraints:
        The input must be a floating-point type.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: The input argument.

    Returns:
        The `acosh` of the input.
    """

    _assert_float64_supported[dtype, "acosh"]()

    comptime if bit_width_of[dtype]() < 32:
        return _acosh_impl(x.cast[.float32]()).cast[dtype]()
    return _acosh_impl(x)


# ===----------------------------------------------------------------------=== #
# asinh
# ===----------------------------------------------------------------------=== #


def asinh[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> type_of(x) where dtype.is_floating_point():
    """Computes the `asinh` of the inputs.

    Constraints:
        The input must be a floating-point type.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: The input argument.

    Returns:
        The `asinh` of the input.
    """

    _assert_float64_supported[dtype, "asinh"]()

    comptime if bit_width_of[dtype]() < 32:
        return _narrow_from_f32[dtype](_asinh_impl(x.cast[.float32]()))
    return _asinh_impl(x)


# ===----------------------------------------------------------------------=== #
# atanh
# ===----------------------------------------------------------------------=== #


@inline(.always)
def atanh[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> type_of(x) where dtype.is_floating_point():
    """Computes the `atanh` of the inputs.

    Constraints:
        The input must be a floating-point type.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: The input argument.

    Returns:
        The `atanh` of the input.
    """

    _assert_float64_supported[dtype, "atanh"]()

    comptime if bit_width_of[dtype]() < 32:
        # We promote the input to float32 and then cast back to the original
        # type. This is done to avoid precision issues that can occur when
        # using the lower-precision floating-point types.
        return _narrow_from_f32[dtype](_atanh_impl(x.cast[.float32]()))
    return _atanh_impl(x)


# ===----------------------------------------------------------------------=== #
# cosh
# ===----------------------------------------------------------------------=== #


def cosh[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> type_of(x) where dtype.is_floating_point():
    """Computes the `cosh` of the inputs.

    Constraints:
        The input must be a floating-point type.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: The input argument.

    Returns:
        The `cosh` of the input.
    """

    # Compute exponential of absolute value: e^|x|
    # Using |x| exploits the even symmetry of cosh
    var e = exp(abs(x))

    # Compute cosh(x) = (e^|x| + e^-|x|) / 2
    # Rearranged as: 0.5 * e^|x| + 0.5 / e^|x|
    # This form is more efficient than (e + 1/e) / 2
    return 0.5 * e + 0.5 / e


# ===----------------------------------------------------------------------=== #
# sinh
# ===----------------------------------------------------------------------=== #


def sinh[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> type_of(x) where dtype.is_floating_point():
    """Computes the `sinh` of the inputs.

    Constraints:
        The input must be a floating-point type.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: The input argument.

    Returns:
        The `sinh` of the input.
    """

    # Compute e = expm1(|x|) = e^|x| - 1
    # Using |x| since we'll apply sign later (sinh is odd)
    # expm1 is more accurate than exp(x) - 1 for small x
    var e = expm1(abs(x))

    # Compute sinh(|x|) using the formula: e * (e+2) / (2(e+1))
    # This is derived from sinh(x) = (e^x - e^-x) / 2
    # Step 1: Compute (e + 2) / (e + 1)
    var y = (e + 2.0) / (e + 1.0)

    # Step 2: Multiply by 0.5 * e to complete the formula
    y = y * 0.5 * e

    # Apply the sign of the original input
    # sinh is an odd function: sinh(-x) = -sinh(x)
    y = copysign(y, x)

    return y


# ===----------------------------------------------------------------------=== #
# expm1
# ===----------------------------------------------------------------------=== #


@inline(.always)
def expm1[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> type_of(x) where dtype.is_floating_point():
    """Computes the `expm1` of the inputs.

    Constraints:
        The input must be a floating-point type.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: The input argument.

    Returns:
        The `expm1` of the input.
    """

    _assert_float64_supported[dtype, "expm1"]()

    comptime if bit_width_of[dtype]() < 32:
        # We promote the input to float32 and then cast back to the original
        # type. This is done to avoid precision issues that can occur when
        # using the lower-precision floating-point types.
        return _narrow_from_f32[dtype](_expm1_accurate(x.cast[.float32]()))
    return _expm1_accurate(x)


# ===----------------------------------------------------------------------=== #
# log10
# ===----------------------------------------------------------------------=== #


def log10[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> type_of(x) where dtype.is_floating_point():
    """Computes the `log10` of the inputs.

    Constraints:
        The input must be a floating-point type.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: The input argument.

    Returns:
        The `log10` of the input.
    """

    _assert_float64_supported[dtype, "log10"]()

    comptime if bit_width_of[dtype]() < 32:
        return _log10_impl(x.cast[.float32]()).cast[dtype]()
    return _log10_impl(x)


# ===----------------------------------------------------------------------=== #
# log1p
# ===----------------------------------------------------------------------=== #


def log1p[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> type_of(x) where dtype.is_floating_point():
    """Computes the `log1p` of the inputs.

    The `log1p(x)` is equivalent to `log(1+x)`, but stays accurate when `x` is
    close to zero, where `1 + x` would round away the low bits of `x`.

    The result is accurate to within 1 ulp for `float32` and `float64`.
    `float16` and `bfloat16` inputs are computed in `float32` and rounded once
    to the narrow type. Special values: `log1p(±0)` is `±0`, `log1p(-1)` is
    `-inf`, `log1p(+inf)` is `+inf`, and NaN is returned for NaN and for
    `x < -1`.

    Constraints:
        The input must be a floating-point type.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: The input argument.

    Returns:
        The `log1p` of the input.
    """

    _assert_float64_supported[dtype, "log1p"]()

    comptime if bit_width_of[dtype]() < 32:
        return _narrow_from_f32[dtype](log1p(x.cast[.float32]()))
    var r = _log1p_accurate(x)
    r = (x.eq(0) | x.eq(inf[dtype]()) | isnan(x)).select(x, r)
    comptime if CurrentPlugin.flushes_subnormals[dtype]:
        # log1p(x) rounds to x for subnormal x, which arithmetic would flush.
        r = _is_subnormal_bits(x).select(x, r)
    r = x.eq(-1).select(-inf[dtype](), r)
    return x.lt(-1).select(nan[dtype](), r)


# ===----------------------------------------------------------------------=== #
# logb
# ===----------------------------------------------------------------------=== #


def logb[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> type_of(x) where dtype.is_floating_point():
    """Computes the `logb` of the inputs.

    Constraints:
        The input must be a floating-point type.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: The input argument.

    Returns:
        The `logb` of the input.
    """

    _assert_float64_supported[dtype, "logb"]()

    comptime if dtype not in [
        DType.float16,
        DType.bfloat16,
        DType.float32,
        DType.float64,
    ]:
        return logb(x.cast[.float32]()).cast[dtype]()
    return _logb_bits(x)


# ===----------------------------------------------------------------------=== #
# cbrt
# ===----------------------------------------------------------------------=== #


def cbrt[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> type_of(x) where dtype.is_floating_point():
    """Computes the `cbrt` of the inputs.

    Constraints:
        The input must be a floating-point type.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: The input argument.

    Returns:
        The `cbrt` of the input.
    """

    _assert_float64_supported[dtype, "cbrt"]()

    comptime if bit_width_of[dtype]() < 32:
        return _cbrt_f32(x.cast[.float32]()).cast[dtype]()
    elif dtype == .float32:
        return _cbrt_f32(x._refine[.float32]())._refine[dtype]()
    return _cbrt_f64(x._refine[.float64]())._refine[dtype]()


# ===----------------------------------------------------------------------=== #
# hypot
# ===----------------------------------------------------------------------=== #


# TODO: implement for variadic inputs as Python.
def hypot[
    dtype: DType, width: SIMDLength, //
](arg0: SIMD[dtype, width], arg1: SIMD[dtype, width]) -> SIMD[
    dtype, width
] where dtype.is_floating_point():
    """Computes the `hypot` of the inputs.

    Constraints:
        The input must be a floating-point type. Float64 is unsupported on
        Apple GPU.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        arg0: The first input argument.
        arg1: The second input argument.

    Returns:
        The `hypot` of the inputs.
    """

    _assert_float64_supported[dtype, "hypot"]()

    comptime if bit_width_of[dtype]() < 32:
        return _narrow_from_f32[dtype](
            _hypot_impl(arg0.cast[.float32](), arg1.cast[.float32]())
        )
    return _hypot_impl(arg0, arg1)


# ===----------------------------------------------------------------------=== #
# erfc
# ===----------------------------------------------------------------------=== #


def erfc[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> type_of(x) where dtype.is_floating_point():
    """Computes the `erfc` of the inputs.

    Constraints:
        The input must be a floating-point type.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: The input argument.

    Returns:
        The `erfc` of the input.
    """

    _assert_float64_supported[dtype, "erfc"]()

    comptime if bit_width_of[dtype]() < 32:
        return _erfc_f32(x.cast[.float32]()).cast[dtype]()
    elif dtype == .float32:
        return _erfc_f32(x._refine[.float32]())._refine[dtype]()
    return _erfc_f64(x._refine[.float64]())._refine[dtype]()


# ===----------------------------------------------------------------------=== #
# lgamma
# ===----------------------------------------------------------------------=== #


def lgamma[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> type_of(x) where dtype.is_floating_point():
    """Computes the `lgamma` of the inputs.

    The error is about 2 ulp or less, including next to the negative real
    zeros of `lgamma` (such as -2.457 and -2.748), where the result is near
    0.

    Constraints:
        The input must be a floating-point type. `float64` is not supported
        on Apple GPU.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: The input argument.

    Returns:
        The `lgamma` of the input.
    """

    _assert_float64_supported[dtype, "lgamma"]()

    comptime if bit_width_of[dtype]() < 32:
        return _lgamma_impl(x.cast[.float32]()).cast[dtype]()
    return _lgamma_impl(x)


# ===----------------------------------------------------------------------=== #
# gamma
# ===----------------------------------------------------------------------=== #


def gamma[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> type_of(x) where dtype.is_floating_point():
    """Computes the Gamma of the input.

    For details, see https://en.wikipedia.org/wiki/Gamma_function.

    Constraints:
        The input must be a floating-point type.
        `float64` is not supported on Apple GPU targets.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: The input argument.

    Returns:
        The Gamma function evaluated at the input.
    """

    _assert_float64_supported[dtype, "gamma"]()

    comptime if bit_width_of[dtype]() < 32:
        return _narrow_from_f32[dtype](_gamma_impl(x.cast[.float32]()))
    return _gamma_impl(x)


# ===----------------------------------------------------------------------=== #
# remainder
# ===----------------------------------------------------------------------=== #


def remainder[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width], y: SIMD[dtype, width]) -> SIMD[
    dtype, width
] where dtype.is_floating_point():
    """Computes the `remainder` of the inputs.

    Constraints:
        The input must be a floating-point type.
        The `float64` type is not supported on Apple GPUs.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: The first input argument.
        y: The second input argument.

    Returns:
        The `remainder` of the inputs.
    """

    _assert_float64_supported[dtype, "remainder"]()

    comptime if dtype == .bfloat16:
        # bfloat16 shares float32's exponent range, so a subnormal bfloat16
        # is a subnormal float32. Widening through the bits keeps it from
        # being flushed to zero by a float conversion on targets that flush
        # subnormals, which would turn a subnormal divisor into NaN.
        comptime I32 = SIMD[.int32, width]
        # A bfloat16 is the top 16 bits of a float32.
        comptime bf16_to_f32_shift = 16
        var xw = SIMD[.float32, width](
            from_bits=x._to_bits_signed().cast[.int32]()
            << I32(bf16_to_f32_shift)
        )
        var yw = SIMD[.float32, width](
            from_bits=y._to_bits_signed().cast[.int32]()
            << I32(bf16_to_f32_shift)
        )
        # The result is exact in bfloat16 but may be subnormal.
        return _narrow_from_f32[dtype](_remainder_impl(xw, yw))
    elif bit_width_of[dtype]() < 32:
        return _remainder_impl(x.cast[.float32](), y.cast[.float32]()).cast[
            dtype
        ]()
    return _remainder_impl(x, y)


# ===----------------------------------------------------------------------=== #
# j0
# ===----------------------------------------------------------------------=== #


def j0[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> type_of(x) where dtype.is_floating_point():
    """Computes the Bessel function of the first kind of order 0 for each input
    value.

    For float32, relative error grows near the zeros of the function;
    absolute error stays small there.

    Constraints:
        The input must be a floating-point type.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: The input vector.

    Returns:
        A vector containing the computed value for each value in the input.
    """

    _assert_float64_supported[dtype, "j0"]()

    comptime if bit_width_of[dtype]() < 32:
        return _j0y0_impl[False](x.cast[.float32]()).cast[dtype]()
    return _j0y0_impl[False](x)


# ===----------------------------------------------------------------------=== #
# j1
# ===----------------------------------------------------------------------=== #


def j1[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> type_of(x) where dtype.is_floating_point():
    """Computes the Bessel function of the first kind of order 1 for each input
    value.

    For float32, relative error grows near the zeros of the function;
    absolute error stays small there.

    Constraints:
        The input must be a floating-point type.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: The input vector.

    Returns:
        A vector containing the computed value for each value in the input.
    """

    _assert_float64_supported[dtype, "j1"]()

    comptime if bit_width_of[dtype]() < 32:
        return _narrow_from_f32[dtype](
            _j1y1_impl[is_y1=False](x.cast[.float32]())
        )
    return _j1y1_impl[is_y1=False](x)


# ===----------------------------------------------------------------------=== #
# y0
# ===----------------------------------------------------------------------=== #


def y0[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> type_of(x) where dtype.is_floating_point():
    """Computes the Bessel function of the second kind of order 0 for each input
    value.

    For float32, relative error grows near the zeros of the function;
    absolute error stays small there.

    Constraints:
        The input must be a floating-point type.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: The input vector.

    Returns:
        A vector containing the computed value for each value in the input.
    """

    _assert_float64_supported[dtype, "y0"]()

    comptime if bit_width_of[dtype]() < 32:
        return _j0y0_impl[True](x.cast[.float32]()).cast[dtype]()
    return _j0y0_impl[True](x)


# ===----------------------------------------------------------------------=== #
# y1
# ===----------------------------------------------------------------------=== #


def y1[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> type_of(x) where dtype.is_floating_point():
    """Computes the Bessel function of the second kind of order 1 for each input
    value.

    For float32, relative error grows near the zeros of the function;
    absolute error stays small there.

    Constraints:
        The input must be a floating-point type.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: The input vector.

    Returns:
        A vector containing the computed value for each value in the input.
    """

    _assert_float64_supported[dtype, "y1"]()

    comptime if bit_width_of[dtype]() < 32:
        return _j1y1_impl[is_y1=True](x.cast[.float32]()).cast[dtype]()
    return _j1y1_impl[is_y1=True](x)


# ===----------------------------------------------------------------------=== #
# scalb
# ===----------------------------------------------------------------------=== #


def scalb[
    dtype: DType, width: SIMDLength, //
](arg0: SIMD[dtype, width], arg1: SIMD[dtype, width]) -> SIMD[
    dtype, width
] where dtype.is_floating_point():
    """Scales each value in `arg0` by 2 raised to its exponent in `arg1`.

    A finite, non-integral exponent produces NaN. A positive infinite
    exponent produces infinity with the sign of `arg0`, except that a zero
    base produces NaN. A negative infinite exponent produces zero with the
    sign of `arg0`, except that an infinite base produces NaN. NaN arguments
    produce NaN.

    For example:

    ```mojo
    from std.math import scalb
    var scaled = scalb(Float32(1.5), Float32(2.0))
    ```

    Constraints:
        The input must be a floating-point type. The `float64` type is not
        supported on Apple GPU.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        arg0: The values to scale.
        arg1: The base-2 exponents.

    Returns:
        The scaled values, computed elementwise.
    """

    _assert_float64_supported[dtype, "scalb"]()

    comptime if bit_width_of[dtype]() < 32:
        return _narrow_from_f32[dtype](
            _scalb_impl(arg0.cast[.float32](), arg1.cast[.float32]())
        )
    return _scalb_impl(arg0, arg1)


# ===----------------------------------------------------------------------=== #
# gcd
# ===----------------------------------------------------------------------=== #


def gcd(m: Int, n: Int, /) -> Int:
    """Compute the greatest common divisor of two integers.

    Args:
        m: The first integer.
        n: The second integrer.

    Returns:
        The greatest common divisor of the two integers.
    """
    var u = abs(m)
    var v = abs(n)
    if u == 0:
        return v
    if v == 0:
        return u

    var uz = count_trailing_zeros(u)
    var vz = count_trailing_zeros(v)
    var shift = min(uz, vz)
    u >>= shift
    while True:
        v >>= vz
        var diff = v - u
        if diff == 0:
            break
        u, v = min(u, v), abs(diff)
        vz = count_trailing_zeros(diff)
    return u << shift


def gcd(s: Span[Int, _], /) -> Int:
    """Computes the greatest common divisor of a span of integers.

    Args:
        s: A span containing a collection of integers.

    Returns:
        The greatest common divisor of all the integers in the span.
    """
    if len(s) == 0:
        return 0
    var result = s[0]
    for item in s[1:]:
        result = gcd(item, result)
        if result == 1:
            return result
    return result


def gcd(*values: Int) -> Int:
    """Computes the greatest common divisor of a variadic number of integers.

    Args:
        values: A variadic list of integers.

    Returns:
        The greatest common divisor of the given integers.
    """
    # TODO: Deduplicate when VariadicList is iterable.
    if len(values) == 0:
        return 0
    var result = values[0]
    for i in range(1, len(values)):
        result = gcd(values[i], result)
        if result == 1:
            return result
    return result


# ===----------------------------------------------------------------------=== #
# lcm
# ===----------------------------------------------------------------------=== #


def lcm(m: Int, n: Int, /) -> Int:
    """Computes the least common multiple of two integers.

    Args:
        m: The first integer.
        n: The second integer.

    Returns:
        The least common multiple of the two integers.
    """
    var d = gcd(m, n)
    if d:
        return abs((m // d) * n if m > n else (n // d) * m)
    return 0


def lcm(s: Span[Int, _], /) -> Int:
    """Computes the least common multiple of a span of integers.

    Args:
        s: A span of integers.

    Returns:
        The least common multiple of the span.
    """
    if len(s) == 0:
        return 1

    var result = s[0]
    for item in s[1:]:
        result = lcm(result, item)
    return result


def lcm(*values: Int) -> Int:
    """Computes the least common multiple of a variadic list of integers.

    Args:
        values: A variadic list of integers.

    Returns:
        The least common multiple of the list.
    """
    # TODO: Deduplicate when VariadicList is iterable.
    if len(values) == 0:
        return 1

    var result = values[0]
    for i in range(1, len(values)):
        result = lcm(result, values[i])
    return result


# ===----------------------------------------------------------------------=== #
# modf
# ===----------------------------------------------------------------------=== #


def modf[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> Tuple[type_of(x), type_of(x)]:
    """Computes the integral and fractional part of the value.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: The input value.

    Returns:
        A tuple containing the integral and fractional part of the value.
    """
    return _modf(x)


# ===----------------------------------------------------------------------=== #
# ulp
# ===----------------------------------------------------------------------=== #


@inline(.always)
def ulp[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    """Computes the ULP (units of last place) or (units of least precision) of
    the number.

    Constraints:
        The element type of the inpiut must be a floating-point type.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: SIMD vector input.

    Returns:
        The ULP of x.
    """

    var nan_mask = isnan(x)
    var xabs = abs(x)
    var inf_mask = isinf(xabs)
    comptime inf_val = SIMD[dtype, width](inf[dtype]())
    var x2 = nextafter(xabs, inf_val)
    var x2_inf_mask = isinf(x2)

    return nan_mask.select(
        x,
        inf_mask.select(
            xabs,
            x2_inf_mask.select(xabs - nextafter(xabs, -inf_val), x2 - xabs),
        ),
    )


# ===----------------------------------------------------------------------=== #
# factorial
# ===----------------------------------------------------------------------=== #


# TODO: implement for IntLiteral
@inline(.always)
def factorial(n: Int) -> Int:
    """Computes the factorial of the integer.

    Args:
        n: The input value. Must be non-negative.

    Returns:
        The factorial of the input. Results are undefined for negative inputs.
    """
    comptime table = StaticTuple[Int, 21](
        1,
        1,
        2,
        6,
        24,
        120,
        720,
        5040,
        40320,
        362880,
        3628800,
        39916800,
        479001600,
        6227020800,
        87178291200,
        1307674368000,
        20922789888000,
        355687428096000,
        6402373705728000,
        121645100408832000,
        2432902008176640000,
    )
    assert (
        0 <= n <= (12 if is_32bit() else 20)
    ), "input value causes an overflow"
    return table[n]


def comb(n: Int, k: Int) -> Int:
    """Computes the number of ways to choose `k` items from `n` items without
    repetition and without order (binomial coefficient).

    Equivalent to Python's `math.comb(n, k)`.

    Args:
        n: The total number of items. Must be non-negative.
        k: The number of items to choose. Must be non-negative.

    Returns:
        The binomial coefficient C(n, k). Returns 0 if `k > n`. Asserts if
        either argument is negative.

    Examples:

    ```mojo
    from std.math import comb
    print(comb(5, 2))  # 10
    print(comb(10, 0)) # 1
    print(comb(3, 5))  # 0
    ```
    """
    assert n >= 0, "n must be non-negative"
    assert k >= 0, "k must be non-negative"
    if k > n:
        return 0
    # Use the smaller of k and n-k to minimise the number of iterations.
    var k2 = k if k <= n - k else n - k
    var result = 1
    for i in range(k2):
        # Reduce (n-i) and (i+1) by their gcd before multiplying to avoid
        # intermediate overflow when the final result fits in Int.
        var num = n - i
        var den = i + 1
        var g = gcd(num, den)
        num //= g
        den //= g
        # After removing gcd(num, den), the remaining den divides result
        # exactly (since C(n, i+1) is always an integer).
        result = (result // den) * num
    return result


def perm(n: Int, k: Int = -1) -> Int:
    """Computes the number of ways to arrange `k` items from `n` items without
    repetition (permutations).

    Equivalent to Python's `math.perm(n, k)`.

    Args:
        n: The total number of items. Must be non-negative.
        k: The number of items to arrange. Must be non-negative and at most `n`.
           If omitted (default), returns `n!` via `factorial(n)`.

    Returns:
        The number of permutations P(n, k) = n! / (n-k)!. Asserts if `n` is
        negative, `k` is negative, or `k > n`.

    Examples:

    ```mojo
    from std.math import perm
    print(perm(5, 2))  # 20
    print(perm(5))     # 120  (same as factorial(5))
    print(perm(5, 0))  # 1
    ```
    """
    assert n >= 0, "n must be non-negative"
    if k == -1:
        return factorial(n)
    assert 0 <= k <= n, "k must be between 0 and n"
    var result = 1
    for i in range(k):
        result *= n - i
    return result


# ===----------------------------------------------------------------------=== #
# clamp
# ===----------------------------------------------------------------------=== #


def clamp[
    dtype: DType, width: SIMDLength, //
](
    val: SIMD[dtype, width],
    lower_bound: type_of(val),
    upper_bound: type_of(val),
) -> type_of(val):
    """Clamps the values in a SIMD vector to be in a certain range.

    Clamp cuts values in the input SIMD vector off at the upper bound and
    lower bound values. For example,  SIMD vector `[0, 1, 2, 3]` clamped to
    a lower bound of 1 and an upper bound of 2 would return `[1, 1, 2, 2]`.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        val: The value to clamp.
        lower_bound: Minimum of the range to clamp to.
        upper_bound: Maximum of the range to clamp to.

    Returns:
        A SIMD vector containing x clamped to be within lower_bound and
        upper_bound.
    """
    return val.clamp(lower_bound, upper_bound)


# ===----------------------------------------------------------------------=== #
# utilities
# ===----------------------------------------------------------------------=== #


@inline(.nodebug)
def _llvm_unary_fn[
    dtype: DType,
    width: SIMDLength,
    //,
    fn_name: StaticString,
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    return llvm_intrinsic[fn_name, type_of(x), has_side_effect=False](x)


@inline(.nodebug)
def _call_libm[
    dtype: DType,
    width: SIMDLength,
    //,
    func_name: StaticString,
](arg: SIMD[dtype, width]) -> SIMD[
    dtype, width
] where dtype.is_floating_point():
    comptime assert (
        not is_gpu()
    ), "libm operations are only available on CPU targets"

    comptime if dtype not in [DType.float32, DType.float64]:
        # Coerce to f32 if the value is not representable by libm.
        var arg_f32 = arg.cast[.float32]()
        return _call_libm[func_name](arg_f32).cast[dtype]()

    comptime libm_name = func_name + ("f" if dtype == .float32 else "")
    var res = SIMD[dtype, width]()

    comptime for i in range(width):
        res[i] = _external_call_const[libm_name, Scalar[dtype]](arg[i])
    return res


# ===----------------------------------------------------------------------=== #
# Ceilable
# ===----------------------------------------------------------------------=== #


trait Ceilable:
    """
    The `Ceilable` trait describes a type that defines a ceiling operation.

    Types that conform to `Ceilable` will work with the builtin `ceil`
    function. The ceiling operation always returns the same type as the input.

    For example:
    ```mojo
    from std.math import Ceilable, ceil

    @fieldwise_init
    struct Complex(Ceilable, ImplicitlyCopyable):
        var re: Float64
        var im: Float64

        def __ceil__(self) -> Self:
            return Self(ceil(self.re), ceil(self.im))
    ```
    """

    # TODO(MOCO-333): Reconsider the signature when we have parametric traits or
    # associated types.
    def __ceil__(self) -> Self:
        """Return the ceiling of the Int value, which is itself.

        Returns:
            The Int value itself.
        """
        ...


# ===----------------------------------------------------------------------=== #
# Floorable
# ===----------------------------------------------------------------------=== #


trait Floorable:
    """
    The `Floorable` trait describes a type that defines a floor operation.

    Types that conform to `Floorable` will work with the builtin `floor`
    function. The floor operation always returns the same type as the input.

    For example:
    ```mojo
    from std.math import Floorable, floor

    @fieldwise_init
    struct Complex(Floorable, ImplicitlyCopyable):
        var re: Float64
        var im: Float64

        def __floor__(self) -> Self:
            return Self(floor(self.re), floor(self.im))
    ```
    """

    # TODO(MOCO-333): Reconsider the signature when we have parametric traits or
    # associated types.
    def __floor__(self) -> Self:
        """Return the floor of the Int value, which is itself.

        Returns:
            The Int value itself.
        """
        ...


# ===----------------------------------------------------------------------=== #
# CeilDivable
# ===----------------------------------------------------------------------=== #


trait CeilDivable:
    """
    The `CeilDivable` trait describes a type that defines a ceil division
    operation.

    Types that conform to `CeilDivable` will work with the `math.ceildiv`
    function.

    For example:
    ```mojo
    from std.math import CeilDivable

    @fieldwise_init
    struct Foo(CeilDivable, ImplicitlyCopyable):
        var x: Float64

        def __ceildiv__(self, denominator: Self) -> Self:
            return Self(self.x // denominator.x)
    ```
    """

    def __ceildiv__(self, denominator: Self) -> Self:
        """Return the rounded-up result of dividing self by denominator.

        Args:
            denominator: The denominator.

        Returns:
            The ceiling of dividing numerator by denominator.
        """
        ...


trait CeilDivableRaising:
    """
    The `CeilDivable` trait describes a type that define a floor division and
    negation operation that can raise.

    Types that conform to `CeilDivableRaising` will work with the `//` operator
    as well as the `math.ceildiv` function.

    For example:
    ```mojo
    from std.math import CeilDivableRaising

    @fieldwise_init
    struct Foo(CeilDivableRaising, ImplicitlyCopyable):
        var x: Float64

        def __ceildiv__(self, denominator: Self) raises -> Self:
            return Self(self.x // denominator.x)
    ```
    """

    def __ceildiv__(self, denominator: Self) raises -> Self:
        """Return the rounded-up result of dividing self by denominator.

        Args:
            denominator: The denominator.

        Returns:
            The ceiling of dividing numerator by denominator.

        Raises:
            If the operation fails.
        """
        ...


# ===----------------------------------------------------------------------=== #
# Truncable
# ===----------------------------------------------------------------------=== #


trait Truncable:
    """
    The `Truncable` trait describes a type that defines a truncation operation.

    Types that conform to `Truncable` will work with the builtin `trunc`
    function. The truncation operation always returns the same type as the
    input.

    For example:
    ```mojo
    from std.math import Truncable, trunc

    @fieldwise_init
    struct Complex(Truncable, ImplicitlyCopyable):
        var re: Float64
        var im: Float64

        def __trunc__(self) -> Self:
            return Self(trunc(self.re), trunc(self.im))
    ```
    """

    # TODO(MOCO-333): Reconsider the signature when we have parametric traits or
    # associated types.
    def __trunc__(self) -> Self:
        """Return the truncated Int value, which is itself.

        Returns:
            The Int value itself.
        """
        ...


# ===----------------------------------------------------------------------=== #
# abs
# ===----------------------------------------------------------------------=== #


trait Absable:
    """
    The `Absable` trait describes a type that defines an absolute value
    operation.

    Types that conform to `Absable` will work with the builtin `abs` function.
    The absolute value operation always returns the same type as the input.

    For example:
    ```mojo
    struct Celsius(Absable):
        var degrees: Float64

        def __init__(out self, degrees: Float64):
            self.degrees = degrees

        def __abs__(self) -> Self:
            return Self(abs(self.degrees))
    ```
    """

    # TODO(MOCO-333): Reconsider the signature when we have parametric traits or
    # associated types.
    def __abs__(self) -> Self:
        """Get the absolute value of this instance.

        Returns:
            The absolute value of the instance.
        """
        ...


@inline(.always)
def abs[T: Absable](value: T) -> T:
    """Get the absolute value of the given object.

    Parameters:
        T: The type conforming to Absable.

    Args:
        value: The object to get the absolute value of.

    Returns:
        The absolute value of the object.
    """
    return value.__abs__()


# ===----------------------------------------------------------------------=== #
# divmod
# ===----------------------------------------------------------------------=== #


trait DivModable(ImplicitlyCopyable):
    """
    The `DivModable` trait describes a type that defines division and
    modulo operations returning both quotient and remainder.

    Types that conform to `DivModable` will work with the builtin `divmod` function,
    which will return the same type as the inputs.

    For example:
    ```mojo
    from std.math import DivModable

    @fieldwise_init
    struct Bytes(DivModable):
        var size: Int

        def __divmod__(self, other: Self) -> Tuple[Self, Self]:
            var quotient_int = self.size // other.size
            var remainder_int = self.size % other.size
            return (Bytes(quotient_int), Bytes(remainder_int))
    ```
    """

    def __divmod__(self, denominator: Self) -> Tuple[Self, Self]:
        """Performs division and returns the quotient and the remainder.

        Args:
            denominator: The value to divide by.

        Returns:
            A `Tuple` containing the quotient and the remainder.
        """
        ...


def divmod[T: DivModable](numerator: T, denominator: T) -> Tuple[T, T]:
    """Performs division and returns the quotient and the remainder.

    Parameters:
        T: A type conforming to the `DivModable` trait.

    Args:
        numerator: The dividend.
        denominator: The divisor.

    Returns:
        A `Tuple` containing the quotient and the remainder.
    """
    return numerator.__divmod__(denominator)


# ===----------------------------------------------------------------------=== #
# max
# ===----------------------------------------------------------------------=== #


@inline(.nodebug)
def max[dtype: DType, //](x: SIMD[dtype, _], y: type_of(x), /) -> type_of(x):
    """Performs elementwise maximum of x and y.

    An element of the result SIMD vector will be the maximum of the
    corresponding elements in x and y.

    Constraints:
        The type of the inputs must be numeric or boolean.

    Parameters:
            dtype: The data type of the SIMD vector.

    Args:
        x: First SIMD vector.
        y: Second SIMD vector.

    Returns:
        A SIMD vector containing the elementwise maximum of x and y.
    """

    comptime assert (
        x.dtype == .bool or x.dtype.is_numeric()
    ), "the SIMD type must be numeric or boolean"

    return {mlir_value = __mlir_op.`pop.max`(x._mlir_value, y._mlir_value)}


@inline(.always)
def max[T: Copyable & Comparable & Deinitable](x: T, *ys: T) -> T:
    """Gets the maximum value from a sequence of values.

    Parameters:
        T: A type that is both copyable and comparable with greater than.

    Args:
        x: The first value to compare.
        ys: Zero or more additional values to compare.

    Returns:
        The maximum value from the input sequence.
    """
    var res = x.copy()
    for y in ys:
        if y > res:
            res = y.copy()
    return res.copy()


# ===----------------------------------------------------------------------=== #
# min
# ===----------------------------------------------------------------------=== #


@inline(.nodebug)
def min[dtype: DType, //](x: SIMD[dtype, _], y: type_of(x), /) -> type_of(x):
    """Gets the elementwise minimum of x and y.

    An element of the result SIMD vector will be the minimum of the
    corresponding elements in x and y.

    Constraints:
        The type of the inputs must be numeric or boolean.

    Parameters:
         dtype: The data type of the SIMD vector.

    Args:
        x: First SIMD vector.
        y: Second SIMD vector.

    Returns:
        A SIMD vector containing the elementwise minimum of x and y.
    """

    comptime assert (
        x.dtype == .bool or x.dtype.is_numeric()
    ), "the SIMD type must be numeric or boolean"

    return {mlir_value = __mlir_op.`pop.min`(x._mlir_value, y._mlir_value)}


@inline(.always)
def min[T: Copyable & Comparable & Deinitable](x: T, *ys: T) -> T:
    """Gets the minimum value from a sequence of values.

    Parameters:
        T: A type that is both copyable and comparable with less than.

    Args:
        x: The first value to compare.
        ys: Zero or more additional values to compare.

    Returns:
        The minimum value from the input sequence.
    """
    var res = x.copy()
    for y in ys:
        if y < res:
            res = y.copy()
    return res.copy()


# ===----------------------------------------------------------------------=== #
# pow
# ===----------------------------------------------------------------------=== #


trait Powable:
    """
    The `Powable` trait describes a type that defines a power operation (i.e.
    exponentiation) with the same base and exponent types.

    Types that conform to `Powable` will work with the builtin `pow` function,
    which will return the same type as the inputs.

    For example:
    ```mojo
    struct Rational(Powable):
        var numerator: Float64
        var denominator: Float64

        def __init__(out self, numerator: Float64, denominator: Float64):
            self.numerator = numerator
            self.denominator = denominator

        def __pow__(self, exp: Self)  -> Self:
            var exp_value = exp.numerator / exp.denominator
            return Self(pow(self.numerator, exp_value), pow(self.denominator, exp_value))
    ```

    You can now use the ** operator to exponentiate objects
    inside generic functions:

    ```mojo
    def exponentiate[T: Powable](base: T, exp: T) -> T:
        return base ** exp

    var base = Rational(Float64(3.0), 5.0)
    var exp = Rational(Float64(1.0), 2.0)
    var res = exponentiate(base, exp)
    ```

    ```plaintext
    raising to power
    ```
    """

    # TODO(MOCO-333): Reconsider the signature when we have parametric traits or
    # associated types.
    def __pow__(self, exp: Self) -> Self:
        """Return the value raised to the power of the given exponent.

        Args:
            exp: The exponent value.

        Returns:
            The value of `self` raised to the power of `exp`.
        """
        ...


def pow[T: Powable](base: T, exp: T) -> T:
    """Computes the `base` raised to the power of the `exp`.

    Parameters:
        T: A type conforming to the `Powable` trait.

    Args:
        base: The base of the power operation.
        exp: The exponent of the power operation.

    Returns:
        The `base` raised to the power of the `exp`.
    """
    return base.__pow__(exp)


def pow(base: SIMD, exp: Int) -> type_of(base):
    """Computes elementwise value of a SIMD vector raised to the power of the
    given integer.

    Args:
        base: The first input argument.
        exp: The second input argument.

    Returns:
        The `base` elementwise raised raised to the power of `exp`.
    """
    return base.__pow__(exp)


# ===----------------------------------------------------------------------=== #
# round
# ===----------------------------------------------------------------------=== #


trait Roundable:
    """
    The `Roundable` trait describes a type that defines a rounding operation.

    Types that conform to `Roundable` will work with the builtin `round`
    function. The round operation always returns the same type as the input.

    For example:
    ```mojo
    @fieldwise_init
    struct Complex(Roundable):
        var re: Float64
        var im: Float64

        def __round__(self) -> Self:
            return Self(round(self.re), round(self.im))

        def __round__(self, ndigits: Int) -> Self:
            return Self(round(self.re, ndigits), round(self.im, ndigits))
    ```
    """

    # TODO(MOCO-333): Reconsider the signature when we have parametric traits or
    # associated types.
    def __round__(self) -> Self:
        """Get a rounded value for the type.

        Returns:
            The rounded value.
        """
        ...

    def __round__(self, ndigits: Int) -> Self:
        """Get a rounded value for the type.

        Args:
            ndigits: Number of digits after the decimal point.

        Returns:
            The rounded value.
        """
        ...


@inline(.always)
def round[T: Roundable, //](number: T) -> T:
    """Get the rounded value of the given object.

    Parameters:
        T: The type conforming to Roundable.

    Args:
        number: The object to get the rounded value of.

    Returns:
        The rounded value of the object.
    """
    return number.__round__()


@inline(.always)
def round[T: Roundable, //](number: T, ndigits: Int) -> T:
    """Get the value of this object, rounded to a specified number of
    digits after the decimal point.

    Parameters:
        T: The type conforming to Roundable.

    Args:
        number: The object to get the rounded value of.
        ndigits: The number of digits to round to.

    Returns:
        The rounded value of the object.
    """
    return number.__round__(ndigits)


# ===----------------------------------------------------------------------=== #
# Pure-Mojo libm implementations
# ===----------------------------------------------------------------------=== #
# These run unchanged on CPU and GPU, so they avoid the target-dependent
# approximations the GPU plugins lower to (`sqrt.approx`, `lg2.approx`,
# `ex2.approx` on NVIDIA) and keep float32 paths free of float64 values, which
# Apple GPUs reject. Algorithm provenance is documented with each
# implementation. GPU backends contract a*b + c into fma where CPU code
# generation does not, so CPU and GPU results may differ in the last ulp
# outside paths that use explicit fma. Apple GPUs may also flush subnormals and
# carry no correct-rounding guarantee for sqrt.

comptime _SQRT2 = 1.41421356237309504880168872420969807856967187537694

comptime _LN2 = 0.69314718055994530941723212145817656807550013436026

# ln2 = _LN2_HI_F64 + _LN2_LO_F64, where _LN2_HI_F64 has 32 significant bits
# (21 trailing zero bits), so k * _LN2_HI_F64 is exact for |k| < 2^21.
comptime _LN2_HI_F64 = 6.93147180369123816490e-01
comptime _LN2_LO_F64 = 1.90821492927058770002e-10

comptime _F32_ABS_MASK = 0x7FFFFFFF


comptime _TWO_OVER_PI_BITS: List[UInt64] = [
    0x0000000000000000,
    0xA2F9836E4E441529,
    0xFC2757D1F534DDC0,
    0xDB6295993C439041,
    0xFE5163ABDEBBC561,
    0xB7246E3A424DD2E0,
    0x06492EEA09D1921C,
    0xFE1DEB1CB129A73E,
    0xE88235F52EBB4484,
    0xE99C7026B45F7E41,
    0x3991D639835339F4,
    0x9C845F8BBDF9283B,
    0x1FF897FFDE05980F,
    0xEF2F118B5A0A6D1F,
    0x6D367ECF27CB09B7,
    0x4F463F669E5FEA2D,
    0x7527BAC7EBE5F17B,
    0x3D0739F78A5292EA,
    0x6BFB5FB11F8D5D08,
    0x56033046FC7B6BAB,
]

comptime _TWO_OVER_PI = 0.63661977236758134307553505349005744813783858296183

comptime _SQRT_TWO_OVER_PI = 0.79788456080286535587989211986876373695171726232987

comptime _PQ_SPLIT_8 = 8.0

comptime _PQ_SPLIT_5 = 4.545452117919922

comptime _PQ_SPLIT_3 = 2.8571414947509766

comptime _BESSEL_ASYMPTOTIC_MIN = 2.0


# ===----------------------------------------------------------------------=== #
# Shared kernels
# ===----------------------------------------------------------------------=== #


@inline(.always)
def _assert_float64_supported[dtype: DType, name: StaticString]():
    comptime assert dtype != .float64 or CurrentPlugin.supports_float64, (
        "`" + name + "` does not support float64 on this target"
    )


# See Licenses/Third-Party-Notices entries 19 and 20 for source notices.
def _cbrt_f32[width: SIMDLength, //](x: SIMD[.float32, width]) -> type_of(x):
    """Computes the cube root of float32 values in pure Mojo.

    Follows FreeBSD msun `s_cbrtf.c` (bit-level estimate, Halley step) but
    stays in float32: the exponent is split as `3q + r` by integer division so
    the iteration runs on `m = 2^r * mantissa` in [1, 8), where nothing can
    overflow, and the last Newton step takes `m - t^3` from FMA-based
    error-free products. Max error is 0.505 ulp over all float32 inputs.
    """
    comptime sign_mask = 0x80000000
    comptime abs_mask = 0x7FFFFFFF
    comptime inf_bits = 0x7F800000
    comptime min_normal_bits = 0x00800000
    comptime mantissa_mask = 0x7FFFFF
    comptime B1 = UInt32(709958130)  # (127 - 127/3 - 0.03306235651) * 2^23

    var ux = bitcast[.uint32](x)
    var hx = ux & abs_mask
    # A subnormal is `hx * 2^-149`; converting the integer `hx` makes its
    # exponent readable without arithmetic on a value that targets flushing
    # subnormals (Apple GPUs) would treat as zero.
    var is_tiny = hx.lt(min_normal_bits)
    var hs = is_tiny.select(bitcast[.uint32](hx.cast[.float32]()), hx)
    # For finite non-zero x, e + 150 lies in [1, 277], so unsigned division
    # by 3 floors. Zero lanes wrap here; the final select discards their
    # intermediates.
    var e_field = hs >> 23
    var eb = is_tiny.select(e_field - (127 + 149 - 150), e_field + (150 - 127))
    var q = eb / 3
    var r = eb - q * 3
    var m = bitcast[.float32]((hs & mantissa_mask) | ((r + 127) << 23))
    # q - 50 is in [-50, 42], so 2^(q - 50) is normal and the scaling exact.
    var scale = bitcast[.float32]((q + (127 - 50)) << 23)

    # ~5 bits, then Halley to ~16 bits.
    var t = bitcast[.float32](bitcast[.uint32](m) / 3 + B1)
    var t3 = t * t * t
    t = t * (m + m + t3) / (m + t3 + t3)

    # Newton on t^3 = m with the residual from t*t = t2 + t2_lo, so the step
    # lands within 0.005 ulp of the root before the final rounding.
    var t2 = t * t
    var t2_lo = fma(t, t, -t2)
    var res = fma(-t2_lo, t, fma(-t2, t, m))
    t = t + res / (3.0 * t2)

    var y = bitcast[.float32](bitcast[.uint32](t * scale) | (ux & sign_mask))
    return (hx.ge(inf_bits) | hx.eq(0)).select(x + x, y)


def _cbrt_f64[width: SIMDLength, //](x: SIMD[.float64, width]) -> type_of(x):
    """Computes the cube root of float64 values in pure Mojo.

    Ported from FreeBSD msun / musl `s_cbrt.c`: a bit-level estimate of the
    exponent, a polynomial for `1/cbrt` to ~23 bits, rounding to 22 bits so
    `t*t` is exact, then one Halley step in float64.
    """
    # Masks and thresholds for the high 32 bits of a float64.
    comptime hi_abs_mask = 0x7FFFFFFF
    comptime hi_inf_bits = 0x7FF00000
    comptime hi_min_normal_bits = 0x00100000
    comptime B1 = UInt32(715094163)  # (1023 - 1023/3 - 0.03306235651) * 2^20
    comptime B2 = UInt32(696219795)  # (1023 - 1023/3 - 54/3 - 0.0330...) * 2^20
    comptime P0 = 1.87595182427177009643
    comptime P1 = -1.88497979543377169875
    comptime P2 = 1.621429720105354466140
    comptime P3 = -0.758397934778766047437
    comptime P4 = 0.145996192886612446982

    var hx = (bitcast[.uint64](x) >> 32).cast[.uint32]() & hi_abs_mask
    var is_special = hx.ge(hi_inf_bits)
    var is_tiny = hx.lt(hi_min_normal_bits)

    # Subnormals are scaled by 2^54 so their exponent is readable.
    comptime two_54 = 18014398509481984.0
    var hx_scaled = (bitcast[.uint64](x * two_54) >> 32).cast[
        .uint32
    ]() & hi_abs_mask
    var is_zero = is_tiny & hx_scaled.eq(0)
    var hi = is_tiny.select(hx_scaled / 3 + B2, hx / 3 + B1)
    var t = bitcast[.float64](
        (bitcast[.uint64](x) & (UInt64(1) << 63)) | (hi.cast[.uint64]() << 32)
    )

    var r = (t * t) * (t / x)
    t = t * polynomial_evaluate[[Float64(P0), P1, P2, P3, P4]](r)

    t = bitcast[.float64](
        (bitcast[.uint64](t) + 0x80000000) & 0xFFFFFFFFC0000000
    )

    var s = t * t
    r = x / s
    var w = t + t
    r = (r - t) / (w + r)
    t = t + t * r

    return (is_special | is_zero).select(x + x, t)


@inline(.always)
def _logb_bits[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> type_of(x) where dtype.is_floating_point():
    # Original algorithm: reads the exponent from the bit pattern alone, with a
    # leading-zero count for subnormals, so targets that flush subnormal inputs
    # still see them. A NaN input returns a NaN with its payload kept; unlike
    # libm, the sign bit is cleared and a signaling NaN is not quieted.
    comptime TInt = SIMD[_integral_type_of[dtype](), width]
    comptime mantissa_width = FPUtils[dtype].mantissa_width()
    comptime bias = FPUtils[dtype].exponent_bias()
    comptime exp_field_max = (1 << FPUtils[dtype].exponent_width()) - 1

    # Shifting out the sign bit leaves exponent and mantissa top-aligned.
    var bits = x.to_bits() << 1
    var exp_field = (bits >> type_of(bits)(mantissa_width + 1)).cast[
        TInt.dtype
    ]()
    # A subnormal whose top set bit is bit b of `bits` has exponent
    # b - bias - mantissa_width, and b = bit_width - 1 - clz.
    var sub_e = (
        TInt(bit_width_of[dtype]() - 1 - mantissa_width - bias)
        - count_leading_zeros(bits).cast[TInt.dtype]()
    )
    var e = exp_field.eq(0).select(sub_e, exp_field - TInt(bias))
    var result = e.cast[dtype]()

    result = exp_field.eq(TInt(exp_field_max)).select(abs(x), result)
    return bits.eq(0).select(-inf[dtype](), result)


def _two_sum[
    dtype: DType, width: SIMDLength, //
](a: SIMD[dtype, width], b: type_of(a)) -> Tuple[type_of(a), type_of(a)]:
    """Returns `(s, e)` with `s = fl(a + b)` and `a + b == s + e` exactly."""
    var s = a + b
    var bb = s - a
    return (s, (a - (s - bb)) + (b - bb))


@inline(.always)
def _exp_kernel[
    dtype: DType, width: SIMDLength, //
](hi: SIMD[dtype, width], lo: type_of(hi)) -> type_of(hi):
    """Returns `exp(hi - lo)` for `|hi - lo| <= ln2/2` (fdlibm e_exp.c,
    musl expf.c)."""
    var r = hi - lo
    var t = r * r
    var p: type_of(hi)
    comptime if dtype == .float64:
        p = polynomial_evaluate[
            [
                Scalar[dtype](1.66666666666666019037e-01),
                -2.77777777770155933842e-03,
                6.61375632143793436117e-05,
                -1.65339022054652515390e-06,
                4.13813679705723846039e-08,
            ]
        ](t)
    else:
        comptime assert dtype == .float32, "expected float32 or float64"
        p = polynomial_evaluate[
            [Scalar[dtype](1.6666625440e-1), -2.7667332906e-3]
        ](t)
    var c = r - t * p
    return 1.0 - ((lo - (r * c) / (2.0 - c)) - hi)


@inline(.always)
def _scale_pow2[
    dtype: DType, width: SIMDLength, //
](m: SIMD[dtype, width], k: type_of(m)) -> type_of(m):
    """Returns `m * 2^k` for integral `|k|` up to about twice the exponent
    range. Of the two power-of-two steps only the second can round, so a
    subnormal result rounds once from `m`. The first step is subnormal only
    when the result is, so on targets that flush subnormals normal results
    are correct and subnormal results flush to zero.
    """
    comptime int_type = FPUtils[dtype].integral_type
    comptime bias = SIMD[int_type, width](FPUtils[dtype].exponent_bias())
    comptime shift = SIMD[int_type, width](FPUtils[dtype].mantissa_width())
    var ki = k.cast[int_type]()
    var k1 = ki >> 1
    var s1 = type_of(m)(from_bits=(k1 + bias) << shift)
    var s2 = type_of(m)(from_bits=(ki - k1 + bias) << shift)
    return (m * s1) * s2


# `_erfc_f32` and `_erfc_f64` follow FreeBSD msun s_erff.c / s_erf.c (Sun
# fdlibm) but evaluate exp(-z*z - 0.5625 + a) as one exp with the fdlibm
# e_exp.c / e_expf.c reduction. z*z and k*ln2_hi are exact and nearly cancel,
# so the reduced argument carries no rounding error from the large -z*z term.
# std `exp` is not used: on GPU its f32 form is `ex2.approx(x * log2e)`, and
# its f64 form clamps at -709 and cannot return subnormals. The result comes
# back as `p * 2**k`, and the caller scales `p / x` by `2**k` last, so a
# subnormal erfc rounds twice: once in the division and once into the
# subnormal range. On targets that flush subnormals, subnormal inputs are
# handled and normal results are correct; subnormal results flush to zero.
@inline(.always)
def _erfc_exp[
    dtype: DType, width: SIMDLength, //
](zz: SIMD[dtype, width], a: type_of(zz)) -> Tuple[type_of(zz), type_of(zz)]:
    """Returns `(p, k)` with `exp(-zz - 0.5625 + a) = p * 2**k`."""
    # ln2_hi has 21 (float64) or 9 (float32) trailing zero bits, so k * ln2_hi
    # is exact over the clamped range of k.
    comptime is_f64 = dtype == .float64
    comptime ln2_hi = Scalar[dtype](_LN2_HI_F64 if is_f64 else 6.9314575195e-01)
    comptime ln2_lo = Scalar[dtype](_LN2_LO_F64 if is_f64 else 1.4286067653e-06)
    # For float32, |k| <= 245 here, within the 2 * 126 that `_scale_pow2`
    # handles.
    comptime bound = Scalar[dtype](1200.0 if is_f64 else 170.0)
    var k = floor((a - zz - 0.5625).clamp(-bound, bound) * log2e + 0.5)
    var hi = ((-zz - k * ln2_hi) - 0.5625) + a
    return (_exp_kernel(hi, k * ln2_lo), k)


@inline(.always)
def _erf_small_ratio_f64[
    width: SIMDLength, //
](z: SIMD[.float64, width]) -> type_of(z):
    """Returns `y` with `erf(x) = x + x*y` for `|x| < 0.84375`, `z = x*x`."""
    return polynomial_evaluate[
        [
            Float64(1.28379167095512558561e-01),
            -3.25042107247001499370e-01,
            -2.84817495755985104766e-02,
            -5.77027029648944159157e-03,
            -2.37630166566501626084e-05,
        ]
    ](z) / polynomial_evaluate[
        [
            Float64(1.0),
            3.97917223959155352819e-01,
            6.50222499887672944485e-02,
            5.08130628187576562776e-03,
            1.32494738004321644526e-04,
            -3.96022827877536812320e-06,
        ]
    ](
        z
    )


@inline(.always)
def _scale_up_subnormal[
    dtype: DType, width: SIMDLength, //, shift: Int
](x: SIMD[dtype, width]) -> type_of(x) where dtype.is_floating_point():
    """Returns `x * 2^shift` for subnormal `x`, built from its integer
    significand so flush-to-zero targets keep the value. `shift` must make
    the result normal (at least `mantissa_width`)."""
    comptime TInt = SIMD[_integral_type_of[dtype](), width]
    comptime mantissa_width = FPUtils[dtype].mantissa_width()
    comptime assert shift >= mantissa_width, "result must be normal"
    comptime emin = 1 - FPUtils[dtype].exponent_bias()
    # The significand `m` converts exactly; `m * 2^(emin - mantissa_width +
    # shift)` has only normal operands and a normal result.
    var m = (x._to_bits_signed() & TInt((1 << mantissa_width) - 1)).cast[
        dtype
    ]()
    comptime scale = SIMD[dtype, width](
        from_bits=TInt(
            (emin - mantissa_width + shift + FPUtils[dtype].exponent_bias())
            << mantissa_width
        )
    )
    return copysign(m * scale, x)


@inline(.always)
def _scale_down_bits[
    dtype: DType, width: SIMDLength, //, shift: Int
](y: SIMD[dtype, width]) -> type_of(y) where dtype.is_floating_point():
    """Returns `y * 2^-shift`, correctly rounded (to nearest even) even when
    the result is subnormal, for finite normal `y`, or for zero `y` when
    `shift >= mantissa_width`. Uses integer operations only, so flush-to-zero
    targets keep subnormal results."""
    comptime TInt = SIMD[_integral_type_of[dtype](), width]
    comptime mantissa_width = FPUtils[dtype].mantissa_width()
    comptime bits = bit_width_of[dtype]()
    comptime sign_mask = TInt(1) << TInt(bits - 1)
    comptime mantissa_mask = TInt((1 << mantissa_width) - 1)
    var b = y._to_bits_signed()
    var sign = b & sign_mask
    var mag = b & ~sign_mask
    var e = mag >> TInt(mantissa_width)
    var new_e = e - TInt(shift)
    # Normal result: only the exponent field changes.
    var normal = (new_e << TInt(mantissa_width)) | (mag & mantissa_mask)
    # Subnormal result: shift the full significand right and round to
    # nearest even; a carry into bit mantissa_width encodes the minimum
    # normal correctly. Shifts past mantissa_width + 1 round to zero.
    var s = min(TInt(1) - new_e, TInt(mantissa_width + 2))
    var sig = (mag & mantissa_mask) | TInt(1 << mantissa_width)
    var q = sig >> s
    var rem = sig & ((TInt(1) << s) - 1)
    var half = TInt(1) << (s - 1)
    var up = rem.gt(half) | (rem.eq(half) & (q & 1).ne(0))
    var sub = q + up.select(TInt(1), TInt(0))
    var r = new_e.ge(1).select(normal, sub)
    return SIMD[dtype, width](from_bits=sign | r)


@inline(.always)
def _narrow_f32_to_bf16_bits[
    width: SIMDLength, //
](x: SIMD[DType.float32, width]) -> SIMD[DType.bfloat16, width]:
    """Rounds float32 to bfloat16 (nearest even) from the bits, keeping
    subnormal results that a flush-to-zero `cast` would drop. NaN stays NaN."""
    var b = x.to_bits()
    var rounded = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16).cast[DType.uint16]()
    var quiet_nan = ((b >> 16) | 0x40).cast[DType.uint16]()
    # Narrow before selecting: selecting between the 32-bit values and then
    # truncating drops the `>> 16` on Metal and yields garbage bits.
    return SIMD[DType.bfloat16, width](
        from_bits=isnan(x).select(quiet_nan, rounded)
    )


# Fraction bits of 2/pi, 64 per word, after one zero word so that a window may
# start before the binary point. The 19 nonzero words reach the deepest bit a
# float64 reduction reads (about 1220).


@inline(.always)
def _umul128[
    width: SIMDLength, //
](a: SIMD[.uint64, width], b: type_of(a)) -> Tuple[type_of(a), type_of(a)]:
    """Returns the low and high words of the 128-bit product `a * b`."""
    var a0 = a & 0xFFFFFFFF
    var a1 = a >> 32
    var b0 = b & 0xFFFFFFFF
    var b1 = b >> 32
    var p00 = a0 * b0
    var p01 = a0 * b1
    var p10 = a1 * b0
    var mid = (p00 >> 32) + (p01 & 0xFFFFFFFF) + (p10 & 0xFFFFFFFF)
    var lo = (mid << 32) | (p00 & 0xFFFFFFFF)
    var hi = a1 * b1 + (p01 >> 32) + (p10 >> 32) + (mid >> 32)
    return (lo, hi)


@inline(.always)
def _rem_pio2[
    dtype: DType, width: SIMDLength, //, quarter: Bool = False
](x: SIMD[dtype, width]) -> Tuple[SIMD[.int32, width], type_of(x), type_of(x)]:
    """Returns `(n mod 4, hi, lo)` with `x = n*pi/2 + hi + lo` and `|hi + lo|`
    at most slightly above pi/4. Lanes with non-finite `x` return unspecified
    values.

    With `quarter`, reduces `x - pi/4` for `x >= 0` instead: `k` becomes
    `n + 1/2`, so the shift is folded into the reduction and the result keeps
    its full relative accuracy where `x - pi/4` is close to a multiple of
    pi/2 (the Bessel asymptotic phases). `k` then carries one more bit, so
    `limit` halves.

    Cody-Waite with pi/2 = C1 + C2 + ... + Cn. `fma(-k, C1, x)` is exact (a
    multiple of ulp(C1)/2 below 1). Below `limit`, k has at most 14 (float32)
    or 20 (float64) bits and C2 .. C(n-1) at most 10 or 33, so each `k * Ci`
    is exact and the two-sums that subtract them are error-free whether or
    not the backend contracts a multiply into a dependent add (NVPTX and
    AMDGPU do, even when the product has other uses). Only `k * Cn` rounds,
    and it lands in the low word through an explicit fma. At the floats
    closest to a multiple of pi/2 the reduced argument is still accurate to
    2^-39 (float32) or 2^-70 (float64) relative. Larger inputs take the
    Payne-Hanek path, run only when some lane needs it.
    """
    comptime assert dtype in (DType.float32, DType.float64)
    comptime T = type_of(x)
    comptime is_f32 = dtype == .float32
    comptime c1 = Scalar[dtype](
        1.5707963705062866 if is_f32 else 1.5707963267948966
    )
    comptime c2 = Scalar[dtype](
        -4.371395334601402e-08 if is_f32 else 6.12323399538461e-17
    )
    # float64 needs only one exact piece after c1.
    comptime c3 = Scalar[dtype](2.5650592760939617e-12 if is_f32 else 0.0)
    comptime c_last = Scalar[dtype](
        -1.7151245100058819e-15 if is_f32 else 3.5215598651832e-27
    )
    comptime limit = Scalar[dtype](
        (12288.0 if is_f32 else 786432.0) if quarter else (
            24576.0 if is_f32 else 1572864.0
        )
    )

    var small = abs(x).lt(limit)
    var ac = small.select(x, T(0))
    var kf: T
    comptime if quarter:
        kf = floor(ac * _TWO_OVER_PI) + 0.5
    else:
        kf = floor(ac * _TWO_OVER_PI + 0.5)
    var s = _two_sum(fma(-kf, c1, ac), -kf * c2)
    var r_hi = s[0]
    var r_lo = s[1]
    comptime if dtype == .float32:
        var s3 = _two_sum(r_hi, -kf * c3)
        r_hi = s3[0]
        r_lo += s3[1]
    var y = _two_sum(r_hi, fma(-kf, c_last, r_lo))
    var n: SIMD[.int32, width]
    comptime if dtype == .float32:
        n = kf.cast[.int32]() & 3
    else:
        n = kf.cast[.int64]().cast[.int32]() & 3
    var hi = y[0]
    var lo = y[1]

    var large = ~small & isfinite(x)
    if large.reduce_or():
        var ax = abs(x)
        var big = _rem_pio2_large[quarter=quarter](large.select(ax, T(limit)))
        var neg = x.lt(0)
        n = large.select(neg.select(-big[0], big[0]) & 3, n)
        hi = large.select(neg.select(-big[1], big[1]), hi)
        lo = large.select(neg.select(-big[2], big[2]), lo)
    return (n, hi, lo)


@inline(.always)
def _tan_impl[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> type_of(x):
    comptime T = type_of(x)
    var red = _rem_pio2(x)
    var y = red[1]
    var yl = red[2]
    var z = y * y
    # tan(y + yl) = y + c with c = y^3 R(y^2) + tan'(y) yl, tan'(y) ~ 1 + y^2.
    var c: T
    comptime if dtype == .float32:
        # Minimax fit of the relative error of tan (max 2.7e-9 after rounding
        # the coefficients) on |y| <= pi/4.
        var p = polynomial_evaluate[
            [
                Scalar[dtype](0.33333349227905273),
                0.13332659006118774,
                0.05406050756573677,
                0.021279029548168182,
                0.010844247415661812,
                7.853933493606746e-05,
                0.004381844773888588,
            ]
        ](z)
        c = fma(y * z, p, fma(yl, z, yl))
    else:
        # Cephes tan.c.
        var p = polynomial_evaluate[
            [
                Scalar[dtype](-1.79565251976484877988e7),
                1.15351664838587416140e6,
                -1.30936939181383777646e4,
            ]
        ](z)
        var q = polynomial_evaluate[
            [
                Scalar[dtype](-5.38695755929454629881e7),
                2.50083801823357915839e7,
                -1.32089234440210967447e6,
                1.36812963470692954678e4,
                1.0,
            ]
        ](z)
        c = fma(y, z * p / q, fma(yl, z, yl))
    # |c| < |y|/2, so t_lo is the exact rounding error of t.
    var t = y + c
    var t_lo = c - (t - y)
    # tan(y + pi/2) = -1/tan(y). As in fdlibm k_tan.c, correct q = -1/t for
    # t_lo: -1/(t + t_lo) ~ q + q (1 + q (t + t_lo)), the residual exact
    # through the fma.
    var q = -1.0 / t
    var cot = fma(q, fma(q, t, 1.0) + q * t_lo, q)
    var res = (red[0] & 1).ne(0).select(cot, t)
    res = x.eq(0).select(x, res)
    return isfinite(x).select(res, x - x)


@inline(.always)
def _fmod_pos_impl[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width], y: SIMD[dtype, width]) -> SIMD[
    dtype, width
] where dtype.is_floating_point():
    """Computes fmod(x, y) exactly for finite x >= 0 and finite normal
    y > 0. Callers lift subnormal divisors first; a subnormal y would
    push the quotient scale past the representable exponent range.

    Each step strips up to `B` quotient bits: with y' = y * 2^k chosen so that
    x / y' < 2^(B+1), q = trunc(x / y') estimates the quotient. `/` may be off
    by a few ulps (GPU fast math), so q is stepped until x - q * y' lies in
    [0, y'). Rounding is monotonic and 0 and y' are representable, so those
    range tests on the rounded fma are exact; once they pass, x - q * y' is a
    multiple of ulp(y') below y', which the fma computes exactly.
    """
    comptime B = 22 if dtype == .float32 else 50
    comptime T = SIMD[dtype, width]
    comptime I32 = SIMD[.int32, width]
    var r = x
    var ey = frexp(y)[1].cast[.int32]()
    var active = r.ge(y)
    while active.reduce_or():
        var ex = frexp(r)[1].cast[.int32]()
        var k = max(ex - ey - I32(B), I32(0))
        # y is normal and k >= 0, so two exact power-of-two factors replace the
        # general `_scalbn_impl`; 2^k itself can exceed the exponent range.
        var kh = (k >> 1).cast[FPUtils[dtype].integral_type]()
        var kl = k.cast[FPUtils[dtype].integral_type]() - kh
        var yk = y * _pow2_impl[dtype, width](kh) * _pow2_impl[dtype, width](kl)
        var q = trunc(r / yk)
        var t = fma(-q, yk, r)
        var neg = t.lt(0)
        var fix = active & (neg | t.ge(yk))
        while fix.reduce_or():
            q = fix.select(q + neg.select(T(-1), T(1)), q)
            t = fma(-q, yk, r)
            neg = t.lt(0)
            fix = active & (neg | t.ge(yk))
        r = active.select(t, r)
        active = r.ge(y)
    return r


@inline(.always)
def _remainder_lifted_impl[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width], y: SIMD[dtype, width]) -> SIMD[
    dtype, width
] where dtype.is_floating_point():
    # `_remainder_impl` for targets that flush subnormals.
    comptime T = SIMD[dtype, width]
    comptime I = SIMD[FPUtils[dtype].integral_type, width]
    comptime mant = FPUtils[dtype].mantissa_width()
    comptime emin = 1 - FPUtils[dtype].exponent_bias()
    var ax = abs(x)
    var p = abs(y)
    var p_bits = p._to_bits_signed()
    var invalid = isnan(x) | isnan(y) | isinf(x) | p_bits.eq(0)

    # The remainder is a multiple of the smaller ulp of |x| and |y|, so for a
    # small |y| it can be subnormal, and targets that flush subnormals would
    # lose it. Such a divisor is lifted by 2^(2 mant), which makes every
    # operand and result of the reduction normal or zero:
    # |x| is reduced modulo the lifted 2|y| (a multiple of 2|y|, so the
    # remainder is unchanged), lifted itself, reduced again, and the result
    # is scaled back with integer operations. A divisor above the limit
    # leaves all intermediates normal.
    comptime lift_limit = Scalar[FPUtils[dtype].integral_type](
        (emin + mant + 3 + FPUtils[dtype].exponent_bias()) << mant
    )
    comptime lift_shift = 2 * mant
    comptime lift = _pow2_impl[dtype, 1](
        Scalar[FPUtils[dtype].integral_type](lift_shift)
    )
    var small_y = p_bits.lt(I(lift_limit))
    var sub_y = p_bits.lt(I(1 << mant))
    p = small_y.select(
        sub_y.select(_scale_up_subnormal[lift_shift](p), p * lift), p
    )
    var p2 = p + p
    var use_fmod = isfinite(p2) & ~invalid
    var a = use_fmod.select(
        _fmod_pos_impl(use_fmod.select(ax, T(0)), use_fmod.select(p2, T(1))),
        ax,
    )
    if small_y.reduce_or():
        var sub_a = a._to_bits_signed().lt(I(1 << mant))
        var lifted = sub_a.select(_scale_up_subnormal[lift_shift](a), a * lift)
        lifted = (use_fmod & small_y).select(lifted, T(0))
        a = small_y.select(_fmod_pos_impl(lifted, use_fmod.select(p2, T(1))), a)

    # p is far above the minimum normal here, so p * 0.5 is exact.
    var p_half = p * 0.5
    var c1 = a.gt(p_half)
    a = c1.select(a - p, a)
    var c2 = c1 & a.ge(p_half)
    a = c2.select(a - p, a)
    if small_y.reduce_or():
        a = small_y.select(
            _is_zero_bits(a).select(a, _scale_down_bits[lift_shift](a)), a
        )

    var r = x._to_bits_signed().lt(0).select(-a, a)
    return invalid.select(nan[dtype](), r)


@inline(.always)
def _sqrt_ieee[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> type_of(x):
    # A plugin may lower `sqrt` to an approximation, but `llvm.sqrt` is
    # correctly rounded on every target.
    return _llvm_unary_fn["llvm.sqrt"](x)


@inline(.always)
def _remainder_impl[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width], y: SIMD[dtype, width]) -> SIMD[
    dtype, width
] where dtype.is_floating_point():
    # Sun fdlibm e_remainder.c: reduce |x| modulo 2|y| exactly, then pick the
    # nearest multiple of |y| with ties to even.
    comptime if CurrentPlugin.flushes_subnormals[dtype]:
        return _remainder_lifted_impl(x, y)
    comptime T = SIMD[dtype, width]
    comptime I = SIMD[FPUtils[dtype].integral_type, width]
    comptime mant = FPUtils[dtype].mantissa_width()
    comptime emin = 1 - FPUtils[dtype].exponent_bias()
    var ax = abs(x)
    var p = abs(y)
    var p_bits = p._to_bits_signed()
    var invalid = isnan(x) | isnan(y) | isinf(x) | p_bits.eq(0)

    # A subnormal |y| = m * 2^(emin - mant) is read through its integer
    # significand and lifted to `m * 2^emin`, that is scaled by 2^mant, so
    # every operand of the reduction is normal. |x| is first reduced modulo
    # the lifted 2|y|, a multiple of 2|y| that leaves the remainder
    # unchanged, then lifted too, and the result scales back exactly. On
    # targets that flush subnormals (Apple GPUs in float32) 2|y| would
    # otherwise read as zero and the result be NaN; with the lift the result
    # is a zero, since the true remainder is at most |y|/2 and so itself
    # subnormal. Whenever the true result is normal, every intermediate is
    # normal or zero, so those results are exact on such targets too.
    var sub_y = (p_bits >> I(mant)).eq(0)
    comptime lift = _pow2_impl[dtype, 1](
        Scalar[FPUtils[dtype].integral_type](mant)
    )
    comptime sub_unit = _pow2_impl[dtype, 1](
        Scalar[FPUtils[dtype].integral_type](emin)
    )
    p = sub_y.select((p_bits & I((1 << mant) - 1)).cast[dtype]() * sub_unit, p)
    var p2 = p + p
    var use_fmod = isfinite(p2) & ~invalid
    var a = use_fmod.select(
        _fmod_pos_impl(use_fmod.select(ax, T(0)), use_fmod.select(p2, T(1))),
        ax,
    )
    if sub_y.reduce_or():
        var lifted = (use_fmod & sub_y).select(a * lift, T(0))
        a = sub_y.select(_fmod_pos_impl(lifted, use_fmod.select(p2, T(1))), a)

    # Below 2^(emin+1), p * 0.5 is inexact, so compare 2a against p instead.
    comptime tiny = _pow2_impl[dtype, 1](
        Scalar[FPUtils[dtype].integral_type](2 - FPUtils[dtype].exponent_bias())
    )
    var small_p = p.lt(T(tiny))
    var p_half = p * 0.5
    var c1 = small_p.select((a + a).gt(p), a.gt(p_half))
    a = c1.select(a - p, a)
    var c2 = c1 & small_p.select((a + a).ge(p), a.ge(p_half))
    a = c2.select(a - p, a)
    a = sub_y.select(a * (1 / lift), a)

    var r = x._to_bits_signed().lt(0).select(-a, a)
    return invalid.select(nan[dtype](), r)


@inline(.always)
def _hypot_impl[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width], y: SIMD[dtype, width]) -> SIMD[
    dtype, width
] where dtype.is_floating_point():
    comptime T = SIMD[dtype, width]
    comptime I = SIMD[FPUtils[dtype].integral_type, width]
    comptime U = SIMD[FPUtils[dtype].uint_type, width]
    comptime mant = FPUtils[dtype].mantissa_width()
    comptime bias = FPUtils[dtype].exponent_bias()

    # Floating min/max may flush subnormal operands before normalization.
    var magnitude_mask = ~U(FPUtils[dtype].sign_mask())
    var ax_bits = x.to_bits[FPUtils[dtype].uint_type]() & magnitude_mask
    var ay_bits = y.to_bits[FPUtils[dtype].uint_type]() & magnitude_mask
    var big_bits = max(ax_bits, ay_bits).cast[FPUtils[dtype].integral_type]()
    var small_bits = min(ax_bits, ay_bits).cast[FPUtils[dtype].integral_type]()

    # Scale by 2^-e (e = exponent of the larger magnitude) so the sum of
    # squares can neither overflow nor lose the larger term to underflow. The
    # clamp keeps both 2^e and 2^-e normal.
    comptime exp_mask = (1 << FPUtils[dtype].exponent_width()) - 1
    var e_big = (big_bits >> I(mant)) & I(exp_mask)
    var e = (e_big - I(bias)).clamp(I(1 - bias), I(bias - 1))
    var down = T(from_bits=(I(bias) - e) << I(mant))
    var up = T(from_bits=(I(bias) + e) << I(mant))
    var e_small = (small_bits >> I(mant)) & I(exp_mask)
    var big_subnormal = e_big.eq(0)
    var small_subnormal = e_small.eq(0)
    var a = T(from_bits=big_subnormal.select(I(0), big_bits)) * down
    var b = T(from_bits=small_subnormal.select(I(0), small_bits)) * down

    # A subnormal operand is `m * 2^(emin - mant)` for its integer significand
    # m, so its scaled value is `m * 2^(emin - mant - e)`, built here without
    # arithmetic on the subnormal itself, which targets that flush subnormals
    # would read as zero. Past e = -mant the unit is not normal, and the
    # operand is below 2^-mant of `a` and negligible.
    comptime sub_exp = 1 - bias - mant
    var unit = _pow2_impl[dtype, width](
        (I(sub_exp) - e).clamp(I(1 - bias), I(bias))
    )
    var mant_mask = I((1 << mant) - 1)
    a = big_subnormal.select((big_bits & mant_mask).cast[dtype]() * unit, a)
    b = (small_subnormal & e.le(I(-mant))).select(
        (small_bits & mant_mask).cast[dtype]() * unit, b
    )
    var s = fma(a, a, b * b)
    var r = _sqrt_ieee(s) * up
    comptime if CurrentPlugin.flushes_subnormals[dtype]:
        # A subnormal product would flush to zero. The result is subnormal
        # only when the exponent clamp is active, so scale it down with
        # integer operations instead.
        r = e.eq(I(1 - bias)).select(
            _scale_down_bits[bias - 1](_sqrt_ieee(s)), r
        )

    # C99 F.10.4.3: hypot(±inf, y) is +inf even when y is NaN.
    r = (isnan(x) | isnan(y)).select(nan[dtype](), r)
    return (isinf(x) | isinf(y)).select(inf[dtype](), r)


def _gamma_max_scalings(n: Int, scale_log2: Int) -> Int:
    """Returns the largest c with 2^(scale_log2 * c) < 2 * n!.

    The running product in `_gamma_impl` is at most 2 * n!: up to n! from the
    factors of magnitude above 1, times one factor in [1, 2) that crosses zero.
    """
    var scale = 1.0
    for _ in range(scale_log2):
        scale *= 2.0
    var p = 2.0
    var count = 0
    for i in range(2, n + 1):
        p *= Float64(i)
        while p > scale:
            p /= scale
            count += 1
    return count


@inline(.always)
def _pow2_impl[
    dtype: DType, width: SIMDLength
](k: SIMD[FPUtils[dtype].integral_type, width]) -> SIMD[
    dtype, width
] where dtype.is_floating_point():
    """Returns 2^k for k in the normal exponent range."""
    comptime I = SIMD[FPUtils[dtype].integral_type, width]
    return SIMD[dtype, width](
        from_bits=(k + I(FPUtils[dtype].exponent_bias()))
        << I(FPUtils[dtype].mantissa_width())
    )


def _has_zero_exponent[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> SIMD[
    DType.bool, width
] where dtype.is_floating_point():
    """Returns true for lanes that are zero or subnormal, read from the bits.

    Unlike a comparison, this is correct on targets that flush subnormal
    operands to zero.
    """
    comptime I = SIMD[FPUtils[dtype].integral_type, width]
    comptime exponent = I(
        ((1 << FPUtils[dtype].exponent_width()) - 1)
        << FPUtils[dtype].mantissa_width()
    )
    return (x._to_bits_signed() & exponent).eq(0)


def _scalbn_impl[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width], n: SIMD[.int32, width]) -> SIMD[
    dtype, width
] where dtype.is_floating_point():
    """Computes x * 2^n with a single final rounding (musl `scalbn`).

    Requires float32 or float64 inputs; `scalb()` promotes narrower types.
    """
    comptime assert dtype in (
        DType.float32,
        DType.float64,
    ), "_scalbn_impl requires float32 or float64; promote narrow inputs first"
    comptime I = SIMD[FPUtils[dtype].integral_type, width]
    comptime emax = FPUtils[dtype].exponent_bias()
    comptime emin = 1 - emax
    # Scaling down by 2^(emin + p) rather than 2^emin keeps the intermediate
    # normal, so a subnormal result is rounded only once.
    comptime mant = FPUtils[dtype].mantissa_width()
    comptime down_exp = emin + mant + 1
    var up = _pow2_impl[dtype, width](I(emax))
    var down = _pow2_impl[dtype, width](I(down_exp))
    # A subnormal x is `m * 2^(emin - mant)` for its integer significand m;
    # converting m is exact and keeps the scaling away from arithmetic on a
    # subnormal, which targets that flush them (Apple GPUs) treat as zero. On
    # such targets subnormal inputs are handled and normal results are
    # correct; a subnormal result is produced by the final multiply and
    # flushes to zero.
    var bits = x._to_bits_signed()
    var sub = _has_zero_exponent(x)
    var m = (bits & I((1 << mant) - 1)).cast[dtype]()
    var y = sub.select(copysign(m, x), x)
    var k = n.cast[FPUtils[dtype].integral_type]() + sub.select(
        I(emin - mant), I(0)
    )
    comptime for _ in range(2):
        var hi = k.gt(I(emax))
        var lo = k.lt(I(emin))
        y = hi.select(y * up, lo.select(y * down, y))
        k = hi.select(k - I(emax), lo.select(k - I(down_exp), k))
    k = k.clamp(I(emin), I(emax))
    var r = y * _pow2_impl[dtype, width](k)
    comptime if CurrentPlugin.flushes_subnormals[dtype]:
        # The multiply above flushes a subnormal result. Rebuild those lanes
        # from a result scaled into the normal range, which `_scale_down_bits`
        # rounds once from its bits.
        var y_exp = (y._to_bits_signed() >> I(mant)) & I(
            (1 << FPUtils[dtype].exponent_width()) - 1
        )
        var tiny = (y_exp + k).lt(I(1))
        var scaled = y * _pow2_impl[dtype, width](k + I(mant + 1))
        r = tiny.select(_scale_down_bits[mant + 1](scaled), r)
    return r


def _scalb_impl[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width], y: SIMD[dtype, width]) -> SIMD[
    dtype, width
] where dtype.is_floating_point():
    """Computes fdlibm `scalb` (Sun `e_scalb.c`): x * 2^y for integral y.

    A non-integral y gives NaN, an infinite y gives x * y except that
    `0 * inf` and `inf * -inf` give NaN, and a NaN operand propagates. y is
    clamped to +-65000, far beyond any finite float32 or float64 scaling.
    Requires float32 or float64; `scalb()` promotes narrower types.
    """
    comptime T = SIMD[dtype, width]
    # Far beyond any finite float32 or float64 scaling, and safe to cast to
    # Int32.
    comptime exp_limit = 65000
    var y_nan = isnan(y)
    var y_inf = isinf(y)
    var n = y_nan.select(T(0), y).clamp(-exp_limit, exp_limit)
    var r = _scalbn_impl(x, n.cast[.int32]())
    # The bit tests below stay correct on a target that flushes subnormal
    # operands (Apple GPUs in float32), where `trunc` and comparisons see a
    # subnormal as zero: a subnormal y is never integral, and a subnormal x
    # must not turn into 0 * inf.
    comptime I = SIMD[FPUtils[dtype].integral_type, width]
    var magnitude = I((1 << (bit_width_of[dtype]() - 1)) - 1)
    var y_sub = _has_zero_exponent(y) & (y._to_bits_signed() & magnitude).ne(0)
    r = (~y_inf & (trunc(y).ne(y) | y_sub)).select(nan[dtype](), r)
    var x_zero = (x._to_bits_signed() & magnitude).eq(0)
    var to_inf = x_zero.select(nan[dtype](), copysign(T(inf[dtype]()), x))
    var to_zero = isinf(x).select(nan[dtype](), copysign(T(0), x))
    r = y_inf.select(y.gt(0).select(to_inf, to_zero), r)
    return (isnan(x) | y_nan).select(x * y, r)


def _atan2_impl[
    dtype: DType, width: SIMDLength, //
](y: SIMD[dtype, width], x: SIMD[dtype, width]) -> type_of(
    x
) where dtype.is_floating_point():
    comptime assert dtype in (DType.float32, DType.float64)
    comptime T = type_of(x)
    comptime is_f32 = dtype == .float32
    comptime pi_lo = Scalar[dtype](
        -8.742278e-8 if is_f32 else 1.2246467991473532e-16
    )
    # 2^(emin + p): above it the division remainder is exactly representable.
    comptime tiny = Scalar[dtype](
        1.9721522630525295e-31 if is_f32 else 2.004168360008973e-292
    )
    comptime scale = Scalar[dtype](
        1.8446744073709552e19 if is_f32 else 3.402823669209385e38
    )
    comptime I = SIMD[FPUtils[dtype].integral_type, width]
    comptime mant = FPUtils[dtype].mantissa_width()
    comptime abs_mask = I(_F32_ABS_MASK if is_f32 else 0x7FFFFFFFFFFFFFFF)
    var ax_bits = x._to_bits_signed() & abs_mask
    var ay_bits = y._to_bits_signed() & abs_mask
    # Integer ordering preserves unequal subnormals under DAZ comparisons.
    var swap = ay_bits.gt(ax_bits)
    var num_bits = swap.select(ax_bits, ay_bits)
    var den_bits = swap.select(ay_bits, ax_bits)
    var num = T(from_bits=num_bits)
    var den = T(from_bits=den_bits)
    var both_inf = isinf(x) & isinf(y)
    # Exact rescale so the remainder below stays exact; if den overflows,
    # num / den was already below the smallest subnormal. A subnormal operand
    # is rebuilt from its integer significand m as `m * 2^(emin - mant)`
    # times the scale, preserving the ratio under arithmetic DAZ/FTZ.
    var small = num_bits.lt(I(tiny._to_bits_signed()))
    comptime unit = Scalar[dtype](
        2.5849394142282115e-26 if is_f32 else 1.681218273811815e-285
    )
    var mant_mask = I((1 << mant) - 1)
    var num_sub = (num_bits >> I(mant)).eq(0)
    var den_sub = (den_bits >> I(mant)).eq(0)
    num = small.select(
        num_sub.select(
            (num_bits & mant_mask).cast[dtype]() * unit, num * scale
        ),
        num,
    )
    den = small.select(
        den_sub.select(
            (den_bits & mant_mask).cast[dtype]() * unit, den * scale
        ),
        den,
    )
    # 0/0 and inf/inf are NaN; the Annex F results are atan(0) and atan(1).
    var t = both_inf.select(T(1.0), den.eq(0.0).select(T(0.0), num / den))
    var a = atan(t)
    # First-order correction for the rounding of num / den, which otherwise
    # adds up to half an ulp on top of atan's error. Divide by den first
    # rather than overflow den * (1 + t*t) for finite large operands.
    var exact_rem = num.ge(tiny) & ~isinf(den)
    var c = exact_rem.select(((-t).fma(den, num) / den) / t.fma(t, 1.0), T(0.0))
    # Unfold the octant in hi/lo pairs; each Fast2Sum is exact because the
    # constant dominates the angle subtracted from it.
    comptime pio2_hi = T(pi / 2.0)
    var s = pio2_hi - a
    var hi = swap.select(s, a)
    var lo = swap.select(((pio2_hi - s) - a) + (T(pi_lo * 0.5) - c), c)
    # The sign bit, not x < 0, so that atan2(+-0, -0) is +-pi.
    var x_neg = copysign(T(1.0), x).lt(0.0)
    var s2 = T(pi) - hi
    lo = x_neg.select(((T(pi) - s2) - hi) + (T(pi_lo) - lo), lo)
    hi = x_neg.select(s2, hi)
    var r = copysign(hi + lo, y)
    comptime if is_f32:
        r = _atan2_tiny_f32(
            y._refine[.float32](), x._refine[.float32](), r._refine[.float32]()
        )._refine[dtype]()
    return (isnan(x) | isnan(y)).select(x + y, r)


def _atan2_tiny_f32[
    width: SIMDLength, //
](
    y: SIMD[.float32, width],
    x: SIMD[.float32, width],
    result: SIMD[.float32, width],
) -> type_of(x):
    comptime T = type_of(x)
    comptime itype = FPUtils[DType.float32].integral_type
    comptime I = SIMD[itype, width]
    comptime sign_mask = -2147483648
    comptime inf_bits = 0x7F800000
    comptime min_normal_bits = 0x00800000
    comptime mantissa_mask = 0x007FFFFF
    comptime one_bits = 0x3F800000
    var xb = x._to_bits_signed() & _F32_ABS_MASK
    var yb = y._to_bits_signed() & _F32_ABS_MASK
    var xs = xb.lt(min_normal_bits)
    var ys = yb.lt(min_normal_bits)
    # Normalize integer significands without consuming a subnormal operand.
    var nx = xs.select(xb.cast[.float32]()._to_bits_signed(), xb)
    var ny = ys.select(yb.cast[.float32]()._to_bits_signed(), yb)
    var ex = (nx >> 23) - xs.select(I(149), I(0))
    var ey = (ny >> 23) - ys.select(I(149), I(0))
    var shift = ey - ex + 149
    var tiny = (
        (ey - ex).le(-126)
        & x._to_bits_signed().gt(0)
        & xb.lt(inf_bits)
        & yb.ne(0)
    )
    # Express the quotient in units of 2^-149. Clamping keeps even unselected
    # SIMD lanes normal, finite, and within the float-to-integer range.
    var scale = T(from_bits=(clamp(shift, I(-1), I(23)) + 127) << 23)
    var num = T(from_bits=(ny & mantissa_mask) | one_bits) * scale
    var den = T(from_bits=(nx & mantissa_mask) | one_bits)
    var q = floor(num / den)
    var rem = (-q).fma(den, num)
    # The residual resolves division rounding near half-integers. At an
    # exact halfway ratio, atan(ratio) is strictly below ratio: round down.
    var bits = q.cast[itype]() + rem.gt(den * 0.5).cast[itype]()
    bits = shift.lt(-1).select(I(0), bits)
    var signed_bits = bits | (y._to_bits_signed() & I(sign_mask))
    return tiny.select(T(from_bits=signed_bits), result)


def _atan_f32[width: SIMDLength, //](x: SIMD[.float32, width]) -> type_of(x):
    comptime T = type_of(x)
    # Reduction breakpoints 7/16, 11/16, 19/16, 39/16.
    comptime break_0 = 0.4375
    comptime break_1 = 0.6875
    comptime break_2 = 1.1875
    comptime break_3 = 2.4375
    # Hi/lo words of atan(1/2), atan(1), atan(3/2), pi/2 from s_atanf.c.
    comptime half_hi = 4.6364760399e-01
    comptime half_lo = 5.0121582440e-09
    comptime quarter_pi_hi = 7.8539812565e-01
    comptime quarter_pi_lo = 3.7748947079e-08
    comptime one_and_half_hi = 9.8279368877e-01
    comptime one_and_half_lo = 3.4473217170e-08
    comptime half_pi_hi = 1.5707962513e00
    comptime half_pi_lo = 7.5497894159e-08
    # Bits of 2^-12: below it, atan(x) rounds to x in float32.
    comptime tiny_bits = 0x39800000
    var ax = abs(x)
    var i0 = ax.ge(break_0)
    var i1 = ax.ge(break_1)
    var i2 = ax.ge(break_2)
    var i3 = ax.ge(break_3)
    # Each numerator is exact by Sterbenz on its interval.
    var num = i3.select(
        T(-1.0), i2.select(ax - 1.5, i1.select(ax - 1.0, ax + ax - 1.0))
    )
    var den = i3.select(
        ax, i2.select(ax.fma(1.5, 1.0), i1.select(ax + 1.0, ax + 2.0))
    )
    var xr = i0.select(num / den, ax)
    var hi = i3.select(
        T(half_pi_hi),
        i2.select(
            T(one_and_half_hi),
            i1.select(T(quarter_pi_hi), i0.select(T(half_hi), 0.0)),
        ),
    )
    var lo = i3.select(
        T(half_pi_lo),
        i2.select(
            T(one_and_half_lo),
            i1.select(T(quarter_pi_lo), i0.select(T(half_lo), 0.0)),
        ),
    )
    var z = xr * xr
    var p = z * polynomial_evaluate[
        [
            Float32(3.3333328366e-01),
            -1.9999158382e-01,
            1.4253635705e-01,
            -1.0648017377e-01,
            6.1687607318e-02,
        ]
    ](z)
    var result = copysign(hi - (xr.fma(p, -lo) - xr), x)
    # For |x| < 2^-12, the cubic correction is below half an ulp.
    # Select the original bits rather than flush a subnormal through arithmetic.
    return (x._to_bits_signed() & _F32_ABS_MASK).lt(tiny_bits).select(x, result)


# Cephes atan: same reduction with a 0.66 split and a [4/5] rational in x^2;
# MOREBITS carries the low part of pi/2 so the reduced branches stay accurate.


def _atan_f64[width: SIMDLength, //](x: SIMD[.float64, width]) -> type_of(x):
    comptime tan_3pi_8 = _SQRT2 + 1
    # Low part of pi/2 (Cephes MOREBITS).
    comptime morebits = 6.123233995736765886130e-17
    # Cephes split between the unreduced and the (x-1)/(x+1) branch.
    comptime mid_split = 0.66
    var ax = abs(x)
    var big = ax.gt(tan_3pi_8)
    var mid = ax.gt(mid_split) & ~big
    var xr = big.select(-1.0 / ax, mid.select((ax - 1.0) / (ax + 1.0), ax))
    var base = big.select(
        type_of(x)(pi / 2.0), mid.select(type_of(x)(pi / 4.0), 0.0)
    )
    var lo = big.select(
        type_of(x)(morebits),
        mid.select(type_of(x)(0.5 * morebits), 0.0),
    )
    var z = xr * xr
    var num = polynomial_evaluate[
        [
            Float64(-6.485021904942025371773e1),
            -1.228866684490136173410e2,
            -7.500855792314704667340e1,
            -1.615753718733365076637e1,
            -8.750608600031904122785e-1,
        ]
    ](z)
    var den = polynomial_evaluate[
        [
            Float64(1.945506571482613964425e2),
            4.853903996359136964868e2,
            4.328810604912902668951e2,
            1.650270098316988542046e2,
            2.485846490142306297962e1,
            1.0,
        ]
    ](z)
    var r = (z * num / den).fma(xr, xr) + lo
    return copysign(base + r, x)


# Reduces to atan(min(|y|,|x|) / max(|y|,|x|)) in [0, pi/4] and unfolds the
# octant with hi/lo splits of pi/2 and pi, following C99 Annex F for zeros,
# infinities and NaN.
def _gamma_2_to_3[
    dtype: DType, width: SIMDLength, //
](t: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Returns Gamma(2 + t) for t in [0, 1]."""
    comptime if dtype == .float64:
        # Cephes gamma.c rational approximation of Gamma(2 + t).
        var p = polynomial_evaluate[
            [
                Scalar[dtype](9.99999999999999996796e-1),
                4.94214826801497100753e-1,
                2.07448227648435975150e-1,
                4.76367800457137231464e-2,
                1.04213797561761569935e-2,
                1.19135147006586384913e-3,
                1.60119522476751861407e-4,
            ]
        ](t)
        var q = polynomial_evaluate[
            [
                Scalar[dtype](1.00000000000000000320e0),
                7.14304917030273074085e-2,
                -2.34591795718243348568e-1,
                3.58236398605498653373e-2,
                1.18139785222060435552e-2,
                -4.45641913851797240494e-3,
                5.39605580493303397842e-4,
                -2.31581873324120129819e-5,
            ]
        ](t)
        return p / q
    else:
        # Relative least-squares fit of Gamma(2 + t) on [0, 1], max error
        # about 1.1 ulp in float32.
        return polynomial_evaluate[
            [
                Scalar[dtype](1.0),
                0.42278435826301575,
                0.41183924674987793,
                0.0815906971693039,
                0.07415854185819626,
                7.92462524259463e-05,
                0.010327345691621304,
                -0.001574849127791822,
                0.0007953942404128611,
            ]
        ](t)


def _gamma_impl[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    """Computes Gamma(x) in float32 or float64 without libm.

    Reduces x to z in [2, 3) with the recurrence Gamma(x + 1) = x Gamma(x),
    accumulating the product of the shift factors in double-word arithmetic so
    the only roundings are the base approximation and the final combination.
    The factors are generated as exact two-sums of x and an integer, and the
    product is rescaled by a power of two whenever it grows large, so neither
    near-overflow results nor underflowing results for negative x lose
    accuracy. This avoids exp/log/pow, whose GPU implementations are not
    accurate enough for a few-ulp Gamma near the overflow threshold.

    Subnormal inputs and results are handled on targets that flush
    subnormals (Apple GPUs in float32): the input is read from its bits and
    the last rescaling of a subnormal result is done with integer operations.
    """
    comptime assert dtype in (DType.float32, DType.float64)
    comptime T = SIMD[dtype, width]
    comptime is_f64 = dtype == .float64
    # Above `max_arg` the result overflows; below `min_arg` it underflows to a
    # signed zero. Both bound the number of recurrence steps.
    comptime max_arg = Scalar[dtype](172.0 if is_f64 else 36.0)
    comptime min_arg = Scalar[dtype](-190.0 if is_f64 else -50.0)
    comptime scale_log2 = 1000 if is_f64 else 64
    comptime scale_up = Scalar[dtype](
        1.0715086071862673e301 if is_f64 else 1.8446744073709552e19
    )  # 2^1000 or 2^64, also the rescaling threshold
    comptime scale_down = Scalar[dtype](
        9.332636185032189e-302 if is_f64 else 5.421010862427522e-20
    )  # 2^-1000 or 2^-64
    # A rescaling fires only once the unscaled running product exceeds
    # scale_up^count, so the count is bounded by the largest product.
    comptime max_scalings = _gamma_max_scalings(
        max(Int(max_arg) - 1, Int(-min_arg)), scale_log2
    )

    var fl = floor(x)
    # Sign of Gamma on (n, n + 1) for negative integer n is (-1)^n.
    var odd = (fl - 2 * floor(fl * 0.5)).ne(0)
    var is_nan = isnan(x)
    var is_zero = x.eq(0)
    var is_pole = x.lt(0) & x.eq(fl)
    var too_big = x.gt(max_arg)
    var too_small = x.lt(min_arg)
    var special = is_nan | is_zero | is_pole | too_big | too_small
    var xc = special.select(T(2.5), x)
    fl = special.select(T(2.0), fl)

    var up = xc.ge(2)
    # For x >= 2: Gamma(x) = Gamma(z) * (x - 1) ... (x - m), with z = x - m.
    # For x < 2:  Gamma(x) = Gamma(z) / (x (x + 1) ... (x + m - 1)), z = x + m.
    var m = up.select(fl - 2, 2 - fl)
    var shift = up.select(-m, m)
    var zs = _two_sum(xc, shift)
    var z = zs[0]
    var z_lo = zs[1]

    var ph = T(1)
    var pl = T(0)
    var nscale = T(0)
    var steps = Int(m.reduce_max())
    for j in range(steps):
        var jf = T(Scalar[dtype](j))
        var active = jf.lt(m)
        var c = up.select(-(jf + 1), jf)
        var f = _two_sum(xc, c)
        var fh = active.select(f[0], T(1))
        var f_lo = active.select(f[1], T(0))
        # Double-word product (ph, pl) * (fh, f_lo). Every rounded product
        # that must stay unfused is an explicit fma, so the result does not
        # depend on whether the backend contracts a*b + c (ptxas does).
        var cross = ph * f_lo + pl * fh
        var hi = fma(ph, fh, cross)
        pl = fma(ph, fh, -hi) + cross
        ph = hi
        var big = abs(ph).gt(scale_up)
        ph = big.select(ph * scale_down, ph)
        pl = big.select(pl * scale_down, pl)
        nscale += big.select(T(1), T(0))

    var t = z - 2
    var r = _gamma_2_to_3(t)
    # z is xc + shift rounded; correct by z_lo using Gamma'/Gamma = digamma,
    # which a line approximates well enough on [2, 3] for this tiny term.
    comptime digamma_2 = 0.42278433509846713  # 1 - Euler's constant
    r = fma(r * (digamma_2 + 0.5 * t), z_lo, r)

    var prod = fma(r, ph, r * pl)
    var q = r / ph
    var quot = q + (fma(-q, ph, r) - q * pl) / ph
    quot = isinf(q).select(q, quot)
    var res = up.select(prod, quot)
    comptime for i in range(max_scalings):
        var apply = nscale.gt(Scalar[dtype](i))
        comptime if CurrentPlugin.flushes_subnormals[dtype]:
            # Scaling down may produce a subnormal, which a multiply flushes.
            res = apply.select(
                up.select(res * scale_up, _scale_down_bits[scale_log2](res)),
                res,
            )
        else:
            res = apply.select(res * up.select(T(scale_up), T(scale_down)), res)

    var tiny = odd.select(T(-0.0), T(0.0))
    res = too_small.select(tiny, res)
    res = too_big.select(T(inf[dtype]()), res)
    res = is_zero.select(copysign(T(inf[dtype]()), x), res)

    # For subnormal x, Gamma(x) = 1/x - euler_gamma + O(x) rounds to 1/x: the
    # correction is 2^-126 (2^-1022) relative and 1/x is never that close to
    # a rounding boundary. Build 1/x from the integer significand m, with
    # x = m * 2^(emin - mant), so a target that flushes subnormal operands
    # (Apple GPUs in float32) still gets the normal or overflowing result.
    comptime I = SIMD[FPUtils[dtype].integral_type, width]
    comptime mant = FPUtils[dtype].mantissa_width()
    comptime bias = FPUtils[dtype].exponent_bias()
    var bits = x._to_bits_signed()
    var sig = (bits & I((1 << mant) - 1)).cast[dtype]()
    comptime exp_mask = (1 << FPUtils[dtype].exponent_width()) - 1
    var sub = ((bits >> I(mant)) & I(exp_mask)).eq(0) & sig.ne(0)
    comptime two_mant = T(from_bits=I(bias + mant) << I(mant))
    comptime two_neg_emin = T(from_bits=I(2 * bias - 1) << I(mant))
    var inv = (two_mant / sig) * two_neg_emin
    res = sub.select(bits.lt(0).select(-inv, inv), res)
    res = (is_nan | is_pole).select(T(nan[dtype]()), res)
    return res


def _expm1_accurate[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> type_of(x) where dtype.is_floating_point():
    # FreeBSD msun / Sun fdlibm `s_expm1.c` and `s_expm1f.c`, made branch-free
    # over lanes. The k = 0 and k = +-1 cases avoid forming 2^k (1 + r) - 1,
    # whose rounded 1 + r loses the low bits the subtraction exposes.
    comptime assert dtype in (DType.float32, DType.float64)
    comptime is_f32 = dtype == .float32
    comptime T = type_of(x)
    comptime int_t = DType.int32 if is_f32 else DType.int64
    comptime uint_t = DType.uint32 if is_f32 else DType.uint64
    comptime mant_bits = Scalar[uint_t](23 if is_f32 else 52)
    comptime bias = Scalar[int_t](127 if is_f32 else 1023)
    # x - k*ln2 = xr + c, with k * ln2_hi exact for every k reached here.
    comptime ln2_hi = Scalar[dtype](6.9313812256e-01 if is_f32 else _LN2_HI_F64)
    comptime ln2_lo = Scalar[dtype](9.0580006145e-06 if is_f32 else _LN2_LO_F64)
    # Largest x with a finite result: log(MAX_FINITE) rounded down.
    comptime o_threshold = Scalar[dtype](
        88.72283172607421875 if is_f32 else 7.09782712893383973096e02
    )
    # Below -27 ln2 (f32) or -56 ln2 (f64), expm1(x) rounds to -1.
    comptime neg_threshold = Scalar[dtype](
        -18.714973875 if is_f32 else -38.816242111356935
    )
    # Below 2^-25 (f32) or 2^-54 (f64), expm1(x) rounds to x.
    comptime tiny = Scalar[dtype](
        2.98023223876953125e-08 if is_f32 else 5.551115123125783e-17
    )
    comptime clamp_lo = Scalar[dtype](-30.0 if is_f32 else -60.0)
    comptime clamp_hi = Scalar[dtype](89.0 if is_f32 else 710.0)
    # fdlibm's cutoff between the y_mid and y_hi forms below.
    comptime k_mid = Scalar[int_t](23 if is_f32 else 20)

    # Clamp (and zero NaN) so the float->int conversion below never sees an
    # out-of-range value; lanes outside the clamp are replaced at the end.
    var xc = isnan(x).select(T(0), x.clamp(clamp_lo, clamp_hi))
    var kf = round(xc * log2e)
    var k = kf.cast[int_t]()

    var hi = xc - kf * ln2_hi
    var lo = kf * ln2_lo
    var xr = hi - lo
    var c = (hi - xr) - lo

    var hfx = 0.5 * xr
    var hxs = xr * hfx
    var q: T
    comptime if is_f32:
        q = polynomial_evaluate[
            [Scalar[dtype](-3.3333212137e-2), 1.5807170421e-3]
        ](hxs)
    else:
        q = polynomial_evaluate[
            [
                Scalar[dtype](-3.33333333333331316428e-02),
                1.58730158725481460165e-03,
                -7.93650757867487942473e-05,
                4.00821782732936239552e-06,
                -2.01099218183624371326e-07,
            ]
        ](hxs)
    var r1 = 1.0 + hxs * q
    var t = 3.0 - r1 * hfx
    var e = hxs * ((r1 - t) / (6.0 - xr * t))

    var res_k0 = xr - (xr * e - hxs)

    e = xr * (e - c) - c
    e = e - hxs

    var res_km1 = 0.5 * (xr - e) - 0.5
    var res_k1 = xr.lt(-0.25).select(
        -2.0 * (e - (xr + 0.5)), 1.0 + 2.0 * (xr - e)
    )

    # 2^(k-1) and 2^-k as bit patterns; scaling by 2 * 2^(k-1) reaches the
    # top binade without overflowing the exponent field.
    var twopk_m1 = bitcast[dtype, width](
        (k + (bias - 1)).cast[uint_t]() << mant_bits
    )
    var kc = k.clamp(-60, 60)
    var twomk = bitcast[dtype, width]((bias - kc).cast[uint_t]() << mant_bits)

    var y_far = (1.0 - (e - xr)) * 2.0 * twopk_m1 - 1.0
    var y_mid = ((1.0 - twomk) - (e - xr)) * 2.0 * twopk_m1
    var y_hi = ((xr - (e + twomk)) + 1.0) * 2.0 * twopk_m1

    var res = (k.le(-2) | k.gt(56)).select(
        y_far, k.lt(k_mid).select(y_mid, y_hi)
    )
    res = k.eq(1).select(res_k1, res)
    res = k.eq(-1).select(res_km1, res)
    res = k.eq(0).select(res_k0, res)

    res = x.gt(o_threshold).select(inf[dtype](), res)
    res = x.lt(neg_threshold).select(T(-1), res)
    # NaN passes through; tiny inputs (including -0 and subnormals) return x
    # exactly. Apple GPUs compare subnormals as zero, which still selects x.
    return (isnan(x) | abs(x).lt(tiny)).select(x, res)


def _atanh_float32(x: SIMD) -> type_of(x) where x.dtype.is_floating_point():
    """This computes the `atanh` of the inputs for float32. It uses the same
    approximation used by Eigen library."""

    comptime nan_val = nan[x.dtype]()
    comptime inf_val = inf[x.dtype]()
    comptime neg_inf_val = -inf[x.dtype]()

    var is_neg = x.lt(0)
    var x_abs = abs(x)
    var x2 = x * x
    var x3 = x2 * x

    # When x is in the range [0, 0.5], we use a polynomial approximation.
    # P(x) = x + x^3*(c[4] + x^2 * (c[3] + x^2 * (... x^2 * c[0]) ... )).
    var p = polynomial_evaluate[
        [
            Scalar[x.dtype](0.3333373963832855224609375),
            0.1997792422771453857421875,
            0.14672131836414337158203125,
            8.2311116158962249755859375e-2,
            0.1819281280040740966796875,
        ]
    ](x2)
    p = x3.fma(p, x)

    # For |x| in the range (0.5, 1), we use the identity:
    # atanh(x) = 0.5 * log((1 + x) / (1 - x))
    var r = 0.5 * log((1.0 + x) / (1.0 - x))

    # If x is 1, then the result is +infinity, and -infinity if x is -1.
    # If |x| > 1, NaN is returned. Otherwise, if |x| > 0.5, we use the r
    # approximation, otherwise we use the p polynomial approximation.
    return x_abs.eq(1).select(
        is_neg.select(neg_inf_val, inf_val),
        x_abs.ge(1).select(
            is_neg.select(nan_val, -nan_val),
            x_abs.gt(0.5).select(r, p),
        ),
    )


@inline(.never)
def _bessel_is_zero_bits[
    dtype: DType, width: SIMDLength, //
](bits: SIMD[dtype, width]) -> SIMD[.bool, width]:
    # Inlining the signless-bit test can turn it into a flushing FP compare.
    comptime assert dtype == .int32 or dtype == .int64
    comptime mask = SIMD[dtype, width]((1 << (bit_width_of[dtype]() - 1)) - 1)
    return (bits & mask).eq(0)


@inline(.always)
def _interval_coef[
    dtype: DType,
    width: SIMDLength,
    //,
    c8: Span[Scalar[dtype], _],
    c5: Span[Scalar[dtype], _],
    c3: Span[Scalar[dtype], _],
    c2: Span[Scalar[dtype], _],
    i: Int,
](
    m8: SIMD[.bool, width], m5: SIMD[.bool, width], m3: SIMD[.bool, width]
) -> SIMD[dtype, width]:
    comptime V = SIMD[dtype, width]
    comptime a8 = c8[i]
    comptime a5 = c5[i]
    comptime a3 = c3[i]
    comptime a2 = c2[i]
    return m8.select(V(a8), m5.select(V(a5), m3.select(V(a3), V(a2))))


@inline(.always)
def _interval_poly[
    dtype: DType,
    width: SIMDLength,
    //,
    c8: Span[Scalar[dtype], _],
    c5: Span[Scalar[dtype], _],
    c3: Span[Scalar[dtype], _],
    c2: Span[Scalar[dtype], _],
](
    m8: SIMD[.bool, width],
    m5: SIMD[.bool, width],
    m3: SIMD[.bool, width],
    z: SIMD[dtype, width],
) -> SIMD[dtype, width]:
    """Evaluates the polynomial in `z` whose coefficient set each lane picks
    from its interval mask (`c8` for x >= 8, then `c5`, `c3`, else `c2`)."""
    comptime n = len(c8)
    var acc = _interval_coef[c8, c5, c3, c2, n - 1](m8, m5, m3)
    comptime for j in range(1, n):
        acc = acc.fma(z, _interval_coef[c8, c5, c3, c2, n - 1 - j](m8, m5, m3))
    return acc


@inline(.always)
def _j0_small[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> type_of(x) where dtype.is_floating_point():
    """Evaluates `j0(x)` for `0 <= x < 2` (fdlibm e_j0.c)."""
    var z = x * x
    var r = z * polynomial_evaluate[
        [
            Scalar[dtype](1.56249999999999947958e-02),
            -1.89979294238854721751e-04,
            1.82954049532700665670e-06,
            -4.61832688532103189199e-09,
        ]
    ](z)
    var s = polynomial_evaluate[
        [
            Scalar[dtype](1.0),
            1.56191029464890010492e-02,
            1.16926784663337450260e-04,
            5.13546550207318111446e-07,
            1.16614003333790000205e-09,
        ]
    ](z)
    var u = 0.5 * x
    return x.lt(1.0).select(
        1.0 + z * (r / s - 0.25), (1.0 + u) * (1.0 - u) + z * (r / s)
    )


@inline(.always)
def _j0y0_asymptotic[
    dtype: DType, width: SIMDLength, //, is_y0: Bool
](x: SIMD[dtype, width]) -> type_of(x) where dtype.is_floating_point():
    """Evaluates `j0(x)` or `y0(x)` for finite `x >= 2` (fdlibm e_j0.c).

    `j0 = sqrt(2/(pi x)) (P cos(x0) - Q sin(x0))` and
    `y0 = sqrt(2/(pi x)) (P sin(x0) + Q cos(x0))` with `x0 = x - pi/4`.
    """
    comptime T = type_of(x)
    var sc = _sincos_minus_pio4(x)
    var pq = _j0y0_pq(x)
    var amp: T
    comptime if is_y0:
        amp = pq[0] * sc[0] + pq[1] * sc[1]
    else:
        amp = pq[0] * sc[1] - pq[1] * sc[0]
    return T(_SQRT_TWO_OVER_PI) * amp / _sqrt_ieee(x)


@inline(.always)
def _j0y0_impl[
    dtype: DType, width: SIMDLength, //, is_y0: Bool
](x: SIMD[dtype, width]) -> type_of(x) where dtype.is_floating_point():
    """Computes `j0` or `y0` in float32 or float64 arithmetic."""
    comptime T = type_of(x)
    var ax = abs(x)
    var big = ax.ge(_BESSEL_ASYMPTOTIC_MIN) & ax.lt(inf[dtype]())
    var res = T(0)

    if any(big):
        res = _j0y0_asymptotic[is_y0](big.select(ax, T(_BESSEL_ASYMPTOTIC_MIN)))

    if not all(big):
        var xs = big.select(T(1), ax)
        var j = _j0_small(xs)
        comptime if is_y0:
            var z = xs * xs
            var u = polynomial_evaluate[
                [
                    Scalar[dtype](-7.38042951086872317523e-02),
                    1.76666452509181115538e-01,
                    -1.38185671945596898896e-02,
                    3.47453432093683650238e-04,
                    -3.81407053724364161125e-06,
                    1.95590137035022920206e-08,
                    -3.98205194132103398453e-11,
                ]
            ](z)
            var v = polynomial_evaluate[
                [
                    Scalar[dtype](1.0),
                    1.27304834834123699328e-02,
                    7.60068627350353253702e-05,
                    2.59150851840457805467e-07,
                    4.41110311332675467403e-10,
                ]
            ](z)
            j = u / v + T(_TWO_OVER_PI) * (j * _log_accurate(xs))
        res = big.select(res, j)

    comptime if is_y0:
        # Sign and zero from the bits, which a flushing compare cannot alter.
        var bits = x._to_bits_signed()
        var is_zero = _bessel_is_zero_bits(bits)
        res = isinf(x).select(T(0), res)
        res = is_zero.select(-inf[dtype](), res)
        return ((bits.lt(0) & ~is_zero) | isnan(x)).select(nan[dtype](), res)
    return isinf(x).select(T(0), res)


@inline(.always)
def _j0y0_pq[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> Tuple[
    type_of(x), type_of(x)
] where dtype.is_floating_point():
    """Returns fdlibm's `(pzero(x), qzero(x))` for `x >= 2`.

    The rational coefficients are from fdlibm e_j0.c; float32 rounds them,
    which reproduces e_j0f.c.
    """
    # Intervals [8, inf), [_PQ_SPLIT_5, 8), [_PQ_SPLIT_3, _PQ_SPLIT_5),
    # [2, _PQ_SPLIT_3).
    comptime PR8: List[Scalar[dtype]] = [
        0.0,
        -7.03124999999900357484e-02,
        -8.08167041275349795626e00,
        -2.57063105679704847262e02,
        -2.48521641009428822144e03,
        -5.25304380490729545272e03,
    ]
    comptime PR5: List[Scalar[dtype]] = [
        -1.14125464691894502584e-11,
        -7.03124940873599280078e-02,
        -4.15961064470587782438e00,
        -6.76747652265167261021e01,
        -3.31231299649172967747e02,
        -3.46433388365604912451e02,
    ]
    comptime PR3: List[Scalar[dtype]] = [
        -2.54704601771951915620e-09,
        -7.03119616381481654654e-02,
        -2.40903221549529611423e00,
        -2.19659774734883086467e01,
        -5.80791704701737572236e01,
        -3.14479470594888503854e01,
    ]
    comptime PR2: List[Scalar[dtype]] = [
        -8.87534333032526411254e-08,
        -7.03030995483624743247e-02,
        -1.45073846780952986357e00,
        -7.63569613823527770791e00,
        -1.11931668860356747786e01,
        -3.23364579351335335033e00,
    ]
    comptime PS8: List[Scalar[dtype]] = [
        1.16534364619668181717e02,
        3.83374475364121826715e03,
        4.05978572648472545552e04,
        1.16752972564375915681e05,
        4.76277284146730962675e04,
    ]
    comptime PS5: List[Scalar[dtype]] = [
        6.07539382692300335975e01,
        1.05125230595704579173e03,
        5.97897094333855784498e03,
        9.62544514357774460223e03,
        2.40605815922939109441e03,
    ]
    comptime PS3: List[Scalar[dtype]] = [
        3.58560338055209726349e01,
        3.61513983050303863820e02,
        1.19360783792111533330e03,
        1.12799679856907414432e03,
        1.73580930813335754692e02,
    ]
    comptime PS2: List[Scalar[dtype]] = [
        2.22202997532088808441e01,
        1.36206794218215208048e02,
        2.70470278658083486789e02,
        1.53875394208320329881e02,
        1.46576176948256193810e01,
    ]
    comptime QR8: List[Scalar[dtype]] = [
        0.0,
        7.32421874999935051953e-02,
        1.17682064682252693899e01,
        5.57673380256401856059e02,
        8.85919720756468632317e03,
        3.70146267776887834771e04,
    ]
    comptime QR5: List[Scalar[dtype]] = [
        1.84085963594515531381e-11,
        7.32421766612684765896e-02,
        5.83563508962056953777e00,
        1.35111577286449829671e02,
        1.02724376596164097464e03,
        1.98997785864605384631e03,
    ]
    comptime QR3: List[Scalar[dtype]] = [
        4.37741014089738620906e-09,
        7.32411180042911447163e-02,
        3.34423137516170720929e00,
        4.26218440745412650017e01,
        1.70808091340565596283e02,
        1.66733948696651168575e02,
    ]
    comptime QR2: List[Scalar[dtype]] = [
        1.50444444886983272379e-07,
        7.32234265963079278272e-02,
        1.99819174093815998816e00,
        1.44956029347885735348e01,
        3.16662317504781540833e01,
        1.62527075710929267416e01,
    ]
    comptime QS8: List[Scalar[dtype]] = [
        1.63776026895689824414e02,
        8.09834494656449805916e03,
        1.42538291419120476348e05,
        8.03309257119514397345e05,
        8.40501579819060512818e05,
        -3.43899293537866615225e05,
    ]
    comptime QS5: List[Scalar[dtype]] = [
        8.27766102236537761883e01,
        2.07781416421392987104e03,
        1.88472887785718085070e04,
        5.67511122894947329769e04,
        3.59767538425114471465e04,
        -5.35434275601944773371e03,
    ]
    comptime QS3: List[Scalar[dtype]] = [
        4.87588729724587182091e01,
        7.09689221056606015736e02,
        3.70414822620111362994e03,
        6.46042516752568917582e03,
        2.51633368920368957333e03,
        -1.49247451836156386662e02,
    ]
    comptime QS2: List[Scalar[dtype]] = [
        3.03655848355219184498e01,
        2.69348118608049844624e02,
        8.44783757595320139444e02,
        8.82935845112488550512e02,
        2.12666388511798828631e02,
        -5.31095493882666946917e00,
    ]

    var m8 = x.ge(_PQ_SPLIT_8)
    var m5 = x.ge(_PQ_SPLIT_5)
    var m3 = x.ge(_PQ_SPLIT_3)
    var z = 1.0 / (x * x)
    var pr = _interval_poly[PR8, PR5, PR3, PR2](m8, m5, m3, z)
    var ps = 1.0 + z * _interval_poly[PS8, PS5, PS3, PS2](m8, m5, m3, z)
    var qr = _interval_poly[QR8, QR5, QR3, QR2](m8, m5, m3, z)
    var qs = 1.0 + z * _interval_poly[QS8, QS5, QS3, QS2](m8, m5, m3, z)
    return (1.0 + pr / ps, (qr / qs - 0.125) / x)


@inline(.always)
def _j1y1_impl[
    dtype: DType, width: SIMDLength, //, is_y1: Bool
](x: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    """Computes `j1(x)` or `y1(x)` following FreeBSD msun e_j1.c / e_j1f.c."""
    comptime V = SIMD[dtype, width]
    comptime is_f64 = dtype == .float64

    var ax = abs(x)
    var asym = ax.ge(_BESSEL_ASYMPTOTIC_MIN) & isfinite(x)

    var res = V(0.0)
    if any(asym):
        # |x| >= 2, with x0 = x - pi/4 and x - 3pi/4 = x0 - pi/2:
        # j1 = sqrt(2/(pi x)) (p1 sin(x0) + q1 cos(x0)),
        # y1 = sqrt(2/(pi x)) (q1 sin(x0) - p1 cos(x0)).
        var xa = asym.select(ax, V(_BESSEL_ASYMPTOTIC_MIN))
        var sc = _sincos_minus_pio4(xa)
        var pq = _j1y1_pone_qone(xa)
        var amp: V
        comptime if is_y1:
            amp = pq[1] * sc[0] - pq[0] * sc[1]
        else:
            amp = pq[0] * sc[0] + pq[1] * sc[1]
        var big = V(_SQRT_TWO_OVER_PI) * amp / _sqrt_ieee(xa)
        comptime if is_y1:
            res = big
        else:
            res = x.lt(0.0).select(-big, big)

    if not all(asym):
        # |x| < 2: j1(x) = x/2 + x z R(z)/S(z), z = x^2.
        var xs = asym.select(V(1.0), x)
        var z = xs * xs
        var r = z * polynomial_evaluate[
            [
                Scalar[dtype](-6.25000000000000000000e-02),
                1.40705666955189706048e-03,
                -1.59955631084035597520e-05,
                4.96727999609584448412e-08,
            ]
        ](z)
        var sd = polynomial_evaluate[
            [
                Scalar[dtype](1.0),
                1.91537599538363460805e-02,
                1.85946785588630915560e-04,
                1.17718464042623683263e-06,
                5.04636257076217042715e-09,
                1.23542274426137913908e-11,
            ]
        ](z)
        var j1_small = (0.5 + r / sd) * xs

        comptime if not is_y1:
            comptime if CurrentPlugin.flushes_subnormals[dtype]:
                # j1(x) = x/2 to float32 precision for |x| < 2^-125, where
                # x/2 is subnormal and would flush, so halve x from its
                # integer significand and round the result from its bits.
                comptime assert (
                    dtype == .float32
                ), "the subnormal-flushing `j1` path supports only float32"
                comptime abs_mask = 0x7FFFFFFF
                comptime two_pow_neg_125_bits = 0x01000000
                comptime min_normal_bits = 0x00800000
                comptime lift_shift = 24  # scales by 2^24
                comptime two_pow_24 = 16777216.0
                var m = x._to_bits_signed() & abs_mask
                var tiny = m.ne(0) & m.lt(two_pow_neg_125_bits)
                var scaled = m.lt(min_normal_bits).select(
                    _scale_up_subnormal[lift_shift](x), x * two_pow_24
                )
                j1_small = tiny.select(
                    _scale_down_bits[lift_shift + 1](scaled), j1_small
                )
            res = asym.select(res, j1_small)
        else:
            # y1(x) = x U(z)/V(z) + (2/pi)(j1(x) log(x) - 1/x).
            comptime tiny = Scalar[dtype](
                5.551115123125783e-17 if is_f64 else 2.9802322387695312e-08
            )  # 2^-54 or 2^-25
            var u = polynomial_evaluate[
                [
                    Scalar[dtype](-1.96057090646238940668e-01),
                    5.04438716639811282616e-02,
                    -1.91256895875763547298e-03,
                    2.35252600561610495928e-05,
                    -9.19099158039878874504e-08,
                ]
            ](z)
            var v = polynomial_evaluate[
                [
                    Scalar[dtype](1.0),
                    1.99167318236649903973e-02,
                    2.02552581025135171496e-04,
                    1.35608801097516229404e-06,
                    6.22741452364621501295e-09,
                    1.66559246207992079114e-11,
                ]
            ](z)
            var log_ok = xs.ge(tiny) & xs.lt(_BESSEL_ASYMPTOTIC_MIN)
            var lg = _log_accurate(log_ok.select(xs, V(1.0)))
            var y1_small = xs.fma(
                u / v, _TWO_OVER_PI * j1_small.fma(lg, -1.0 / xs)
            )
            # Below `tiny`, y1(x) = -2/(pi x). A subnormal x = m * 2^(emin - mant)
            # is divided through its integer significand, scaled so that the
            # quotient stays normal: a target that flushes subnormal operands
            # (Apple GPUs in float32) would otherwise divide by zero, and 1/x
            # alone overflows where y1 is still finite.
            comptime I = SIMD[FPUtils[dtype].integral_type, width]
            comptime mant = FPUtils[dtype].mantissa_width()
            comptime bias = FPUtils[dtype].exponent_bias()
            comptime exp_mask = (1 << FPUtils[dtype].exponent_width()) - 1
            comptime two_mant = V(from_bits=I(bias + mant) << I(mant))
            comptime two_neg_emin = V(from_bits=I(2 * bias - 1) << I(mant))
            comptime c_scaled = V(_TWO_OVER_PI) * two_mant
            var bits = x._to_bits_signed()
            var sig = (bits & I((1 << mant) - 1)).cast[dtype]()
            var sub = ((bits >> I(mant)) & I(exp_mask)).eq(0)
            var tiny_y1 = sub.select(
                -(c_scaled / sig) * two_neg_emin, -_TWO_OVER_PI / xs
            )
            y1_small = xs.lt(tiny).select(tiny_y1, y1_small)
            res = asym.select(res, y1_small)

    comptime if not is_y1:
        # +0 at either infinity matches j0's convention and musl.
        return isinf(x).select(V(0.0), res)
    var bits = x._to_bits_signed()
    var is_zero = _bessel_is_zero_bits(bits)
    res = is_zero.select(V(-inf[dtype]()), res)
    res = (bits.lt(0) & ~is_zero).select(V(nan[dtype]()), res)
    return x.eq(inf[dtype]()).select(V(0.0), res)


@inline(.always)
def _j1y1_pone_qone[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> Tuple[SIMD[dtype, width], SIMD[dtype, width]]:
    """Returns fdlibm's `(pone(x), qone(x))` for `x >= 2` (FreeBSD msun
    e_j1.c; the float32 path uses the same coefficients rounded to float)."""
    # Intervals [8, inf), [_PQ_SPLIT_5, 8), [_PQ_SPLIT_3, _PQ_SPLIT_5),
    # [2, _PQ_SPLIT_3).
    comptime PR8: List[Scalar[dtype]] = [
        0.00000000000000000000e00,
        1.17187499999988647970e-01,
        1.32394806593073575129e01,
        4.12051854307378562225e02,
        3.87474538913960532227e03,
        7.91447954031891731574e03,
    ]
    comptime PS8: List[Scalar[dtype]] = [
        1.14207370375678408436e02,
        3.65093083420853463394e03,
        3.69562060269033463555e04,
        9.76027935934950801311e04,
        3.08042720627888811578e04,
    ]
    comptime PR5: List[Scalar[dtype]] = [
        1.31990519556243522749e-11,
        1.17187493190614097638e-01,
        6.80275127868432871736e00,
        1.08308182990189109773e02,
        5.17636139533199752805e02,
        5.28715201363337541807e02,
    ]
    comptime PS5: List[Scalar[dtype]] = [
        5.92805987221131331921e01,
        9.91401418733614377743e02,
        5.35326695291487976647e03,
        7.84469031749551231769e03,
        1.50404688810361062679e03,
    ]
    comptime PR3: List[Scalar[dtype]] = [
        3.02503916137373618024e-09,
        1.17186865567253592491e-01,
        3.93297750033315640650e00,
        3.51194035591636932736e01,
        9.10550110750781271918e01,
        4.85590685197364919645e01,
    ]
    comptime PS3: List[Scalar[dtype]] = [
        3.47913095001251519989e01,
        3.36762458747825746741e02,
        1.04687139975775130551e03,
        8.90811346398256432622e02,
        1.03787932439639277504e02,
    ]
    comptime PR2: List[Scalar[dtype]] = [
        1.07710830106873743082e-07,
        1.17176219462683348094e-01,
        2.36851496667608785174e00,
        1.22426109148261232917e01,
        1.76939711271687727390e01,
        5.07352312588818499250e00,
    ]
    comptime PS2: List[Scalar[dtype]] = [
        2.14364859363821409488e01,
        1.25290227168402751090e02,
        2.32276469057162813669e02,
        1.17679373287147100768e02,
        8.36463893371618283368e00,
    ]
    comptime QR8: List[Scalar[dtype]] = [
        0.00000000000000000000e00,
        -1.02539062499992714161e-01,
        -1.62717534544589987888e01,
        -7.59601722513950107896e02,
        -1.18498066702429587167e04,
        -4.84385124285750353010e04,
    ]
    comptime QS8: List[Scalar[dtype]] = [
        1.61395369700722909556e02,
        7.82538599923348465381e03,
        1.33875336287249578163e05,
        7.19657723683240939863e05,
        6.66601232617776375264e05,
        -2.94490264303834643215e05,
    ]
    comptime QR5: List[Scalar[dtype]] = [
        -2.08979931141764104297e-11,
        -1.02539050241375426231e-01,
        -8.05644828123936029840e00,
        -1.83669607474888380239e02,
        -1.37319376065508163265e03,
        -2.61244440453215656817e03,
    ]
    comptime QS5: List[Scalar[dtype]] = [
        8.12765501384335777857e01,
        1.99179873460485964642e03,
        1.74684851924908907677e04,
        4.98514270910352279316e04,
        2.79480751638918118260e04,
        -4.71918354795128470869e03,
    ]
    comptime QR3: List[Scalar[dtype]] = [
        -5.07831226461766561369e-09,
        -1.02537829820837089745e-01,
        -4.61011581139473403113e00,
        -5.78472216562783643212e01,
        -2.28244540737631695038e02,
        -2.19210128478909325622e02,
    ]
    comptime QS3: List[Scalar[dtype]] = [
        4.76651550323729509273e01,
        6.73865112676699709482e02,
        3.38015286679526343505e03,
        5.54772909720722782367e03,
        1.90311919338810798763e03,
        -1.35201191444307340817e02,
    ]
    comptime QR2: List[Scalar[dtype]] = [
        -1.78381727510958865572e-07,
        -1.02517042607985553460e-01,
        -2.75220568278187460720e00,
        -1.96636162643703720221e01,
        -4.23253133372830490089e01,
        -2.13719211703704061733e01,
    ]
    comptime QS2: List[Scalar[dtype]] = [
        2.95333629060523854548e01,
        2.52981549982190529136e02,
        7.57502834868645436472e02,
        7.39393205320467245656e02,
        1.55949003336666123687e02,
        -4.95949898822628210127e00,
    ]

    var m8 = x.ge(_PQ_SPLIT_8)
    var m5 = x.ge(_PQ_SPLIT_5)
    var m3 = x.ge(_PQ_SPLIT_3)
    var z = 1.0 / (x * x)
    var pr = _interval_poly[PR8, PR5, PR3, PR2](m8, m5, m3, z)
    var ps = 1.0 + z * _interval_poly[PS8, PS5, PS3, PS2](m8, m5, m3, z)
    var qr = _interval_poly[QR8, QR5, QR3, QR2](m8, m5, m3, z)
    var qs = 1.0 + z * _interval_poly[QS8, QS5, QS3, QS2](m8, m5, m3, z)
    return (1.0 + pr / ps, (0.375 + qr / qs) / x)


@inline(.always)
def _log10_impl[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> type_of(x) where dtype.is_floating_point():
    # FreeBSD msun e_log10.c / e_log10f.c with k_log.h / k_logf.h. The
    # reduced log is split into hi/lo halves so the products with log10(e) and
    # log10(2) stay well under 1 ULP.
    comptime is_f32 = dtype == .float32
    comptime assert is_f32 or dtype == .float64, "expected float32 or float64"
    comptime ivln10hi = Scalar[dtype](
        4.3432617188e-01 if is_f32 else 4.34294481878168880939e-01
    )
    comptime ivln10lo = Scalar[dtype](
        -3.1689971365e-05 if is_f32 else 2.50829467116452752298e-11
    )
    comptime log10_2hi = Scalar[dtype](
        3.0102920532e-01 if is_f32 else 3.01029995663611771306e-01
    )
    comptime log10_2lo = Scalar[dtype](
        7.9034151668e-07 if is_f32 else 3.69423907715893078616e-13
    )
    # Keeps the high half of the significand so `hi * ivln10hi` is exact.
    comptime U: DType = .uint32 if is_f32 else .uint64
    comptime hi_mask = Scalar[U](0xFFFFF000 if is_f32 else 0xFFFFFFFF00000000)

    var y, f = _log_reduce(x)
    var hfsq = 0.5 * f * f
    var r = _log_kernel(f)

    var hi = bitcast[dtype](bitcast[U](f - hfsq) & hi_mask)
    var lo = (f - hi) - hfsq + r
    var val_hi = hi * ivln10hi
    var y2 = y * log10_2hi
    var val_lo = y * log10_2lo + (lo + hi) * ivln10lo + lo * ivln10hi
    var w = y2 + val_hi
    val_lo += (y2 - w) + val_hi
    var result = val_lo + w

    # Zero and sign come from the bits so that a subnormal input is not taken
    # for zero on targets whose comparisons flush subnormal operands.
    var bits = x._to_bits_signed()
    var is_zero = (bits + bits).eq(0)
    result = x.eq(inf[dtype]()).select(x, result)
    result = is_zero.select(-inf[dtype](), result)
    return ((bits.lt(0) & ~is_zero) | isnan(x)).select(nan[dtype](), result)


def _log1p_impl[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> type_of(x) where dtype.is_floating_point():
    # This uses the approximation from cephes to compute log1p via the approximation
    # log(1+x) = x - x**2/2 + x**3 P(x)/Q(x)
    # in the domain 1/sqrt(2) <= x < sqrt(2)

    comptime P: List[Scalar[dtype]] = [
        2.0039553499201281259648e1,
        5.7112963590585538103336e1,
        6.0949667980987787057556e1,
        2.9911919328553073277375e1,
        6.5787325942061044846969e0,
        4.9854102823193375972212e-1,
        4.5270000862445199635215e-5,
    ]
    comptime Q: List[Scalar[dtype]] = [
        6.0118660497603843919306e1,
        2.1642788614495947685003e2,
        3.0909872225312059774938e2,
        2.2176239823732856465394e2,
        8.3047565967967209469434e1,
        1.5062909083469192043167e1,
    ]

    # Sqrt(1/2)
    comptime sqrt2_div_2 = 0.70710678118654752440
    # Sqrt(2)
    comptime sqrt2 = 1.41421356237309504880

    var z = 1.0 + x
    var log1x = log(z)

    var in_domain_mask = z.lt(sqrt2_div_2) | z.gt(sqrt2)
    if all(in_domain_mask):
        return log1x

    z = x * x
    z = -0.5 * z + x * (
        z * polynomial_evaluate[P](x) / polynomial_evaluate[Q](x)
    )

    return in_domain_mask.select(log1x, x + z)


def _narrow_from_f32[
    width: SIMDLength, //, dtype: DType
](x: SIMD[.float32, width]) -> SIMD[dtype, width]:
    """Narrows a float32 result to a 16-bit `dtype`, keeping bfloat16
    subnormals that a plain `cast` flushes on targets that flush float32
    subnormals (every bfloat16 subnormal is a float32 subnormal)."""
    comptime if dtype == .bfloat16 and CurrentPlugin.flushes_subnormals[
        .float32
    ]:
        return _narrow_f32_to_bf16_bits(x)._refine[dtype]()
    return x.cast[dtype]()


@inline(.always)
def _rem_pio2_large[
    dtype: DType, width: SIMDLength, //, quarter: Bool = False
](ax: SIMD[dtype, width]) -> Tuple[
    SIMD[.int32, width], type_of(ax), type_of(ax)
]:
    """Payne-Hanek: returns `(n mod 4, hi, lo)` with `ax = n*pi/2 + hi + lo`
    and `|hi + lo| <= pi/4`, for finite `ax >= 1`. With `quarter`, reduces
    `ax - pi/4` instead by subtracting 1/2 from the fixed-point `ax * 2/pi`,
    so the shift costs no precision.

    Uses only integer and `dtype` arithmetic, so the float32 path never
    touches float64.
    """
    comptime T = type_of(ax)
    comptime U = SIMD[.uint64, width]
    comptime I = SIMD[.int64, width]
    comptime mant = FPUtils[dtype].mantissa_width()
    comptime bias = FPUtils[dtype].exponent_bias()
    comptime P = mant + 1

    comptime assert dtype in (DType.float32, DType.float64)
    var bits = ax.to_bits().cast[.uint64]()

    # ax = m * 2^(E - bias - mant). Fraction bits of 2/pi weighing 2^-j with
    # j <= E - bias - mant - 2 only add multiples of 4 quadrants, so a 192-bit
    # window from bit E - bias - mant - 1 (table bit E - bias - mant + 62)
    # keeps the quadrant and far more fraction bits than the closest approach
    # of any float to a multiple of pi/2 needs.
    var m = (bits & U((1 << mant) - 1)) | U(1 << mant)
    var e = bits >> U(mant)
    var pos = e + U(62 - bias - mant)
    var q = pos >> 6
    var r = pos & 63
    comptime max_q = (2 * bias + 62 - bias - mant) >> 6
    var w = StaticTuple[U, 4](fill=U(0))
    comptime for k in range(max_q + 1):
        var hit = q.eq(U(k))
        comptime for j in range(4):
            comptime word = _TWO_OVER_PI_BITS[k + j]
            w[j] = hit.select(U(word), w[j])
    # Shift in two steps so that r == 0 never shifts by 64.
    var v0 = (w[0] << r) | ((w[1] >> 1) >> (63 - r))
    var v1 = (w[1] << r) | ((w[2] >> 1) >> (63 - r))
    var v2 = (w[2] << r) | ((w[3] >> 1) >> (63 - r))

    # (m * v) mod 2^192 is ax * 2/pi mod 4 with two integer bits.
    var p2 = _umul128(m, v2)
    var p1 = _umul128(m, v1)
    var r1 = p2[1] + p1[0]
    var r0 = p1[1] + m * v0 + r1.lt(p1[0]).select(U(1), U(0))
    var h = (r0 << 2) | (r1 >> 62)
    var l = (r1 << 2) | (p2[0] >> 62)
    # Rounding to the nearest quadrant leaves the signed fraction h:l in
    # [-1/2, 1/2), split into P-bit chunks that convert to `dtype` exactly.
    var n: U
    comptime if quarter:
        # frac - 1/2 is already in [-1/2, 1/2): flip the half bit and keep the
        # integer quadrant.
        n = (r0 >> 62) & 3
        h ^= U(1) << 63
    else:
        n = ((r0 >> 62) + (h >> 63)) & 3
    var c0 = bitcast[.int64, width](h) >> I(64 - P)
    var c1 = bitcast[.int64, width](
        ((h << U(P)) | (l >> U(64 - P))) >> U(64 - P)
    )
    var c2: I
    comptime if 2 * P < 64:
        c2 = bitcast[.int64, width](
            ((h << U(2 * P)) | (l >> U(64 - 2 * P))) >> U(64 - P)
        )
    else:
        c2 = bitcast[.int64, width]((l << U(2 * P - 64)) >> U(64 - P))

    @inline(.always)
    def to_float(c: I) -> T:
        # The chunks fit in 24 bits for float32; convert through int32.
        comptime if dtype == .float32:
            return c.cast[.int32]().cast[dtype]()
        return c.cast[dtype]()

    comptime s1 = 1 / Scalar[dtype](1 << P)
    var f = _two_sum(to_float(c0) * s1, to_float(c1) * (s1 * s1))
    var f_lo = f[1] + to_float(c2) * (s1 * s1 * s1)

    comptime is_f32 = dtype == .float32
    comptime pio2_hi = Scalar[dtype](
        1.5707963705062866 if is_f32 else 1.5707963267948966
    )
    comptime pio2_lo = Scalar[dtype](
        -4.371138828673793e-08 if is_f32 else 6.123233995736766e-17
    )
    # (f + f_lo) * pi/2 as hi + lo. No rounded product feeds an add: one that
    # did would be contracted into an fma by GPU backends, while its error
    # term still assumed the rounded value.
    var t = fma(f[0], pio2_lo, f_lo * pio2_hi)
    var hi = fma(f[0], pio2_hi, t)
    var lo = fma(f[0], pio2_hi, -hi) + t
    return (n.cast[.int32](), hi, lo)


@inline(.always)
def _sincos_kernel[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width], y: type_of(x)) -> Tuple[type_of(x), type_of(x)]:
    """Returns `(sin(x + y), cos(x + y))` for `|x + y| <~ pi/4` and tiny `y`.

    fdlibm k_sin.c / k_cos.c for float64; Cephes sinf/cosf polynomials with
    the same tail correction for float32.
    """
    var z = x * x
    var sin_v: type_of(x)
    var cos_r: type_of(x)
    comptime if dtype == .float64:
        var w = z * z
        var r = polynomial_evaluate[
            [
                Scalar[dtype](8.33333333332248946124e-03),
                -1.98412698298579493134e-04,
                2.75573137070700676789e-06,
            ]
        ](z) + z * w * polynomial_evaluate[
            [
                Scalar[dtype](-2.50507602534068634195e-08),
                1.58969099521155010221e-10,
            ]
        ](
            z
        )
        var v = z * x
        sin_v = x - (
            (z * (0.5 * y - v * r) - y) - v * -1.66666666666666324348e-01
        )
        cos_r = z * polynomial_evaluate[
            [
                Scalar[dtype](4.16666666666666019037e-02),
                -1.38888888888741095749e-03,
                2.48015872894767294178e-05,
            ]
        ](z) + w * w * polynomial_evaluate[
            [
                Scalar[dtype](-2.75573143513906633035e-07),
                2.08757232129817482790e-09,
                -1.13596475577881948265e-11,
            ]
        ](
            z
        )
    else:
        comptime assert dtype == .float32, "expected float32 or float64"
        var sp = polynomial_evaluate[
            [Scalar[dtype](-1.6666654611e-1), 8.3321608736e-3, -1.9515295891e-4]
        ](z)
        sin_v = x + (x * z * sp + y * (1.0 - 0.5 * z))
        cos_r = z * polynomial_evaluate[
            [
                Scalar[dtype](4.166664568298827e-2),
                -1.388731625493765e-3,
                2.443315711809948e-5,
            ]
        ](z)
    var hz = 0.5 * z
    var w1 = 1.0 - hz
    return (sin_v, w1 + (((1.0 - w1) - hz) + (z * cos_r - x * y)))


@inline(.always)
def _sincos_minus_pio4[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> Tuple[type_of(x), type_of(x)]:
    """Returns `(sin(x - pi/4), cos(x - pi/4))` for finite `x >= 2`.

    The Bessel asymptotic forms need these where `sin(x) -+ cos(x)` cancels.
    fdlibm recovers the cancelling side through a `cos(2x)` identity; here
    the pi/4 shift is folded into the argument reduction, so the reduced
    argument keeps its relative accuracy even where `x - pi/4` lies within
    2^-60 of a multiple of pi/2 (subtracting a double-word pi/4 after the
    reduction lost 62 ulp there).
    """
    var red = _rem_pio2[quarter=True](x)
    var n = red[0]
    var sc = _sincos_kernel(red[1], red[2])
    var swap = (n & 1).eq(1)
    var sin_v = swap.select(sc[1], sc[0])
    var cos_v = swap.select(-sc[0], sc[1])
    var flip = n.ge(2)
    return (flip.select(-sin_v, sin_v), flip.select(-cos_v, cos_v))


# fdlibm's pzero/qzero and pone/qone breakpoints (high words 0x40122E8B and
# 0x4006DB6D); both are exact in float32.


@inline(.always)
def _log_kernel[
    dtype: DType, width: SIMDLength, //
](f: SIMD[dtype, width]) -> type_of(f):
    """Returns `log(1 + f) - f + f*f/2` for `1 + f` in `[sqrt(2)/2, sqrt(2))`.

    FreeBSD msun k_log.h (float64) and k_logf.h / e_logf.c (float32):
    `s = f / (2 + f)` and a minimax polynomial in `s^2`.

    Uses Sun fdlibm k_log.h (float64) and FreeBSD msun e_logf.c (float32),
    also carried by historical musl logf.c: `s = f / (2 + f)` and a minimax
    polynomial in `s^2`.
    """
    var s = f / (2.0 + f)
    var z = s * s
    var w = z * z
    var r: type_of(f)
    comptime if dtype == .float64:
        r = w * polynomial_evaluate[
            [
                Scalar[dtype](3.999999999940941908e-01),
                2.222219843214978396e-01,
                1.531383769920937332e-01,
            ]
        ](w) + z * polynomial_evaluate[
            [
                Scalar[dtype](6.666666666666735130e-01),
                2.857142874366239149e-01,
                1.818357216161805012e-01,
                1.479819860511658591e-01,
            ]
        ](
            w
        )
    else:
        comptime assert dtype == .float32, "expected float32 or float64"
        r = w * polynomial_evaluate[
            [Scalar[dtype](0.40000972152), 0.24279078841]
        ](w) + z * polynomial_evaluate[
            [Scalar[dtype](0.66666662693), 0.28498786688]
        ](
            w
        )
    return s * (0.5 * f * f + r)


@inline(.always)
def _is_zero_bits[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> SIMD[
    DType.bool, width
] where dtype.is_floating_point():
    """Returns whether `x` is +-0, from its bits: Apple GPUs compare float32
    subnormals as zero."""
    comptime TInt = SIMD[_integral_type_of[dtype](), width]
    comptime magnitude = TInt((1 << (bit_width_of[dtype]() - 1)) - 1)
    return (x._to_bits_signed() & magnitude).eq(0)


@inline(.always)
def _is_subnormal_bits[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> SIMD[
    DType.bool, width
] where dtype.is_floating_point():
    """Returns whether `x` is a nonzero subnormal, from its bits."""
    comptime TInt = SIMD[_integral_type_of[dtype](), width]
    comptime mantissa_width = FPUtils[dtype].mantissa_width()
    comptime magnitude = TInt((1 << (bit_width_of[dtype]() - 1)) - 1)
    var m = x._to_bits_signed() & magnitude
    return m.ne(0) & m.lt(TInt(1 << mantissa_width))


@inline(.always)
def _frexp_bits[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> Tuple[
    type_of(x), type_of(x)
] where dtype.is_floating_point():
    """Returns `frexp(x)` for finite nonzero `x`, reading subnormal exponents
    from the bit pattern so flush-to-zero targets keep them."""
    comptime T = type_of(x)
    comptime TInt = SIMD[_integral_type_of[dtype](), width]
    comptime mantissa_width = FPUtils[dtype].mantissa_width()
    comptime exponent_bias = FPUtils[dtype].exponent_bias()
    comptime min_normal = T(from_bits=TInt(1 << mantissa_width))

    var exponent_mask = _frexp_mask1[dtype, width]()
    var half_exponent = _frexp_mask2[dtype, width]()
    var bits = x._to_bits_signed()
    var is_subnormal = (bits & exponent_mask).eq(0)
    # A subnormal is `m * 2^(emin - mantissa_width)` for an integer significand
    # `m`. Converting `m` is exact and `m * 2^emin` is normal, so the value
    # survives targets that flush subnormal operands to zero.
    var m = (bits & TInt((1 << mantissa_width) - 1)).cast[dtype]()
    var x0 = is_subnormal.select(copysign(m * min_normal, x), x)
    var bits0 = x0._to_bits_signed()
    var e = ((bits0 & exponent_mask) >> TInt(mantissa_width)) - TInt(
        exponent_bias - 1
    )
    e -= is_subnormal.select(TInt(mantissa_width), TInt(0))
    var fraction = T(from_bits=(bits0 & ~exponent_mask) | half_exponent)
    return (fraction, e.cast[dtype]())


@inline(.always)
def _log_reduce[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> Tuple[
    type_of(x), type_of(x)
] where dtype.is_floating_point():
    """Returns `(k, f)` with `x = 2^k (1 + f)`, `1 + f` in `[sqrt(2)/2, sqrt(2))`,
    for positive finite `x` (subnormals included)."""
    var fr = _frexp_bits(x)
    var below = fr[0].lt(_SQRT2 / 2)
    return (
        below.select(fr[1] - 1.0, fr[1]),
        below.select(fr[0] + fr[0], fr[0]) - 1.0,
    )


@inline(.always)
def _log_combine[
    dtype: DType, width: SIMDLength, //
](k: SIMD[dtype, width], f: type_of(k), c: type_of(k)) -> type_of(k):
    """Returns `k*ln2 + log(1 + f) + c` for small `c`; ln2 is split so that
    `k * ln2_hi` is exact."""
    comptime is_f64 = dtype == .float64
    comptime ln2_hi = Scalar[dtype](_LN2_HI_F64 if is_f64 else 6.9313812256e-01)
    comptime ln2_lo = Scalar[dtype](_LN2_LO_F64 if is_f64 else 9.0580006145e-06)
    var hfsq = 0.5 * f * f
    return k.fma(ln2_hi, -((hfsq - (_log_kernel(f) + k.fma(ln2_lo, c))) - f))


@inline(.always)
def _log_accurate[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> type_of(x) where dtype.is_floating_point():
    """Computes the natural log of positive finite `x` to within about 1 ulp."""
    var exponent, fraction = _log_reduce(x)
    return _log_combine(exponent, fraction, 0)


def _cbrtf(x: Float32) -> Float32:
    """Compute the cube root of a 32-bit floating-point number.

    This function implements an efficient algorithm for computing the cube root (∛x)
    of a single-precision floating-point value. The algorithm works by:

    1. Extracting the binary exponent `e` from the input `x` such that x = 2^e * mantissa.
    2. Normalizing the input by scaling `x` by 2^(-e) to obtain a mantissa in [0.5, 1).
    3. Dividing the exponent `e` by 3 to obtain the cube root exponent, and computing the remainder.
    4. Applying a correction factor based on the remainder to improve accuracy.
    5. Reconstructing the result by scaling the cube root of the mantissa by 2^(e/3) and the correction factor.

    Args:
        x: Input value (Float32) for which to compute the cube root.

    Returns:
        The cube root of `x` as a Float32.
    """
    # Initialize correction factor q (may be adjusted based on exponent remainder)
    var q = Float32(1.0)

    # Extract exponent e such that x = 2^e * mantissa
    # Add 1 to exponent for normalization purposes
    var e = _ilogb(abs(x)) + 1  # Get binary exponent

    # Normalize input: scale x by 2^(-e) to get mantissa in [0.5, 1)
    var d = ldexp(x, -e)

    # Compute exponent division by 3 with remainder
    # We need to split e into: e = 3*qu + re where re ∈ {0, 1, 2}
    # Add offset 6144 to ensure positive values for easier integer division
    var t = Float32(e) + 6144.0
    var qu = Int(t / 3.0)  # Quotient: e // 3
    var re = Int(t - Float32(qu) * 3.0)  # Remainder: e % 3

    comptime CBRT_2 = 1.2599210498948731647672106
    comptime CBRT_4 = 1.5874010519681994747517056

    # Apply correction factors based on remainder
    # If e % 3 == 1: need to multiply by 2^(1/3) = cbrt(2)
    if re == 1:
        q = CBRT_2
    # If e % 3 == 2: need to multiply by 2^(2/3) = cbrt(4)
    elif re == 2:
        q = CBRT_4

    # Scale q by 2^(qu - 2048) to reconstruct proper exponent
    # Subtract 2048 to compensate for the 6144 offset used earlier
    q = ldexp(q, Int32(qu - 2048))

    # Apply sign to correction factor (cube root preserves sign)
    q = copysign(q, x)

    # Work with absolute value for polynomial approximation
    d = abs(d)

    # Polynomial approximation for cbrt(d) where d ∈ [0.5, 1)
    # Using Horner's method for efficient evaluation
    var poly = polynomial_evaluate[
        [
            Scalar[x.dtype](2.2241256237030029296875),
            -3.8095417022705078125,
            5.898262500762939453125,
            -5.532182216644287109375,
            2.8208892345428466796875,
            -0.601564466953277587890625,
        ]
    ](d)

    # Newton-Raphson refinement step
    # Formula: y_new = y - (y³ - d) / (3y²)
    # Rearranged as: y_new = (2y³ + d) / (3y²)
    var y = d * poly * poly  # d * (poly ** 2), where poly ≈ cbrt(d)

    # Apply Newton iteration: y = y - (2/3) * y * (y*x - 1)
    # This refines the approximation
    y = y - (2.0 / 3.0) * y * (y * poly - 1.0)

    # Multiply by the correction factor q to get final result
    y = y * q

    # Handle special cases
    if isinf(x):
        y = copysign(Float32.MAX, x)  # cbrt(±∞) = ±∞
    if x == 0.0:
        y = copysign(Float32(0.0), x)  # cbrt(±0) = ±0

    return y


@inline(.always)
def _dd_add[
    dtype: DType, width: SIMDLength, //
](
    ah: SIMD[dtype, width], al: type_of(ah), bh: type_of(ah), bl: type_of(ah)
) -> Tuple[type_of(ah), type_of(ah)]:
    """Returns the double-word sum `(ah + al) + (bh + bl)`."""
    var s = _two_sum(ah, bh)
    return (s[0], s[1] + (al + bl))


@inline(.always)
def _dd_div[
    dtype: DType, width: SIMDLength, //
](
    nh: SIMD[dtype, width], nl: type_of(nh), dh: type_of(nh), dl: type_of(nh)
) -> Tuple[type_of(nh), type_of(nh)]:
    """Returns the double-word quotient `(nh + nl) / (dh + dl)`."""
    var q = nh / dh
    return (q, ((fma(-q, dh, nh) + nl) - q * dl) / dh)


@inline(.always)
def _dd_log[
    dtype: DType, width: SIMDLength, //
](h: SIMD[dtype, width], l: type_of(h)) -> Tuple[
    type_of(h), type_of(h)
] where dtype.is_floating_point():
    """Returns `log(h + l)` as a double-word for positive normal `h`.

    As in SLEEF logk2, `log(2^k m) = k log(2) + 2 atanh((m - 1) / (m + 1))`
    with the reduced argument and its square carried as double-words.
    """
    comptime T = type_of(h)
    comptime is_f32 = dtype == .float32
    comptime ln2_hi = Scalar[dtype](
        0.69314718246459960938 if is_f32 else 0.693147180559945286226764
    )
    comptime ln2_lo = Scalar[dtype](
        -1.904654323148236017e-09 if is_f32 else 2.319046813846299558417771e-17
    )
    var kf = _log_reduce(h)
    var m = kf[1] + 1.0
    # m / h is the exact power of two that scaled h to m.
    var ml = l * (m / h)
    var den = _two_sum(m, T(1))
    var x = _dd_div(kf[1], ml, den[0], den[1] + ml)
    var x2 = _dd_mul(x[0], x[1], x[0], x[1])
    # Taylor series 2x + 2x^3/3 + 2x^5/5 + ... for |x| <= 3 - 2 sqrt(2); the
    # 2/3 and 2/5 terms are added in double-word arithmetic.
    var t: T
    comptime if is_f32:
        t = polynomial_evaluate[
            [
                Scalar[dtype](0.2857142857142857),
                0.2222222222222222,
                0.18181818181818182,
                0.15384615384615385,
                0.13333333333333333,
                0.11764705882352941,
                0.10526315789473684,
                0.09523809523809523,
            ]
        ](x2[0])
    else:
        t = polynomial_evaluate[
            [
                Scalar[dtype](0.2857142857142857),
                0.2222222222222222,
                0.18181818181818182,
                0.15384615384615385,
                0.13333333333333333,
                0.11764705882352941,
                0.10526315789473684,
                0.09523809523809523,
                0.08695652173913043,
                0.08,
                0.07407407407407407,
                0.06896551724137931,
            ]
        ](x2[0])
    comptime two_5_hi = Scalar[dtype](0.4000000059604645 if is_f32 else 0.4)
    comptime two_5_lo = Scalar[dtype](
        -5.9604645663569045e-09 if is_f32 else -2.2204460492503132e-17
    )
    comptime two_3_hi = Scalar[dtype](
        0.6666666865348816 if is_f32 else 0.6666666666666666
    )
    comptime two_3_lo = Scalar[dtype](
        -1.9868215517249155e-08 if is_f32 else 3.700743415417188e-17
    )
    var p = _dd_mul(t, T(0), x2[0], x2[1])
    p = _dd_add(p[0], p[1], T(two_5_hi), T(two_5_lo))
    p = _dd_mul(p[0], p[1], x2[0], x2[1])
    p = _dd_add(p[0], p[1], T(two_3_hi), T(two_3_lo))
    var x3 = _dd_mul(x2[0], x2[1], x[0], x[1])
    p = _dd_mul(p[0], p[1], x3[0], x3[1])
    var s = _dd_mul(T(ln2_hi), T(ln2_lo), kf[0], T(0))
    s = _dd_add(s[0], s[1], x[0] + x[0], x[1] + x[1])
    return _dd_add(s[0], s[1], p[0], p[1])


@inline(.always)
def _dd_mul[
    dtype: DType, width: SIMDLength, //
](
    ah: SIMD[dtype, width], al: type_of(ah), bh: type_of(ah), bl: type_of(ah)
) -> Tuple[type_of(ah), type_of(ah)]:
    """Returns the double-word product `(ah + al) * (bh + bl)`."""
    # The high part is an fma that already holds the cross terms, so a
    # backend that contracts a*b + c cannot fuse it into a later add and
    # desynchronize it from the exact remainder below.
    var cross = ah * bl + al * bh
    var hi = fma(ah, bh, cross)
    return (hi, fma(ah, bh, -hi) + cross)


def _ilogb[width: SIMDLength](x: SIMD[.float32, width]) -> SIMD[.int32, width]:
    """Extract binary exponent from floating-point number.

    Args:
        x: Input floating-point value.

    Returns:
        Integer binary exponent of x.
    """

    @inline(.always)
    def extract(x: SIMD[.float32, width]) -> SIMD[.int32, width]:
        """Internal helper function to extract binary exponent from float.

        Args:
            x: Input floating-point value (assumed positive).

        Returns:
            Unbiased binary exponent as integer.
        """
        # Check if d is subnormal (very small, < 2^-64 ≈ 5.421010862427522e-20)
        var is_subnormal_mask = x.lt(5.421010862427522e-20)
        var d = is_subnormal_mask.select(
            x * 1.8446744073709552e19,  # Scale by 2^64
            x,
        )

        # Extract exponent bits from IEEE 754 representation
        # Step 1: Reinterpret float as 32-bit integer
        var bits = rebind[SIMD[.int32, width]](d._to_bits_signed())

        # Step 2: Right shift by 23 to move exponent to lower bits
        # This moves bits [30:23] to positions [7:0]
        var exponent_bits = (bits >> 23) & 0xFF  # Mask to get 8 bits

        # Step 3: Remove bias to get true exponent
        # IEEE 754 bias for single precision is 127 (0x7f)
        # If subnormal path was taken, also subtract 64 to compensate for scaling

        return is_subnormal_mask.select(
            exponent_bits - (64 + 0x7F),  # Remove bias and scaling offset
            exponent_bits - 0x7F,  # Remove bias only
        )

    comptime FP_ILOGB0 = (-2147483647 - 1)
    comptime FP_ILOGBNAN = 2147483647

    # Extract the binary exponent from |x|
    # For x = m × 2^e where m ∈ [1, 2), this returns e
    var e = extract(abs(x))

    comptime for i in range(width):
        var d = x[i]

        # Special case: ilogb(±0) returns FP_ILOGB0
        if d == 0.0:
            e[i] = FP_ILOGB0

        # Special case: ilogb(NaN) returns FP_ILOGBNAN
        if isnan(d):
            e[i] = FP_ILOGBNAN

        # Special case: ilogb(±∞) returns Int32.MAX
        if isinf(d):
            e[i] = Int32.MAX

    return e


@inline(.always)
def _lgamma_impl[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    # Port of fdlibm e_lgamma_r.c (f64) / e_lgammaf_r.c (f32): every
    # interval is evaluated and the per-lane result picked with `select`.
    comptime T = SIMD[dtype, width]
    comptime is_f32 = dtype == .float32
    comptime tiny = Scalar[dtype](
        4.76837158203125e-07 if is_f32 else 8.470329472543003e-22
    )  # 2**-21 / 2**-70
    comptime huge = Scalar[dtype](2.8823037615171174e17)  # 2**58
    comptime tc = Scalar[dtype](1.46163214496836224576)
    comptime tf = Scalar[dtype](
        -1.2148628384e-01 if is_f32 else -1.21486290535849611461e-01
    )
    comptime tt = Scalar[dtype](
        6.6971006518e-09 if is_f32 else -3.63867699703950536541e-18
    )

    # fdlibm boundaries inside (0, 2): below split_09 the expansions are for
    # lgamma(x + 1) and shifted by 1 above it.
    comptime split_09 = 0.9
    comptime near_1_bound = 0.73163
    comptime near_2_hi_bound = 1.73163
    comptime near_tc_bound = 0.23164
    comptime near_tc_hi_bound = 1.23164

    comptime TInt = SIMD[_integral_type_of[dtype](), width]
    comptime sign_mask = TInt(FPUtils[dtype].sign_mask())
    var bits = x._to_bits_signed()
    var ax_bits = bits & ~sign_mask
    var is_zero = _lgamma_is_zero_bits(bits)
    var ax = T(from_bits=ax_bits)
    var neg = bits.lt(0)
    var lt09 = ax.lt(split_09)
    var lt2 = ax.lt(2.0)
    var lt8 = ax.lt(8.0)

    # [2, 8): lgamma(2 + y) via a rational fit, then the recurrence up to ax.
    var fl = floor(ax)
    var ym = ax - fl
    var zm = T(1)
    comptime for k in range(3, 8):
        zm *= fl.ge(Scalar[dtype](k)).select(ym + Scalar[dtype](k - 1), T(1))
    var mid = (~lt2) & lt8
    var log_x = _log_accurate(mid.select(zm, ax))

    # (0, 2): three expansions around 1, the minimum tc, and 2.
    var sel0 = lt09.select(ax.ge(near_1_bound), ax.ge(near_2_hi_bound))
    var sel1 = lt09.select(ax.ge(near_tc_bound), ax.ge(near_tc_hi_bound))
    var y0 = lt09.select(1 - ax, 2 - ax)
    var y1 = lt09.select(ax - (tc - 1), ax - tc)
    var y2 = lt09.select(ax, ax - 1)

    var z0 = y0 * y0
    var p0 = y0 * polynomial_evaluate[
        [
            Scalar[dtype](7.72156649015328655494e-02),
            6.73523010531292681824e-02,
            7.38555086081402883957e-03,
            1.19270763183362067845e-03,
            2.20862790713908385557e-04,
            2.52144565451257326939e-05,
        ]
    ](z0) + z0 * polynomial_evaluate[
        [
            Scalar[dtype](3.22467033424113591611e-01),
            2.05808084325167332806e-02,
            2.89051383673415629091e-03,
            5.10069792153511336608e-04,
            1.08011567247583939954e-04,
            4.48640949618915160150e-05,
        ]
    ](
        z0
    )
    var r0 = p0 - 0.5 * y0

    var z1 = y1 * y1
    var w1 = z1 * y1
    var p11 = polynomial_evaluate[
        [
            Scalar[dtype](4.83836122723810047042e-01),
            -3.27885410759859649565e-02,
            6.10053870246291332635e-03,
            -1.40346469989232843813e-03,
            3.15632070903625950361e-04,
        ]
    ](w1)
    var p12 = polynomial_evaluate[
        [
            Scalar[dtype](-1.47587722994593911752e-01),
            1.79706750811820387126e-02,
            -3.68452016781138256760e-03,
            8.81081882437654011382e-04,
            -3.12754168375120860518e-04,
        ]
    ](w1)
    var p13 = polynomial_evaluate[
        [
            Scalar[dtype](6.46249402391333854778e-02),
            -1.03142241298341437450e-02,
            2.25964780900612472250e-03,
            -5.38595305356740546715e-04,
            3.35529192635519073543e-04,
        ]
    ](w1)
    var r1 = tf + (z1 * p11 - (tt - w1 * (p12 + y1 * p13)))

    var p21 = y2 * polynomial_evaluate[
        [
            Scalar[dtype](-7.72156649015328655494e-02),
            6.32827064025093366517e-01,
            1.45492250137234768737e00,
            9.77717527963372745603e-01,
            2.28963728064692451092e-01,
            1.33810918536787660377e-02,
        ]
    ](y2)
    var p22 = polynomial_evaluate[
        [
            Scalar[dtype](1.0),
            2.45597793713041134822e00,
            2.12848976379893395361e00,
            7.69285150456672783825e-01,
            1.04222645593369134254e-01,
            3.21709242282423911810e-03,
        ]
    ](y2)
    var r2 = p21 / p22 - 0.5 * y2

    var r_small = sel0.select(r0, sel1.select(r1, r2))
    r_small = lt09.select(r_small - log_x, r_small)

    var pm = ym * polynomial_evaluate[
        [
            Scalar[dtype](-7.72156649015328655494e-02),
            2.14982415960608852501e-01,
            3.25778796408930981787e-01,
            1.46350472652464452805e-01,
            2.66422703033638609560e-02,
            1.84028451407337715652e-03,
            3.19475326584100867617e-05,
        ]
    ](ym)
    var qm = polynomial_evaluate[
        [
            Scalar[dtype](1.0),
            1.39200533467621045958e00,
            7.21935547567138069525e-01,
            1.71933865632803078993e-01,
            1.86459191715652901344e-02,
            7.77942496381893596434e-04,
            7.32668430744625636189e-06,
        ]
    ](ym)
    var r_mid = (0.5 * ym + pm / qm) + log_x

    # [8, inf): Stirling series in 1/x.
    # log(2 pi) / 2 - 1/2
    comptime half_log_2pi_m_half = 4.18938533204672725052e-01
    var zb = 1 / ax
    var wb = zb.fma(
        polynomial_evaluate[
            [
                Scalar[dtype](8.33333333333329678849e-02),
                -2.77777777728775536470e-03,
                7.93650558643019558500e-04,
                -5.95187557450339963135e-04,
                8.36339918996282139126e-04,
                -1.63092934096575273989e-03,
            ]
        ](zb * zb),
        half_log_2pi_m_half,
    )
    var r_big = ax.lt(huge).select(
        (ax - 0.5) * (log_x - 1) + wb, ax * (log_x - 1)
    )

    var r = lt2.select(r_small, lt8.select(r_mid, r_big))

    if neg.reduce_or():
        r = neg.select(_lgamma_neg(ax), r)
        # Past this bound the absolute error of `_lgamma_neg` stays below
        # 0.25 ulp of the result.
        comptime near_zero = Scalar[dtype](2.0**-13 if is_f32 else 2.0**-14)
        var near = neg & abs(r).lt(near_zero)
        if near.reduce_or():
            r = near.select(_lgamma_neg_zero(ax, r), r)

    r = (ax.eq(1) | ax.eq(2)).select(T(0), r)
    r = ax.lt(tiny).select(-log_x, r)
    var pole = is_zero | (neg & ax.ge(1) & fl.eq(ax))
    r = pole.select(inf[dtype](), r)
    # inf -> +inf, nan -> nan.
    return (isinf(x) | isnan(x)).select(x * x, r)


@inline(.never)
def _lgamma_is_zero_bits[
    dtype: DType, width: SIMDLength, //
](bits: SIMD[dtype, width]) -> SIMD[.bool, width]:
    comptime assert dtype.is_integral(), "expected integer bit pattern"
    # Keep the bit classification opaque to floating-point canonicalization:
    # a floating zero comparison can misclassify subnormal inputs under DAZ.
    comptime sign_mask = SIMD[dtype, width](1) << SIMD[dtype, width](
        bit_width_of[dtype]() - 1
    )
    return bits.eq(0) | bits.eq(sign_mask)


@inline(.always)
def _lgamma_neg[
    dtype: DType, width: SIMDLength, //
](ax: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    """Returns `lgamma(-ax)` for finite non-integer `ax > 0`.

    SLEEF gammak / xlgamma_u1: `log(pi) - lgamma(1 + ax) - log|sin(pi ax)|`
    with every term, the logs, and the final sum in double-word arithmetic,
    because the terms cancel near the zeros of lgamma on the negative axis.
    """
    comptime T = type_of(ax)
    comptime is_f32 = dtype == .float32
    # Recurrence steps that lift 1 + ax in [2.3, shift_bound] far enough for
    # the Stirling correction to err by about 2^-70 (f64) or 2^-41 (f32);
    # SLEEF's shorter shifts leave 2^-60 or 2^-38, which dominates near the
    # zeros.
    comptime nshift = 6 if is_f32 else 10
    comptime shift_bound = Scalar[dtype](8 if is_f32 else 12)
    comptime log_pi_hi = Scalar[dtype](
        1.1447298526763916 if is_f32 else 1.1447298858494001639
    )
    comptime log_pi_lo = Scalar[dtype](
        3.317300922844879e-08 if is_f32 else 1.026595116270782638e-17
    )
    comptime half_log_2pi_hi = Scalar[dtype](
        0.9189385175704956 if is_f32 else 0.91893853320467278056
    )
    comptime half_log_2pi_lo = Scalar[dtype](
        1.563417661998301e-08 if is_f32 else -3.8782941580672414498e-17
    )
    # Coefficients, constant term first, for 1 + ax >= 2.3 (Stirling series
    # in 1/x), 1 + ax near 1, and 1 + ax near 2. The two leading ones of each
    # set are kept apart and applied in double-word arithmetic.
    comptime b0_hi = Scalar[dtype](
        0.0833333358168602 if is_f32 else 0.08333333333333333
    )  # 1/12
    comptime b0_lo = Scalar[dtype](
        -2.4835269396561444e-09 if is_f32 else 4.625929269271485e-18
    )
    comptime b1 = Scalar[dtype](0.003472222222222222222175164840)
    comptime c10 = Scalar[dtype](
        0.2705872357 if is_f32 else 0.2705808084277815939
    )
    comptime c11 = Scalar[dtype](
        -0.2072454542 if is_f32 else -0.2073855510284092762
    )
    comptime c20 = Scalar[dtype](
        0.2058077045e-1 if is_f32 else 0.2058080842778455335e-1
    )
    comptime c21 = Scalar[dtype](
        -0.7385451812e-2 if is_f32 else -0.7385551028674461858e-2
    )
    comptime CB: List[Scalar[dtype]] = [
        -0.002681327160493827160473958490,
        -0.000229472093621399176949318732,
        0.000784039221720066627493314301,
        6.97281375836585777403743539e-05,
        -0.000592166437353693882857342347,
        -5.17179090826059219329394422e-05,
        0.000839498720672087279971000786,
    ] if is_f32 else [
        -0.002681327160493827160473958490,
        -0.000229472093621399176949318732,
        0.000784039221720066627493314301,
        6.97281375836585777403743539e-05,
        -0.000592166437353693882857342347,
        -5.17179090826059219329394422e-05,
        0.000839498720672087279971000786,
        7.20489541602001055898311517e-05,
        -0.001914438498565477526465972390,
        -0.000162516262783915816896611252,
        0.006403362833808069794787256200,
        0.000540164767892604515196325186,
        -0.029527880945699120504851034100,
        -0.002481743600264997730942489280,
        0.179540117061234856098844714000,
        0.015056113040026424412918973400,
        -1.391801093265337481495562410000,
        -0.116546276599463200848033357000,
        13.39798545514258921833306020000,
        1.120804464289911606838558160000,
        -156.801412704022726379848862,
    ]
    comptime C1: List[Scalar[dtype]] = [
        0.1697509140e0,
        -0.1493408978e0,
        0.1013825238e0,
        -0.8855129778e-1,
        0.4826702476e0,
        0.8670063615e0,
        0.9435157776e0,
    ] if is_f32 else [
        0.1695571770041949811e0,
        -0.1440498967843054368e0,
        0.1255096673213020875e0,
        -0.1113342861544207724e0,
        0.1000996313575929358e0,
        -0.9094964931456242518e-1,
        0.8337492023017314957e-1,
        -0.7712481339961671511e-1,
        0.6828831828341884458e-1,
        -0.8211558669746804595e-1,
        0.8396709124579147809e-1,
        0.7270275473996180571e0,
        0.5766792106140076868e1,
        0.2512856643080930752e2,
        0.7748735764030416817e2,
        0.1728670414673559605e3,
        0.2818145867730348186e3,
        0.3287022855685790432e3,
        0.2617544025784515043e3,
        0.1281459691827820109e3,
        0.2947916772827614196e2,
    ]
    comptime C2: List[Scalar[dtype]] = [
        0.2891599433e-2,
        -0.1193488017e-2,
        0.4962242092e-3,
        -0.2355068718e-3,
        0.1528468856e-3,
        0.8160019934e-4,
        0.1102489550e-3,
    ] if is_f32 else [
        0.2890510330742210310e-2,
        -0.1192753911667886971e-2,
        0.5096695247101967622e-3,
        -0.2231547599034983196e-3,
        0.9945751236071875931e-4,
        -0.4492620183431184018e-4,
        0.2050727030376389804e-4,
        -0.9439951268304008677e-5,
        0.4373363141819725815e-5,
        -0.2036886057225966011e-5,
        0.9724785897406779555e-6,
        -0.4343931277157336040e-6,
        0.1358883821470355377e-6,
        -0.3189252471452599844e-6,
        -0.6554816306542489902e-7,
        0.4744167749884993937e-6,
        0.1288143074933901020e-5,
        0.1508349150733329167e-5,
        0.1040114641628246946e-5,
        0.4009244333008730443e-6,
        0.7074816000864609279e-7,
    ]

    # Coefficients of t^3, t^2, t about 1 and about 2.
    comptime z3_1 = Scalar[dtype](
        -0.400686534596170958447352690395 if is_f32 else -0.4006856343865314862
    )
    comptime z3_2 = Scalar[dtype](
        -0.673523028297382446749257758235e-1 if is_f32 else -0.6735230105319810201e-1
    )
    comptime z2_1 = Scalar[dtype](
        0.822466960142643054450325495997 if is_f32 else 0.8224670334241132030
    )
    comptime z2_2 = Scalar[dtype](
        0.322467033928981157743538726901 if is_f32 else 0.3224670334241132030
    )
    comptime z1_1 = Scalar[dtype](
        -0.577215665946766039837398973297 if is_f32 else -0.5772156649015328655
    )
    comptime z1_2 = Scalar[dtype](
        0.422784335087484338986941629852 if is_f32 else 0.4227843350984671345
    )
    comptime o1_bound = Scalar[dtype](1.2 if is_f32 else 1.1)
    comptime stirling_bound = 2.3

    var x = _two_sum(T(1), ax)
    var o1 = x[0].le(o1_bound)
    var o2 = x[0].ge(stirling_bound)

    var prod = x
    comptime for k in range(1, nshift):
        var f = _dd_add(x[0], x[1], T(k), T(0))
        prod = _dd_mul(f[0], f[1], prod[0], prod[1])
        prod = _two_sum(prod[0], prod[1])
    var shifted = o2 & x[0].le(shift_bound)
    var num = (shifted.select(prod[0], T(1)), shifted.select(prod[1], T(0)))
    x = _dd_add(x[0], x[1], shifted.select(T(nshift), T(0)), T(0))

    # t is 1/x for Stirling, else x - 1 or x - 2, as a double-word: near
    # the zeros u * t must be accurate well below one ulp of lgamma(1 + ax).
    var xm = _dd_add(x[0], x[1], o1.select(T(-1), T(-2)), T(0))
    xm = _two_sum(xm[0], xm[1])
    var rx = _dd_div(T(1), T(0), x[0], x[1])
    var t = o2.select(rx[0], xm[0])
    var tl = o2.select(rx[1], xm[1])
    var u = _dd_mul(_interval_poly[CB, C1, C2, C2](o2, o1, o1, t), T(0), t, tl)
    u = _dd_add(u[0], u[1], o2.select(T(b1), o1.select(T(c11), T(c21))), T(0))
    u = _dd_mul(u[0], u[1], t, tl)
    u = _dd_add(
        u[0],
        u[1],
        o2.select(T(b0_hi), o1.select(T(c10), T(c20))),
        o2.select(T(b0_lo), T(0)),
    )
    var ut = _dd_mul(u[0], u[1], t, tl)

    # Stirling: (x - 1/2) log(x) - x + log(2 pi) / 2.
    var lx = _dd_log(x[0], x[1])
    var xh = _dd_add(x[0], x[1], T(-0.5), T(0))
    var st = _dd_mul(xh[0], xh[1], lx[0], lx[1])
    st = _dd_add(st[0], st[1], -x[0], -x[1])
    st = _dd_add(st[0], st[1], T(half_log_2pi_hi), T(half_log_2pi_lo))

    # Taylor series of lgamma about 1 or 2, with the tail ut.
    var z = _dd_add(ut[0], ut[1], o1.select(T(z3_1), T(z3_2)), T(0))
    z = _dd_mul(z[0], z[1], t, tl)
    z = _dd_add(z[0], z[1], o1.select(T(z2_1), T(z2_2)), T(0))
    z = _dd_mul(z[0], z[1], t, tl)
    z = _dd_add(z[0], z[1], o1.select(T(z1_1), T(z1_2)), T(0))
    z = _dd_mul(z[0], z[1], t, tl)

    # lgamma(1 + ax) is st + log(1 + ut) - log(prod) or z; fold it with
    # log(pi) and log|sin(pi ax)| into one log of a quotient.
    var c = _dd_add(
        T(log_pi_hi),
        T(log_pi_lo),
        -o2.select(st[0], z[0]),
        -o2.select(st[1], z[1]),
    )
    var corr = _dd_add(ut[0], ut[1], T(1), T(0))
    var sp = _lgamma_sinpi(ax)
    var den = _dd_mul(
        o2.select(corr[0], T(1)), o2.select(corr[1], T(0)), sp[0], sp[1]
    )
    var q = _dd_div(num[0], num[1], den[0], den[1])
    var lq = _dd_log(q[0], q[1])
    var r = _dd_add(c[0], c[1], lq[0], lq[1])
    return r[0] + r[1]


@inline(.always)
def _lgamma_neg_zero[
    dtype: DType, width: SIMDLength, //
](ax: SIMD[dtype, width], r: type_of(ax)) -> type_of(
    ax
) where dtype.is_floating_point():
    """Returns `lgamma(-ax)` for lanes next to a zero of `lgamma` on the
    negative axis, and `r` for the other lanes.

    `_lgamma_neg` has an absolute error near 2^-70 (f64) or 2^-39 (f32),
    which is large relative to the result next to a zero. There
    `lgamma(-(a + t))` is its Taylor series in `t` about the zero `-a`, with
    `a` split into three words so that `t` keeps full relative precision
    however close `ax` is to `a`. Rows are the zeros on `(-n - 1, -n)` for
    `n` = 2..6 (f32) or 2..13 (f64), larger root first; past those every
    float is far enough from the zeros for `_lgamma_neg`. The coefficients
    are `(-1)^j polygamma(j - 1, -a) / j!` and the window of each row covers
    `|lgamma| <= 2^-12` with a truncation error below 2^-56 (f64) or 2^-33
    (f32) relative.
    """
    comptime T = type_of(ax)
    comptime is_f32 = dtype == .float32
    comptime nq = 2 if is_f32 else 4
    comptime A_HI: List[Scalar[dtype]] = [
        2.747682571411133,
        2.4570248126983643,
        3.955294370651245,
        3.143580913543701,
        4.991544723510742,
        4.039361953735352,
        5.998607635498047,
        5.008218288421631,
        6.9998016357421875,
        6.00138521194458,
    ] if is_f32 else [
        2.7476826467274127,
        2.4570247382208006,
        3.955294284858598,
        3.14358088834998,
        4.991544640560048,
        4.039361839740537,
        5.998607480080875,
        5.0082181683225935,
        6.999801507890638,
        6.001385294453155,
        7.999975197095821,
        7.000198333407325,
        8.999997244250977,
        8.000024800270682,
        9.99999972442663,
        9.000002755714823,
        10.99999997494789,
        10.000000275573013,
        11.999999997912324,
        11.000000025052106,
        12.99999999983941,
        12.000000002087676,
        13.99999999998853,
        13.00000000016059,
    ]
    comptime A_LO: List[Scalar[dtype]] = [
        7.531627943535568e-08,
        -7.447756189549182e-08,
        -8.579264942909504e-08,
        -2.519372088727323e-08,
        -8.295069164887536e-08,
        -1.1399481536500389e-07,
        -1.5541716891220858e-07,
        -1.2009903116450005e-07,
        -1.2785154979155777e-07,
        8.250857774783071e-08,
    ] if is_f32 else [
        -9.055340329338315e-17,
        3.7075610815513266e-17,
        1.999428391746348e-17,
        2.1818179852331714e-16,
        -1.5174411760571722e-16,
        -2.1143995503980602e-16,
        3.311862478893795e-16,
        4.3926353491015815e-17,
        -1.0550130037400023e-17,
        -6.415847287933042e-17,
        5.261737128572354e-17,
        -2.504354173632409e-16,
        2.2185620509727132e-16,
        4.354586297860107e-16,
        -4.883037618642443e-16,
        9.491348611623208e-17,
        -1.9843998306985407e-16,
        3.4909708332642057e-16,
        1.0020693920103036e-16,
        6.850849812286175e-16,
        -6.747262033096337e-16,
        -1.2222548112048185e-16,
        -8.094860741926607e-16,
        6.745919484964342e-16,
    ]
    comptime A_LO2: List[Scalar[dtype]] = [
        3.5353580266508195e-16,
        -1.739281254386864e-15,
        2.240440261970251e-15,
        -2.2590741927164574e-16,
        -2.8162793057768523e-15,
        6.76738455516481e-16,
        -2.33334897661397e-15,
        -6.173322656289216e-15,
        -1.0550130366200395e-17,
        -2.7286937696901527e-15,
    ] if is_f32 else [
        -3.322761057167369e-33,
        1.3622663121726005e-33,
        -6.2357435447617e-34,
        1.1246581285745781e-32,
        9.643515906617392e-34,
        3.5961421111626576e-33,
        -3.4720224807210337e-33,
        2.68183947324466e-33,
        4.08696427365735e-34,
        1.116080967205309e-33,
        2.0441803623138533e-33,
        -2.413795840298293e-32,
        -7.336677520259467e-33,
        -2.3599860861934562e-32,
        -3.548028340923709e-32,
        5.762352109706189e-33,
        4.778979059252407e-33,
        1.2687206116063323e-32,
        -6.563612372549864e-34,
        2.753413969507158e-33,
        -3.2387758664429733e-32,
        -2.4017170001173477e-33,
        -9.034244883215544e-33,
        -4.8554922539526397e-32,
    ]
    comptime D1_HI: List[Scalar[dtype]] = [
        1.914350152015686,
        -1.5156034231185913,
        20.725061416625977,
        -7.781884670257568,
        116.53578186035156,
        -26.790481567382812,
        716.2454223632812,
        -123.3621826171875,
        5035.96728515625,
        -723.7366333007812,
    ] if is_f32 else [
        1.9143501856115988,
        -1.5156034480216574,
        20.725060845803707,
        -7.781884658131351,
        116.53578161624363,
        -26.790480886140593,
        716.2454304275473,
        -123.3621845633534,
        5035.967373768125,
        -723.7366299252801,
        40315.71854218779,
        -5044.029941110829,
        362875.4964746711,
        -40324.281108124356,
        3628795.296492739,
        -362884.5034850277,
        39916795.114676446,
        -3628804.7035030955,
        479001594.94800997,
        -39916804.88532317,
        6227020794.794164,
        -479001605.05199003,
        87178291194.6513,
        -6227020805.205836,
    ]
    comptime D1_LO: List[Scalar[dtype]] = [
        3.3595913606632166e-08,
        -2.4903066275783203e-08,
        -5.708222943212604e-07,
        1.2126217363572778e-08,
        -2.4410792320850305e-07,
        6.812422270741081e-07,
        8.064265784923919e-06,
        -1.9461658666841686e-06,
        8.861187234288082e-05,
        3.3755011372704757e-06,
    ] if is_f32 else [
        6.288473508186805e-17,
        4.0695290379659363e-17,
        -1.4319348367658305e-15,
        1.2366266971852707e-16,
        5.71604654736871e-15,
        -7.29303362861673e-16,
        2.978324292002825e-14,
        1.7685712092825869e-15,
        3.431412146498085e-13,
        -1.1249184125179834e-14,
        1.3528231836249092e-12,
        -2.1055091809068177e-13,
        -1.3486075072383962e-11,
        2.389812710483955e-12,
        1.2671744970551317e-10,
        -5.4195819812064025e-12,
        -7.737126660349542e-10,
        3.2764123120909294e-11,
        -1.1134489574140605e-08,
        1.9950818162239775e-09,
        1.1429843684124307e-07,
        2.2289997782624084e-08,
        5.242382227630101e-07,
        1.1693576000219276e-07,
    ]
    # d2..d(nq + 1) of each row, row after row.
    comptime DQ: List[Scalar[dtype]] = [
        9.575189590454102,
        20.095134735107422,
        4.858321189880371,
        -1.4112911224365234,
        251.7146759033203,
        3730.604736328125,
        25.83133888244629,
        -112.26898956298828,
        6995.23583984375,
        551420.0,
        324.25531005859375,
        -5465.69189453125,
        257851.59375,
        123445280.0,
        7404.76220703125,
        -600556.1875,
        12690641.0,
        42623488000.0,
        260548.40625,
        -125386984.0,
    ] if is_f32 else [
        9.575189475709667,
        20.095134916842603,
        62.627282713513715,
        194.76615530344623,
        4.8583209516339965,
        -1.41129114307798,
        8.721782583815346,
        -5.800414566599873,
        251.7146825868894,
        3730.6047156806126,
        62588.08191876606,
        1119991.501665522,
        25.831338372387957,
        -112.268986297176,
        588.8907422380014,
        -3277.1937854953417,
        6995.235954894064,
        551419.9978512884,
        48911582.709059045,
        4627747171.739249,
        324.25532293784715,
        -5465.691820777134,
        104145.55520145541,
        -2116669.417577535,
        257851.58963956262,
        123445282.15783688,
        66486633581.99334,
        38196442388133.234,
        7404.762432228682,
        -600556.168452811,
        54807486.35665586,
        -5335250743.384385,
        12690641.16604718,
        42623489764.75812,
        161052333144473.78,
        6.491032158885829e17,
        260548.41030309396,
        -125386984.94090366,
        67884656938.77096,
        -39203019565186.83,
        812764889.3354839,
        21845960238437.15,
        6.605867627581421e17,
        2.1306755305163004e22,
        12710958.833951395,
        -42725890801.879196,
        161568434353282.84,
        -6.517043660273723e17,
        65840130084.02046,
        1.592794543191949e16,
        4.3349227292710556e21,
        1.258437599020075e27,
        812937510.664516,
        -21852920330413.64,
        6.608673936664925e17,
        -2.1318070343392603e22,
        6584086185960.205,
        1.5928210978304629e19,
        4.335019090413096e25,
        1.258472566431477e32,
        65841764315.97954,
        -1.5928538462012788e16,
        4.33513792803131e21,
        -1.2585156903461966e27,
        796675363616762.2,
        2.120048613893936e22,
        6.346916349938978e29,
        2.026788600430993e37,
        6584103254039.795,
        -1.5928272914951848e19,
        4.335041565983639e25,
        -1.2584807223665154e32,
        1.1472126519132435e17,
        3.6634446193922216e25,
        1.3160968687098167e34,
        5.043300020340304e42,
        796675558623237.8,
        -2.1200493922973838e22,
        6.346919457076453e29,
        -2.0267898407008447e37,
        1.9387894005607895e19,
        8.048587946043371e28,
        3.758904339726865e38,
        1.872542039912427e48,
        1.1472126761123565e17,
        -3.66344473530636e25,
        1.3160969242328746e34,
        -5.0433002862966394e42,
        3.8000272280428536e21,
        2.208532534960574e32,
        1.4440206933867054e43,
        1.0070980520242424e54,
        1.9387894038024745e19,
        -8.048587966229412e28,
        3.758904352296755e38,
        -1.872542047739723e48,
    ]

    var res = r
    comptime for j in range(len(A_HI)):
        comptime a_hi = A_HI[j]
        comptime a_lo = A_LO[j]
        comptime a_lo2 = A_LO2[j]
        comptime d1h = D1_HI[j]
        comptime d1l = D1_LO[j]
        # Four times the window: rows never overlap, the closest pair of
        # zeros (around -8) being 5e-5 apart.
        comptime tol = Scalar[dtype](2.0**-10) / abs(d1h)
        # Exact by Sterbenz's lemma inside the window.
        var t1 = ax - a_hi
        var s = _two_sum(t1, T(-a_lo))
        var th = s[0]
        var tl = s[1] - a_lo2
        comptime dn = DQ[j * nq + nq - 1]
        var q = T(dn)
        comptime for i in range(1, nq):
            comptime di = DQ[j * nq + nq - 1 - i]
            q = q.fma(th, T(di))
        # One fma rounds d1 * t together with the small terms.
        var v = fma(th, T(d1h), th * d1l + tl * d1h + th * th * q)
        res = abs(t1).lt(tol).select(v, res)
    return res


@inline(.always)
def _lgamma_sinpi[
    dtype: DType, width: SIMDLength, //
](ax: SIMD[dtype, width]) -> Tuple[
    type_of(ax), type_of(ax)
] where dtype.is_floating_point():
    """Returns `|sin(pi * ax)|` as a double-word for `ax >= 0`.

    The reduction to `w` in `[0, 1/2]` is exact. The Taylor series of
    `sin(pi/4 * t)` or `cos(pi/4 * t)`, `t` in `[0, 1]`, is summed with its
    four leading coefficients in double-word arithmetic.
    """
    comptime T = type_of(ax)
    comptime is_f32 = dtype == .float32
    # (pi/4)^(2k+1) / (2k+1)! and (pi/4)^(2k+2) / (2k+2)! with alternating
    # signs, k = 0..3, split into high and low words.
    comptime S_HI: List[Scalar[dtype]] = [
        0.7853981852531433,
        -0.08074551075696945,
        0.002490394515916705,
        -3.657620254671201e-05,
    ] if is_f32 else [
        0.7853981633974483,
        -0.08074551218828079,
        0.0024903945701927202,
        -3.657620418217725e-05,
    ]
    comptime S_LO: List[Scalar[dtype]] = [
        -2.1855694143368964e-08,
        -1.4313112917108128e-09,
        5.427601423857453e-11,
        -1.6354652007666282e-12,
    ] if is_f32 else [
        3.061616997868383e-17,
        3.541503566529601e-18,
        -7.745123384107965e-20,
        1.7364661332386583e-21,
    ]
    comptime C_HI: List[Scalar[dtype]] = [
        -0.3084251284599304,
        0.01585434377193451,
        -0.0003259918885305524,
        3.5908603877032874e-06,
    ] if is_f32 else [
        -0.30842513753404244,
        0.015854344243815502,
        -0.00032599188692739,
        3.59086044859151e-06,
    ]
    comptime C_LO: List[Scalar[dtype]] = [
        -9.074112128359957e-09,
        4.718809787362943e-10,
        1.6031623728193778e-12,
        6.088822480064537e-14,
    ] if is_f32 else [
        -1.9579048464811598e-17,
        -1.0391406571991658e-18,
        7.76739206351315e-22,
        -1.9200234339484294e-23,
    ]
    comptime S_TAIL: List[Scalar[dtype]] = [
        3.1336168903781217e-07,
        -1.757247673443401e-09,
        6.948453273886629e-12,
        -2.0410263396641442e-14,
        4.628704628834683e-17,
    ] if is_f32 else [
        3.1336168903781217e-07,
        -1.757247673443401e-09,
        6.948453273886629e-12,
        -2.0410263396641442e-14,
        4.628704628834683e-17,
        -8.348589834811673e-20,
        1.2261499847700472e-22,
    ]
    comptime C_TAIL: List[Scalar[dtype]] = [
        -2.4611369504942e-08,
        1.1501159127974052e-10,
        -3.8980731712596753e-13,
        1.001886461636272e-15,
        -2.019653396886682e-18,
    ] if is_f32 else [
        -2.4611369504942e-08,
        1.1501159127974052e-10,
        -3.8980731712596753e-13,
        1.001886461636272e-15,
        -2.019653396886682e-18,
        3.2784835616098472e-21,
        -4.377345209491838e-24,
    ]

    var y = ax - floor(ax)
    var w = min(y, 1 - y)
    var use_cos = w.gt(0.25)
    var t = use_cos.select(2 - 4 * w, 4 * w)
    var s = _dd_mul(t, T(0), t, T(0))
    var v = (
        _interval_poly[C_TAIL, S_TAIL, S_TAIL, S_TAIL](
            use_cos, use_cos, use_cos, s[0]
        ),
        T(0),
    )
    comptime for j in range(4):
        comptime ch = C_HI[3 - j]
        comptime cl = C_LO[3 - j]
        comptime sh = S_HI[3 - j]
        comptime sl = S_LO[3 - j]
        v = _dd_mul(v[0], v[1], s[0], s[1])
        v = _dd_add(
            v[0],
            v[1],
            use_cos.select(T(ch), T(sh)),
            use_cos.select(T(cl), T(sl)),
        )
    v = _dd_mul(v[0], v[1], use_cos.select(s[0], t), use_cos.select(s[1], T(0)))
    var vc = _dd_add(T(1), T(0), v[0], v[1])
    return (use_cos.select(vc[0], v[0]), use_cos.select(vc[1], v[1]))


def _log1p_accurate[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> type_of(x) where dtype.is_floating_point():
    """`log(1 + x)` for finite `x > -1` to within about 1 ulp (musl log1p).

    Outside `1 + x` in `[sqrt(2)/2, sqrt(2))` it reduces `u = fl(1 + x)` and
    adds back the rounding error of `1 + x` relative to `u`.
    """
    comptime T = type_of(x)
    var u = 1.0 + x
    var kf = _log_reduce(u)
    var c = kf[0].ge(2.0).select(1.0 - (u - x), x - (u - 1.0)) / u
    var near = x.gt(_SQRT2 / 2 - 1) & x.lt(_SQRT2 - 1)
    return _log_combine(
        near.select(T(0), kf[0]),
        near.select(x, kf[1]),
        near.select(T(0), c),
    )


@inline(.always)
def _erf_mid_ratio_f64[
    width: SIMDLength, //
](s: SIMD[.float64, width]) -> Tuple[type_of(s), type_of(s)]:
    """Returns `(p, q)` with `erf(1 + s) - erx = p / q` for
    `0.84375 <= 1 + s < 1.25`."""
    var p = polynomial_evaluate[
        [
            Float64(-2.36211856075265944077e-03),
            4.14856118683748331666e-01,
            -3.72207876035701323847e-01,
            3.18346619901161753674e-01,
            -1.10894694282396677476e-01,
            3.54783043256182359371e-02,
            -2.16637559486879084300e-03,
        ]
    ](s)
    var q = polynomial_evaluate[
        [
            Float64(1.0),
            1.06420880400844228286e-01,
            5.40397917702171048937e-01,
            7.18286544141962662868e-02,
            1.26171219808761642112e-01,
            1.36370839120290507362e-02,
            1.19844998467991074170e-02,
        ]
    ](s)
    return (p, q)


def _erf_f64[width: SIMDLength, //](x: SIMD[.float64, width]) -> type_of(x):
    # FreeBSD msun s_erf.c. For 1.25 <= |x| < 6 it computes 1 - erfc(|x|)
    # from the same exp(-x*x - 0.5625 + R/S) / |x| tail as erfc.
    comptime erx = 8.45062911510467529297e-01
    var ax = abs(x)
    var small = x.fma(_erf_small_ratio_f64(x * x), x)
    var pq = _erf_mid_ratio_f64(ax - 1.0)
    var r = ax.lt(1.25).select(erx + pq[0] / pq[1], 1.0 - _erfc_f64(ax))
    return ax.lt(0.84375).select(small, copysign(r, x))


def _erfc_f64[width: SIMDLength, //](x: SIMD[.float64, width]) -> type_of(x):
    var ax = abs(x)
    var neg = x.lt(0)

    # |x| < 0.84375
    var z = x * x
    var y = _erf_small_ratio_f64(z)
    # Above 0.25, 0.5 - (x*y + (x - 0.5)) as fdlibm, but keeping the rounding
    # errors of the inner sum s and of 0.5 - s: near 0.84375 erfc is under
    # 0.25 while s is over it, so half an ulp of s is a whole ulp of erfc.
    # d - s is exact there (Sterbenz).
    var d = x - 0.5
    var s_mid = x.fma(y, d)
    var e_mid = x.fma(y, d - s_mid)
    var h = _two_sum(type_of(x)(0.5), -s_mid)
    var small = x.lt(0.25).select(1.0 - (x + x * y), h[0] + (h[1] - e_mid))

    # 0.84375 <= |x| < 1.25
    comptime erx = 8.45062911510467529297e-01
    var pn_qd = _erf_mid_ratio_f64(ax - 1.0)
    var pn = pn_qd[0]
    var qd = pn_qd[1]
    # erfc is down to 0.07 here while pq reaches 0.08, so keep the rounding
    # errors of pq and of (1 - erx) - pq.
    var pq = pn / qd
    var pq_lo = qd.fma(-pq, pn) / qd
    var m = _two_sum(type_of(x)(1.0 - erx), -pq)
    var mid = neg.select(1.0 + (erx + pq), m[0] + (m[1] - pq_lo))

    # 1.25 <= |x| < 28: erfc(x) = exp(-x*x - 0.5625 + R/S) / x, with x*x
    # split around z (x with the low 32 bits cleared, so z*z is exact).
    var w = 1.0 / (ax * ax)
    var ra = polynomial_evaluate[
        [
            Float64(-9.86494403484714822705e-03),
            -6.93858572707181764372e-01,
            -1.05586262253232909814e01,
            -6.23753324503260060396e01,
            -1.62396669462573470355e02,
            -1.84605092906711035994e02,
            -8.12874355063065934246e01,
            -9.81432934416914548592e00,
        ]
    ](w) / polynomial_evaluate[
        [
            Float64(1.0),
            1.96512716674392571292e01,
            1.37657754143519042600e02,
            4.34565877475229228821e02,
            6.45387271733267880336e02,
            4.29008140027567833386e02,
            1.08635005541779435134e02,
            6.57024977031928170135e00,
            -6.04244152148580987438e-02,
        ]
    ](
        w
    )
    var rb = polynomial_evaluate[
        [
            Float64(-9.86494292470009928597e-03),
            -7.99283237680523006574e-01,
            -1.77579549177547519889e01,
            -1.60636384855821916062e02,
            -6.37566443368389627722e02,
            -1.02509513161107724954e03,
            -4.83519191608651397019e02,
        ]
    ](w) / polynomial_evaluate[
        [
            Float64(1.0),
            3.03380607434824582924e01,
            3.25792512996573918826e02,
            1.53672958608443695994e03,
            3.19985821950859553908e03,
            2.55305040643316442583e03,
            4.74528541206955367215e02,
            -2.24409524465858183362e01,
        ]
    ](
        w
    )
    # 0x4006DB6D00000000, about 1/0.35.
    var split = SIMD[.float64, width](from_bits=UInt64(0x4006DB6D00000000))
    var zh = type_of(x)(from_bits=ax.to_bits[.uint64]() & 0xFFFFFFFF00000000)
    # As `_erfc_exp`, but the exponent and the exp result are carried in
    # double-word up to the division by x: one rounding in p and another in
    # p / x can add to 1.6 ulp when p / x falls just below a power of 2.
    var zz = zh * zh
    var a = _two_sum(ax.lt(split).select(ra, rb), (zh - ax) * (zh + ax))
    var k = floor((a[0] - zz - 0.5625).clamp(-1200.0, 1200.0) * log2e + 0.5)
    var b = _two_sum(-zz - k * _LN2_HI_F64, type_of(x)(-0.5625))
    var c = _two_sum(b[0], a[0])
    var e_hi = c[0]
    var e_lo = k * _LN2_LO_F64 - ((b[1] + c[1]) + a[1])
    # fdlibm e_exp.c kernel: exp(e_hi - e_lo) = 1 - q.
    var r = e_hi - e_lo
    var t = r * r
    var rc = r - t * polynomial_evaluate[
        [
            Float64(1.66666666666666019037e-01),
            -2.77777777770155933842e-03,
            6.61375632143793436117e-05,
            -1.65339022054652515390e-06,
            4.13813679705723846039e-08,
        ]
    ](t)
    var p = _two_sum(type_of(x)(1.0), (e_hi - (e_lo - (r * rc) / (2.0 - rc))))
    var quot = p[0] / ax
    quot += (ax.fma(-quot, p[0]) + p[1]) / ax
    var tail = _scale_pow2(quot, k)
    var big = neg.select(2.0 - tail, tail)

    var result = ax.lt(0.84375).select(
        small,
        ax.lt(1.25).select(
            mid,
            ax.lt(28.0).select(big, neg.select(type_of(x)(2.0), 0.0)),
        ),
    )
    # erfc(x) rounds to 2 for x <= -6.
    result = x.le(-6.0).select(type_of(x)(2.0), result)
    return isnan(x).select(x, result)


def _erfc_f32[width: SIMDLength, //](x: SIMD[.float32, width]) -> type_of(x):
    var ax = abs(x)
    var neg = x.lt(0)

    # |x| < 0.84375
    var z = x * x
    var y = polynomial_evaluate[
        [Float32(1.28379166e-01), -3.36030394e-01, -1.86261395e-03]
    ](z) / polynomial_evaluate[
        [Float32(1.0), 3.12324315e-01, 2.16070414e-02, -1.98859372e-03]
    ](
        z
    )
    var small = x.lt(0.25).select(1.0 - (x + x * y), 0.5 - (x * y + (x - 0.5)))

    # 0.84375 <= |x| < 1.25
    comptime erx = Float32(8.42697144e-01)
    var s = ax - 1.0
    var pq = polynomial_evaluate[
        [
            Float32(3.65041046e-06),
            4.15109307e-01,
            -2.09395722e-01,
            8.67677554e-02,
        ]
    ](s) / polynomial_evaluate[
        [Float32(1.0), 4.95560974e-01, 3.71248513e-01, 3.92478965e-02]
    ](
        s
    )
    var mid = neg.select(1.0 + (erx + pq), (1.0 - erx) - pq)

    # 1.25 <= |x| < 11, with z = x with the low 13 bits cleared.
    var w = 1.0 / (ax * ax)
    var ra = polynomial_evaluate[
        [
            Float32(-9.88156721e-03),
            -5.43658376e-01,
            -1.66828310e00,
            -6.91554189e-01,
        ]
    ](w) / polynomial_evaluate[
        [Float32(1.0), 4.48581553e00, 4.10799170e00, 5.53855181e-01]
    ](
        w
    )
    var rb = polynomial_evaluate[
        [Float32(-9.86496918e-03), -5.48049808e-01, -1.84115684e00]
    ](w) / polynomial_evaluate[
        [Float32(1.0), 4.87132740e00, 3.04982710e00, -7.61900663e-01]
    ](
        w
    )
    # 0x4036db8c, about 1/0.35.
    var split = SIMD[.float32, width](from_bits=UInt32(0x4036DB8C))
    var zh = type_of(x)(from_bits=ax.to_bits[.uint32]() & 0xFFFFE000)
    var e = _erfc_exp(
        zh * zh, (zh - ax) * (zh + ax) + ax.lt(split).select(ra, rb)
    )
    var tail = _scale_pow2(e[0] / ax, e[1])
    var big = neg.select(2.0 - tail, tail)

    var result = ax.lt(0.84375).select(
        small,
        ax.lt(1.25).select(
            mid,
            ax.lt(11.0).select(big, neg.select(type_of(x)(2.0), 0.0)),
        ),
    )
    # erfc(x) rounds to 2 for x <= -5.
    result = x.le(-5.0).select(type_of(x)(2.0), result)
    return isnan(x).select(x, result)


@inline(.always)
def _atanh_impl[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> type_of(x) where dtype.is_floating_point():
    """Returns the inverse hyperbolic tangent of `x`.

    FreeBSD msun `e_atanh.c` / `e_atanhf.c` over `_log1p_accurate`. Returns
    NaN for NaN and `|x| > 1`, `+-inf` for `x == +-1`, and `x` itself for
    `|x| < 2^-28` (preserving the sign of zero). Measured error is within
    2 ulp (`float32`) and 3 ulp (`float64`) of the system libm.
    """
    var t = abs(x)
    var one_minus_t = 1.0 - t
    var two_t = t + t
    var arg = t.lt(0.5).select(
        two_t + two_t * t / one_minus_t, two_t / one_minus_t
    )
    var res = 0.5 * _log1p_accurate(arg)
    res = t.eq(1.0).select(inf[dtype](), t.gt(1.0).select(nan[dtype](), res))
    res = copysign(res, x)
    # Below 2^-28, atanh(x) rounds to x.
    comptime tiny = Scalar[dtype](3.725290298461914e-09)
    return (isnan(x) | t.lt(tiny)).select(x, res)


@inline(.always)
def _log1p_dw[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width], xl: type_of(x)) -> type_of(
    x
) where dtype.is_floating_point():
    """`log(1 + x + xl)` for finite `x > -1` and `|xl| <= ulp(x)`.

    Uses the fdlibm log1p rounded-sum correction with `xl` folded into the
    correction term, so the low word costs no extra rounding.
    """
    comptime T = type_of(x)
    var u = 1.0 + x
    var kf = _log_reduce(u)
    var c = (kf[0].ge(2.0).select(1.0 - (u - x), x - (u - 1.0)) + xl) / u
    var near = x.gt(_SQRT2 / 2 - 1) & x.lt(_SQRT2 - 1)
    return _log_combine(
        near.select(T(0), kf[0]),
        near.select(x, kf[1]),
        near.select(xl / u, c),
    )


# Above these, acosh(x) and asinh(x) round to log(x) + ln2.
comptime _ACOSH_ASINH_HUGE_F64 = 268435456.0  # 2^28
comptime _ACOSH_ASINH_HUGE_F32 = 4096.0  # 2^12


@inline(.always)
def _acosh_impl[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> type_of(x) where dtype.is_floating_point():
    # fdlibm e_acosh.c / musl acoshf.c, with the near-one argument
    # t + sqrt(t^2 + 2t) carried in double-word.
    comptime huge = Scalar[dtype](
        _ACOSH_ASINH_HUGE_F64 if dtype == .float64 else _ACOSH_ASINH_HUGE_F32
    )
    comptime T = type_of(x)
    var t = x - 1.0
    # v + v_lo = t^2 + 2t; 2t - v is exact (Sterbenz) for t < 2.
    var v = t.fma(t, t + t)
    var v_lo = t.fma(t, (t + t) - v)
    var s = _sqrt_ieee(v)
    var s_lo = v.gt(0).select(((-s).fma(s, v) + v_lo) / (s + s), T(0))
    var a = _two_sum(t, s)
    var r_near_one = _log1p_dw(a[0], a[1] + s_lo)
    var mid_arg = 2.0 * x - 1.0 / (x + _sqrt_ieee(x.fma(x, -1.0)))
    # One log serves both ranges; above `huge` it is log(x) + ln2.
    var is_huge = x.ge(huge)
    var r_far = _log_accurate(is_huge.select(x, mid_arg)) + is_huge.select(
        T(_LN2), T(0)
    )
    var r = x.lt(2.0).select(r_near_one, r_far)
    r = x.lt(inf[dtype]()).select(r, x)
    return x.ge(1.0).select(r, nan[dtype]())


@inline(.always)
def _asinh_impl[
    dtype: DType, width: SIMDLength, //
](x: SIMD[dtype, width]) -> type_of(x) where dtype.is_floating_point():
    # fdlibm s_asinh.c / musl asinhf.c.
    # Below these, asinh(x) rounds to x.
    comptime tiny_f64 = 3.725290298461914e-09  # 2^-28
    comptime tiny_f32 = 2.44140625e-04  # 2^-12
    comptime tiny = Scalar[dtype](tiny_f64 if dtype == .float64 else tiny_f32)
    comptime huge = Scalar[dtype](
        _ACOSH_ASINH_HUGE_F64 if dtype == .float64 else _ACOSH_ASINH_HUGE_F32
    )
    comptime T = type_of(x)
    var a = abs(x)
    var a2 = a * a
    var arg = _two_sum(a, a2 / (1.0 + _sqrt_ieee(1.0 + a2)))
    var r_small = _log1p_dw(arg[0], arg[1])
    var mid_arg = 2.0 * a + 1.0 / (_sqrt_ieee(a2 + 1.0) + a)
    var is_huge = a.ge(huge)
    var r_far = _log_accurate(is_huge.select(a, mid_arg)) + is_huge.select(
        T(_LN2), T(0)
    )
    var r = a.lt(2.0).select(r_small, r_far)
    r = x.lt(0.0).select(-r, r)
    # Tiny, infinite, and NaN inputs return x itself.
    var regular = a.ge(tiny) & a.lt(inf[dtype]())
    return regular.select(r, x)
