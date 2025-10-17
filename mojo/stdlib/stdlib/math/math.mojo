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
"""Defines math utilities.

You can import these APIs from the `math` package. For example:

```mojo
from math import floor
```
"""

from sys import (
    CompilationTarget,
    bit_width_of,
    is_amd_gpu,
    is_apple_gpu,
    is_compile_time,
    is_gpu,
    is_nvidia_gpu,
    llvm_intrinsic,
    simd_width_of,
    size_of,
)
from sys._assembly import inlined_assembly
from sys.ffi import _external_call_const
from sys.info import _is_sm_9x_or_newer, is_32bit

from algorithm import vectorize
from bit import count_trailing_zeros
from builtin.dtype import _integral_type_of
from builtin.simd import _modf, _simd_apply
from memory import Span

from utils.numerics import FPUtils, isnan, nan
from utils.static_tuple import StaticTuple

from .constants import log2e
from .polynomial import polynomial_evaluate

# ===----------------------------------------------------------------------=== #
# floor
# ===----------------------------------------------------------------------=== #


@always_inline
fn floor[T: Floorable, //](value: T) -> T:
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


@always_inline
fn ceil[T: Ceilable, //](value: T) -> T:
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


@always_inline
fn ceildiv[T: CeilDivable, //](numerator: T, denominator: T) -> T:
    """Return the rounded-up result of dividing numerator by denominator.

    Parameters:
        T: A type that support floor division.

    Args:
        numerator: The numerator.
        denominator: The denominator.

    Returns:
        The ceiling of dividing numerator by denominator.
    """
    # return -(numerator // -denominator)
    return numerator.__ceildiv__(denominator)


@always_inline
fn ceildiv[T: CeilDivableRaising, //](numerator: T, denominator: T) raises -> T:
    """Return the rounded-up result of dividing numerator by denominator, potentially raising.

    Parameters:
        T: A type that support floor division.

    Args:
        numerator: The numerator.
        denominator: The denominator.

    Returns:
        The ceiling of dividing numerator by denominator.
    """
    return numerator.__ceildiv__(denominator)


# NOTE: this overload is needed because IntLiteral promotes to a runtime type
# before overload resolution.
@always_inline("builtin")
fn ceildiv(
    numerator: IntLiteral, denominator: IntLiteral
) -> __type_of(numerator.__ceildiv__(denominator)):
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


@always_inline
fn trunc[T: Truncable, //](value: T) -> T:
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


@always_inline
fn sqrt(x: Int) -> Int:
    """Performs square root on an integer.

    Args:
        x: The integer value to perform square root on.

    Returns:
        The square root of x.
    """
    if x < 0:
        return 0

    var r = 0
    var r2 = 0

    @parameter
    for p in reversed(range(bit_width_of[Int]() // 2)):
        var dr2 = (r << (p + 1)) + (1 << (p + p))
        if r2 <= x - dr2:
            r2 += dr2
            r |= 1 << p

    return r


@always_inline
fn _sqrt_nvvm(x: SIMD, out res: __type_of(x)):
    constrained[
        x.dtype in (DType.float32, DType.float64), "must be f32 or f64 type"
    ]()
    alias instruction = "llvm.nvvm.sqrt.approx.ftz.f" if x.dtype is DType.float32 else "llvm.nvvm.sqrt.approx.d"
    res = {}

    @parameter
    for i in range(x.size):
        res[i] = _llvm_unary_fn[instruction](x[i])


@always_inline
fn sqrt[
    dtype: DType, width: Int, //
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
    constrained[
        dtype.is_numeric() or dtype is DType.bool,
        "type must be arithmetic or boolean",
    ]()

    @parameter
    if dtype is DType.bool:
        return x
    elif dtype.is_integral():
        var res = SIMD[dtype, width]()

        @parameter
        for i in range(width):
            res[i] = sqrt(Int(x[i]))
        return res
    elif is_nvidia_gpu():

        @parameter
        if dtype in (DType.float16, DType.bfloat16):
            return _sqrt_nvvm(x.cast[DType.float32]()).cast[dtype]()
        return _sqrt_nvvm(x)
    elif is_apple_gpu():
        return _llvm_unary_fn["llvm.air.sqrt"](x)

    return _llvm_unary_fn["llvm.sqrt"](x)


# ===----------------------------------------------------------------------=== #
# rsqrt
# ===----------------------------------------------------------------------=== #


@always_inline
fn _rsqrt_nvvm(x: SIMD, out res: __type_of(x)):
    constrained[
        x.dtype in (DType.float32, DType.float64), "must be f32 or f64 type"
    ]()

    alias instruction = "llvm.nvvm.rsqrt.approx.ftz.f" if x.dtype is DType.float32 else "llvm.nvvm.rsqrt.approx.d"
    res = {}

    @parameter
    for i in range(x.size):
        res[i] = _llvm_unary_fn[instruction](x[i])


@always_inline
fn rsqrt[dtype: DType, width: Int, //](x: SIMD[dtype, width]) -> __type_of(x):
    """Performs elementwise reciprocal square root on a SIMD vector.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: SIMD vector to perform reciprocal square root on.

    Returns:
        The elementwise reciprocal square root of x.
    """
    constrained[dtype.is_floating_point(), "type must be floating point"]()

    @parameter
    if is_nvidia_gpu():

        @parameter
        if dtype in (DType.float16, DType.bfloat16):
            return _rsqrt_nvvm(x.cast[DType.float32]()).cast[dtype]()

        return _rsqrt_nvvm(x)
    elif is_amd_gpu():

        @parameter
        if dtype in (DType.float16, DType.float32, DType.float64):
            return _call_amdgcn_intrinsic[
                String("llvm.amdgcn.rsq.", _get_amdgcn_type_suffix[dtype]())
            ](x)

        return rsqrt(x.cast[DType.float32]()).cast[dtype]()
    elif is_apple_gpu():
        return _llvm_unary_fn["llvm.air.rsqrt"](x)

    return 1 / sqrt(x)


# ===----------------------------------------------------------------------=== #
# recip
# ===----------------------------------------------------------------------=== #


@always_inline
fn _recip_nvvm(x: SIMD, out res: __type_of(x)):
    constrained[
        x.dtype in (DType.float32, DType.float64), "must be f32 or f64 type"
    ]()

    alias instruction = "llvm.nvvm.rcp.approx.ftz.f" if x.dtype is DType.float32 else "llvm.nvvm.rcp.approx.ftz.d"
    res = {}

    @parameter
    for i in range(x.size):
        res[i] = _llvm_unary_fn[instruction](x[i])


@always_inline
fn recip[dtype: DType, width: Int, //](x: SIMD[dtype, width]) -> __type_of(x):
    """Performs elementwise reciprocal on a SIMD vector.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: SIMD vector to perform reciprocal on.

    Returns:
        The elementwise reciprocal of x.
    """
    constrained[dtype.is_floating_point(), "type must be floating point"]()

    @parameter
    if is_nvidia_gpu():

        @parameter
        if dtype in (DType.float16, DType.bfloat16):
            return _recip_nvvm(x.cast[DType.float32]()).cast[dtype]()

        return _recip_nvvm(x)
    elif is_amd_gpu():

        @parameter
        if dtype in (DType.float16, DType.float32, DType.float64):
            return _call_amdgcn_intrinsic[
                String("llvm.amdgcn.rcp.", _get_amdgcn_type_suffix[dtype]())
            ](x)

        return recip(x.cast[DType.float32]()).cast[dtype]()

    return 1 / x


# ===----------------------------------------------------------------------=== #
# exp2
# ===----------------------------------------------------------------------=== #


@always_inline
fn exp2[
    dtype: DType, width: Int, //
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
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

    @parameter
    if is_nvidia_gpu():

        @parameter
        if dtype is DType.float16:

            @parameter
            if _is_sm_9x_or_newer():
                return _call_ptx_intrinsic[
                    scalar_instruction="ex2.approx.f16",
                    vector2_instruction="ex2.approx.f16x2",
                    scalar_constraints="=h,h",
                    vector_constraints="=r,r",
                ](x)
            else:
                return _call_ptx_intrinsic[
                    instruction="ex2.approx.f16", constraints="=h,h"
                ](x)
        elif dtype is DType.bfloat16 and _is_sm_9x_or_newer():
            return _call_ptx_intrinsic[
                scalar_instruction="ex2.approx.ftz.bf16",
                vector2_instruction="ex2.approx.ftz.bf16x2",
                scalar_constraints="=h,h",
                vector_constraints="=r,r",
            ](x)
        elif dtype is DType.float32:
            return _call_ptx_intrinsic[
                instruction="ex2.approx.ftz.f32", constraints="=f,f"
            ](x)

    @parameter
    if is_amd_gpu() and dtype in (DType.float16, DType.float32):
        return _call_amdgcn_intrinsic[
            String("llvm.amdgcn.exp2.", _get_amdgcn_type_suffix[dtype]())
        ](x)

    @parameter
    if is_apple_gpu() and dtype in (DType.float16, DType.float32):
        return _llvm_unary_fn["llvm.air.exp2"](x)

    @parameter
    if dtype is DType.float32:
        return _exp2_float32(x._refine[DType.float32]())._refine[dtype]()
    elif dtype is DType.float64:
        return 2**x
    else:
        return exp2(x.cast[DType.float32]()).cast[dtype]()


@always_inline
fn _exp2_float32(x: SIMD[DType.float32, _]) -> __type_of(x):
    alias u32 = DType.uint32
    var xc = x.clamp(-126, 126)
    var m = xc.cast[DType.int32]()
    xc -= m.cast[x.dtype]()

    var r = polynomial_evaluate[
        List[Float32](
            1.0,
            0.693144857883,
            0.2401793301105,
            5.551834031939e-2,
            9.810352697968e-3,
            1.33336498402e-3,
        ),
    ](xc)
    return __type_of(x)(
        from_bits=r.to_bits[u32]()
        + (m.cast[u32]() << FPUtils[DType.float32].mantissa_width())
    )


# ===----------------------------------------------------------------------=== #
# ldexp
# ===----------------------------------------------------------------------=== #


@always_inline
fn _ldexp_impl[
    dtype: DType, width: Int, //
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

    alias hardware_width = simd_width_of[dtype]()

    @parameter
    if (
        CompilationTarget.has_avx512f()
        and dtype is DType.float32
        and width >= hardware_width
    ):
        var res: SIMD[dtype, width] = 0
        var zero: SIMD[dtype, hardware_width] = 0

        @parameter
        for idx in range(width // hardware_width):
            alias i = idx * hardware_width
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

    alias integral_type = FPUtils[dtype].integral_type
    var m = exp.cast[integral_type]() + FPUtils[dtype].exponent_bias()

    return x * __type_of(x)(from_bits=m << FPUtils[dtype].mantissa_width())


@always_inline
fn ldexp[
    dtype: DType, width: Int, //
](x: SIMD[dtype, width], exp: SIMD[DType.int32, width]) -> SIMD[dtype, width]:
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

    fn __exp__(self) -> Self:
        """Computes the exponential of the input value.

        Returns:
            The exponential of the input value.
        """
        ...


@always_inline
fn _exp_taylor[
    dtype: DType, width: Int, //
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    alias coefficients = List[Scalar[dtype]](
        1.0,
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
    )
    return polynomial_evaluate[
        coefficients if dtype is DType.float64 else coefficients[:8],
    ](x)


@always_inline
fn exp[
    dtype: DType, width: Int, //
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
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
    constrained[dtype.is_floating_point(), "must be a floating point value"]()
    alias neg_ln2 = -0.69314718055966295651160180568695068359375

    @parameter
    if is_gpu():

        @parameter
        if dtype in (DType.float16, DType.float32):
            return exp2(x * log2e)

    @parameter
    if dtype not in (DType.float32, DType.float64):
        return exp(x.cast[DType.float32]()).cast[dtype]()

    var min_val: SIMD[dtype, width]
    var max_val: SIMD[dtype, width]

    @parameter
    if dtype is DType.float64:
        min_val = -709.436139303
        max_val = 709.437
    else:
        min_val = -88.3762626647949
        max_val = 88.3762626647950

    var xc = x.clamp(min_val, max_val)
    var k = floor(xc.fma(log2e, 0.5))
    var r = k.fma(neg_ln2, xc)
    return max(_ldexp_impl(_exp_taylor(r), k), xc)


@always_inline
fn exp[T: _Expable](x: T) -> T:
    """Computes the exponential of the input value.

    Parameters:
        T: The type of the input value.

    Args:
        x: The input value.

    Returns:
        The exponential of the input value.
    """
    return x.__exp__()


@always_inline
fn _exp2_approx_f32[
    W: Int
](x: SIMD[DType.float32, W]) -> SIMD[DType.float32, W]:
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
    alias ROUND_BIAS_F32 = 3 * FPUtils[DType.float32].mantissa_mask()
    alias NEG_ROUND_BIAS_F32 = -ROUND_BIAS_F32

    # Lower clamp for exp2 range reduction:
    # The float32 exponent bias is 127. Clamping at −127 keeps n from becoming
    # too negative (extreme subnormals/FTZ) and maintains accuracy of the cubic.
    # If you require strictly normal outputs, use −126.0 instead.
    alias EXP2_MIN_INPUT = -FPUtils[DType.float32].exponent_bias()
    # --- Kernel ---------------------------------------------------------------

    # 1) clamp in float
    var x_min = max(x, EXP2_MIN_INPUT)

    # 2) bias trick: vi = round(x_min) in float via +bias then −bias
    # (works for |x| < 2^23; we use 1.5*2^23 to behave well around 0 and negatives)
    var vb = x_min + ROUND_BIAS_F32
    var vi = vb + NEG_ROUND_BIAS_F32

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
        List[Float32](
            1.0000000000,
            0.6951461434,
            0.2275643945,
            0.0771190897,
        ),
    ](r)

    # 5) exponent as int lanes (no extra clamp needed due to early float clamp)
    var n = SIMD[DType.int32, W](vi)

    # result: 2^x ≈ 2^n * p
    return ldexp(p, n)


# ---------- e^x helpers ----------
@always_inline
fn exp_approx_f32[W: Int](x: SIMD[DType.float32, W]) -> SIMD[DType.float32, W]:
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
    return _exp2_approx_f32[W](x * SIMD[DType.float32, W](log2e))


# ===----------------------------------------------------------------------=== #
# frexp
# ===----------------------------------------------------------------------=== #


@always_inline
fn _frexp_mask1[
    dtype: DType, width: Int
]() -> SIMD[_integral_type_of[dtype](), width]:
    @parameter
    if dtype is DType.float16:
        return 0x7C00
    elif dtype is DType.bfloat16:
        return 0x7F80
    elif dtype is DType.float32:
        return 0x7F800000
    else:
        constrained[dtype is DType.float64, "unhandled fp type"]()
        return 0x7FF0000000000000


@always_inline
fn _frexp_mask2[
    dtype: DType, width: Int
]() -> SIMD[_integral_type_of[dtype](), width]:
    @parameter
    if dtype is DType.float16:
        return 0x3800
    elif dtype is DType.bfloat16:
        return 0x3F00
    elif dtype is DType.float32:
        return 0x3F000000
    else:
        constrained[dtype is DType.float64, "unhandled fp type"]()
        return 0x3FE0000000000000


@always_inline
fn frexp[
    dtype: DType, width: Int, //
](x: SIMD[dtype, width]) -> StaticTuple[SIMD[dtype, width], 2]:
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
    constrained[dtype.is_floating_point(), "must be a floating point value"]()
    alias T = SIMD[dtype, width]
    alias zero = T(0)
    # Add one to the resulting exponent up by subtracting 1 from the bias
    alias exponent_bias = FPUtils[dtype].exponent_bias() - 1
    alias mantissa_width = FPUtils[dtype].mantissa_width()
    var mask1 = _frexp_mask1[dtype, width]()
    var mask2 = _frexp_mask2[dtype, width]()
    var x_int = x._to_bits_signed()
    var selector = x.ne(zero)
    var exp = selector.select(
        (((mask1 & x_int) >> mantissa_width) - exponent_bias).cast[dtype](),
        zero,
    )
    var frac = selector.select(T(from_bits=x_int & ~mask1 | mask2), zero)
    return StaticTuple[size=2](frac, exp)


# ===----------------------------------------------------------------------=== #
# log
# ===----------------------------------------------------------------------=== #


@always_inline
fn _log_base[
    dtype: DType, width: Int, //, base: Int
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
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
    alias sqrt2_div_2 = 0.70710678118654752440

    constrained[base == 2 or base == 27, "input base must be either 2 or 27"]()

    var frexp_result = frexp(x)
    var x1 = frexp_result[0]
    var exp = frexp_result[1]
    exp = x1.lt(sqrt2_div_2).select(exp - 1, exp)
    x1 = x1.lt(sqrt2_div_2).select(x1 + x1, x1) - 1

    var x2 = x1 * x1
    var x3 = x2 * x1

    var y = (
        polynomial_evaluate[
            List[Scalar[dtype]](
                3.3333331174e-1,
                -2.4999993993e-1,
                2.0000714765e-1,
                -1.6668057665e-1,
                1.4249322787e-1,
                -1.2420140846e-1,
                1.1676998740e-1,
                -1.1514610310e-1,
                7.0376836292e-2,
            ),
        ](x1)
        * x3
    )
    y = x1 + x2.fma(-0.5, y)

    # TODO: fix this hack
    @parameter
    if base == 27:  # Natural log
        alias ln2 = 0.69314718055994530942
        y = exp.fma(ln2, y)
    else:
        y = y.fma(log2e, exp)
    return x.eq(0).select(Scalar[dtype].MIN, x.gt(0).select(y, nan[dtype]()))


@always_inline
fn log[dtype: DType, width: Int, //](x: SIMD[dtype, width]) -> __type_of(x):
    """Performs elementwise natural log (base E) of a SIMD vector.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: Vector to perform logarithm operation on.

    Returns:
        Vector containing result of performing natural log base E on x.
    """

    @parameter
    if size_of[dtype]() < size_of[DType.float32]():
        return log(x.cast[DType.float32]()).cast[dtype]()

    if is_compile_time():
        return _log_base[27](x)

    @parameter
    if is_nvidia_gpu() and dtype is DType.float32:
        alias ln2 = 0.69314718055966295651160180568695068359375
        return (
            _call_ptx_intrinsic[
                instruction="lg2.approx.f32", constraints="=f,f"
            ](x)
            * ln2
        )

    return _log_base[27](x)


# ===----------------------------------------------------------------------=== #
# log2
# ===----------------------------------------------------------------------=== #


@always_inline
fn log2[dtype: DType, width: Int, //](x: SIMD[dtype, width]) -> __type_of(x):
    """Performs elementwise log (base 2) of a SIMD vector.

    Args:
        x: Vector to perform logarithm operation on.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Returns:
        Vector containing result of performing log base 2 on x.
    """

    @parameter
    if size_of[dtype]() < size_of[DType.float32]() and not (
        is_amd_gpu() and dtype is DType.float16
    ):
        return log2(x.cast[DType.float32]()).cast[dtype]()

    if is_compile_time():
        return _log_base[2](x)

    @parameter
    if is_nvidia_gpu() and dtype is DType.float32:
        return _call_ptx_intrinsic[
            instruction="lg2.approx.f32", constraints="=f,f"
        ](x)
    elif is_amd_gpu() and dtype in (DType.float32, DType.float16):
        return _call_amdgcn_intrinsic[
            String("llvm.amdgcn.log.", _get_amdgcn_type_suffix[dtype]())
        ](x)

    return _log_base[2](x)


# ===----------------------------------------------------------------------=== #
# copysign
# ===----------------------------------------------------------------------=== #


@always_inline
fn copysign[
    dtype: DType, width: Int, //
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
    constrained[dtype.is_numeric(), "operands must be a numeric type"]()

    @parameter
    if dtype.is_unsigned():
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


@always_inline
fn erf[
    dtype: DType, width: Int, //
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
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
    constrained[dtype.is_floating_point(), "must be a floating point value"]()
    var x_abs = abs(x)

    var r_large = polynomial_evaluate[
        List[Scalar[dtype]](
            1.28717512e-1,
            6.34846687e-1,
            1.06777847e-1,
            -2.42545605e-2,
            3.88393435e-3,
            -3.83208680e-4,
            1.72948930e-5,
        ),
    ](min(x_abs, 3.925))

    r_large = r_large.fma(x_abs, x_abs)
    r_large = copysign(1 - exp(-r_large), x)

    var r_small = polynomial_evaluate[
        List[Scalar[dtype]](
            1.28379151e-1,
            -3.76124859e-1,
            1.12818025e-1,
            -2.67667342e-2,
            4.99339588e-3,
            -5.99104969e-4,
        ),
    ](x_abs * x_abs).fma(x, x)

    return x_abs.gt(0.921875).select[dtype](r_large, r_small)


# ===----------------------------------------------------------------------=== #
# tanh
# ===----------------------------------------------------------------------=== #


@always_inline
fn tanh[
    dtype: DType, width: Int, //
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Performs elementwise evaluation of the tanh function.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: The vector to perform the elementwise tanh on.

    Returns:
        The result of the elementwise tanh operation.
    """

    constrained[
        dtype.is_floating_point(), "the input type must be floating point"
    ]()

    @parameter
    if is_nvidia_gpu():
        alias instruction = "tanh.approx.f32"

        @parameter
        if dtype is DType.float16:
            return _call_ptx_intrinsic[
                scalar_instruction="tanh.approx.f16",
                vector2_instruction="tanh.approx.f16x2",
                scalar_constraints="=h,h",
                vector_constraints="=r,r",
            ](x)

        elif dtype is DType.bfloat16:

            @parameter
            if _is_sm_9x_or_newer():
                return _call_ptx_intrinsic[
                    scalar_instruction="tanh.approx.bf16",
                    vector2_instruction="tanh.approx.bf16x2",
                    scalar_constraints="=h,h",
                    vector_constraints="=r,r",
                ](x)
            else:
                return _call_ptx_intrinsic[
                    instruction="tanh.approx.f32", constraints="=f,f"
                ](x.cast[DType.float32]()).cast[dtype]()

        elif dtype is DType.float32:
            return _call_ptx_intrinsic[
                instruction="tanh.approx.f32", constraints="=f,f"
            ](x)

    var xc = x.clamp(-9, 9)
    var x_squared = xc * xc

    var numerator = xc * polynomial_evaluate[
        List[Scalar[dtype]](
            4.89352455891786e-03,
            6.37261928875436e-04,
            1.48572235717979e-05,
            5.12229709037114e-08,
            -8.60467152213735e-11,
            2.00018790482477e-13,
            -2.76076847742355e-16,
        ),
    ](x_squared)

    var denominator = polynomial_evaluate[
        List[Scalar[dtype]](
            4.89352518554385e-03,
            2.26843463243900e-03,
            1.18534705686654e-04,
            1.19825839466702e-06,
        ),
    ](x_squared)

    return numerator / denominator


# ===----------------------------------------------------------------------=== #
# isclose
# ===----------------------------------------------------------------------=== #


@always_inline
fn isclose[
    dtype: DType,
    width: Int,
    *,
    symmetrical: Bool = True,
](
    a: SIMD[dtype, width],
    b: SIMD[dtype, width],
    *,
    atol: Float64 = 1e-08,
    rtol: Float64 = 1e-05,
    equal_nan: Bool = False,
) -> SIMD[DType.bool, width]:
    """Returns a boolean SIMD vector indicating which element pairs of `a` and
    `b` are equal within a given tolerance.

    For floating-point dtypes, the following criteria apply:

    - Symmetric (Python `math.isclose` style), when `symmetrical` is true:
        ```
        |a - b| ≤ max(atol, rtol * max(|a|, |b|))
        ```
    - Asymmetric (NumPy style), when `symmetrical` is false:
        ```
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
    constrained[
        a.dtype.is_floating_point(),
        "isclose only supports floating-point types",
    ]()
    alias T = __type_of(a)

    var check_nan = isnan(a) & isnan(b)
    var check_fin: T._Mask
    var in_range: T._Mask

    @parameter
    if symmetrical:
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


@always_inline
fn iota[
    dtype: DType, width: Int
](offset: Scalar[dtype] = 0) -> SIMD[dtype, width]:
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

    @parameter
    if width == 1:
        return offset

    alias step_dtype = dtype if dtype.is_integral() else DType.int
    var step: SIMD[step_dtype, width]
    if is_compile_time():
        step = 0

        @parameter
        for i in range(width):
            step[i] = i
    else:
        step = llvm_intrinsic[
            "llvm.stepvector", SIMD[step_dtype, width], has_side_effect=False
        ]()
    return step.cast[dtype]() + offset


fn iota[
    dtype: DType, //
](buff: UnsafePointer[Scalar[dtype], mut=True, **_], len: Int, offset: Int = 0):
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

    @always_inline
    @__copy_capture(offset, buff)
    @parameter
    fn fill[width: Int](i: Int):
        buff.store(i, iota[dtype, width](offset + i))

    vectorize[fill, simd_width_of[dtype]()](len)


fn iota[dtype: DType, //](mut v: List[Scalar[dtype], *_], offset: Int = 0):
    """Fill a list with consecutive numbers starting from the specified offset.

    Parameters:
        dtype: DType of the underlying data.

    Args:
        v: The list to fill with numbers.
        offset: The starting value to fill at index 0.
    """
    iota(v.unsafe_ptr(), len(v), offset)


fn iota(mut v: List[Int, *_], offset: Int = 0):
    """Fill a list with consecutive numbers starting from the specified offset.

    Args:
        v: The list to fill with numbers.
        offset: The starting value to fill at index 0.
    """
    var buff = v.unsafe_ptr().bitcast[Scalar[DType.int]]()
    iota(buff, len(v), offset=offset)


# ===----------------------------------------------------------------------=== #
# fma
# ===----------------------------------------------------------------------=== #


@always_inline
fn fma(a: Int, b: Int, c: Int) -> Int:
    """Performs `fma` (fused multiply-add) on the inputs.

    The result is `(a * b) + c`.

    Args:
        a: The first input.
        b: The second input.
        c: The third input.

    Returns:
        `(a * b) + c`.
    """
    return a * b + c


@always_inline
fn fma(a: UInt, b: UInt, c: UInt) -> UInt:
    """Performs `fma` (fused multiply-add) on the inputs.

    The result is `(a * b) + c`.

    Args:
        a: The first input.
        b: The second input.
        c: The third input.

    Returns:
        `(a * b) + c`.
    """
    return a * b + c


@always_inline("nodebug")
fn fma[
    dtype: DType, width: Int, //
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


@always_inline
fn align_down(value: Int, alignment: Int) -> Int:
    """Returns the closest multiple of alignment that is less than or equal to
    value.

    Args:
        value: The value to align.
        alignment: Value to align to.

    Returns:
        Closest multiple of the alignment that is less than or equal to the
        input value. In other words, floor(value / alignment) * alignment.
    """
    debug_assert(alignment != 0, "zero alignment")
    return (value // alignment) * alignment


@always_inline
fn align_down(value: UInt, alignment: UInt) -> UInt:
    """Returns the closest multiple of alignment that is less than or equal to
    value.

    Args:
        value: The value to align.
        alignment: Value to align to.

    Returns:
        Closest multiple of the alignment that is less than or equal to the
        input value. In other words, floor(value / alignment) * alignment.
    """
    debug_assert(alignment != 0, "zero alignment")
    return (value // alignment) * alignment


# ===----------------------------------------------------------------------=== #
# align_up
# ===----------------------------------------------------------------------=== #


@always_inline
fn align_up(value: Int, alignment: Int) -> Int:
    """Returns the closest multiple of alignment that is greater than or equal
    to value.

    Args:
        value: The value to align.
        alignment: Value to align to.

    Returns:
        Closest multiple of the alignment that is greater than or equal to the
        input value. In other words, ceiling(value / alignment) * alignment.
    """
    debug_assert(alignment != 0, "zero alignment")
    return ceildiv(value, alignment) * alignment


@always_inline
fn align_up(value: UInt, alignment: UInt) -> UInt:
    """Returns the closest multiple of alignment that is greater than or equal
    to value.

    Args:
        value: The value to align.
        alignment: Value to align to.

    Returns:
        Closest multiple of the alignment that is greater than or equal to the
        input value. In other words, ceiling(value / alignment) * alignment.
    """
    debug_assert(alignment != 0, "zero alignment")
    return ceildiv(value, alignment) * alignment


# ===----------------------------------------------------------------------=== #
# acos
# ===----------------------------------------------------------------------=== #


fn acos[dtype: DType, width: Int, //](x: SIMD[dtype, width]) -> __type_of(x):
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

    constrained[
        dtype.is_floating_point(), "input type must be floating point"
    ]()

    @parameter
    if size_of[dtype]() < size_of[DType.float32]():
        return acos(x.cast[DType.float32]()).cast[dtype]()
    elif dtype is DType.float64:
        return _llvm_unary_fn["llvm.acos"](x)

    # For F32 types, use the Remez approximation found in Sleef with range
    # splitting to improve accuracy.

    # Determine which approximation method to use based on domain.
    var x_abs = clamp(abs(x), 0, 1)
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
    d = x_abs.eq(1).select(__type_of(x)(0.0), d)

    # Evaluate Remez polynomial using Horner's method
    # Coefficients derived to minimize maximum absolute error
    var poly = polynomial_evaluate[
        List[Scalar[x.dtype]](
            0.1666677296e0,
            0.7495029271e-1,
            0.4547423869e-1,
            0.2424046025e-1,
            0.4197454825e-1,
        )
    ](x_squared)

    # Final polynomial term: poly * x² * d
    poly *= x_squared * d

    # Small domain: compute π/2 - asin(x) where asin(x) = d + poly with sign.
    var y = (pi / 2.0) - (copysign(d, x) + copysign(poly, x))

    # Large domain: compute 2 * asin(sqrt((1-|x|)/2)) = 2 * (d + poly)
    var d_plus_poly = d + poly

    # Select result based on domain
    var result = directed_polynomial_mask.select(y, 2 * d_plus_poly)

    # Large domain with negative x: apply π - result transformation.
    return (~directed_polynomial_mask & x.lt(0)).select(pi - result, result)


# ===----------------------------------------------------------------------=== #
# asin
# ===----------------------------------------------------------------------=== #


fn asin[dtype: DType, width: Int, //](x: SIMD[dtype, width]) -> __type_of(x):
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

    constrained[
        dtype.is_floating_point(), "input type must be floating point"
    ]()

    @parameter
    if size_of[dtype]() < size_of[DType.float32]():
        return asin(x.cast[DType.float32]()).cast[dtype]()
    elif dtype is DType.float64:
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
    var d2 = directed_polynomial_mask.select(x * x, (1 - x_abs) * 0.5)

    # Compute d for evaluation:
    # - For |x| < 0.5: d = |x|
    # - For |x| >= 0.5: d = sqrt((1-|x|)/2)
    #   (using identity: asin(d) = π/2 - 2*asin(sqrt((1-d)/2)))
    var d = directed_polynomial_mask.select(x_abs, sqrt(d2))

    # Evaluate Remez polynomial approximation using Horner's method
    # This approximates the series: asin(x)/x ≈ 1 + x²/6 + 3x⁴/40 + ...
    var poly = polynomial_evaluate[
        List[Scalar[x.dtype]](
            0.1666677296e0,
            0.7495029271e-1,
            0.4547423869e-1,
            0.2424046025e-1,
            0.4197454825e-1,
        )
    ](d2)

    # Final polynomial evaluation: poly*x*x² + x = x*(poly*x² + 1)
    poly = poly.fma(d * d2, d)

    # Compute final result based on domain:
    # - For |x| < 0.5: result = poly  (direct approximation)
    # - For |x| >= 0.5: result = π/2 - 2*poly  (using identity)
    var result = directed_polynomial_mask.select(poly, pi / 2 - 2 * poly)

    return copysign(result, x)


# ===----------------------------------------------------------------------=== #
# atan
# ===----------------------------------------------------------------------=== #


fn atan[dtype: DType, width: Int, //](x: SIMD[dtype, width]) -> __type_of(x):
    """Computes the `atan` of the inputs.

    Constraints:
        The input must be a floating-point type.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: The input argument.

    Returns:
        The `atan` of the input.
    """
    return _call_libm["atan"](x)


# ===----------------------------------------------------------------------=== #
# atan2
# ===----------------------------------------------------------------------=== #


fn atan2[
    dtype: DType, width: Int, //
](y: SIMD[dtype, width], x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Computes the `atan2` of the inputs.

    Constraints:
        The input must be a floating-point type.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        y: The first input argument.
        x: The second input argument.

    Returns:
        The `atan2` of the inputs.
    """

    @always_inline("nodebug")
    @parameter
    fn _float32_dispatch[
        lhs_type: DType, rhs_type: DType, result_type: DType
    ](arg0: Scalar[lhs_type], arg1: Scalar[rhs_type]) -> Scalar[result_type]:
        return _external_call_const["atan2f", Scalar[result_type]](arg0, arg1)

    @always_inline("nodebug")
    @parameter
    fn _float64_dispatch[
        lhs_type: DType, rhs_type: DType, result_type: DType
    ](arg0: Scalar[lhs_type], arg1: Scalar[rhs_type]) -> Scalar[result_type]:
        return _external_call_const["atan2", Scalar[result_type]](arg0, arg1)

    constrained[
        dtype.is_floating_point(), "input type must be floating point"
    ]()

    @parameter
    if dtype is DType.float64:
        return _simd_apply[_float64_dispatch, result_dtype=dtype](y, x)
    else:
        return _simd_apply[_float32_dispatch, result_dtype=dtype](y, x)


# ===----------------------------------------------------------------------=== #
# cos
# ===----------------------------------------------------------------------=== #


fn cos[
    dtype: DType, width: Int, //
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
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

    @parameter
    if size_of[dtype]() < size_of[DType.float32]():
        return cos(x.cast[DType.float32]()).cast[dtype]()

    if is_compile_time():
        return _llvm_unary_fn["llvm.cos"](x)

    @parameter
    if is_nvidia_gpu() and dtype is DType.float32:
        return _call_ptx_intrinsic[
            instruction="cos.approx.ftz.f32", constraints="=f,f"
        ](x)
    elif is_apple_gpu():
        return _llvm_unary_fn["llvm.air.cos"](x)
    else:
        return _llvm_unary_fn["llvm.cos"](x)


# ===----------------------------------------------------------------------=== #
# sin
# ===----------------------------------------------------------------------=== #


fn sin[
    dtype: DType, width: Int, //
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
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

    @parameter
    if size_of[dtype]() < size_of[DType.float32]():
        return sin(x.cast[DType.float32]()).cast[dtype]()

    if is_compile_time():
        return _llvm_unary_fn["llvm.sin"](x)

    @parameter
    if is_nvidia_gpu() and dtype is DType.float32:
        return _call_ptx_intrinsic[
            instruction="sin.approx.ftz.f32", constraints="=f,f"
        ](x)
    elif is_apple_gpu():
        return _llvm_unary_fn["llvm.air.sin"](x)
    else:
        return _llvm_unary_fn["llvm.sin"](x)


# ===----------------------------------------------------------------------=== #
# tan
# ===----------------------------------------------------------------------=== #


fn tan[dtype: DType, width: Int, //](x: SIMD[dtype, width]) -> __type_of(x):
    """Computes the `tan` of the inputs.

    Constraints:
        The input must be a floating-point type.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: The input argument.

    Returns:
        The `tan` of the input.
    """
    return _call_libm["tan"](x)


# ===----------------------------------------------------------------------=== #
# acosh
# ===----------------------------------------------------------------------=== #


fn acosh[dtype: DType, width: Int, //](x: SIMD[dtype, width]) -> __type_of(x):
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
    return _call_libm["acosh"](x)


# ===----------------------------------------------------------------------=== #
# asinh
# ===----------------------------------------------------------------------=== #


fn asinh[dtype: DType, width: Int, //](x: SIMD[dtype, width]) -> __type_of(x):
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
    return _call_libm["asinh"](x)


# ===----------------------------------------------------------------------=== #
# atanh
# ===----------------------------------------------------------------------=== #


fn _atanh_float32(x: SIMD) -> __type_of(x):
    """This computes the `atanh` of the inputs for float32. It uses the same
    approximation used by Eigen library."""

    alias nan_val = nan[x.dtype]()
    alias inf_val = inf[x.dtype]()
    alias neg_inf_val = -inf[x.dtype]()

    var is_neg = x.lt(0)
    var x_abs = abs(x)
    var x2 = x * x
    var x3 = x2 * x

    # When x is in the range [0, 0.5], we use a polynomial approximation.
    # P(x) = x + x^3*(c[4] + x^2 * (c[3] + x^2 * (... x^2 * c[0]) ... )).
    var p = polynomial_evaluate[
        List[Scalar[x.dtype]](
            0.3333373963832855224609375,
            0.1997792422771453857421875,
            0.14672131836414337158203125,
            8.2311116158962249755859375e-2,
            0.1819281280040740966796875,
        )
    ](x2)
    p = x3.fma(p, x)

    # For |x| in the range [0.5, 1), we use the identity:
    # atanh(x) = 0.5 * log((1 + x) / (1 - x))
    var r = 0.5 * log((1 + x) / (1 - x))

    # If If x is >= 1, NaN is returned.
    # If x is 1, then the result is +infinity if x is negative, and -infinity
    # if x is positive. If x is >= 1, NaN is returned. Otherwise, if x is >= 0.5,
    # we use the r approximation, otherwise we use the p polynomial approximation.
    return x_abs.eq(1).select(
        is_neg.select(neg_inf_val, inf_val),
        x_abs.ge(1).select(
            is_neg.select(nan_val, -nan_val),
            x_abs.gt(0.5).select(r, p),
        ),
    )


@always_inline
fn atanh[dtype: DType, width: Int, //](x: SIMD[dtype, width]) -> __type_of(x):
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
    constrained[
        dtype.is_floating_point(), "input type must be floating point"
    ]()

    @parameter
    if dtype.bit_width() <= 16:
        # We promote the input to float32 and then cast back to the original
        # type. This is done to avoid precision issues that can occur when
        # using the lower-precision floating-point types.
        return _atanh_float32(x.cast[DType.float32]()).cast[dtype]()
    elif dtype is DType.float32:
        return _atanh_float32(x)

    # Otherwise, this is a double and we can just call the libm function.
    return _call_libm["atanh"](x)


# ===----------------------------------------------------------------------=== #
# cosh
# ===----------------------------------------------------------------------=== #


fn cosh[dtype: DType, width: Int, //](x: SIMD[dtype, width]) -> __type_of(x):
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
    return _call_libm["cosh"](x)


# ===----------------------------------------------------------------------=== #
# sinh
# ===----------------------------------------------------------------------=== #


fn sinh[dtype: DType, width: Int, //](x: SIMD[dtype, width]) -> __type_of(x):
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
    return _call_libm["sinh"](x)


# ===----------------------------------------------------------------------=== #
# expm1
# ===----------------------------------------------------------------------=== #


@always_inline
fn expm1[dtype: DType, width: Int, //](x: SIMD[dtype, width]) -> __type_of(x):
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
    return _call_libm["expm1"](x)


# ===----------------------------------------------------------------------=== #
# log10
# ===----------------------------------------------------------------------=== #


fn log10[dtype: DType, width: Int, //](x: SIMD[dtype, width]) -> __type_of(x):
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

    @parameter
    if is_nvidia_gpu():
        alias log10_2 = 0.301029995663981195213738894724493027

        @parameter
        if size_of[dtype]() < size_of[DType.float32]():
            return log10(x.cast[DType.float32]()).cast[dtype]()
        elif dtype is DType.float32:
            return (
                _call_ptx_intrinsic[
                    instruction="lg2.approx.f32", constraints="=f,f"
                ](x)
                * log10_2
            )
    elif is_amd_gpu():
        return _llvm_unary_fn["llvm.log10"](x)
    elif is_apple_gpu():
        return _llvm_unary_fn["llvm.air.log10"](x)

    return _call_libm["log10"](x)


# ===----------------------------------------------------------------------=== #
# log1p
# ===----------------------------------------------------------------------=== #


@always_inline
fn _log1p_f64[width: Int, //](x: SIMD[DType.float64, width]) -> __type_of(x):
    # This uses the approximation from cephes to compute log1p via the approximation
    # log(1+x) = x - x**2/2 + x**3 P(x)/Q(x)
    # in the domain 1/sqrt(2) <= x < sqrt(2)

    alias P = [
        2.0039553499201281259648e1,
        5.7112963590585538103336e1,
        6.0949667980987787057556e1,
        2.9911919328553073277375e1,
        6.5787325942061044846969e0,
        4.9854102823193375972212e-1,
        4.5270000862445199635215e-5,
    ]
    alias Q = [
        6.0118660497603843919306e1,
        2.1642788614495947685003e2,
        3.0909872225312059774938e2,
        2.2176239823732856465394e2,
        8.3047565967967209469434e1,
        1.5062909083469192043167e1,
    ]

    # Sqrt(1/2)
    alias sqrt2_div_2 = 0.70710678118654752440
    # Sqrt(2)
    alias sqrt2 = 1.41421356237309504880

    var z = 1 + x
    var log1x = log(z)

    var in_domain_mask = z.lt(sqrt2_div_2) | z.gt(sqrt2)
    if all(in_domain_mask):
        return log1x

    z = x * x
    z = -0.5 * z + x * (
        z * polynomial_evaluate[P](x) / polynomial_evaluate[Q](x)
    )

    return in_domain_mask.select(log1x, x + z)


fn log1p[dtype: DType, width: Int, //](x: SIMD[dtype, width]) -> __type_of(x):
    """Computes the `log1p` of the inputs.

    The `log1p(x)` is equivalent to `log(1+x)`.

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

    constrained[
        dtype.is_floating_point(), "input type must be floating point"
    ]()

    return _log1p_f64(x.cast[DType.float64]()).cast[dtype]()


# ===----------------------------------------------------------------------=== #
# logb
# ===----------------------------------------------------------------------=== #


fn logb[dtype: DType, width: Int, //](x: SIMD[dtype, width]) -> __type_of(x):
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
    return _call_libm["logb"](x)


# ===----------------------------------------------------------------------=== #
# cbrt
# ===----------------------------------------------------------------------=== #


fn _ilogb[
    width: Int
](x: SIMD[DType.float32, width]) -> SIMD[DType.int32, width]:
    """Extract binary exponent from floating-point number.

    Args:
        x: Input floating-point value.

    Returns:
        Integer binary exponent of x.
    """

    @always_inline
    fn extract(x: SIMD[DType.float32, width]) -> SIMD[DType.int32, width]:
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
        var bits = rebind[SIMD[DType.int32, width]](d._to_bits_signed())

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

    alias FP_ILOGB0 = (-2147483647 - 1)
    alias FP_ILOGBNAN = 2147483647

    # Extract the binary exponent from |x|
    # For x = m × 2^e where m ∈ [1, 2), this returns e
    var e = extract(abs(x))

    @parameter
    for i in range(width):
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


fn _cbrtf(x: Float32) -> Float32:
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

    alias CBRT_2 = 1.2599210498948731647672106
    alias CBRT_4 = 1.5874010519681994747517056

    # Apply correction factors based on remainder
    # If e % 3 == 1: need to multiply by 2^(1/3) = cbrt(2)
    if re == 1:
        q = CBRT_2
    # If e % 3 == 2: need to multiply by 2^(2/3) = cbrt(4)
    elif re == 2:
        q = CBRT_4

    # Scale q by 2^(qu - 2048) to reconstruct proper exponent
    # Subtract 2048 to compensate for the 6144 offset used earlier
    q = ldexp(q, qu - 2048)

    # Apply sign to correction factor (cube root preserves sign)
    q = copysign(q, x)

    # Work with absolute value for polynomial approximation
    d = abs(d)

    # Polynomial approximation for cbrt(d) where d ∈ [0.5, 1)
    # Using Horner's method for efficient evaluation
    var poly = polynomial_evaluate[
        List[Scalar[x.dtype]](
            2.2241256237030029296875,
            -3.8095417022705078125,
            5.898262500762939453125,
            -5.532182216644287109375,
            2.8208892345428466796875,
            -0.601564466953277587890625,
        )
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


fn cbrt[dtype: DType, width: Int, //](x: SIMD[dtype, width]) -> __type_of(x):
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

    constrained[
        dtype.is_floating_point(), "input type must be floating point"
    ]()

    @parameter
    if size_of[dtype]() < size_of[DType.float32]():
        return cbrt(x.cast[DType.float32]()).cast[dtype]()
    elif dtype is DType.float64:
        return _call_libm["cbrt"](x)

    var result = SIMD[DType.float32, width]()

    for i in range(width):
        result[i] = _cbrtf(rebind[Float32](x[i]))

    return rebind[__type_of(x)](result)


# ===----------------------------------------------------------------------=== #
# hypot
# ===----------------------------------------------------------------------=== #


# TODO: implement for variadic inputs as Python.
fn hypot[
    dtype: DType, width: Int, //
](arg0: SIMD[dtype, width], arg1: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Computes the `hypot` of the inputs.

    Constraints:
        The input must be a floating-point type.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        arg0: The first input argument.
        arg1: The second input argument.

    Returns:
        The `hypot` of the inputs.
    """

    @always_inline("nodebug")
    @parameter
    fn _float32_dispatch[
        lhs_type: DType, rhs_type: DType, result_type: DType
    ](arg0: Scalar[lhs_type], arg1: Scalar[rhs_type]) -> Scalar[result_type]:
        return _external_call_const["hypotf", Scalar[result_type]](arg0, arg1)

    @always_inline("nodebug")
    @parameter
    fn _float64_dispatch[
        lhs_type: DType, rhs_type: DType, result_type: DType
    ](arg0: Scalar[lhs_type], arg1: Scalar[rhs_type]) -> Scalar[result_type]:
        return _external_call_const["hypot", Scalar[result_type]](arg0, arg1)

    constrained[
        dtype.is_floating_point(), "input type must be floating point"
    ]()

    @parameter
    if dtype is DType.float64:
        return _simd_apply[_float64_dispatch, result_dtype=dtype](arg0, arg1)
    return _simd_apply[_float32_dispatch, result_dtype=dtype](arg0, arg1)


# ===----------------------------------------------------------------------=== #
# erfc
# ===----------------------------------------------------------------------=== #


fn _erfcf(x: Float32) -> Float32:
    """Fast single-precision complementary error function (erfc) approximation.

    The complementary error function is defined as:
    erfc(x) = 1 - erf(x) = (2/√π) ∫[x,∞] e^(-t²) dt

    Uses domain splitting with different rational approximations for accuracy:
    - Domain 1: |x| < 1.0  - Direct polynomial for erfc
    - Domain 2: 1.0 ≤ |x| < 2.2 - Polynomial approximation
    - Domain 3: 2.2 ≤ |x| < 4.3 - Scaled rational approximation with 1/x
    - Domain 4: 4.3 ≤ |x| < 10.1 - Further scaled rational approximation
    - Domain 5: |x| ≥ 10.1 - Returns 0 (underflow)

    Each domain uses optimized polynomial coefficients to minimize error.


    Args:
        x: Input value, any real number.

    Returns:
        Complementary error function erfc(x), range [0, 2].
    """

    # Handle NaN input
    if isnan(x):
        return x

    # Save original input for sign handling
    var s = x

    # Work with absolute value for domain classification
    var a = abs(x)

    # Domain classification based on input magnitude
    var o0 = a < 1.0  # Small values: direct approximation
    var o1 = a < 2.2  # Medium-small values
    var o2 = a < 4.3  # Medium values
    var o3 = a < 10.1  # Medium-large values
    # o3 false means a >= 10.1: very large values (return 0)

    # Choose transformation: u = a for small values, u = 1/a for large values
    # This improves numerical stability in different regions
    var u: Float32
    if o1:
        u = a  # Direct evaluation for a < 2.2
    else:
        u = 1.0 / a  # Use reciprocal for a >= 2.2

    # Coefficients are domain-specific for optimal accuracy
    alias coeffs0: List[Float32] = [
        -0.112837917790537404939545770596e1,
        -0.636619483208481931303752546439e0,
        -0.102775359343930288081655368891e0,
        0.1914106123e-1,
        0.1795156277e-3,
        -0.1665703603e-2,
        0.6000166177e-3,
        -0.8638041618e-4,
    ]
    alias coeffs1: List[Float32] = [
        -0.112855987376668622084547028949e1,
        -0.635609463574589034216723775292e0,
        -0.105247583459338632253369014063e0,
        0.2260518074e-1,
        -0.2851036377e-2,
        0.6002851478e-5,
        0.5749821503e-4,
        -0.6236977242e-5,
    ]
    alias coeffs2: List[Float32] = [
        -0.572319781150472949561786101080e0,
        -0.134450203224533979217859332703e-2,
        -0.482365310333045318680618892669e0,
        -0.1328857988e0,
        0.1249150872e1,
        -0.1816803217e1,
        0.1288077235e1,
        -0.3869504035e0,
    ]
    alias coeffs3: List[Float32] = [
        -0.572364030327966044425932623525e0,
        -0.471199543422848492080722832666e-4,
        -0.498961546254537647970305302739e0,
        -0.1262947265e-1,
        0.7155663371e0,
        -0.3667259514e0,
        -0.9454904199e0,
        0.1115344167e1,
    ]

    # Evaluate polynomial using Horner's method
    var d: Float32
    if o0:
        d = polynomial_evaluate[coeffs0](u)
    elif o1:
        d = polynomial_evaluate[coeffs1](u)
    elif o2:
        d = polynomial_evaluate[coeffs2](u)
    else:
        d = polynomial_evaluate[coeffs3](u)

    # Compute argument for exponential
    # For small a: x = d * a (using original a value)
    # For large a: x = -a² + d
    var exp_arg = d * a if o1 else (-a * a) + d

    # Compute exponential: exp(exp_arg)
    var result = exp(exp_arg)

    # For large a, multiply by u = 1/a
    if not o1:
        result *= u

    # Return 0 for very large values (a >= 10.1)
    if not o3:
        result = 0.0

    # Apply symmetry: erfc(-x) = 2 - erfc(x)
    if s < 0:
        result = 2.0 - result

    return result


fn erfc[dtype: DType, width: Int, //](x: SIMD[dtype, width]) -> __type_of(x):
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
    constrained[
        dtype.is_floating_point(), "input type must be floating point"
    ]()

    @parameter
    if size_of[dtype]() < size_of[DType.float32]():
        return erfc(x.cast[DType.float32]()).cast[dtype]()
    elif dtype is DType.float64:
        return _call_libm["erfc"](x)

    var result = SIMD[DType.float32, width]()

    for i in range(width):
        result[i] = _erfcf(rebind[Float32](x[i]))

    return rebind[__type_of(x)](result)


# ===----------------------------------------------------------------------=== #
# lgamma
# ===----------------------------------------------------------------------=== #


fn lgamma[dtype: DType, width: Int, //](x: SIMD[dtype, width]) -> __type_of(x):
    """Computes the `lgamma` of the inputs.

    Constraints:
        The input must be a floating-point type.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: The input argument.

    Returns:
        The `lgamma` of the input.
    """
    return _call_libm["lgamma"](x)


# ===----------------------------------------------------------------------=== #
# gamma
# ===----------------------------------------------------------------------=== #


fn gamma[dtype: DType, width: Int, //](x: SIMD[dtype, width]) -> __type_of(x):
    """Computes the Gamma of the input.

    For details, see https://en.wikipedia.org/wiki/Gamma_function.

    Constraints:
        The input must be a floating-point type.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: The input argument.

    Returns:
        The Gamma function evaluated at the input.
    """
    return _call_libm["tgamma"](x)


# ===----------------------------------------------------------------------=== #
# remainder
# ===----------------------------------------------------------------------=== #


fn remainder[
    dtype: DType, width: Int, //
](x: SIMD[dtype, width], y: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Computes the `remainder` of the inputs.

    Constraints:
        The input must be a floating-point type.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: The first input argument.
        y: The second input argument.

    Returns:
        The `remainder` of the inputs.
    """

    @always_inline("nodebug")
    @parameter
    fn _float32_dispatch[
        lhs_type: DType, rhs_type: DType, result_type: DType
    ](arg0: Scalar[lhs_type], arg1: Scalar[rhs_type]) -> Scalar[result_type]:
        return _external_call_const["remainderf", Scalar[result_type]](
            arg0, arg1
        )

    @always_inline("nodebug")
    @parameter
    fn _float64_dispatch[
        lhs_type: DType, rhs_type: DType, result_type: DType
    ](arg0: Scalar[lhs_type], arg1: Scalar[rhs_type]) -> Scalar[result_type]:
        return _external_call_const["remainder", Scalar[result_type]](
            arg0, arg1
        )

    constrained[
        dtype.is_floating_point(), "input type must be floating point"
    ]()

    @parameter
    if dtype is DType.float64:
        return _simd_apply[_float64_dispatch, result_dtype=dtype](x, y)
    return _simd_apply[_float32_dispatch, result_dtype=dtype](x, y)


# ===----------------------------------------------------------------------=== #
# j0
# ===----------------------------------------------------------------------=== #


fn j0[dtype: DType, width: Int, //](x: SIMD[dtype, width]) -> __type_of(x):
    """Computes the Bessel function of the first kind of order 0 for each input
    value.

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
    return _call_libm["j0"](x)


# ===----------------------------------------------------------------------=== #
# j1
# ===----------------------------------------------------------------------=== #


fn j1[dtype: DType, width: Int, //](x: SIMD[dtype, width]) -> __type_of(x):
    """Computes the Bessel function of the first kind of order 1 for each input
    value.

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
    return _call_libm["j1"](x)


# ===----------------------------------------------------------------------=== #
# y0
# ===----------------------------------------------------------------------=== #


fn y0[dtype: DType, width: Int, //](x: SIMD[dtype, width]) -> __type_of(x):
    """Computes the Bessel function of the second kind of order 0 for each input
    value.

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
    return _call_libm["y0"](x)


# ===----------------------------------------------------------------------=== #
# y1
# ===----------------------------------------------------------------------=== #


fn y1[dtype: DType, width: Int, //](x: SIMD[dtype, width]) -> __type_of(x):
    """Computes the Bessel function of the second kind of order 1 for each input
    value.

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
    return _call_libm["y1"](x)


# ===----------------------------------------------------------------------=== #
# scalb
# ===----------------------------------------------------------------------=== #


fn scalb[
    dtype: DType, width: Int, //
](arg0: SIMD[dtype, width], arg1: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Computes the `scalb` of the inputs.

    Constraints:
        The input must be a floating-point type.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        arg0: The first input argument.
        arg1: The second input argument.

    Returns:
        The `scalb` of the inputs.
    """

    @always_inline("nodebug")
    @parameter
    fn _float32_dispatch[
        lhs_type: DType, rhs_type: DType, result_type: DType
    ](arg0: Scalar[lhs_type], arg1: Scalar[rhs_type]) -> Scalar[result_type]:
        return _external_call_const["scalbf", Scalar[result_type]](arg0, arg1)

    @always_inline("nodebug")
    @parameter
    fn _float64_dispatch[
        lhs_type: DType, rhs_type: DType, result_type: DType
    ](arg0: Scalar[lhs_type], arg1: Scalar[rhs_type]) -> Scalar[result_type]:
        return _external_call_const["scalb", Scalar[result_type]](arg0, arg1)

    constrained[
        dtype.is_floating_point(), "input type must be floating point"
    ]()

    @parameter
    if dtype is DType.float64:
        return _simd_apply[_float64_dispatch, result_dtype=dtype](arg0, arg1)
    return _simd_apply[_float32_dispatch, result_dtype=dtype](arg0, arg1)


# ===----------------------------------------------------------------------=== #
# gcd
# ===----------------------------------------------------------------------=== #


fn gcd(m: Int, n: Int, /) -> Int:
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


fn gcd(s: Span[Int], /) -> Int:
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


@always_inline
fn gcd(l: List[Int, *_], /) -> Int:
    """Computes the greatest common divisor of a list of integers.

    Args:
        l: A list containing a collection of integers.

    Returns:
        The greatest common divisor of all the integers in the list.
    """
    return gcd(Span(l))


fn gcd(*values: Int) -> Int:
    """Computes the greatest common divisor of a variadic number of integers.

    Args:
        values: A variadic list of integers.

    Returns:
        The greatest common divisor of the given integers.
    """
    # TODO: Deduplicate when we can create a Span from VariadicList
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


fn lcm(m: Int, n: Int, /) -> Int:
    """Computes the least common multiple of two integers.

    Args:
        m: The first integer.
        n: The second integer.

    Returns:
        The least common multiple of the two integers.
    """
    if d := gcd(m, n):
        return abs((m // d) * n if m > n else (n // d) * m)
    return 0


fn lcm(s: Span[Int], /) -> Int:
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


@always_inline
fn lcm(l: List[Int, *_], /) -> Int:
    """Computes the least common multiple of a list of integers.

    Args:
        l: A list of integers.

    Returns:
        The least common multiple of the list.
    """
    return lcm(Span(l))


fn lcm(*values: Int) -> Int:
    """Computes the least common multiple of a variadic list of integers.

    Args:
        values: A variadic list of integers.

    Returns:
        The least common multiple of the list.
    """
    # TODO: Deduplicate when we can create a Span from VariadicList
    if len(values) == 0:
        return 1

    var result = values[0]
    for i in range(1, len(values)):
        result = lcm(result, values[i])
    return result


# ===----------------------------------------------------------------------=== #
# modf
# ===----------------------------------------------------------------------=== #


fn modf[
    dtype: DType, width: Int, //
](x: SIMD[dtype, width]) -> Tuple[__type_of(x), __type_of(x)]:
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


@always_inline
fn ulp[
    dtype: DType, width: Int, //
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
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

    constrained[dtype.is_floating_point(), "the type must be floating point"]()

    var nan_mask = isnan(x)
    var xabs = abs(x)
    var inf_mask = isinf(xabs)
    alias inf_val = SIMD[dtype, width](inf[dtype]())
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
@always_inline
fn factorial(n: Int) -> Int:
    """Computes the factorial of the integer.

    Args:
        n: The input value. Must be non-negative.

    Returns:
        The factorial of the input. Results are undefined for negative inputs.
    """
    alias table = StaticTuple[Int, 21](
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
    debug_assert(
        0 <= n <= (12 if is_32bit() else 20), "input value causes an overflow"
    )
    return table[n]


# ===----------------------------------------------------------------------=== #
# clamp
# ===----------------------------------------------------------------------=== #


fn clamp(
    val: Int, lower_bound: __type_of(val), upper_bound: __type_of(val)
) -> __type_of(val):
    """Clamps the integer value vector to be in a certain range.

    Args:
        val: The value to clamp.
        lower_bound: Minimum of the range to clamp to.
        upper_bound: Maximum of the range to clamp to.

    Returns:
        An integer clamped to be within lower_bound and upper_bound.
    """
    return max(min(val, upper_bound), lower_bound)


fn clamp(
    val: UInt, lower_bound: __type_of(val), upper_bound: __type_of(val)
) -> __type_of(val):
    """Clamps the integer value vector to be in a certain range.

    Args:
        val: The value to clamp.
        lower_bound: Minimum of the range to clamp to.
        upper_bound: Maximum of the range to clamp to.

    Returns:
        An integer clamped to be within lower_bound and upper_bound.
    """
    return max(min(val, upper_bound), lower_bound)


fn clamp[
    dtype: DType, width: Int, //
](
    val: SIMD[dtype, width],
    lower_bound: __type_of(val),
    upper_bound: __type_of(val),
) -> __type_of(val):
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


@always_inline("nodebug")
fn _llvm_unary_fn[
    dtype: DType,
    width: Int, //,
    fn_name: StaticString,
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    return llvm_intrinsic[fn_name, __type_of(x), has_side_effect=False](x)


@always_inline("nodebug")
fn _call_libm[
    dtype: DType,
    width: Int, //,
    func_name: StaticString,
](arg: SIMD[dtype, width]) -> SIMD[dtype, width]:
    constrained[
        dtype.is_floating_point(), "argument type must be floating point"
    ]()
    constrained[
        not is_gpu(), "libm operations are only available on CPU targets"
    ]()

    @parameter
    if dtype not in [DType.float32, DType.float64]:
        # Coerce to f32 if the value is not representable by libm.
        var arg_f32 = arg.cast[DType.float32]()
        return _call_libm[func_name](arg_f32).cast[dtype]()

    alias libm_name = func_name + ("f" if dtype is DType.float32 else "")
    var res = SIMD[dtype, width]()

    @parameter
    for i in range(width):
        res[i] = _external_call_const[libm_name, Scalar[dtype]](arg[i])
    return res


fn _call_ptx_intrinsic_scalar[
    dtype: DType, //,
    *,
    instruction: StaticString,
    constraints: StaticString,
](arg: Scalar[dtype]) -> Scalar[dtype]:
    return inlined_assembly[
        instruction + " $0, $1;",
        Scalar[dtype],
        constraints=constraints,
        has_side_effect=False,
    ](arg)


fn _call_ptx_intrinsic_scalar[
    dtype: DType, //,
    *,
    instruction: StaticString,
    constraints: StaticString,
](arg0: Scalar[dtype], arg1: Scalar[dtype]) -> Scalar[dtype]:
    return inlined_assembly[
        instruction + " $0, $1, $2;",
        Scalar[dtype],
        constraints=constraints,
        has_side_effect=False,
    ](arg0, arg1)


fn _call_ptx_intrinsic[
    dtype: DType,
    width: Int, //,
    *,
    instruction: StaticString,
    constraints: StaticString,
](arg: SIMD[dtype, width]) -> SIMD[dtype, width]:
    @parameter
    if width == 1:
        return _call_ptx_intrinsic_scalar[
            instruction=instruction, constraints=constraints
        ](arg[0])

    var res = SIMD[dtype, width]()

    @parameter
    for i in range(width):
        res[i] = _call_ptx_intrinsic_scalar[
            instruction=instruction, constraints=constraints
        ](arg[i])
    return res


fn _call_ptx_intrinsic[
    dtype: DType,
    width: Int, //,
    *,
    scalar_instruction: StaticString,
    vector2_instruction: StaticString,
    scalar_constraints: StaticString,
    vector_constraints: StaticString,
](arg: SIMD[dtype, width]) -> SIMD[dtype, width]:
    @parameter
    if width == 1:
        return _call_ptx_intrinsic_scalar[
            instruction=scalar_instruction, constraints=scalar_constraints
        ](arg[0])

    var res = SIMD[dtype, width]()

    @parameter
    for i in range(0, width, 2):
        res = res.insert[offset=i](
            inlined_assembly[
                vector2_instruction + " $0, $1;",
                SIMD[dtype, 2],
                constraints=vector_constraints,
                has_side_effect=False,
            ](arg.slice[2, offset=i]())
        )

    return res


fn _call_ptx_intrinsic[
    dtype: DType,
    width: Int, //,
    *,
    scalar_instruction: StaticString,
    vector2_instruction: StaticString,
    scalar_constraints: StaticString,
    vector_constraints: StaticString,
](arg0: SIMD[dtype, width], arg1: SIMD[dtype, width]) -> SIMD[dtype, width]:
    @parameter
    if width == 1:
        return _call_ptx_intrinsic_scalar[
            instruction=scalar_instruction, constraints=scalar_constraints
        ](arg0[0], arg1[0])

    var res = SIMD[dtype, width]()

    @parameter
    for i in range(0, width, 2):
        res = res.insert[offset=i](
            inlined_assembly[
                vector2_instruction + " $0, $1; $2;",
                SIMD[dtype, 2],
                constraints=vector_constraints,
                has_side_effect=False,
            ](arg0.slice[2, offset=i](), arg1.slice[2, offset=i]())
        )

    return res


@always_inline
fn _call_amdgcn_intrinsic[intrin: StaticString](x: SIMD, out res: __type_of(x)):
    res = {}

    @parameter
    for i in range(x.size):
        res[i] = _llvm_unary_fn[intrin](x[i])


@always_inline
fn _get_amdgcn_type_suffix[dtype: DType]() -> StaticString:
    @parameter
    if dtype is DType.float16:
        return "f16"
    elif dtype is DType.float32:
        return "f32"
    elif dtype is DType.float64:
        return "f64"
    else:
        constrained[False, "Extend to support additional dtypes."]()
        return ""


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
    from math import Ceilable, ceil

    @fieldwise_init
    struct Complex(Ceilable, ImplicitlyCopyable):
        var re: Float64
        var im: Float64

        fn __ceil__(self) -> Self:
            return Self(ceil(self.re), ceil(self.im))
    ```
    """

    # TODO(MOCO-333): Reconsider the signature when we have parametric traits or
    # associated types.
    fn __ceil__(self) -> Self:
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
    from math import Floorable, floor

    @fieldwise_init
    struct Complex(Floorable, ImplicitlyCopyable):
        var re: Float64
        var im: Float64

        fn __floor__(self) -> Self:
            return Self(floor(self.re), floor(self.im))
    ```
    """

    # TODO(MOCO-333): Reconsider the signature when we have parametric traits or
    # associated types.
    fn __floor__(self) -> Self:
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
    from math import CeilDivable

    @fieldwise_init
    struct Foo(CeilDivable, ImplicitlyCopyable):
        var x: Float64

        fn __ceildiv__(self, denominator: Self) -> Self:
            return Self(self.x // denominator.x)
    ```
    """

    fn __ceildiv__(self, denominator: Self) -> Self:
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
    from math import CeilDivableRaising

    @fieldwise_init
    struct Foo(CeilDivableRaising, ImplicitlyCopyable):
        var x: Float64

        fn __ceildiv__(self, denominator: Self) raises -> Self:
            return Self(self.x // denominator.x)
    ```
    """

    fn __ceildiv__(self, denominator: Self) raises -> Self:
        """Return the rounded-up result of dividing self by denominator.

        Args:
            denominator: The denominator.

        Returns:
            The ceiling of dividing numerator by denominator.
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
    from math import Truncable, trunc

    @fieldwise_init
    struct Complex(Truncable, ImplicitlyCopyable):
        var re: Float64
        var im: Float64

        fn __trunc__(self) -> Self:
            return Self(trunc(self.re), trunc(self.im))
    ```
    """

    # TODO(MOCO-333): Reconsider the signature when we have parametric traits or
    # associated types.
    fn __trunc__(self) -> Self:
        """Return the truncated Int value, which is itself.

        Returns:
            The Int value itself.
        """
        ...
