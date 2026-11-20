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
"""Provides calls into the host libm, the reference for math tests."""

from std.ffi import _external_call_const, external_call
from std.math import isinf, isnan, nan, trunc
from std.math.math import _call_libm
from std.simd import _simd_apply
from std.sys import CompilationTarget, size_of


@inline(.always)
def libm_call[
    dtype: DType,
    width: SIMDLength,
    //,
    fn_fp32: StaticString,
    fn_fp64: StaticString,
](arg: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Calls a libm function with the appropriate float32 or float64 version.

    Parameters:
        dtype: The data type (must be float32 or float64).
        width: The SIMD width.
        fn_fp32: Name of the float32 version of the libm function.
        fn_fp64: Name of the float64 version of the libm function.

    Args:
        arg: The input SIMD vector.

    Returns:
        The result of calling the libm function.
    """

    @inline(.nodebug)
    def _float32_dispatch[
        input_type: DType, result_type: DType
    ](arg: Scalar[input_type]) -> Scalar[result_type]:
        return external_call[fn_fp32, Scalar[result_type]](arg)

    @inline(.nodebug)
    def _float64_dispatch[
        input_type: DType, result_type: DType
    ](arg: Scalar[input_type]) -> Scalar[result_type]:
        return external_call[fn_fp64, Scalar[result_type]](arg)

    comptime match dtype:
        case .float32:
            return _simd_apply[_float32_dispatch, result_dtype=dtype](arg)
        case .float64:
            return _simd_apply[_float64_dispatch, result_dtype=dtype](arg)
        case _:
            comptime assert False, "input dtype must be float32 or float64"


def libm_call_binary[
    dtype: DType, //, fn_fp32: StaticString, fn_fp64: StaticString
](x: Scalar[dtype], y: Scalar[dtype]) -> Scalar[
    dtype
] where dtype.is_floating_point():
    """Calls a two-argument libm function on the host.

    float16 and bfloat16 arguments go through the float32 function, and its
    result is rounded to `dtype`.

    Parameters:
        dtype: The floating-point type of the arguments.
        fn_fp32: Name of the float32 version of the libm function.
        fn_fp64: Name of the float64 version of the libm function.

    Args:
        x: The first argument.
        y: The second argument.

    Returns:
        The libm result.
    """
    comptime match dtype:
        case .float64:
            return _external_call_const[fn_fp64, Scalar[dtype]](x, y)
        case .float32:
            return _external_call_const[fn_fp32, Scalar[dtype]](x, y)
        case .float16 | .bfloat16:
            return _external_call_const[fn_fp32, Float32](
                x.cast[.float32](), y.cast[.float32]()
            ).cast[dtype]()
        case _:
            comptime assert False, "expected a floating point"


def libm_call_f64[
    dtype: DType, width: SIMDLength, //, name: StaticString
](x: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    """Calls the float64 version of a libm function and rounds the result to
    `dtype`.

    Parameters:
        dtype: The floating-point type of the argument and the result.
        width: The SIMD width.
        name: Name of the float64 version of the libm function.

    Args:
        x: The argument.

    Returns:
        The float64 libm result, rounded once to `dtype`.
    """
    return _call_libm[name](x.cast[.float64]()).cast[dtype]()


def libm_scalb[
    dtype: DType, //
](x: Scalar[dtype], y: Scalar[dtype]) -> Scalar[
    dtype
] where dtype.is_floating_point():
    """Returns `x * 2^y` as libm `scalb` computes it.

    macOS libm has no `scalbf`, and its `scalb` does not return NaN for a
    non-integral exponent, so off Linux this follows fdlibm/glibc `scalb`
    using libm `ldexp` instead.

    Parameters:
        dtype: The floating-point type of the arguments.

    Args:
        x: The value to scale.
        y: The exponent, which must be integral for a non-NaN result.

    Returns:
        The libm result, rounded through float32 for float16 and bfloat16.
    """
    comptime if CompilationTarget.is_linux():
        return libm_call_binary["scalbf", "scalb"](x, y)
    # Scaling a float32 or narrower value is exact in float64, so the rounded
    # `ldexp` result is correct.
    var x64 = x.cast[.float64]()
    var y64 = y.cast[.float64]()
    var r: Float64
    if isnan(x64) or isnan(y64):
        r = x64 * y64
    elif isinf(y64):
        r = x64 * y64 if y64 > 0 else x64 / -y64
    elif trunc(y64) != y64:
        r = nan[.float64]()
    else:
        # Clamping the exponent this far out saturates `ldexp` for every
        # dtype without overflowing `Int32`.
        var n = Int32(y64.clamp(-65000.0, 65000.0))
        r = _external_call_const["ldexp", Float64](x64, n)
    comptime if size_of[dtype]() < size_of[DType.float32]():
        return r.cast[.float32]().cast[dtype]()
    return r.cast[dtype]()
