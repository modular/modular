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
from std.simd import _simd_apply


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
