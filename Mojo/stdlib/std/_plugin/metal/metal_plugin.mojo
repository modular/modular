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

from std._plugin._trait import (
    PluginHooks,
    StdPlugin,
    _PrintEmitPluginHookFnType,
    _UnaryMathPluginHookFnType,
)
from std.collections import OptionalReg
from std.sys.info import CompilationTarget
from std.collections.string.string_span import (
    _get_kgen_string,
    get_static_string,
)
from std.ffi import CStringSpan
from std.io.file_descriptor import RawFileDescriptor
from std.sys import llvm_intrinsic
from std.sys._metal_print import _metal_print_write


def _print_emit[O: Origin](cstr: CStringSpan[O], file: RawFileDescriptor):
    """Writes `print()` output from a Metal kernel to the os_log stream.

    Metal kernels have a single os_log stream, so `file` is ignored.
    """
    _metal_print_write(StringSlice(unsafe_from_utf8=cstr.as_bytes_with_nul()))


def _has_air_op[name: StaticString, dtype: DType]() -> Bool:
    comptime if name == "exp2":
        return dtype in (DType.float16, DType.float32)
    return name == "sqrt" or name == "rsqrt" or name == "sin" or name == "cos"


def _air_unary_math[
    dtype: DType, width: SIMDLength, //, name: StaticString
](x: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Lowers the `std.math` function `name` to its `air.*` intrinsic.

    AIR has no bf16 overloads of these intrinsics and Metal rejects them, so
    bf16 inputs round-trip through f32.
    """
    comptime if dtype == .bfloat16:
        return _air_unary_math[name=name](x.cast[.float32]()).cast[dtype]()
    return llvm_intrinsic[
        get_static_string["llvm.air.", name](),
        type_of(x),
        has_side_effect=False,
    ](x)


struct MetalPlugin(StdPlugin):
    """`PluginHooks` implementation for Apple Metal backends."""

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["metal"]()
    comptime HookFor[target: CompilationTarget] = MetalPluginHooks[target]


struct MetalPluginHooks[target: CompilationTarget](PluginHooks):
    comptime print_emit_fn: OptionalReg[
        _PrintEmitPluginHookFnType
    ] = _print_emit

    comptime unary_math_fn[
        name: StaticString, dtype: DType, width: Int
    ]: OptionalReg[_UnaryMathPluginHookFnType] = (
        _air_unary_math[name=name] if _has_air_op[name, dtype]() else None
    )

    comptime flushes_subnormals[dtype: DType]: Bool = dtype == DType.float32
