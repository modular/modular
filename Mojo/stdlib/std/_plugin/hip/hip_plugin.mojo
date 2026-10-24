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

from std._gpu.primitives.id import _GPUIdPluginHookFnType
from std._plugin._trait import (
    TargetDeviceKind,
    PluginHooks,
    StdPlugin,
    _GlobalPerfCounterNsPluginHookFnType,
    _PrintEmitPluginHookFnType,
    _PrintfEmitPluginHookFnType,
    _SleepPluginHookFnType,
    _UnaryMathPluginHookFnType,
)
from std.collections import OptionalReg
from std.simd import _CastPluginHookFnType
from std.sys.info import CompilationTarget
from std.collections.string.string_span import (
    _get_kgen_string,
    get_static_string,
)
from std.ffi import CStringSpan
from std.io.file_descriptor import RawFileDescriptor
from std.math.math import _UnaryMathOp
from std.memory import bitcast
from std.reflection.location import SourceLocation

from .hip_cast import _amdgcn_cast_fn, _amdgcn_native_float8_cast
from .hip_gpu_id import _amdgcn_gpu_id_fn
from .hip_hostcall import (
    printf_append_args,
    printf_append_string_n,
    printf_begin,
)
from .hip_math import _amdgcn_unary_math_fn
from .hip_time import _global_perf_counter_ns, _sleep


def _print_emit[O: Origin](cstr: CStringSpan[O], file: RawFileDescriptor):
    """Writes `print()` output from an AMD kernel through the hostcall
    fprintf service.

    The service writes to the host's stdout, so `file` is ignored.
    """
    var msg = printf_begin()
    _ = printf_append_string_n(msg, cstr.as_bytes_with_nul(), is_last=True)


@inline(.always)
def _printf_emit[
    fmt: StaticString, *types: AnyType
](*args: *types, file: RawFileDescriptor):
    """Writes `_printf` output from an AMD kernel through the hostcall
    fprintf service.

    The service writes to a single output stream, so `file` is ignored.
    """

    # This is adapted from Triton's third party method for lowering
    # AMD printf calls:
    # https://github.com/triton-lang/triton/blob/1c28e08971a0d70c4331432994338ee05d31e633/third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.cpp#L321
    def _to_uint64[T: AnyType, //](value: T) -> UInt64:
        comptime if T == UInt64:
            return rebind[UInt64](value)
        elif T == UInt32:
            return UInt64(rebind[UInt32](value))
        elif T == UInt16:
            return UInt64(rebind[UInt16](value))
        elif T == UInt8:
            return UInt64(rebind[UInt8](value))
        elif T == Int64:
            return UInt64(rebind[Int64](value))
        elif T == Int32:
            return UInt64(rebind[Int32](value))
        elif T == Int16:
            return UInt64(rebind[Int16](value))
        elif T == Int8:
            return UInt64(rebind[Int8](value))
        elif T == Float16:
            return bitcast[.uint64](Float64(rebind[Float16](value)))
        elif T == Float32:
            return bitcast[.uint64](Float64(rebind[Float32](value)))
        elif T == Float64:
            return bitcast[.uint64](rebind[Float64](value))
        elif T == Int:
            return UInt64(rebind[Int](value))
        elif T == UInt:
            return UInt64(rebind[UInt](value))
        return 0

    comptime args_len = types.length

    var message = printf_begin()
    # `get_static_string` guarantees a trailing nul in static memory (just
    # past the returned range); include it so the AMD fprintf service sees a
    # terminated format string even when `len(fmt)` is a multiple of 8.
    # `as_bytes()` alone drops the nul, corrupting output (MSTDL-1597).
    var fmt_str = get_static_string[fmt]()
    message = printf_append_string_n(
        message,
        Span(
            unsafe_ptr=fmt_str.as_bytes().unsafe_ptr(),
            length=fmt_str.byte_length() + 1,
        ),
        args_len == 0,
    )
    comptime k_args_per_group = 7

    comptime for group in range(0, args_len, k_args_per_group):
        comptime bound = min(group + k_args_per_group, args_len)
        comptime num_args = bound - group

        var arguments = Array[UInt64, k_args_per_group](fill=0)

        comptime for i in range(num_args):
            arguments[i] = _to_uint64(args[group + i])
        message = printf_append_args(
            message,
            UInt32(num_args),
            arguments[0],
            arguments[1],
            arguments[2],
            arguments[3],
            arguments[4],
            arguments[5],
            arguments[6],
            Int32(Int(bound == args_len)),
        )


# TODO(MSTDL-1783): Use `_printf` once it passes runtime `%s` arguments on
# AMD GPUs.
@inline(.always)
def _debug_assert_emit[
    O: Origin
](message: Pointer[Byte, O], length: Int, loc: SourceLocation):
    """Writes a failed `debug_assert` message, with the failing block and
    thread, through the hostcall fprintf service.

    The format string, file name, integer arguments, and message are appended
    to the fprintf message one piece at a time.
    """
    # The compile-time interpreter cannot run the hostcall.
    if __is_run_in_comptime_interpreter:
        print("At: ", loc, ": Assert Error: ", message, sep="")
        return

    from std._gpu.primitives.id import block_idx, thread_idx

    comptime fmt = (
        "At: %s:%llu:%llu: block: [%llu,%llu,%llu] thread: [%llu,%llu,%llu]"
        " Assert Error: %s\n"
    )

    var fd = printf_begin()
    # Each appended string must carry its own nul terminator so the AMD
    # fprintf service can delimit it; `as_bytes()` omits the nul, which
    # corrupts output when a string's length is a multiple of 8.
    # `get_static_string` guarantees a trailing nul in static memory
    # just past the returned range.
    var fmt_str = get_static_string[fmt]()
    _ = printf_append_string_n(
        fd,
        Span(
            unsafe_ptr=fmt_str.as_bytes().unsafe_ptr(),
            length=fmt_str.byte_length() + 1,
        ),
        False,
    )
    # `file_name()` is a string literal, so its trailing nul lives in static
    # memory just past the range.
    var file_name = loc.file_name()
    _ = printf_append_string_n(
        fd,
        Span(
            unsafe_ptr=file_name.as_bytes().unsafe_ptr(),
            length=file_name.byte_length() + 1,
        ),
        False,
    )
    # The service takes at most 7 integer arguments per append.
    _ = printf_append_args(
        fd,
        7,
        UInt64(loc.line()),
        UInt64(loc.column()),
        UInt64(block_idx.x),
        UInt64(block_idx.y),
        UInt64(block_idx.z),
        UInt64(thread_idx.x),
        UInt64(thread_idx.y),
        0,
    )
    _ = printf_append_args(fd, 1, UInt64(thread_idx.z), 0, 0, 0, 0, 0, 0, 0)
    _ = printf_append_string_n(
        fd, Span(unsafe_ptr=message, length=length), True
    )


struct HIPPlugin(StdPlugin):
    """`PluginHooks` implementation for AMD HIP backends."""

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["hip"]()
    comptime HookFor[target: CompilationTarget] = HIPPluginHooks[target]


struct HIPPluginHooks[target: CompilationTarget](PluginHooks):
    comptime device_kind = TargetDeviceKind.gpu

    comptime print_emit_fn: OptionalReg[
        _PrintEmitPluginHookFnType
    ] = _print_emit

    comptime global_perf_counter_ns_fn: OptionalReg[
        _GlobalPerfCounterNsPluginHookFnType
    ] = _global_perf_counter_ns

    comptime sleep_fn: OptionalReg[_SleepPluginHookFnType] = _sleep

    @staticmethod
    @inline(.always)
    def debug_assert_emit_fn[
        O: Origin
    ](message: Pointer[Byte, O], length: Int, loc: SourceLocation):
        _debug_assert_emit(message, length, loc)

    comptime _handles_debug_assert: Bool = True

    comptime printf_emit_fn[fmt: StaticString, *types: AnyType]: OptionalReg[
        _PrintfEmitPluginHookFnType[*types]
    ] = _printf_emit[fmt, *types]

    comptime unary_math_fn[
        op: _UnaryMathOp, dtype: DType, width: Int
    ]: OptionalReg[_UnaryMathPluginHookFnType] = _amdgcn_unary_math_fn[
        op, dtype
    ]()

    comptime gpu_id_fn[name: StaticString, dim: StaticString]: OptionalReg[
        _GPUIdPluginHookFnType
    ] = _amdgcn_gpu_id_fn[name, dim]()

    comptime cast_fn[src: DType, dst: DType]: OptionalReg[
        _CastPluginHookFnType[dst]
    ] = _amdgcn_cast_fn[src, dst]()

    comptime native_float8_cast[
        src: DType, dst: DType
    ]: Bool = _amdgcn_native_float8_cast[src, dst, Self.target]()

    comptime has_native_float8: Bool = True
