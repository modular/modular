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
from std.ffi import CStringSpan, external_call
from std.io.file_descriptor import RawFileDescriptor
from std.math.math import _UnaryMathOp
from std.io.io import _printf
from std.reflection.location import SourceLocation
from std.sys._assembly import inlined_assembly
from std.sys.intrinsics import PrefetchOptions, _PrefetchPluginHookFnType

from .cuda_cast import (
    _nvptx_cast_fn,
    _nvptx_has_native_float8,
    _nvptx_native_float8_cast,
)
from .cuda_gpu_id import _nvptx_gpu_id_fn
from .cuda_math import _nvptx_unary_math_fn
from .cuda_time import _global_perf_counter_ns, _sleep


def _print_emit[O: Origin](cstr: CStringSpan[O], file: RawFileDescriptor):
    """Writes `print()` output from an NVIDIA kernel through device `printf`.

    Device `printf` has a single output stream, so `file` is ignored.
    """
    _printf["%s"](cstr.ptr())


@inline(.always)
def _debug_assert_emit[
    O: Origin
](message: Pointer[Byte, O], length: Int, loc: SourceLocation):
    """Writes a failed `debug_assert` message, with the failing block and
    thread, through device `printf`.
    """
    # `_debug_assert_msg` hands every assertion to this hook, including ones
    # evaluated at compile time, so it keeps the stdlib's compile-time message.
    if __is_run_in_comptime_interpreter:
        print("At: ", loc, ": Assert Error: ", message, sep="")
        return

    from std._gpu.primitives.id import block_idx, thread_idx

    _printf[
        "At: %s:%llu:%llu: block: [%llu,%llu,%llu] thread:"
        " [%llu,%llu,%llu] Assert Error: %s\n"
    ](
        loc.file_name().as_c_string_span(),
        loc.line(),
        loc.column(),
        UInt(block_idx.x),
        UInt(block_idx.y),
        UInt(block_idx.z),
        UInt(thread_idx.x),
        UInt(thread_idx.y),
        UInt(thread_idx.z),
        message,
    )


@inline(.always)
def _printf_emit[
    fmt: StaticString, *types: AnyType
](*args: *types, file: RawFileDescriptor):
    """Formats `args` into `fmt` with device `printf`.

    Device `printf` has a single output stream, so `file` is ignored.
    """
    # The argument pack will contain references for each value in the pack,
    # but we want to pass their values directly into the C printf call. Load
    # all the members of the pack.
    var loaded_pack = args.get_loaded_kgen_pack()

    _ = external_call["vprintf", Int32](
        get_static_string[fmt]().as_c_string_span(),
        Pointer(to=loaded_pack),
    )


@inline(.always)
def _prefetch[
    address_space: AddressSpace
](addr: Pointer[NoneType, ImmUntrackedOrigin, address_space=address_space]):
    """Prefetches `addr` into the L2 cache."""
    # PTX `prefetch` takes a global, local, or generic address. Other address
    # spaces, such as shared memory, have no form, and dropping a hint keeps
    # the program correct.
    comptime match address_space:
        case .GLOBAL:
            _prefetch_l2["prefetch.global.L2 [$0];", "l,~{memory}"](addr)
        case .LOCAL:
            _prefetch_l2["prefetch.local.L2 [$0];", "r,~{memory}"](addr)
        case .GENERIC:
            _prefetch_l2["prefetch.L2 [$0];", "l,~{memory}"](addr)
        case _:
            pass


@inline(.always)
def _prefetch_l2[
    asm: StaticString, constraints: StaticString
](addr: Pointer[NoneType, ImmUntrackedOrigin, address_space=_]):
    inlined_assembly[
        asm,
        NoneType,
        constraints=constraints,
        has_side_effect=True,
    ](addr)


struct CUDAPlugin(StdPlugin):
    """`PluginHooks` implementation for NVIDIA CUDA backends."""

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["cuda"]()
    comptime HookFor[target: CompilationTarget] = CUDAPluginHooks[target]


struct CUDAPluginHooks[target: CompilationTarget](PluginHooks):
    comptime device_kind = TargetDeviceKind.gpu

    comptime print_emit_fn: OptionalReg[
        _PrintEmitPluginHookFnType
    ] = _print_emit

    comptime global_perf_counter_ns_fn: OptionalReg[
        _GlobalPerfCounterNsPluginHookFnType
    ] = _global_perf_counter_ns

    comptime sleep_fn: OptionalReg[_SleepPluginHookFnType] = _sleep

    comptime atomic_float_minmax_via_int: Bool = True

    comptime prefetch_fn[
        address_space: AddressSpace, params: PrefetchOptions
    ]: OptionalReg[_PrefetchPluginHookFnType[address_space]] = _prefetch[
        address_space
    ]

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
    ]: OptionalReg[_UnaryMathPluginHookFnType] = _nvptx_unary_math_fn[
        op, dtype, Self.target
    ]()

    comptime gpu_id_fn[name: StaticString, dim: StaticString]: OptionalReg[
        _GPUIdPluginHookFnType
    ] = _nvptx_gpu_id_fn[name, dim, Self.target]()

    comptime cast_fn[src: DType, dst: DType]: OptionalReg[
        _CastPluginHookFnType[dst]
    ] = _nvptx_cast_fn[src, dst, Self.target]()

    comptime native_float8_cast[
        src: DType, dst: DType
    ]: Bool = _nvptx_native_float8_cast[src, dst, Self.target]()

    comptime has_native_float8: Bool = _nvptx_has_native_float8[Self.target]()
