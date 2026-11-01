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

from std._gpu.primitives.id import _GPUIdPluginHookFnType, _is_cluster_id
from std.math.uutils import ufloordiv
from std._plugin._trait import (
    TargetDeviceKind,
    PluginHooks,
    StdPlugin,
    _BinaryMathPluginHookFnType,
    _PrintEmitPluginHookFnType,
    _UnaryMathPluginHookFnType,
)
from std.atomic import Ordering
from std.collections import OptionalReg
from std.sys.info import CompilationTarget
from std.collections.string.string_span import (
    _get_kgen_string,
    get_static_string,
)
from std.ffi import CStringSpan
from std.io.file_descriptor import RawFileDescriptor
from std.math.math import _BinaryMathOp, _UnaryMathOp
from std.sys import bit_width_of, llvm_intrinsic
from std.sys._metal_print import _metal_print_write
from std.sys.intrinsics import PrefetchOptions, _PrefetchPluginHookFnType


def _print_emit[O: Origin](cstr: CStringSpan[O], file: RawFileDescriptor):
    """Writes `print()` output from a Metal kernel to the os_log stream.

    Metal kernels have a single os_log stream, so `file` is ignored.
    """
    _metal_print_write(StringSlice(unsafe_from_utf8=cstr.as_bytes_with_nul()))


@inline(.always)
def _drop_prefetch[
    address_space: AddressSpace
](addr: Pointer[NoneType, ImmUntrackedOrigin, address_space=address_space]):
    # Metal has no prefetch instruction, and dropping a hint keeps the program
    # correct.
    pass


@inline(.always)
def _air_read[intrinsic: StaticString, dtype: DType]() -> Int:
    return Int(
        llvm_intrinsic[intrinsic, Scalar[dtype], has_side_effect=False]()
    )


def _air_gpu_id_fn[
    name: StaticString, dim: StaticString
]() -> OptionalReg[_GPUIdPluginHookFnType]:
    """Returns the AIR read of `name` along `dim`, or `None` for a thread
    block cluster id, which Metal does not have."""
    comptime if _is_cluster_id(name):
        return None
    else:
        return comptime (
            OptionalReg[_GPUIdPluginHookFnType](_air_gpu_id[name, dim])
        )


@inline(.always)
def _air_gpu_id[name: StaticString, dim: StaticString]() -> Int:
    """Reads the GPU index or size `name` along `dim` from its `air.*`
    intrinsic.
    """
    comptime if name == "lane_id":
        return _air_read["llvm.air.thread_index_in_simdgroup", DType.int32]()
    elif name == "thread_idx":
        return _air_read[
            get_static_string[
                "llvm.air.thread_position_in_threadgroup.", dim
            ](),
            DType.uint32,
        ]()
    elif name == "block_idx":
        return _air_read[
            get_static_string["llvm.air.threadgroup_position_in_grid.", dim](),
            DType.uint32,
        ]()
    elif name == "block_dim":
        return _air_read[
            get_static_string["llvm.air.threads_per_threadgroup.", dim](),
            DType.int32,
        ]()
    else:
        comptime assert name == "grid_dim", "unknown GPU id"
        # Metal reports the grid size in threads; the other backends count
        # blocks.
        return ufloordiv(
            _air_read[
                get_static_string["llvm.air.threads_per_grid.", dim](),
                DType.int32,
            ](),
            _air_gpu_id["block_dim", dim](),
        )


comptime _MathFn = OptionalReg[_UnaryMathPluginHookFnType]


def _air_unary_math_fn[op: _UnaryMathOp, dtype: DType]() -> _MathFn:
    """Returns the AIR lowering of the `std.math` function `op` for `dtype`,
    or `None` to keep the stdlib's own lowering."""
    comptime match op:
        case .exp2:
            return comptime (
                _MathFn(_air_unary_math[name="exp2"]) if dtype
                in (DType.float16, DType.float32) else _MathFn(None)
            )
        case .sqrt:
            return comptime (_MathFn(_air_unary_math[name="sqrt"]))
        case .rsqrt:
            return comptime (_MathFn(_air_unary_math[name="rsqrt"]))
        case .sin:
            return comptime (_MathFn(_air_unary_math[name="sin"]))
        case .cos:
            return comptime (_MathFn(_air_unary_math[name="cos"]))
        case .exp | .log | .log2 | .recip | .tanh:
            return None


def _air_binary_math_fn[
    op: _BinaryMathOp
]() -> OptionalReg[_BinaryMathPluginHookFnType]:
    """Returns the AIR lowering of the two-argument function `op`."""
    comptime match op:
        case .pow:
            return comptime (
                OptionalReg[_BinaryMathPluginHookFnType](
                    _air_binary_math[name="pow"]
                )
            )


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


def _air_binary_math[
    dtype: DType, width: SIMDLength, //, name: StaticString
](x: SIMD[dtype, width], y: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Lowers the two-argument function `name` to its `air.*` intrinsic.

    As in `_air_unary_math`, bf16 inputs round-trip through f32.
    """
    comptime if dtype == .bfloat16:
        return _air_binary_math[name=name](
            x.cast[.float32](), y.cast[.float32]()
        ).cast[dtype]()
    return llvm_intrinsic[
        get_static_string["llvm.air.", name](),
        type_of(x),
        has_side_effect=False,
    ](x, y)


struct MetalPlugin(StdPlugin):
    """`PluginHooks` implementation for Apple Metal backends."""

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["metal"]()
    comptime HookFor[target: CompilationTarget] = MetalPluginHooks[target]


struct MetalPluginHooks[target: CompilationTarget](PluginHooks):
    comptime device_kind = TargetDeviceKind.gpu

    comptime gpu_id_fn[name: StaticString, dim: StaticString]: OptionalReg[
        _GPUIdPluginHookFnType
    ] = _air_gpu_id_fn[name, dim]()

    comptime print_emit_fn: OptionalReg[
        _PrintEmitPluginHookFnType
    ] = _print_emit

    comptime supports_atomic_rmw[dtype: DType]: Bool = (
        bit_width_of[dtype, target=Self.target]() == 32
    )

    comptime unary_math_fn[
        op: _UnaryMathOp, dtype: DType, width: Int
    ]: OptionalReg[_UnaryMathPluginHookFnType] = _air_unary_math_fn[op, dtype]()

    comptime binary_math_fn[
        op: _BinaryMathOp, dtype: DType, width: Int
    ]: OptionalReg[_BinaryMathPluginHookFnType] = _air_binary_math_fn[op]()

    comptime supports_float64: Bool = False

    comptime flushes_subnormals[dtype: DType]: Bool = dtype == DType.float32

    comptime lowers_subvector_intrinsics: Bool = False

    comptime has_generic_address_space: Bool = False

    comptime prefetch_fn[
        address_space: AddressSpace, params: PrefetchOptions
    ]: OptionalReg[_PrefetchPluginHookFnType[address_space]] = _drop_prefetch[
        address_space
    ]

    comptime default_atomic_ordering: Ordering = Ordering.RELAXED
