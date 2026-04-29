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

from std.sys.info import _TargetType, _current_target

from std.utils.index import Index, IndexList, StaticTuple


# Named function-type alias so trait and DefaultPlugin reference the *same*
# nominal type for `print_emit_fn` — Mojo's trait conformance treats freshly
# spelled-out `def[O: Origin](...)` types as distinct even when syntactically
# identical, so duplicating the signature in both places fails to conform.
#
# `file_value` is `FileDescriptor.value` (raw integer fd) rather than
# `FileDescriptor` itself — referencing `FileDescriptor` here would cycle
# through `std.io`, which imports `CurrentPlugin` from this package.
comptime _PrintEmitFn = def[O: Origin](
    ptr: UnsafePointer[UInt8, O],
    length: Int,
    file_value: Int,
) thin -> None


trait PluginHooks:
    """Compile-time hook interface for pluggable stdlib behavior.

    Each hook is a `comptime Optional[Callable]` field. Call sites invoke
    `comptime if CurrentPlugin.xxx_fn: return comptime(CurrentPlugin.xxx_fn.value())(...)`,
    so implementors that leave a hook at `None` add zero cost.
    """

    comptime exp_fn: Optional[
        def[
            dtype: DType, width: Int, //
        ](SIMD[dtype, width]) thin -> SIMD[dtype, width]
    ]
    """Elementwise exponential override.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: The input SIMD vector.

    Returns:
        Elementwise `exp(x)` computed on the vendor backend.
    """

    comptime stack_allocation_fn[address_space: AddressSpace]: Optional[
        def[
            count: Int,
            type: AnyType,
            /,
            name: Optional[StaticString],
            alignment: Int,
        ]() thin -> UnsafePointer[
            type, MutExternalOrigin, address_space=address_space
        ]
    ]

    comptime reduce_generator_fn[target: StaticString]: Optional[
        ReduceGeneratorFnType
    ]

    comptime print_emit_fn: Optional[_PrintEmitFn]
    """Plugin hook for emitting a `print()` UTF-8 byte buffer to a file
    descriptor."""


comptime ReduceGeneratorFnType = (
    def[
        num_reductions: Int,
        init_type: DType,
        input_0_fn: def[dtype: DType, width: Int, rank: Int](
            IndexList[rank]
        ) capturing[_] -> SIMD[dtype, width],
        output_0_fn: def[dtype: DType, width: Int, rank: Int](
            IndexList[rank], StaticTuple[SIMD[dtype, width], num_reductions]
        ) capturing[_] -> None,
        reduce_function: def[ty: DType, width: Int, reduction_idx: Int](
            SIMD[ty, width], SIMD[ty, width]
        ) capturing[_] -> SIMD[ty, width],
    ](
        shape: IndexList[_, element_type=DType.int64],
        init: StaticTuple[Scalar[init_type], num_reductions],
        reduce_dim: Int,
    ) thin
)


# ===-----------------------------------------------------------------------===#
# DefaultPlugin
# ===-----------------------------------------------------------------------===#


struct DefaultPlugin(PluginHooks):
    """Default `PluginHooks` implementation used when no plugin is active."""

    comptime exp_fn: Optional[
        def[
            dtype: DType, width: Int, //
        ](SIMD[dtype, width]) thin -> SIMD[dtype, width]
    ] = None

    comptime stack_allocation_fn[address_space: AddressSpace]: Optional[
        def[
            count: Int,
            type: AnyType,
            /,
            name: Optional[StaticString],
            alignment: Int,
        ]() thin -> UnsafePointer[
            type, MutExternalOrigin, address_space=address_space
        ]
    ] = None

    comptime reduce_generator_fn[target: StaticString]: Optional[
        ReduceGeneratorFnType
    ] = None

    comptime print_emit_fn: Optional[_PrintEmitFn] = None
