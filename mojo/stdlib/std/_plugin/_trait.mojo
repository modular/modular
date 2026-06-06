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

from std.reflection.location import SourceLocation
from std.sys.info import _TargetType, _current_target
from std.io import FileDescriptor
from std.ffi import CStringSlice
from std.gpu import PDLLevel
from std.gpu.host import DeviceContext

from std.utils.index import Index, IndexList, StaticTuple
from std.math.math import _ExpPluginHookFnType, _TanhPluginHookFnType


trait PluginHooks:
    """Compile-time hook interface for pluggable stdlib behavior.

    Most hooks are `comptime Optional[Callable]` fields; call sites invoke
    `comptime if CurrentPlugin.xxx_fn: return comptime(CurrentPlugin.xxx_fn.value())(...)`,
    so implementors that leave a hook at `None` add zero cost.

    A few hooks (`abort_fn`, `debug_assert_emit_fn`) are required
    `@staticmethod` trait methods rather than `Optional` fields, because
    their dispatch sites lie on `Optional.value()`'s own instantiation
    path — an `Optional` field would re-enter that template via its own
    `debug_assert` and deadlock comptime instantiation.
    """

    comptime exp_fn: Optional[_ExpPluginHookFnType]
    """Elementwise exponential override.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: The input SIMD vector.

    Returns:
        Elementwise `exp(x)` computed on the vendor backend.
    """

    comptime tanh_fn[dtype: DType, width: Int]: Optional[_TanhPluginHookFnType]
    """Elementwise hyperbolic tangent override.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: The input SIMD vector.

    Returns:
        Elementwise `tanh(x)` computed on the vendor backend.
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

    comptime unsafe_dangling_fn: Optional[def[alignment: Int]() thin -> Int]
    """`UnsafePointer.unsafe_dangling()` address override.

    Parameters:
        alignment: The natural alignment of the pointee type, which the
            stdlib default uses as the dangling address.

    Returns:
        The raw integer address used to construct the dangling pointer.
    """

    comptime print_emit_fn: Optional[PrintEmitFnType]
    """Plugin hook for emitting a `print()` UTF-8 byte buffer to a file
    descriptor."""

    comptime reduce_generator_fn[target: StaticString]: Optional[
        ReduceGeneratorFnType
    ]

    @staticmethod
    def abort_fn():
        """`abort()` override, called before the default trap. If the hook
        doesn't return (e.g. via `longjmp`), the trap is dead code."""
        ...

    @staticmethod
    def debug_assert_emit_fn[
        O: Origin
    ](message: UnsafePointer[Byte, O], length: Int, loc: SourceLocation):
        """Assertion-message emitter for targets without a usable `_printf`.

        Parameters:
            O: The origin of the message pointer.

        Args:
            message: Pointer to the nul-terminated message bytes.
            length: Length in bytes (excluding the trailing nul).
            loc: Source location of the failing assertion.

        Only invoked when `_handles_debug_assert` is `True`.
        """
        ...

    comptime _handles_debug_assert: Bool
    """If `True`, `_debug_assert_msg` dispatches to `debug_assert_emit_fn`
    and comptime-elides its `_printf` fallback. Required because the
    fallback's transitive `Optional.value()` → `debug_assert` recurses
    back through `_debug_assert_msg` and deadlocks instantiation when
    assertions are enabled."""

    @staticmethod
    def elementwise_fn[
        target: StaticString,
        rank: Int,
        simd_width: Int,
        *,
        pdl_level: PDLLevel = PDLLevel.ON,
    ](
        func: Some[
            def[
                width: Int, rank: Int, alignment: Int = 1
            ](IndexList[rank]) -> None
        ],
        shape: IndexList[rank, ...],
        ctx: DeviceContext,
    ) raises:
        """Per-target plugin hook for `elementwise[..., target=target]`.

        Parameters:
            target: The dispatch target (e.g. `"cpu"`, `"gpu"`, `"npu"`).
            rank: The rank of the work domain.
            simd_width: The SIMD lane count for bulk invocations.
            pdl_level: PDL level for overlap control.

        Args:
            func: The body closure to invoke per index.
            shape: The shape of the work domain.
            ctx: The device context to dispatch on.
        """
        ...

    comptime _handles_elementwise[target: StaticString]: Bool
    """If `True` for a given `target`, `_elementwise_impl` dispatches to
    `elementwise_fn[target, ...]`."""


# FIXME(MOCO-3871): Alias is to workaround function type comparison bug.
comptime PrintEmitFnType = def[O: Origin](
    cstr: CStringSlice[O],
    file_value: FileDescriptor,
) thin -> None


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

    comptime exp_fn: Optional[_ExpPluginHookFnType] = None

    comptime tanh_fn[dtype: DType, width: Int]: Optional[
        _TanhPluginHookFnType
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

    comptime unsafe_dangling_fn: Optional[
        def[alignment: Int]() thin -> Int
    ] = None

    comptime print_emit_fn: Optional[PrintEmitFnType] = None

    comptime reduce_generator_fn[target: StaticString]: Optional[
        ReduceGeneratorFnType
    ] = None

    @staticmethod
    def abort_fn():
        pass

    @staticmethod
    def debug_assert_emit_fn[
        O: Origin
    ](message: UnsafePointer[Byte, O], length: Int, loc: SourceLocation):
        pass

    comptime _handles_debug_assert: Bool = False

    @staticmethod
    def elementwise_fn[
        target: StaticString,
        rank: Int,
        simd_width: Int,
        *,
        pdl_level: PDLLevel = PDLLevel.ON,
    ](
        func: Some[
            def[
                width: Int, rank: Int, alignment: Int = 1
            ](IndexList[rank]) -> None
        ],
        shape: IndexList[rank, ...],
        ctx: DeviceContext,
    ) raises:
        pass

    comptime _handles_elementwise[target: StaticString]: Bool = False
