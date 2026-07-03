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
from std.gpu import PDLLevel
from std.gpu.host import DeviceContext

from std.utils.index import IndexList
from std.math.math import _ExpPluginHookFnType, _TanhPluginHookFnType
from std.memory.stack_allocation import _StackAllocationPluginHookFnType
from std.memory.unsafe_pointer import _UnsafeDanglingPluginHookFnType
from std.io.io import _PrintEmitPluginHookFnType
from std.algorithm.reduction import _ReduceGeneratorPluginHookFnType


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
        _StackAllocationPluginHookFnType[address_space]
    ]

    comptime address_space_fn[name: StaticString]: Optional[AddressSpace]
    """Target-specific named address-space lookup.

    Resolves an address-space *name* that has no built-in constant on
    `AddressSpace` (the GPU spaces `GENERIC`/`GLOBAL`/`SHARED`/...) to its
    target-specific value — for example an accelerator-specific scratchpad
    space. `AddressSpace.<NAME>` consults this hook for any such name; leaving it
    `None` (the default) makes the name a compile-time error. This keeps the
    set of valid address-space names open and target-extensible rather than a
    fixed portable enum.

    Parameters:
        name: The address-space name being looked up.

    Returns:
        The backend's `AddressSpace` for `name`, or `None` if the backend does
        not define it.
    """

    comptime unsafe_dangling_fn: Optional[_UnsafeDanglingPluginHookFnType]
    """`UnsafePointer.unsafe_dangling()` address override.

    Parameters:
        alignment: The natural alignment of the pointee type, which the
            stdlib default uses as the dangling address.

    Returns:
        The raw integer address used to construct the dangling pointer.
    """

    comptime print_emit_fn: Optional[_PrintEmitPluginHookFnType]
    """Plugin hook for emitting a `print()` UTF-8 byte buffer to a file
    descriptor."""

    comptime reduce_generator_fn[target: StaticString]: Optional[
        _ReduceGeneratorPluginHookFnType
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
        _StackAllocationPluginHookFnType[address_space]
    ] = None

    comptime address_space_fn[name: StaticString]: Optional[AddressSpace] = None

    comptime unsafe_dangling_fn: Optional[
        _UnsafeDanglingPluginHookFnType
    ] = None

    comptime print_emit_fn: Optional[_PrintEmitPluginHookFnType] = None

    comptime reduce_generator_fn[target: StaticString]: Optional[
        _ReduceGeneratorPluginHookFnType
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
