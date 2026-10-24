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

from std.atomic import Ordering
from std.collections import OptionalReg
from std.reflection.location import SourceLocation

from std.utils.coord import Coord
from std.utils.index import IndexList
from std.math.math import (
    _BinaryMathOp,
    _BinaryMathPluginHookFnType,
    _UnaryMathOp,
    _UnaryMathPluginHookFnType,
)
from std.memory.stack_allocation import _StackAllocationPluginHookFnType
from std.simd import _CastPluginHookFnType
from std.memory.unsafe_pointer import _UnsafeDanglingPluginHookFnType
from std.sys.intrinsics import PrefetchOptions, _PrefetchPluginHookFnType
from std._gpu.primitives.id import _GPUIdPluginHookFnType
from std.io.io import _PrintEmitPluginHookFnType, _PrintfEmitPluginHookFnType
from std.time.time import (
    _GlobalPerfCounterNsPluginHookFnType,
    _SleepPluginHookFnType,
)
from std.collections.string.string_span import (
    _get_kgen_string,
)
from std.utils import StaticTuple

from .selector import Plugin
from std.sys.info import CompilationTarget

comptime _ReduceGeneratorPluginHookFnType = (
    def[
        num_reductions: Int,
        init_type: DType,
        InputFnType: ImplicitlyCopyable
        & RegisterPassable
        & def[dtype: DType, width: Int](Coord) -> SIMD[dtype, width],
        OutputFnType: ImplicitlyCopyable
        & RegisterPassable
        & def[dtype: DType, width: SIMDLength](
            Coord, StaticTuple[SIMD[dtype, width], num_reductions]
        ) -> None,
        ReduceFnType: ImplicitlyCopyable
        & RegisterPassable
        & def[ty: DType, width: SIMDLength, reduction_idx: Int](
            SIMD[ty, width], SIMD[ty, width]
        ) -> SIMD[ty, width],
    ](
        shape: IndexList[_, element_type=DType.int64],
        init: StaticTuple[Scalar[init_type], num_reductions],
        reduce_dim: Int,
        input_0_fn: InputFnType,
        output_0_fn: OutputFnType,
        reduce_function: ReduceFnType,
    ) thin
)
"""Plugin-hook signature for `PluginHooks.reduce_generator_fn`; keep in sync with `_reduce_generator`."""


__enum TargetDeviceKind(ImplicitlyCopyable):
    """The kind of device a compilation target is.

    `CompilationTarget.is_gpu()` and `is_accelerator()` are derived from it.
    Generic code should keep asking those questions rather than match on the
    kind, and should not branch on an `other` kind's name, which is for
    diagnostics.
    """

    case cpu
    """A host CPU."""
    case gpu
    """A SIMT GPU, where the thread, block and warp primitives apply."""
    case other(StaticString, Bool)
    """Any other device, such as an NPU: its name, then whether it is an
    accelerator rather than a host."""


trait StdPlugin(Plugin):
    comptime HookFor[target: CompilationTarget]: PluginHooks


trait PluginHooks:
    """Compile-time hook interface for pluggable stdlib behavior.

    Most hooks are `comptime OptionalReg[Callable]` fields; call sites invoke
    `comptime if CurrentPlugin.xxx_fn: return comptime(CurrentPlugin.xxx_fn.value())(...)`,
    so implementors that leave a hook at `None` add zero cost.

    A few hooks (`abort_fn`, `debug_assert_emit_fn`) are required
    `@staticmethod` trait methods rather than `OptionalReg` fields, because
    their dispatch sites lie on `OptionalReg.value()`'s own instantiation
    path — an `OptionalReg` field would re-enter that template via its own
    `debug_assert` and deadlock comptime instantiation.

    An implementation keeps each hook's logic in a module-level function and
    binds it in its `PluginHooks` struct, either as a `comptime` field or as a
    static method that forwards to it, so the struct reads as a list of the
    hooks the plugin overrides.
    """

    comptime device_kind: TargetDeviceKind
    """The kind of device the target is.

    `CompilationTarget.is_gpu()`, `is_accelerator()` and `device_kind()` read
    this. It has no default: most plugins are for accelerators, and a missing
    kind would otherwise make their targets silently count as host CPUs. It is
    a hook rather than a property of the plugin, so one plugin can serve
    targets of different kinds.
    """

    # ===------------------------------------------------------------------=== #
    # Math & Algorithm Operations
    # ===------------------------------------------------------------------=== #

    comptime unary_math_fn[
        op: _UnaryMathOp, dtype: DType, width: Int
    ]: OptionalReg[_UnaryMathPluginHookFnType] = None
    """Elementwise override for the `std.math` function `op`.

    Consulted by `exp`, `exp2`, `log`, `log2`, `sqrt`, `rsqrt`, `recip`,
    `sin`, `cos`, and `tanh` for floating-point inputs, before the stdlib
    converts the input to another `dtype`, so the hook sees the caller's
    `dtype`. Returning `None` for an `op`, `dtype`, and `width` keeps the
    stdlib's own lowering for that case. The hook is not called when the
    function is evaluated at compile time, because the compile-time
    interpreter cannot run backend intrinsics.

    An implementation matches on `op` without a wildcard case, naming the
    functions it leaves to the stdlib, so that adding a function to
    `_UnaryMathOp` makes every plugin decide how to lower it.

    Parameters:
        op: The `std.math` function being lowered, such as `.sqrt`.
        dtype: The `dtype` of the input and output SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        x: The input SIMD vector.

    Returns:
        Elementwise `op(x)` computed on the vendor backend.
    """

    comptime binary_math_fn[
        op: _BinaryMathOp, dtype: DType, width: Int
    ]: OptionalReg[_BinaryMathPluginHookFnType] = None
    """Elementwise override for the two-argument function `op`.

    Consulted by `pow` when both arguments are floating point, under the same
    rules as `unary_math_fn`: the hook sees the caller's `dtype`, `None` keeps
    the stdlib's own lowering, and compile-time evaluation skips the hook.

    Parameters:
        op: The function being lowered, such as `.pow`.
        dtype: The `dtype` of the inputs and output SIMD vectors.
        width: The width of the inputs and output SIMD vectors.

    Args:
        x: The first input SIMD vector.
        y: The second input SIMD vector.

    Returns:
        Elementwise `op(x, y)` computed on the vendor backend.
    """

    comptime supports_float64: Bool = True
    """Whether the target supports float64 arithmetic.

    Where it is `False`, `std.math` functions that compute float64 results in
    software, such as `log` and `atan2`, reject float64 inputs with a
    compile-time error instead of emitting float64 code the backend cannot
    lower.
    """

    comptime flushes_subnormals[dtype: DType]: Bool = False
    """Whether the target flushes `dtype` subnormals to zero.

    A target that does reads subnormal operands as zero and flushes subnormal
    results of arithmetic and conversions. `std.math` then rebuilds those
    values from their bits so subnormal results are not lost.

    Parameters:
        dtype: The floating-point `dtype` being queried.
    """

    comptime reduce_generator_fn: OptionalReg[
        _ReduceGeneratorPluginHookFnType
    ] = None

    # ===------------------------------------------------------------------=== #
    # SIMD Operations
    # ===------------------------------------------------------------------=== #

    comptime lowers_subvector_intrinsics: Bool = True
    """Whether the backend lowers LLVM's subvector intrinsics.

    These are `llvm.vector.extract`, `llvm.vector.insert`, and
    `llvm.vector.splice.{left,right}`, which `SIMD.slice`, `SIMD.insert`,
    `SIMD.rotate_{left,right}`, and `SIMD.shift_{left,right}` use. Where this
    is `False`, those methods build the result from element accesses and
    `shuffle` instead.
    """

    comptime cast_fn[src: DType, dst: DType]: OptionalReg[
        _CastPluginHookFnType[dst]
    ] = None
    """Override for converting `src` elements to `dst` with `SIMD.cast`.

    Consulted before the stdlib's own conversion, so the hook sees the
    caller's `src` and `dst`. The float8_e8m0fnu conversion helpers that
    kernels call directly consult it too. Returning `None` for a `src` and
    `dst` keeps the stdlib's own conversion. The hook is not called when the
    cast is evaluated at compile time, because the compile-time interpreter
    cannot run backend instructions.

    Parameters:
        src: The `dtype` of the input SIMD vector.
        dst: The `dtype` of the output SIMD vector.

    Args:
        x: The input SIMD vector.

    Returns:
        `x` converted elementwise to `dst` on the vendor backend.
    """

    comptime native_float8_cast[src: DType, dst: DType]: Bool = False
    """Whether the backend converts `src` to `dst` with a native instruction,
    where one of them is a float8 type.

    Where this is `True`, `SIMD.cast` lowers the conversion directly, and
    the backend selects the instruction. Elsewhere the stdlib converts in
    software. A conversion from float8 to float32 that is not native goes
    through float16 when that one is.

    Parameters:
        src: The `dtype` converted from.
        dst: The `dtype` converted to.
    """

    comptime has_native_float8: Bool = False
    """Whether the GPU supports float8 natively, for example in its matrix
    instructions."""

    # ===------------------------------------------------------------------=== #
    # Memory Operations
    # ===------------------------------------------------------------------=== #

    comptime has_generic_address_space: Bool = True
    """Whether a `GENERIC` pointer can reach memory in every address space.

    NVIDIA and AMD GPUs resolve a generic pointer's address space at run time.
    A target without a generic address space, such as Metal, cannot: a pointer
    rebuilt from an integer address does not reach threadgroup memory. Where
    this is `False`, the stdlib keeps the typed pointer instead of
    round-tripping it through an integer.
    """

    comptime prefetch_fn[
        address_space: AddressSpace, params: PrefetchOptions
    ]: OptionalReg[_PrefetchPluginHookFnType[address_space]] = None
    """Override for `prefetch` on a pointer in `address_space`.

    `prefetch` lowers to `llvm.prefetch` where this is `None`. A backend that
    cannot lower `llvm.prefetch` provides a hook that emits nothing, since a
    prefetch is only a hint.

    Parameters:
        address_space: The address space of the pointer being prefetched.
        params: The prefetch options the caller passed to `prefetch`.

    Args:
        addr: The address to prefetch.
    """

    comptime default_atomic_ordering: Ordering = Ordering.SEQUENTIAL
    """The ordering `Atomic` read-modify-write and compare-exchange
    operations use when the caller does not name one.

    A target whose backend rejects sequentially consistent atomics, such as
    Metal, sets this to the strongest ordering it lowers. Loads, stores, and
    fences default to `Ordering.SEQUENTIAL` regardless.
    """

    comptime atomic_float_minmax_via_int: Bool = False
    """Whether `Atomic.max` and `Atomic.min` on floating-point values use
    integer atomics.

    Where it is `True`, `Atomic.max` with a non-negative operand is a signed
    integer atomic max on the value's bits, and with a negative operand an
    unsigned integer atomic min; `Atomic.min` swaps the two. Each update is
    then one integer atomic instruction. Where it is `False`, the
    floating-point read-modify-write goes to the backend, which may expand it
    into a compare-exchange loop.
    """

    comptime stack_allocation_fn[address_space: AddressSpace]: OptionalReg[
        _StackAllocationPluginHookFnType[address_space]
    ] = None

    comptime supports_atomic_rmw[dtype: DType]: Bool = True
    """Whether the backend lowers `Atomic` read-modify-write and
    compare-exchange operations on `dtype`.

    Metal, for example, only lowers them on 32-bit types.
    """

    comptime address_space_fn[name: StaticString]: OptionalReg[
        AddressSpace
    ] = None
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

    comptime unsafe_dangling_fn: OptionalReg[
        _UnsafeDanglingPluginHookFnType
    ] = None
    """`Pointer.unsafe_dangling()` address override.

    Parameters:
        alignment: The natural alignment of the pointee type, which the
            stdlib default uses as the dangling address.

    Returns:
        The raw integer address used to construct the dangling pointer.
    """

    # ===------------------------------------------------------------------=== #
    # GPU Operations
    # ===------------------------------------------------------------------=== #

    comptime gpu_id_fn[name: StaticString, dim: StaticString]: OptionalReg[
        _GPUIdPluginHookFnType
    ] = None
    """Reads a GPU execution index or size.

    `lane_id`, `thread_idx`, `block_idx`, `block_dim`, `grid_dim`,
    `cluster_dim`, `cluster_idx`, and `block_id_in_cluster` read through this
    hook, and raise an unsupported-target error on a GPU where it is `None`.
    A plugin returns `None` for the cluster ids when its GPU has no thread
    block clusters. The hook must inline into the kernel.

    Parameters:
        name: One of `"lane_id"`, `"thread_idx"`, `"block_idx"`,
            `"block_dim"`, `"grid_dim"`, `"cluster_dim"`, `"cluster_idx"`,
            or `"block_id_in_cluster"`.
        dim: `"x"`, `"y"`, or `"z"`; empty for `"lane_id"`.
    """

    # ===------------------------------------------------------------------=== #
    # IO & System Operations
    # ===------------------------------------------------------------------=== #

    comptime print_emit_fn: OptionalReg[_PrintEmitPluginHookFnType] = None
    """Plugin hook for emitting a `print()` UTF-8 byte buffer to a file
    descriptor."""

    comptime printf_emit_fn[fmt: StaticString, *types: AnyType]: OptionalReg[
        _PrintfEmitPluginHookFnType[*types]
    ] = None
    """Override for `_printf`, which formats its arguments into `fmt` with the
    target's own `printf`.

    Where this is `None`, `_printf` calls the C `fprintf` on a CPU and writes
    `fmt` through `print_emit_fn` on a GPU.

    Parameters:
        fmt: The C `printf` format string.
        types: The types of the values to format.

    Args:
        args: The values to format into `fmt`.
        file: The file descriptor `_printf` was asked to write to.
    """

    comptime global_perf_counter_ns_fn: OptionalReg[
        _GlobalPerfCounterNsPluginHookFnType
    ] = None
    """Reads the device-wide nanosecond timer behind `global_perf_counter_ns()`.

    On GPU targets, `perf_counter_ns()`, `perf_counter()`, and `monotonic()`
    read this timer too. A GPU target that leaves it `None` rejects those
    functions at compile time; CPU targets use the host's monotonic clock.

    Returns:
        The current time in nanoseconds, from a timer shared by every block on
        the device.
    """

    comptime sleep_fn: OptionalReg[_SleepPluginHookFnType] = None
    """Suspends the calling thread for `time.sleep()`.

    `time.sleep()` returns before calling the hook when `sec` is not
    positive. A GPU target that leaves it `None` rejects `time.sleep()` at
    compile time; CPU targets call the C library's `nanosleep`.

    Args:
        sec: The number of seconds to sleep for, always positive.
    """

    @staticmethod
    def abort_fn():
        """`abort()` override, called before the default trap. If the hook
        doesn't return (e.g. via `longjmp`), the trap is dead code.

        The default is a no-op (the stdlib trap runs).
        """
        pass

    @staticmethod
    def debug_assert_emit_fn[
        O: Origin
    ](message: Pointer[Byte, O], length: Int, loc: SourceLocation):
        """Emits a failed assertion's message in the target's own format.

        GPU targets use it to add the failing block and thread ids, and
        targets without a usable `_printf` use it in place of the generic
        `_printf` fallback.

        Parameters:
            O: The origin of the message pointer.

        Args:
            message: Pointer to the nul-terminated message bytes.
            length: Length in bytes (excluding the trailing nul).
            loc: Source location of the failing assertion.

        Only invoked when `_handles_debug_assert` is `True`; the default is
        never called and is a no-op.
        """
        pass

    comptime _handles_debug_assert: Bool = False
    """If `True`, `_debug_assert_msg` dispatches to `debug_assert_emit_fn`
    and comptime-elides its `_printf` fallback. A target without a usable
    `_printf` must set it, because the fallback's transitive
    `OptionalReg.value()` → `debug_assert` recurses back through
    `_debug_assert_msg` and deadlocks instantiation when assertions are
    enabled."""


# ===-----------------------------------------------------------------------===#
# DefaultPlugin
# ===-----------------------------------------------------------------------===#


struct DefaultPlugin(StdPlugin):
    """Default `PluginHooks` implementation used when no plugin is active.

    Every hook is left at its `PluginHooks` default, so the built-in stdlib
    code paths are preserved.
    """

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["default"]()
    comptime HookFor[target: CompilationTarget] = DefaultPluginHooks[target]


struct DefaultPluginHooks[target: CompilationTarget](PluginHooks):
    comptime device_kind = TargetDeviceKind.cpu
