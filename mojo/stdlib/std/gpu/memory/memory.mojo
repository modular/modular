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
"""This module provides GPU memory operations and utilities.

The module implements low-level memory operations for GPU programming, with a focus on:

- Memory address space abstractions (global, shared, constant)
- Cache control operations and policies
- Memory access patterns and optimizations
- Memory alignment and pointer manipulation

It provides a unified interface for memory operations across different GPU architectures,
with specialized implementations for NVIDIA and AMD GPUs where needed.

The module is designed for performance-critical code and requires careful usage to
achieve optimal memory access patterns and cache utilization.
"""

from collections.optional import Optional, OptionalReg
from collections.string import StaticString
from collections.string.string_slice import _get_kgen_string, get_static_string
from sys import (
    align_of,
    bit_width_of,
    is_apple_gpu,
    is_amd_gpu,
    is_gpu,
    is_nvidia_gpu,
    llvm_intrinsic,
    size_of,
)
from sys._assembly import inlined_assembly
from sys.info import (
    CompilationTarget,
    _is_sm_9x_or_newer,
    _is_sm_100x_or_newer,
    is_apple_gpu,
)
from sys.intrinsics import _RegisterPackType

from builtin.dtype import _uint_type_of_width
from memory.pointer import AddressSpace, GPUAddressSpace
from memory.unsafe import bitcast

from utils import IndexList, StaticTuple
from utils.numerics import get_accum_type

from .._utils import (
    to_i16,
    to_i32,
    to_llvm_ptr,
    to_llvm_shared_mem_ptr,
    to_llvm_shared_cluster_mem_ptr,
)
from ..intrinsics import Scope

# ===-----------------------------------------------------------------------===#
# CacheOperation
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct CacheOperation(Equatable, TrivialRegisterPassable):
    """Represents different GPU cache operation policies.

    This struct defines various caching behaviors for GPU memory operations,
    controlling how data is cached and evicted at different cache levels.
    The policies affect performance and memory coherency.
    """

    var _value: Int

    comptime ALWAYS = Self(0)
    """Cache at all levels. This will be accessed again.

    Best for data that will be frequently reused across multiple threads.
    Provides fastest subsequent access but uses the most cache space.
    """

    comptime GLOBAL = Self(1)
    """Cache at global level.

    Caches data only in the L2 cache, bypassing L1.
    Good for data shared between different thread blocks.
    """

    comptime STREAMING = Self(2)
    """Streaming, this is likely to be accessed once.

    Optimizes for streaming access patterns where data is only read once.
    May bypass certain cache levels for better throughput.
    """

    comptime LAST_USE = Self(4)
    """Indicates the cache line will not be used again.

    Hints to the cache that this data can be evicted after this access.
    Helps optimize cache utilization.
    """

    comptime VOLATILE = Self(8)
    """Don't cache, and fetch again.

    Forces reads/writes to bypass cache and go directly to memory.
    Useful for memory-mapped I/O or when cache coherency is required.
    """

    comptime WRITE_BACK = Self(16)
    """Write back at all coherent levels.

    Updates all cache levels and eventually writes to memory.
    Most efficient for multiple writes to same location.
    """

    comptime WRITE_THROUGH = Self(32)
    """Write through to system memory.

    Immediately writes updates to memory while updating cache.
    Provides stronger consistency but lower performance than write-back.
    """

    comptime WORKGROUP = Self(64)
    """Workgroup level coherency.

    Caches data in the L1 cache and streams it to the wave.
    """

    fn __eq__(self, other: Self) -> Bool:
        """Tests if two `CacheOperation` instances are equal.

        Args:
            other: The `CacheOperation` to compare against.

        Returns:
            True if the operations are equal, False otherwise.
        """
        return self._value == other._value

    fn __or__(self, other: Self) -> Self:
        """Returns the bitwise OR of two `CacheOperation` instances.

        Args:
            other: The other `CacheOperation` to OR with.

        Returns:
            A new `CacheOperation` representing the bitwise OR of the two values.
        """
        return Self(self._value | other._value)

    @always_inline
    fn mnemonic(self) -> StaticString:
        """Returns the PTX mnemonic string for this cache operation.

        Converts the cache operation into its corresponding PTX assembly
        mnemonic string used in GPU instructions.

        Returns:
            A string literal containing the PTX mnemonic for this operation.
        """
        if self == Self.ALWAYS:
            return "ca"
        if self == Self.GLOBAL:
            return "cg"
        if self == Self.STREAMING:
            return "cs"
        if self == Self.LAST_USE:
            return "lu"
        if self == Self.VOLATILE:
            return "cv"
        if self == Self.WRITE_BACK:
            return "wb"
        if self == Self.WRITE_THROUGH:
            return "wt"
        if self == Self.WORKGROUP:
            return "wg"

        return "unknown cache operation"


# ===-----------------------------------------------------------------------===#
# CacheEviction
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct CacheEviction(Equatable, TrivialRegisterPassable):
    """Represents cache eviction policies for GPU memory operations.

    This struct defines different cache eviction priorities that control how data is
    evicted from cache when space is needed. The policies affect cache utilization
    and performance by controlling which data gets evicted first.
    """

    var _value: Int

    comptime EVICT_NORMAL = Self(0)
    """Default cache eviction priority.

    Data cached with normal priority follows standard cache replacement policies.
    This is the default behavior and suitable for most general-purpose data access
    patterns where no special caching requirements exist.
    """

    comptime EVICT_FIRST = Self(1)
    """Highest eviction priority - data will be evicted first.

    Data cached with this priority is marked as the first candidate for eviction
    when cache space is needed. This is optimal for:
    - Streaming data that will not be reused
    - Single-pass algorithms
    - Data with low temporal locality
    """

    comptime EVICT_LAST = Self(2)
    """Lowest eviction priority - data will be evicted last.

    Data cached with this priority remains in cache until all higher priority data
    is evicted. Best used for:
    - Frequently accessed data
    - Data needed across multiple kernel launches
    - Critical data structures that benefit from cache persistence
    """

    comptime EVICT_UNCHANGED = Self(3)
    """Preserves existing cache eviction priority.

    When this policy is used:
    - Existing cache entries maintain their current eviction priority
    - No changes are made to the cache replacement order
    - Useful for operations that should not affect caching behavior
    """

    comptime NO_ALLOCATE = Self(4)
    """Prevents cache allocation for accessed data.

    Data is not cached when using this policy. Optimal for:
    - Large sequential reads/writes
    - Data that will only be accessed once
    - Preserving cache space for more critical data
    - Streaming operations with no data reuse
    """

    fn __eq__(self, other: Self) -> Bool:
        """Tests if two CacheEviction instances are equal.

        Args:
            other: The CacheEviction to compare against.

        Returns:
            True if the eviction policies are equal, False otherwise.
        """
        return self._value == other._value

    @always_inline
    fn mnemonic(self) -> StaticString:
        """Returns the string mnemonic for this cache eviction policy.

        Converts the cache eviction policy into its corresponding string
        representation used in GPU instructions and debugging.

        Returns:
            A string literal containing the mnemonic for this eviction policy.
        """
        if self == Self.EVICT_NORMAL:
            return "evict_normal"
        if self == Self.EVICT_FIRST:
            return "evict_first"
        if self == Self.EVICT_LAST:
            return "evict_last"
        if self == Self.EVICT_UNCHANGED:
            return "evict_unchanged"
        if self == Self.NO_ALLOCATE:
            return "no_allocate"
        return "unknown cache eviction"


# ===-----------------------------------------------------------------------===#
# Fill
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct Fill(Equatable, TrivialRegisterPassable):
    """Represents memory fill patterns for GPU memory operations.

    This struct defines different fill patterns that can be used when allocating or
    initializing GPU memory. The patterns control how memory is initialized, which
    can be important for debugging and performance optimization.
    """

    var _value: Int

    comptime NONE = Self(0)
    """No fill pattern - memory is left uninitialized."""

    comptime ZERO = Self(1)
    """Fill memory with zeros."""

    comptime NAN = Self(2)
    """Fill memory with NaN values. Useful for debugging floating point computations."""

    fn __eq__(self, other: Self) -> Bool:
        """Tests if two Fill instances have the same fill pattern.

        Args:
            other: The Fill instance to compare against.

        Returns:
            True if the fill patterns are equal, False otherwise.
        """
        return self._value == other._value

    @no_inline
    fn __str__(self) -> String:
        """Returns a string representation of the fill pattern.

        Converts the fill pattern into a human-readable string for debugging
        and display purposes.

        Returns:
            A string describing the fill pattern.
        """
        if self == Self.NONE:
            return "none"
        if self == Self.ZERO:
            return "zero"
        if self == Self.NAN:
            return "nan"
        return "unknown fill"


# ===-----------------------------------------------------------------------===#
# Consistency
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct Consistency(Equatable, TrivialRegisterPassable):
    """Represents memory consistency models for GPU memory operations.

    This struct defines different memory consistency levels that control how memory
    operations are ordered and synchronized between threads. The consistency model
    affects both performance and correctness of parallel algorithms.
    """

    var _value: Int

    comptime WEAK = Self(0)
    """Weakest consistency model with minimal ordering guarantees.

    Provides maximum flexibility for hardware/compiler optimizations but requires
    careful synchronization by the programmer."""

    comptime RELAXED = Self(1)
    """Relaxed consistency with basic ordering guarantees.

    Provides some ordering guarantees while still allowing optimizations.
    Suitable for operations that don't require strict ordering."""

    comptime ACQUIRE = Self(2)
    """Acquire consistency for synchronization operations.

    Ensures all subsequent memory operations are ordered after this operation.
    Used in producer-consumer patterns."""

    comptime RELEASE = Self(3)
    """Release consistency for synchronization operations.

    Ensures all previous memory operations are ordered before this operation.
    Paired with acquire operations for synchronization."""

    fn __eq__(self, other: Self) -> Bool:
        """Tests if two Consistency instances are equal.

        Args:
            other: The Consistency instance to compare against.

        Returns:
            True if the consistency levels are equal, False otherwise.
        """
        return self._value == other._value

    fn __str__(self) -> String:
        """Returns a string representation of the consistency level.

        Returns:
            A string describing the consistency level.
        """
        return String(self.mnemonic())

    @always_inline
    fn mnemonic(self) -> StaticString:
        """Returns the mnemonic string for the consistency level.

        Returns:
            A string literal containing the consistency level mnemonic.
        """
        if self == Self.WEAK:
            return "weak"
        if self == Self.RELAXED:
            return "relaxed"
        if self == Self.ACQUIRE:
            return "acquire"
        if self == Self.RELEASE:
            return "release"

        return "unknown consistency"


# ===-----------------------------------------------------------------------===#
# ReduceOp
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct ReduceOp(Equatable, TrivialRegisterPassable):
    """Represents reduction operations for parallel reduction algorithms.

    This struct defines different reduction operations that can be performed
    across multiple threads in parallel. These operations are commonly used
    in parallel reduction algorithms on GPUs.
    """

    var _value: Int

    comptime ADD = Self(0)
    """Addition reduction operation.

    Combines values by adding them together."""

    comptime MIN = Self(1)
    """Minimum reduction operation.

    Finds the minimum value across all inputs."""

    comptime MAX = Self(2)
    """Maximum reduction operation.

    Finds the maximum value across all inputs."""

    comptime AND = Self(3)
    """Bitwise AND reduction operation.

    Performs bitwise AND across all inputs."""

    comptime OR = Self(4)
    """Bitwise OR reduction operation.

    Performs bitwise OR across all inputs."""

    comptime XOR = Self(5)
    """Bitwise XOR reduction operation.

    Performs bitwise XOR across all inputs."""

    fn __eq__(self, other: Self) -> Bool:
        """Tests if two ReduceOp instances are equal.

        Args:
            other: The ReduceOp instance to compare against.

        Returns:
            True if the reduction operations are equal, False otherwise.
        """
        return self._value == other._value

    fn __is__(self, other: Self) -> Bool:
        """Tests if two ReduceOp instances are identical.

        Args:
            other: The ReduceOp instance to compare against.

        Returns:
            True if the reduction operations are identical, False otherwise.
        """
        return self == other

    @no_inline
    fn __str__(self) -> String:
        """Returns a string representation of the reduction operation.

        Returns:
            A string describing the reduction operation.
        """
        return String(self.mnemonic())

    @always_inline
    fn mnemonic(self) -> StaticString:
        """Returns the mnemonic string for the reduction operation.

        Returns:
            A string literal containing the reduction operation mnemonic.
        """
        if self == Self.ADD:
            return "add"
        if self == Self.MIN:
            return "min"
        if self == Self.MAX:
            return "max"
        if self == Self.AND:
            return "and"
        if self == Self.OR:
            return "or"
        if self == Self.XOR:
            return "xor"

        return "unknown reduce operation"


# ===-----------------------------------------------------------------------===#
# cp.async
# ===-----------------------------------------------------------------------===#


@always_inline
fn _mark_eviction[
    eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL
]() -> UInt64:
    """Returns the eviction policy value for GPU cache operations.

    Parameters:
        eviction_policy: The cache eviction policy to use.

    Returns:
        A 64-bit handle encoding the eviction policy:
            - 0 for normal eviction
            - Non-zero handle for fractional L2 eviction policy.
    """

    comptime if eviction_policy == CacheEviction.EVICT_NORMAL:
        return 0
    elif eviction_policy == CacheEviction.EVICT_LAST:
        return inlined_assembly[
            "createpolicy.fractional.L2::evict_last.b64 $0;",
            UInt64,
            constraints="=l",
        ]()
    else:
        comptime assert (
            eviction_policy == CacheEviction.EVICT_FIRST
        ), "invalid eviction policy, only support normal, first, and last"
        return inlined_assembly[
            "createpolicy.fractional.L2::evict_first.b64 $0;",
            UInt64,
            constraints="=l",
        ]()


@always_inline("nodebug")
fn async_copy[
    dtype: DType,
    //,
    size: Int,
    *,
    fill: Optional[Scalar[dtype]] = None,
    bypass_L1_16B: Bool = True,
    l2_prefetch: Optional[Int] = None,
    eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
](
    src: UnsafePointer[Scalar[dtype], address_space = AddressSpace.GLOBAL],
    dst: UnsafePointer[
        mut=True, Scalar[dtype], address_space = AddressSpace.SHARED
    ],
    src_size: Int32 = Int32(size),
    predicate: Bool = False,
):
    """Asynchronously copies data from global memory to shared memory.

    This function provides a high-performance asynchronous memory copy operation with
    configurable caching behavior, prefetching, and fill values. It maps directly to
    the PTX cp.async instruction on NVIDIA GPUs.

    Parameters:
        dtype: The data type to copy (e.g. float32, int32).
        size: Number of bytes to copy (must be 4, 8, or 16).
        fill: Optional fill value for uncopied bytes when src_size < size.
        bypass_L1_16B: If True, bypasses L1 cache for 16-byte copies.
        l2_prefetch: Optional L2 prefetch size (64, 128, or 256 bytes).
        eviction_policy: Cache eviction policy for the copy operation.

    Args:
        src: Source pointer in global memory.
        dst: Destination pointer in shared memory.
        src_size: Actual bytes to copy from src (remaining bytes use fill value).
        predicate: Optional predicate to conditionally execute the copy.

    Constraints:
        - Fill value only supported for dtypes <= 32 bits.
        - Size must be 4, 8, or 16 bytes.
        - Cannot enable both L2 prefetch and L1 bypass.
        - L2 prefetch size must be 64, 128, or 256 bytes.
    """
    comptime assert (
        not fill or size_of[dtype]() <= size_of[Int32]()
    ), "if the fill value is specified, then the dtype must be 32bit or less"
    comptime assert size in (4, 8, 16)
    comptime assert not (
        l2_prefetch.__bool__() == bypass_L1_16B == True
    ), "both enable l2 prefetching and l1 bypass cannot be True"
    comptime assert not l2_prefetch or l2_prefetch.value() in (
        64,
        128,
        256,
    ), "the l2 prefetch size must be in bounds"

    comptime if is_amd_gpu() or is_apple_gpu():
        # Use sync load and stores for now
        # TODO(KERN-1249): add async memcopy to AMD
        comptime n_scalars = size // size_of[dtype]()
        var n_src_scalars = src_size // Int32(size_of[dtype]())

        comptime if fill:
            for i in range(n_src_scalars):
                dst.store(i, src.load(i))
            for i in range(n_src_scalars, n_scalars):
                dst.store(i, fill.value())
        else:
            comptime for i in range(n_scalars):
                dst.store(i, src.load(i))
        return
    # Cache always: cache data in L1 first, then copy to shared memory.
    # Cache global: bypass L1 cache
    # We always do the latter.
    comptime cache_op = CacheOperation.GLOBAL.mnemonic() if (
        bypass_L1_16B and size == 16
    ) else CacheOperation.ALWAYS.mnemonic()
    comptime access_size = _int_to_str[size]()

    comptime cache_hint = ".L2::cache_hint" if eviction_policy != CacheEviction.EVICT_NORMAL else StaticString(
        ""
    )
    var cache_policy = _mark_eviction[eviction_policy]()

    comptime l2_prefetch_substr = ".L2::" + _int_to_str[
        l2_prefetch.value()
    ]() + "B" if l2_prefetch else ""

    comptime cp_async_asm = "cp.async." + cache_op + ".shared.global" + cache_hint + l2_prefetch_substr

    comptime if Bool(fill) and fill.value() == 0:
        debug_assert(
            not predicate, "Predicate bit has to be set False for zero fill."
        )

        comptime args_with_fill = " [$0], [$1], $2, $3"
        comptime asm = cp_async_asm + args_with_fill

        comptime if eviction_policy == CacheEviction.EVICT_NORMAL:
            inlined_assembly[asm + ";", NoneType, constraints="r,l,n,r"](
                Int32(Int(dst)), src, Int32(size), Int32(src_size)
            )
        else:
            inlined_assembly[asm + ", $4;", NoneType, constraints="r,l,n,r,l"](
                Int32(Int(dst)), src, Int32(size), Int32(src_size), cache_policy
            )
    elif fill:
        comptime assert (
            size == 16
        ), "Non zero filling is supported only for 16B access."

        # Pack filling values into 4B registers.
        @always_inline
        fn _i32_repr[fill: Scalar[dtype]]() -> Int32:
            comptime if size_of[dtype]() == 1:
                return bitcast[DType.int32, 1](
                    SIMD[dtype, 4](fill, fill, fill, fill)
                )
            elif size_of[dtype]() == 2:
                return bitcast[DType.int32, 1](SIMD[dtype, 2](fill, fill))
            elif size_of[dtype]() == 4:
                return bitcast[DType.int32](fill)

            return 0

        var fill_val = _i32_repr[fill.value()]()
        comptime header_asm = "{\n.reg .pred p;\nsetp.ne.b32 p, $0, 0;\n"
        comptime footer_asm = "@!p st.shared.v4.b32 [$1], {$4, $5, $6, $7};\n}\n"
        comptime args_with_fill = " [$1], [$2], $3"
        comptime copy_asm = header_asm + "@p " + cp_async_asm + args_with_fill

        comptime if eviction_policy == CacheEviction.EVICT_NORMAL:
            inlined_assembly[
                copy_asm + ";\n" + footer_asm,
                NoneType,
                constraints="r,r,l,n,r,r,r,r",
            ](
                Int32(Int(predicate)),
                Int32(Int(dst)),
                src,
                Int32(size),
                fill_val,
                fill_val,
                fill_val,
                fill_val,
            )
        else:
            inlined_assembly[
                copy_asm + ", $8;\n" + footer_asm,
                NoneType,
                constraints="r,r,l,n,r,r,r,r,l",
            ](
                Int32(Int(predicate)),
                Int32(Int(dst)),
                src,
                Int32(size),
                fill_val,
                fill_val,
                fill_val,
                fill_val,
                cache_policy,
            )

    else:
        debug_assert(
            not predicate, "Predicate bit has to set False for no fill."
        )

        comptime args = " [$0], [$1], $2"
        comptime asm = cp_async_asm + args

        comptime if eviction_policy == CacheEviction.EVICT_NORMAL:
            inlined_assembly[asm + ";", NoneType, constraints="r,l,n"](
                Int32(Int(dst)), src, Int32(size)
            )
        else:
            inlined_assembly[asm + ", $3;", NoneType, constraints="r,l,n,l"](
                Int32(Int(dst)), src, Int32(size), cache_policy
            )


@always_inline
fn async_copy_commit_group():
    """Commits all prior initiated but uncommitted cp.async instructions into a cp.async-group.

    This function creates a new cp.async-group containing all previously initiated but uncommitted
    asynchronous copy operations. The group can then be waited on using async_copy_wait_group().

    Notes:

    - Only supported on NVIDIA GPUs
    - Maps to the cp.async.commit.group PTX instruction
    - Used for managing asynchronous memory transfers
    - Should be paired with async_copy_wait_group() or async_copy_wait_all()
    """

    comptime if is_nvidia_gpu():
        llvm_intrinsic["llvm.nvvm.cp.async.commit.group", NoneType]()
    elif is_amd_gpu() or is_apple_gpu() or not is_gpu():
        # This operation is a no-op on AMD and CPU.
        pass
    else:
        return CompilationTarget.unsupported_target_error[
            operation = __get_current_function_name()
        ]()


@always_inline
fn async_copy_wait_group(n: Int32):
    """Waits for the completion of `n` most recently committed cp.async-groups.

    This function blocks execution until the specified number of previously committed
    cp.async-groups have completed their memory transfers.

    Args:
        n: The number of pending cp.async-groups to wait for. Must be > 0.

    Notes:

    - Only supported on NVIDIA GPUs.
    - Maps to the cp.async.wait.group PTX instruction.
    - Provides fine-grained control over asynchronous transfer synchronization.
    - Can be used to implement a pipeline of asynchronous transfers.
    """

    comptime if is_nvidia_gpu():
        llvm_intrinsic["llvm.nvvm.cp.async.wait.group", NoneType](n)
    elif is_amd_gpu() or is_apple_gpu() or not is_gpu():
        # This operation is a no-op on AMD and CPU.
        pass
    else:
        return CompilationTarget.unsupported_target_error[
            operation = __get_current_function_name()
        ]()


@always_inline
fn async_copy_wait_all():
    """Waits for completion of all committed cp.async-groups.

    This function blocks execution until all previously committed cp.async-groups
    have completed their memory transfers. It provides a barrier to ensure all
    asynchronous copies are finished.

    Notes:

    - Only supported on NVIDIA GPUs.
    - Maps to the cp.async.wait.all PTX instruction.
    - Ensures all outstanding asynchronous transfers are complete.
    - More coarse-grained than `async_copy_wait_group()`.
    """

    comptime if is_nvidia_gpu():
        llvm_intrinsic["llvm.nvvm.cp.async.wait.all", NoneType]()
    elif is_amd_gpu() or is_apple_gpu() or not is_gpu():
        # This operation is a no-op on AMD and CPU.
        pass
    else:
        return CompilationTarget.unsupported_target_error[
            operation = __get_current_function_name()
        ]()


@always_inline
fn external_memory[
    dtype: TrivialRegisterPassable,
    *,
    address_space: AddressSpace,
    alignment: Int,
    name: StaticString = "extern_ptr_syml",
]() -> UnsafePointer[dtype, MutAnyOrigin, address_space=address_space]:
    """Gets a pointer to dynamically allocated external memory.

    This function returns a pointer to external memory that can be used for dynamic
    shared memory allocations in GPU kernels. The memory is allocated in the specified
    address space with the given alignment requirements.

    Parameters:
        dtype: The dtype of elements stored in the memory. Must be a trivial register dtype.
        address_space: The memory address space to allocate in (e.g. shared, global).
        alignment: The minimum alignment requirement in bytes for the allocated memory.
        name: Optional symbolic name for the external memory allocation. Defaults to
            "extern_ptr_syml".

    Returns:
        A properly aligned pointer to the allocated external memory in the
        specified address space.

    Note:

    - The memory is not initialized and must be explicitly written before reading.
    - The allocation size is determined at kernel launch time.
    - The pointer is only valid within the GPU kernel execution context.
    - Care must be taken to respect alignment requirements when accessing the memory.
    """
    comptime assert (
        not is_apple_gpu()
    ), "external memory is not supported on Apple GPU"
    var extern_ptr_symbol = UnsafePointer[
        StaticTuple[dtype, 0], MutAnyOrigin, address_space=address_space
    ](
        __mlir_op.`pop.extern_ptr_symbol`[
            _type = UnsafePointer[
                StaticTuple[dtype, 0],
                MutAnyOrigin,
                address_space=address_space,
            ]._mlir_type,
            name = _get_kgen_string[name](),
            alignment = alignment._mlir_value,
        ]()
    )
    return extern_ptr_symbol.bitcast[dtype]()


# ===-----------------------------------------------------------------------===#
# TMA
# ===-----------------------------------------------------------------------===#


@always_inline
fn fence_proxy_tensormap_generic_sys_acquire[
    dtype: AnyType,
](
    ptr: UnsafePointer[mut=True, dtype, address_space = AddressSpace.GENERIC],
    size: Int32,
):
    """Acquires a system-wide memory fence for tensor map operations.

    This function establishes a memory fence that ensures proper synchronization
    between tensor map operations and system memory. It guarantees that all previous
    memory operations are completed before subsequent tensor map accesses.

    Parameters:
        dtype: The data type of the tensor map object being synchronized.

    Args:
        ptr: Pointer to the tensor map object in system memory that needs to be synchronized.
        size: The size in bytes of the tensor map object being synchronized.

    Note:

    This is a low-level synchronization primitive typically used in conjunction with
    TMA (Tensor Memory Access) operations on NVIDIA GPUs.
    """
    llvm_intrinsic[
        "llvm.nvvm.fence.proxy.tensormap_generic.acquire.sys", NoneType
    ](ptr, size)


@always_inline
fn fence_proxy_tensormap_generic_sys_release():
    """Releases the system-wide memory fence for tensor map operations.

    This function releases the memory fence previously established by the acquire operation.
    It ensures that all tensor map operations are completed and visible to the system
    before proceeding.

    Note:

    Should be called after tensor map operations are complete to maintain proper
    memory ordering semantics.
    """
    llvm_intrinsic[
        "llvm.nvvm.fence.proxy.tensormap_generic.release.sys", NoneType
    ]()


@always_inline
fn fence_async_view_proxy():
    """Establishes a memory fence for shared memory view operations.

    This function creates a memory barrier that ensures all previous shared memory
    stores are completed before subsequent shared memory view operations begin.
    This is crucial for maintaining memory consistency.

    Note:

    This fence specifically targets the CTA (Cooperative Thread Array) scope
    and is used to synchronize async shared memory operations.
    """
    __mlir_op.`nvvm.fence.proxy`[
        _properties = __mlir_attr.`{ kind = #nvvm.proxy_kind<async.shared>, space = #nvvm.shared_space<cta>}`
    ]()


@always_inline
fn fence_mbarrier_init():
    """Creates a memory fence after mbarrier initialization.

    This function establishes a memory barrier that ensures the proper initialization
    of memory barriers (mbarrier) before they are used. It guarantees that the
    mbarrier initialization is complete and visible to all threads before subsequent
    operations.

    Note:

    Should be called immediately after mbarrier initialization to ensure proper
    synchronization semantics.
    """
    __mlir_op.`nvvm.fence.mbarrier.init`[_type=None]()


@always_inline("nodebug")
fn cp_async_bulk_tensor_shared_cluster_global[
    dst_type: AnyType,  # Type of the destination memory
    mbr_type: AnyType,  # Type of the memory barrier
    rank: Int,  # Dimensionality of the tensor (1, 2, or 3)
    /,
    *,
    cta_group: Int = 1,
    eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
](
    dst_mem: UnsafePointer[
        mut=True, dst_type, address_space = AddressSpace.SHARED
    ],
    tma_descriptor: OpaquePointer[mut=False],
    mem_bar: UnsafePointer[
        mut=False, mbr_type, address_space = AddressSpace.SHARED
    ],
    coords: IndexList[rank],
):
    """Initiates an asynchronous bulk copy operation of tensor data from global memory to shared memory.

    This function performs an asynchronous copy of tensor data using NVIDIA's Tensor Memory Access (TMA)
    mechanism. It supports both rank-1 and rank-2 tensors and uses cluster-level synchronization for
    efficient data movement.

    Parameters:
        dst_type: The data type of the destination memory.
        mbr_type: The data type of the memory barrier.
        rank: The dimensionality of the tensor (1, 2, or 3).
        cta_group: The CTA group to use for the copy operation. Must be 1 or 2.
        eviction_policy: Optional cache eviction policy that controls how the data is handled
                        in the cache hierarchy. Defaults to EVICT_NORMAL.

    Args:
        dst_mem: Pointer to the destination in shared memory where the tensor data will be copied.
                Must be properly aligned according to TMA requirements.
        tma_descriptor: Pointer to the TMA descriptor that contains metadata about the tensor layout
                       and memory access patterns.
        mem_bar: Pointer to a shared memory barrier used for synchronizing the asynchronous copy
                operation across threads in the cluster.
        coords: Coordinates specifying which tile of the tensor to copy. For rank-1 tensors,
               this is a single coordinate. For rank-2 tensors, this contains both row and
               column coordinates.

    Notes:

    - This operation is asynchronous - use appropriate memory barriers to ensure
      copy completion.
    - Only supports rank-1 and rank-2 tensors.
    - Requires NVIDIA GPU with TMA support.
    - The memory barrier should be properly initialized before use.
    """
    comptime assert (
        rank <= 5
    ), "Expecting rank-1, rank-2, rank-3, rank-4, or rank-5 tensors"

    comptime assert cta_group in (1, 2), "cta_group must be 1 or 2"
    comptime assert cta_group == 1 or _is_sm_100x_or_newer()
    comptime cache_hint: Bool = eviction_policy != CacheEviction.EVICT_NORMAL
    comptime assert not cache_hint or cta_group == 1
    comptime tma_asm = String(
        "cp.async.bulk.tensor.",
        rank,
        "d.cta_group::2" if cta_group == 2 else "d",
        ".shared::cluster.global.tile.mbarrier::complete_tx::bytes",
        ".L2::cache_hint" if cache_hint else "",
    )

    comptime if cache_hint:
        comptime if rank == 3:
            inlined_assembly[
                tma_asm + " [$0], [$1, {$3, $4, $5}], [$2], $6;",
                NoneType,
                constraints="r,l,r,r,r,r,l",
            ](
                Int32(Int(dst_mem)),
                tma_descriptor,
                Int32(Int(mem_bar)) & 0xFEFFFFFF,
                Int32(coords[0]),
                Int32(coords[1]),
                Int32(coords[2]),
                Int64(eviction_policy._value),
            )
        elif rank == 2:
            inlined_assembly[
                tma_asm + " [$0], [$1, {$3, $4}], [$2], $5;",
                NoneType,
                constraints="r,l,r,r,r,l",
            ](
                Int32(Int(dst_mem)),
                tma_descriptor,
                Int32(Int(mem_bar)) & 0xFEFFFFFF,
                Int32(coords[0]),
                Int32(coords[1]),
                Int64(eviction_policy._value),
            )
        elif rank == 1:
            inlined_assembly[
                tma_asm + " [$0], [$1, {$3}], [$2], $4;",
                NoneType,
                constraints="r,l,r,r,l",
            ](
                Int32(Int(dst_mem)),
                tma_descriptor,
                Int32(Int(mem_bar)) & 0xFEFFFFFF,
                Int32(coords[0]),
                Int64(eviction_policy._value),
            )
        elif rank == 4:
            inlined_assembly[
                tma_asm + " [$0], [$1, {$3, $4, $5, $6}], [$2], $7;",
                NoneType,
                constraints="r,l,r,r,r,r,r,l",
            ](
                Int32(Int(dst_mem)),
                tma_descriptor,
                Int32(Int(mem_bar)) & 0xFEFFFFFF,
                Int32(coords[0]),
                Int32(coords[1]),
                Int32(coords[2]),
                Int32(coords[3]),
                Int64(eviction_policy._value),
            )
        else:  # rank == 5
            inlined_assembly[
                tma_asm + " [$0], [$1, {$3, $4, $5, $6, $7}], [$2], $8;",
                NoneType,
                constraints="r,l,r,r,r,r,r,r,l",
            ](
                Int32(Int(dst_mem)),
                tma_descriptor,
                Int32(Int(mem_bar)) & 0xFEFFFFFF,
                Int32(coords[0]),
                Int32(coords[1]),
                Int32(coords[2]),
                Int32(coords[3]),
                Int32(coords[4]),
                Int64(eviction_policy._value),
            )
    else:
        comptime if rank == 3:
            inlined_assembly[
                tma_asm + " [$0], [$1, {$3, $4, $5}], [$2];",
                NoneType,
                constraints="r,l,r,r,r,r",
            ](
                Int32(Int(dst_mem)),
                tma_descriptor,
                Int32(Int(mem_bar)) & 0xFEFFFFFF,
                Int32(coords[0]),
                Int32(coords[1]),
                Int32(coords[2]),
            )
        elif rank == 2:
            inlined_assembly[
                tma_asm + " [$0], [$1, {$3, $4}], [$2];",
                NoneType,
                constraints="r,l,r,r,r",
            ](
                Int32(Int(dst_mem)),
                tma_descriptor,
                Int32(Int(mem_bar)) & 0xFEFFFFFF,
                Int32(coords[0]),
                Int32(coords[1]),
            )
        elif rank == 1:
            inlined_assembly[
                tma_asm + " [$0], [$1, {$3}], [$2];",
                NoneType,
                constraints="r,l,r,r",
            ](
                Int32(Int(dst_mem)),
                tma_descriptor,
                Int32(Int(mem_bar)) & 0xFEFFFFFF,
                Int32(coords[0]),
            )
        elif rank == 4:
            inlined_assembly[
                tma_asm + " [$0], [$1, {$3, $4, $5, $6}], [$2];",
                NoneType,
                constraints="r,l,r,r,r,r,r",
            ](
                Int32(Int(dst_mem)),
                tma_descriptor,
                Int32(Int(mem_bar)) & 0xFEFFFFFF,
                Int32(coords[0]),
                Int32(coords[1]),
                Int32(coords[2]),
                Int32(coords[3]),
            )
        else:  # rank == 5
            inlined_assembly[
                tma_asm + " [$0], [$1, {$3, $4, $5, $6, $7}], [$2];",
                NoneType,
                constraints="r,l,r,r,r,r,r,r",
            ](
                Int32(Int(dst_mem)),
                tma_descriptor,
                Int32(Int(mem_bar)) & 0xFEFFFFFF,
                Int32(coords[0]),
                Int32(coords[1]),
                Int32(coords[2]),
                Int32(coords[3]),
                Int32(coords[4]),
            )


@always_inline("nodebug")
fn cp_async_bulk_tensor_shared_cluster_global_im2col[
    dst_type: AnyType,
    mbr_type: AnyType,
    tensor_rank: Int,
    /,
    *,
    cta_group: Int = 1,
](
    dst_mem: UnsafePointer[
        mut=True, dst_type, address_space = AddressSpace.SHARED
    ],
    tma_descriptor: OpaquePointer[mut=False],
    mem_bar: UnsafePointer[
        mut=False, mbr_type, address_space = AddressSpace.SHARED
    ],
    coords: IndexList[tensor_rank],
    filter_offsets: IndexList[tensor_rank - 2],
):
    """Initiates an asynchronous TMA load with im2col addressing for convolution.

    This function performs a TMA load using im2col mode, which applies coordinate
    transformation suitable for implicit GEMM convolution. The TMA descriptor must
    be created with cuTensorMapEncodeIm2col.

    For 2D convolution with 4D NHWC tensor:
    - coords: (c, w, h, n) - channel, output spatial, batch
    - filter_offsets: (offset_w, offset_h) - position within filter window

    PTX instruction formats differ based on cta_group:
    - cta_group=1: Uses SM90-style PTX (no cta_group modifier)
      cp.async.bulk.tensor.4d.shared::cluster.global.im2col...
    - cta_group=2: Uses SM100-style PTX with cta_group::2 (from CUTLASS)
      cp.async.bulk.tensor.4d.im2col.cta_group::2.shared::cluster.global...

    Parameters:
        dst_type: The data type of the destination memory.
        mbr_type: The data type of the memory barrier.
        tensor_rank: The rank of the tensor (3, 4, or 5).
        cta_group: The CTA group to use for the copy operation. Must be 1 or 2.

    Args:
        dst_mem: Pointer to the destination in shared memory.
        tma_descriptor: Pointer to the TMA im2col descriptor.
        mem_bar: Pointer to the shared memory barrier.
        coords: Tensor coordinates (c, w, h, n for 4D).
        filter_offsets: Filter window offsets (offset_w, offset_h for 4D).
    """
    comptime assert tensor_rank in (
        3,
        4,
        5,
    ), "Im2col TMA expects 3D, 4D, or 5D tensor"
    comptime assert cta_group in (1, 2), "cta_group must be 1 or 2"
    comptime assert cta_group == 1 or _is_sm_100x_or_newer()

    comptime if cta_group == 1:
        # SM90-style PTX: no cta_group, no L2 hint, no Sm100MemDescDefault
        # Format: cp.async.bulk.tensor.Nd.shared::cluster.global.im2col.mbarrier::complete_tx::bytes
        comptime tma_asm_sm90 = String(
            "cp.async.bulk.tensor.",
            tensor_rank,
            "d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes",
        )

        comptime if tensor_rank == 4:
            inlined_assembly[
                tma_asm_sm90 + " [$0], [$1, {$3, $4, $5, $6}], [$2], {$7, $8};",
                NoneType,
                constraints="r,l,r,r,r,r,r,h,h",
            ](
                Int32(Int(dst_mem)),
                tma_descriptor,
                Int32(Int(mem_bar)),
                Int32(coords[0]),
                Int32(coords[1]),
                Int32(coords[2]),
                Int32(coords[3]),
                UInt16(filter_offsets[0]),
                UInt16(filter_offsets[1]),
            )
        elif tensor_rank == 3:
            inlined_assembly[
                tma_asm_sm90 + " [$0], [$1, {$3, $4, $5}], [$2], {$6};",
                NoneType,
                constraints="r,l,r,r,r,r,h",
            ](
                Int32(Int(dst_mem)),
                tma_descriptor,
                Int32(Int(mem_bar)),
                Int32(coords[0]),
                Int32(coords[1]),
                Int32(coords[2]),
                UInt16(filter_offsets[0]),
            )
        else:  # tensor_rank == 5
            inlined_assembly[
                tma_asm_sm90
                + " [$0], [$1, {$3, $4, $5, $6, $7}], [$2], {$8, $9, $10};",
                NoneType,
                constraints="r,l,r,r,r,r,r,r,h,h,h",
            ](
                Int32(Int(dst_mem)),
                tma_descriptor,
                Int32(Int(mem_bar)),
                Int32(coords[0]),
                Int32(coords[1]),
                Int32(coords[2]),
                Int32(coords[3]),
                Int32(coords[4]),
                UInt16(filter_offsets[0]),
                UInt16(filter_offsets[1]),
                UInt16(filter_offsets[2]),
            )
    else:  # cta_group == 2
        # SM100-style PTX: cta_group::2, L2 hint, Sm100MemDescDefault, peer bit mask
        # Format: cp.async.bulk.tensor.Nd.im2col.cta_group::2.shared::cluster.global...L2::cache_hint
        comptime tma_asm_sm100 = String(
            "cp.async.bulk.tensor.",
            tensor_rank,
            "d.im2col.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint",
        )

        # Sm100MemDescDefault constant from CUTLASS
        comptime Sm100MemDescDefault: UInt64 = 0x1000000000000000
        # Sm100MmaPeerBitMask for mbarrier (apply to clear peer bit)
        comptime Sm100MmaPeerBitMask: Int32 = Int32(0xFEFFFFFF)

        comptime if tensor_rank == 4:
            inlined_assembly[
                tma_asm_sm100
                + " [$0], [$1, {$3, $4, $5, $6}], [$2], {$7, $8}, $9;",
                NoneType,
                constraints="r,l,r,r,r,r,r,h,h,l",
            ](
                Int32(Int(dst_mem)),
                tma_descriptor,
                Int32(Int(mem_bar)) & Sm100MmaPeerBitMask,
                Int32(coords[0]),
                Int32(coords[1]),
                Int32(coords[2]),
                Int32(coords[3]),
                UInt16(filter_offsets[0]),
                UInt16(filter_offsets[1]),
                Sm100MemDescDefault,
            )
        elif tensor_rank == 3:
            inlined_assembly[
                tma_asm_sm100 + " [$0], [$1, {$3, $4, $5}], [$2], {$6}, $7;",
                NoneType,
                constraints="r,l,r,r,r,r,h,l",
            ](
                Int32(Int(dst_mem)),
                tma_descriptor,
                Int32(Int(mem_bar)) & Sm100MmaPeerBitMask,
                Int32(coords[0]),
                Int32(coords[1]),
                Int32(coords[2]),
                UInt16(filter_offsets[0]),
                Sm100MemDescDefault,
            )
        else:  # tensor_rank == 5
            inlined_assembly[
                tma_asm_sm100
                + " [$0], [$1, {$3, $4, $5, $6, $7}], [$2], {$8, $9, $10},"
                " $11;",
                NoneType,
                constraints="r,l,r,r,r,r,r,r,h,h,h,l",
            ](
                Int32(Int(dst_mem)),
                tma_descriptor,
                Int32(Int(mem_bar)) & Sm100MmaPeerBitMask,
                Int32(coords[0]),
                Int32(coords[1]),
                Int32(coords[2]),
                Int32(coords[3]),
                Int32(coords[4]),
                UInt16(filter_offsets[0]),
                UInt16(filter_offsets[1]),
                UInt16(filter_offsets[2]),
                Sm100MemDescDefault,
            )


@always_inline("nodebug")
fn cp_async_bulk_tensor_shared_cluster_global_im2col_multicast[
    dst_type: AnyType,
    mbr_type: AnyType,
    tensor_rank: Int,
    /,
    *,
    cta_group: Int = 1,
](
    dst_mem: UnsafePointer[
        mut=True, dst_type, address_space = AddressSpace.SHARED
    ],
    tma_descriptor: OpaquePointer[mut=False],
    mem_bar: UnsafePointer[
        mut=False, mbr_type, address_space = AddressSpace.SHARED
    ],
    coords: IndexList[tensor_rank],
    filter_offsets: IndexList[tensor_rank - 2],
    multicast_mask: UInt16,
):
    """Initiates an asynchronous multicast TMA load with im2col addressing.

    This combines im2col addressing with multicast, distributing the loaded
    data to multiple CTAs in a cluster.

    For 2D convolution with 4D NHWC tensor:
    - coords: (c, w, h, n) - channel, output spatial, batch
    - filter_offsets: (offset_w, offset_h) - position within filter window

    PTX instruction formats differ based on cta_group:
    - cta_group=1: Uses SM90-style multicast im2col PTX (no cta_group modifier)
      cp.async.bulk.tensor.4d.shared::cluster.global.im2col...multicast::cluster
    - cta_group=2: Uses SM100-style multicast im2col PTX with cta_group::2 (from CUTLASS)
      cp.async.bulk.tensor.4d.im2col.cta_group::2.shared::cluster.global...multicast::cluster...

    Parameters:
        dst_type: The data type of the destination memory.
        mbr_type: The data type of the memory barrier.
        tensor_rank: The rank of the tensor (3, 4, or 5).
        cta_group: The CTA group to use for the copy operation. Must be 1 or 2.

    Args:
        dst_mem: Pointer to the destination in shared memory.
        tma_descriptor: Pointer to the TMA im2col descriptor.
        mem_bar: Pointer to the shared memory barrier.
        coords: Tensor coordinates (c, w, h, n for 4D).
        filter_offsets: Filter window offsets (offset_w, offset_h for 4D).
        multicast_mask: Bitmask specifying target CTAs for multicast.
    """
    comptime assert tensor_rank in (
        3,
        4,
        5,
    ), "Im2col TMA expects 3D, 4D, or 5D tensor"
    comptime assert cta_group in (1, 2), "cta_group must be 1 or 2"

    comptime if cta_group == 1:
        # SM90-style multicast im2col PTX: no cta_group, no L2 hint, no Sm100MemDescDefault
        # Format: cp.async.bulk.tensor.Nd.shared::cluster.global.im2col.mbarrier::complete_tx::bytes.multicast::cluster
        comptime tma_asm_sm90 = String(
            "cp.async.bulk.tensor.",
            tensor_rank,
            "d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes.multicast::cluster",
        )

        comptime if tensor_rank == 4:
            inlined_assembly[
                tma_asm_sm90
                + " [$0], [$1, {$3, $4, $5, $6}], [$2], {$7, $8}, $9;",
                NoneType,
                constraints="r,l,r,r,r,r,r,h,h,h",
            ](
                Int32(Int(dst_mem)),
                tma_descriptor,
                Int32(Int(mem_bar)),
                Int32(coords[0]),
                Int32(coords[1]),
                Int32(coords[2]),
                Int32(coords[3]),
                UInt16(filter_offsets[0]),
                UInt16(filter_offsets[1]),
                multicast_mask,
            )
        elif tensor_rank == 3:
            inlined_assembly[
                tma_asm_sm90 + " [$0], [$1, {$3, $4, $5}], [$2], {$6}, $7;",
                NoneType,
                constraints="r,l,r,r,r,r,h,h",
            ](
                Int32(Int(dst_mem)),
                tma_descriptor,
                Int32(Int(mem_bar)),
                Int32(coords[0]),
                Int32(coords[1]),
                Int32(coords[2]),
                UInt16(filter_offsets[0]),
                multicast_mask,
            )
        else:  # tensor_rank == 5
            inlined_assembly[
                tma_asm_sm90
                + " [$0], [$1, {$3, $4, $5, $6, $7}], [$2], {$8, $9, $10},"
                " $11;",
                NoneType,
                constraints="r,l,r,r,r,r,r,r,h,h,h,h",
            ](
                Int32(Int(dst_mem)),
                tma_descriptor,
                Int32(Int(mem_bar)),
                Int32(coords[0]),
                Int32(coords[1]),
                Int32(coords[2]),
                Int32(coords[3]),
                Int32(coords[4]),
                UInt16(filter_offsets[0]),
                UInt16(filter_offsets[1]),
                UInt16(filter_offsets[2]),
                multicast_mask,
            )
    else:  # cta_group == 2
        # SM100-style multicast im2col PTX: cta_group::2, L2 hint, Sm100MemDescDefault, peer bit mask
        # Format: cp.async.bulk.tensor.Nd.im2col.cta_group::2.shared::cluster.global...multicast::cluster.L2::cache_hint
        comptime tma_asm_sm100 = String(
            "cp.async.bulk.tensor.",
            tensor_rank,
            "d.im2col.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint",
        )

        # Sm100MemDescDefault constant from CUTLASS
        comptime Sm100MemDescDefault: UInt64 = 0x1000000000000000
        # Sm100MmaPeerBitMask for mbarrier (apply to clear peer bit)
        comptime Sm100MmaPeerBitMask: Int32 = Int32(0xFEFFFFFF)

        comptime if tensor_rank == 4:
            inlined_assembly[
                tma_asm_sm100
                + " [$0], [$1, {$3, $4, $5, $6}], [$2], {$7, $8}, $9, $10;",
                NoneType,
                constraints="r,l,r,r,r,r,r,h,h,h,l",
            ](
                Int32(Int(dst_mem)),
                tma_descriptor,
                Int32(Int(mem_bar)) & Sm100MmaPeerBitMask,
                Int32(coords[0]),
                Int32(coords[1]),
                Int32(coords[2]),
                Int32(coords[3]),
                UInt16(filter_offsets[0]),
                UInt16(filter_offsets[1]),
                multicast_mask,
                Sm100MemDescDefault,
            )
        elif tensor_rank == 3:
            inlined_assembly[
                tma_asm_sm100
                + " [$0], [$1, {$3, $4, $5}], [$2], {$6}, $7, $8;",
                NoneType,
                constraints="r,l,r,r,r,r,h,h,l",
            ](
                Int32(Int(dst_mem)),
                tma_descriptor,
                Int32(Int(mem_bar)) & Sm100MmaPeerBitMask,
                Int32(coords[0]),
                Int32(coords[1]),
                Int32(coords[2]),
                UInt16(filter_offsets[0]),
                multicast_mask,
                Sm100MemDescDefault,
            )
        else:  # tensor_rank == 5
            inlined_assembly[
                tma_asm_sm100
                + " [$0], [$1, {$3, $4, $5, $6, $7}], [$2], {$8, $9, $10}, $11,"
                " $12;",
                NoneType,
                constraints="r,l,r,r,r,r,r,r,h,h,h,h,l",
            ](
                Int32(Int(dst_mem)),
                tma_descriptor,
                Int32(Int(mem_bar)) & Sm100MmaPeerBitMask,
                Int32(coords[0]),
                Int32(coords[1]),
                Int32(coords[2]),
                Int32(coords[3]),
                Int32(coords[4]),
                UInt16(filter_offsets[0]),
                UInt16(filter_offsets[1]),
                UInt16(filter_offsets[2]),
                multicast_mask,
                Sm100MemDescDefault,
            )


@always_inline
fn cp_async_bulk_tensor_shared_cluster_global_multicast[
    dst_type: AnyType,
    mbr_type: AnyType,
    rank: Int,
    /,
    *,
    cta_group: Int = 1,
](
    dst_mem: UnsafePointer[
        mut=True, dst_type, address_space = AddressSpace.SHARED
    ],
    tma_descriptor: OpaquePointer[mut=False],
    mem_bar: UnsafePointer[
        mut=False, mbr_type, address_space = AddressSpace.SHARED
    ],
    coords: IndexList[rank],
    multicast_mask: UInt16,
):
    """Initiates an asynchronous multicast load operation using NVIDIA's Tensor Memory Access (TMA)
    to copy tensor data from global memory to shared memories of multiple CTAs in a cluster.

    This function performs an optimized multicast copy operation where a single global memory read
    can be distributed to multiple CTAs' shared memories simultaneously, reducing memory bandwidth
    usage. It supports both rank-1 and rank-2 tensors and uses cluster-level synchronization.

    Parameters:
        dst_type: The data type of the destination tensor elements.
        mbr_type: The data type of the memory barrier.
        rank: The dimensionality of the tensor (must be 1 or 2).
        cta_group: The CTA group to use for the copy operation. Must be 1 or 2.

    Args:
        dst_mem: Pointer to the destination in shared memory where the tensor data will be copied.
                Must be properly aligned according to TMA requirements.
        tma_descriptor: Pointer to the TMA descriptor containing metadata about tensor layout
                       and memory access patterns.
        mem_bar: Pointer to a shared memory barrier used for synchronizing the asynchronous copy
                operation across threads in the cluster.
        coords: Coordinates specifying which tile of the tensor to copy. For rank-1 tensors,
               this is a single coordinate. For rank-2 tensors, this contains both row and
               column coordinates.
        multicast_mask: A 16-bit bitmask where each bit corresponds to a CTA in the cluster.
                       Set bits indicate which CTAs will receive a copy of the loaded data.
                       This enables efficient data sharing across multiple CTAs.

    Notes:

    - This operation is asynchronous - use appropriate memory barriers to ensure copy completion.
    - Only supports rank-1 and rank-2 tensors.
    - Requires NVIDIA GPU with TMA support.
    - The memory barrier should be properly initialized before use.
    - The multicast_mask must be properly configured based on cluster size and desired distribution.
    """
    comptime assert rank in (
        1,
        2,
        3,
    ), "Expecting rank-1, rank-2, or rank-3 tensors"

    comptime assert cta_group in (1, 2), "cta_group must be 1 or 2"
    comptime tma_asm = String(
        "cp.async.bulk.tensor.",
        rank,
        "d",
        ".cta_group::",
        cta_group,
        ".shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster",
    )

    comptime if rank == 3:
        comptime if cta_group == 1:
            var dst_mem_cluster = dst_mem.address_space_cast[
                AddressSpace.SHARED_CLUSTER
            ]()
            __mlir_op.`nvvm.cp.async.bulk.tensor.shared.cluster.global`[
                _properties = __mlir_attr.`{operandSegmentSizes = array<i32: 1,1,3,1,0,1,0,0>}`
            ](
                to_llvm_shared_cluster_mem_ptr(dst_mem_cluster),
                to_llvm_ptr(tma_descriptor),
                to_i32(Int32(coords[0])),
                to_i32(Int32(coords[1])),
                to_i32(Int32(coords[2])),
                to_llvm_shared_mem_ptr(mem_bar),
                to_i16(multicast_mask),
            )
        else:
            inlined_assembly[
                tma_asm + " [$0], [$1, {$4, $5, $6}], [$2], $3;",
                NoneType,
                constraints="r,l,r,h,r,r,r",
            ](
                Int32(Int(dst_mem)),
                tma_descriptor,
                Int32(Int(mem_bar)) & 0xFEFFFFFF,
                multicast_mask,
                Int32(coords[0]),
                Int32(coords[1]),
                Int32(coords[2]),
            )

    elif rank == 2:
        comptime if cta_group == 1:
            var dst_mem_cluster = dst_mem.address_space_cast[
                AddressSpace.SHARED_CLUSTER
            ]()
            __mlir_op.`nvvm.cp.async.bulk.tensor.shared.cluster.global`[
                _properties = __mlir_attr.`{operandSegmentSizes = array<i32: 1,1,2,1,0,1,0,0>}`
            ](
                to_llvm_shared_cluster_mem_ptr(dst_mem_cluster),
                to_llvm_ptr(tma_descriptor),
                to_i32(Int32(coords[0])),
                to_i32(Int32(coords[1])),
                to_llvm_shared_mem_ptr(mem_bar),
                to_i16(multicast_mask),
            )
        else:
            inlined_assembly[
                tma_asm + " [$0], [$1, {$4, $5}], [$2], $3;",
                NoneType,
                constraints="r,l,r,h,r,r",
            ](
                Int32(Int(dst_mem)),
                tma_descriptor,
                Int32(Int(mem_bar)) & 0xFEFFFFFF,
                multicast_mask,
                Int32(coords[0]),
                Int32(coords[1]),
            )
    else:
        comptime if cta_group == 1:
            var dst_mem_cluster = dst_mem.address_space_cast[
                AddressSpace.SHARED_CLUSTER
            ]()
            __mlir_op.`nvvm.cp.async.bulk.tensor.shared.cluster.global`[
                _properties = __mlir_attr.`{operandSegmentSizes = array<i32: 1,1,1,1,0,1,0,0>}`
            ](
                to_llvm_shared_cluster_mem_ptr(dst_mem_cluster),
                to_llvm_ptr(tma_descriptor),
                to_i32(Int32(coords[0])),
                to_llvm_shared_mem_ptr(mem_bar),
                to_i16(multicast_mask),
            )
        else:
            inlined_assembly[
                tma_asm + " [$0], [$1, {$4}], [$2], $3;",
                NoneType,
                constraints="r,l,r,h,r",
            ](
                Int32(Int(dst_mem)),
                tma_descriptor,
                Int32(Int(mem_bar)) & 0xFEFFFFFF,
                multicast_mask,
                Int32(coords[0]),
            )


@always_inline
fn cp_async_bulk_tensor_global_shared_cta[
    src_type: AnyType,
    rank: Int,
    /,
    eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
](
    src_mem: UnsafePointer[src_type, address_space = AddressSpace.SHARED],
    tma_descriptor: OpaquePointer[mut=False],
    coords: IndexList[rank],
):
    """Initiates an asynchronous copy operation to transfer tensor data from shared CTA
    memory to global memory using NVIDIA's Tensor Memory Access (TMA) mechanism.

    This function provides an efficient way to write data back from shared memory to global
    memory using TMA. It supports both rank-1, rank-2, and rank-3 tensors and allows control over
    cache eviction policy.

    Parameters:
        src_type: The data type of the source tensor elements.
        rank: The dimensionality of the tensor (must be 1, 2, or 3).
        eviction_policy: Optional cache eviction policy that controls how the data is handled
                        in the cache hierarchy. Defaults to EVICT_NORMAL.

    Args:
        src_mem: Pointer to the source data in shared memory that will be copied to global
                memory. Must be properly aligned according to TMA requirements.
        tma_descriptor: Pointer to the TMA descriptor containing metadata about tensor layout
                       and memory access patterns.
        coords: Coordinates specifying which tile of the tensor to copy. For rank-1 tensors,
               this is a single coordinate. For rank-2 tensors, this contains both row and
               column coordinates. For rank-3 tensors, this contains both row, column, and depth coordinates.

    Notes:

    - This operation is asynchronous - use appropriate memory barriers to ensure completion.
    - Only supports rank-1 and rank-2 tensors.
    - Requires NVIDIA GPU with TMA support.
    - The source memory must be properly aligned for TMA operations.
    - The TMA descriptor must be properly initialized before use.
    """
    comptime assert rank in (
        1,
        2,
        3,
        4,
        5,
    ), "Expecting rank-1, 2, 3, 4, or 5 tensors"

    comptime cache_hint: Bool = eviction_policy != CacheEviction.EVICT_NORMAL

    comptime if rank == 5:
        llvm_intrinsic["llvm.nvvm.cp.async.bulk.tensor.s2g.tile.5d", NoneType](
            src_mem,
            tma_descriptor,
            Int32(coords[0]),
            Int32(coords[1]),
            Int32(coords[2]),
            Int32(coords[3]),
            Int32(coords[4]),
            eviction_policy._value,
            cache_hint,
        )
    elif rank == 4:
        llvm_intrinsic["llvm.nvvm.cp.async.bulk.tensor.s2g.tile.4d", NoneType](
            src_mem,
            tma_descriptor,
            Int32(coords[0]),
            Int32(coords[1]),
            Int32(coords[2]),
            Int32(coords[3]),
            eviction_policy._value,
            cache_hint,
        )

    elif rank == 3:
        llvm_intrinsic["llvm.nvvm.cp.async.bulk.tensor.s2g.tile.3d", NoneType](
            src_mem,
            tma_descriptor,
            Int32(coords[0]),
            Int32(coords[1]),
            Int32(coords[2]),
            eviction_policy._value,
            cache_hint,
        )
    elif rank == 2:
        llvm_intrinsic["llvm.nvvm.cp.async.bulk.tensor.s2g.tile.2d", NoneType](
            src_mem,
            tma_descriptor,
            Int32(coords[0]),
            Int32(coords[1]),
            eviction_policy._value,
            cache_hint,
        )
    else:
        llvm_intrinsic["llvm.nvvm.cp.async.bulk.tensor.s2g.tile.1d", NoneType](
            src_mem,
            tma_descriptor,
            Int32(coords[0]),
            eviction_policy._value,
            cache_hint,
        )


@always_inline
fn cp_async_bulk_tensor_reduce[
    src_type: AnyType,
    rank: Int,
    /,
    *,
    reduction_kind: ReduceOp,
    eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
](
    src_mem: UnsafePointer[src_type, address_space = AddressSpace.SHARED],
    tma_descriptor: OpaquePointer[mut=False],
    coords: IndexList[rank],
):
    """Initiates an asynchronous reduction operation between shared CTA memory and global memory
    using NVIDIA's Tensor Memory Access (TMA) mechanism.

    This function performs an in-place reduction operation, combining data from shared memory
    with data in global memory using the specified reduction operation. The operation is
    performed asynchronously and uses TMA's tile mode for efficient memory access.

    Parameters:
        src_type: The data type of the source tensor elements.
        rank: The dimensionality of the tensor (must be 1 or 2).
        reduction_kind: The dtype of reduction operation to perform. Supported operations are:
                       "add", "min", "max", "inc", "dec", "and", "or", "xor".
        eviction_policy: Optional cache eviction policy that controls how the data is handled
                        in the cache hierarchy. Defaults to `EVICT_NORMAL`.

    Args:
        src_mem: Pointer to the source data in shared memory that will be reduced with the
                global memory data. Must be properly aligned according to TMA requirements.
        tma_descriptor: Pointer to the TMA descriptor containing metadata about tensor layout
                       and memory access patterns.
        coords: Coordinates specifying which tile of the tensor to operate on. For rank-1
               tensors, this is a single coordinate. For rank-2 tensors, this contains both
               row and column coordinates.

    Notes:

    - This operation is asynchronous - use appropriate memory barriers to ensure completion.
    - Only supports rank-1 and rank-2 tensors.
    - Requires NVIDIA GPU with TMA support.
    - The source memory must be properly aligned for TMA operations.
    - The TMA descriptor must be properly initialized before use.
    - The reduction operation is performed atomically to ensure correctness.
    """
    comptime assert rank == 1 or rank == 2, "Expecting rank-1 or rank-2 tensors"
    comptime cache_hint: Bool = eviction_policy != CacheEviction.EVICT_NORMAL

    comptime if rank == 2:
        llvm_intrinsic[
            "llvm.nvvm.cp.async.bulk.tensor.reduce."
            + reduction_kind.mnemonic()
            + ".tile.2d",
            NoneType,
        ](
            src_mem,
            tma_descriptor,
            Int32(coords[0]),
            Int32(coords[1]),
            UInt64(eviction_policy._value),
            cache_hint,
        )
    else:
        llvm_intrinsic[
            "llvm.nvvm.cp.async.bulk.tensor.reduce."
            + reduction_kind.mnemonic()
            + ".tile.1d",
            NoneType,
        ](
            src_mem,
            tma_descriptor,
            Int32(coords[0]),
            UInt64(eviction_policy._value),
            cache_hint,
        )


# ===-----------------------------------------------------------------------===#
# load
# ===-----------------------------------------------------------------------===#


@always_inline
fn _load_impl[
    dtype: DType,
    //,
    width: Int = 1,
    *,
    read_only: Bool = False,
    prefetch_size: Optional[Int] = None,
    cache_policy: CacheOperation = CacheOperation.ALWAYS,
    eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    alignment: Int = align_of[Scalar[dtype]](),
](
    ptr: UnsafePointer[Scalar[dtype], address_space = AddressSpace.GENERIC]
) -> SIMD[dtype, width]:
    """Internal implementation of vectorized memory loads from global memory.

    This function provides low-level control over cache behavior and memory access patterns
    for loading data from global memory into vector registers.

    Parameters:
        dtype: The data type to load.
        width: Vector width (number of elements to load).
        read_only: If True, marks the load as read-only for cache optimization.
        prefetch_size: Optional L2 cache prefetch size (64, 128, or 256 bytes).
        cache_policy: Cache operation policy for the load.
        eviction_policy: Cache eviction policy.
        alignment: Memory alignment in bytes.

    Args:
        ptr: Pointer to global memory to load from.

    Returns:
        SIMD vector containing the loaded data.

    Constraints:
        - Must be used with global memory pointers.
        - Type must be numeric.
        - Prefetch size must be 64, 128, or 256 bytes if specified.
        - Read-only not supported on AMD GPUs.
    """
    comptime assert dtype.is_numeric(), "type must be numeric"

    comptime if is_amd_gpu():
        # TODO: KERN-1230
        comptime assert read_only == False
        return ptr.load[width=width]()

    comptime if prefetch_size:
        comptime assert prefetch_size.value() in (64, 128, 256)

    comptime bytes_to_load = size_of[dtype]() * width
    comptime dtype_bitwidth = bit_width_of[dtype]()

    comptime if bytes_to_load < size_of[DType.uint32]():
        return ptr.load[width=width, alignment=alignment]()

    comptime if dtype.is_floating_point() or dtype.is_signed():
        return bitcast[dtype, width](
            _load_impl[
                width=width,
                prefetch_size=prefetch_size,
                cache_policy=cache_policy,
                eviction_policy=eviction_policy,
                alignment=alignment,
            ](ptr.bitcast[Scalar[_uint_type_of_width[dtype_bitwidth]()]]())
        )

    comptime if (
        dtype_bitwidth <= 16
        and size_of[DType.uint32]() <= bytes_to_load < size_of[DType.uint64]()
    ):
        return bitcast[dtype, width](
            _load_impl[
                width = (bytes_to_load // size_of[DType.uint32]()),
                prefetch_size=prefetch_size,
                cache_policy=cache_policy,
                eviction_policy=eviction_policy,
                alignment=alignment,
            ](ptr.bitcast[UInt32]())
        )

    comptime dtype_mnemonic = "u" + _int_to_str[dtype_bitwidth]()
    comptime cache_policy_mnemonic = cache_policy.mnemonic()
    comptime eviction_policy_mnemonic = (
        ".L1::" + eviction_policy.mnemonic()
    ) if eviction_policy != CacheEviction.EVICT_NORMAL else ""
    comptime pretch_size_mnemonic = (
        ".L2::" + _int_to_str[prefetch_size.value()]() + "B"
    ) if prefetch_size else ""
    comptime cache_operation = ".nc" if read_only else ""

    comptime cache_policy_inst = (
        "" if cache_policy
        == CacheOperation.ALWAYS else ("." + cache_policy_mnemonic)
    )
    comptime v_width = ("" if width == 1 else ".v" + _int_to_str[width]())

    comptime instruction_name = "ld.global" + cache_policy_inst + cache_operation + eviction_policy_mnemonic + pretch_size_mnemonic + v_width + "." + dtype_mnemonic

    var res = SIMD[dtype, width]()

    comptime if width == 1:
        var tmp = inlined_assembly[
            "ld.global " + cache_policy_inst + cache_operation + " $0, [$2];",
            Scalar[dtype],
            constraints="=r,l,r",
            has_side_effect=True,
        ](ptr.bitcast[NoneType](), res[0])
        return SIMD[dtype, width](tmp)
    elif width == 2:
        var tmp = inlined_assembly[
            instruction_name + " {$0, $1}, [$2];",
            _RegisterPackType[Scalar[dtype], Scalar[dtype]],
            constraints="=r,=r,l,r,r",
            has_side_effect=True,
        ](ptr.bitcast[NoneType](), res[0], res[1])
        return SIMD[dtype, width](tmp[0], tmp[1])
    elif width == 4:
        var tmp = inlined_assembly[
            instruction_name + " {$0, $1, $2, $3}, [$4];",
            _RegisterPackType[
                Scalar[dtype], Scalar[dtype], Scalar[dtype], Scalar[dtype]
            ],
            constraints="=r,=r,=r,=r,l,r,r,r,r",
            has_side_effect=True,
        ](ptr.bitcast[NoneType](), res[0], res[1], res[2], res[3])
        return SIMD[dtype, width](tmp[0], tmp[1], tmp[2], tmp[3])
    else:
        var lhs = _load_impl[
            width = width // 2,
            prefetch_size=prefetch_size,
            cache_policy=cache_policy,
            eviction_policy=eviction_policy,
            alignment=alignment,
        ](ptr)
        var rhs = _load_impl[
            width = width // 2,
            prefetch_size=prefetch_size,
            cache_policy=cache_policy,
            eviction_policy=eviction_policy,
            alignment=alignment,
        ](ptr + width // 2)
        return lhs.join(rhs)._refine[new_size=width]()


@always_inline
fn load[
    dtype: DType,
    //,
    width: Int = 1,
    *,
    read_only: Bool = False,
    prefetch_size: Optional[Int] = None,
    cache_policy: CacheOperation = CacheOperation.ALWAYS,
    eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    alignment: Int = align_of[Scalar[dtype]]() if is_nvidia_gpu() else 1,
](
    ptr: UnsafePointer[Scalar[dtype], address_space = AddressSpace.GENERIC]
) -> SIMD[dtype, width]:
    """Loads data from global memory into a SIMD vector.

    Provides a high-level interface for vectorized memory loads with configurable
    cache behavior and memory access patterns.

    Parameters:
        dtype: The data type to load.
        width: Vector width (number of elements to load).
        read_only: If True, marks the load as read-only for cache optimization.
        prefetch_size: Optional L2 cache prefetch size (64, 128, or 256 bytes).
        cache_policy: Cache operation policy for the load.
        eviction_policy: Cache eviction policy.
        alignment: Memory alignment in bytes.

    Args:
        ptr: Pointer to global memory to load from.

    Returns:
        SIMD vector containing the loaded data.
    """
    return _load_impl[
        width=width,
        read_only=read_only,
        prefetch_size=prefetch_size,
        cache_policy=cache_policy,
        eviction_policy=eviction_policy,
        alignment=alignment,
    ](ptr)


@always_inline
fn load[
    OffsetType: Indexer,
    dtype: DType,
    //,
    width: Int = 1,
    *,
    read_only: Bool = False,
    prefetch_size: Optional[Int] = None,
    cache_policy: CacheOperation = CacheOperation.ALWAYS,
    eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    alignment: Int = align_of[Scalar[dtype]]() if is_nvidia_gpu() else 1,
](
    ptr: UnsafePointer[Scalar[dtype], address_space = AddressSpace.GENERIC],
    offset: OffsetType,
) -> SIMD[dtype, width]:
    """Loads data from global memory with an offset into a SIMD vector.

    Provides a high-level interface for vectorized memory loads with configurable
    cache behavior and memory access patterns, supporting offset-based addressing.

    Parameters:
        OffsetType: Type of the offset value.
        dtype: The data type to load.
        width: Vector width (number of elements to load).
        read_only: If True, marks the load as read-only for cache optimization.
        prefetch_size: Optional L2 cache prefetch size (64, 128, or 256 bytes).
        cache_policy: Cache operation policy for the load.
        eviction_policy: Cache eviction policy.
        alignment: Memory alignment in bytes.

    Args:
        ptr: Base pointer to global memory.
        offset: Offset from base pointer in elements.

    Returns:
        SIMD vector containing the loaded data.
    """
    return _load_impl[
        width=width,
        prefetch_size=prefetch_size,
        cache_policy=cache_policy,
        eviction_policy=eviction_policy,
        alignment=alignment,
    ](ptr + offset)


# ===-----------------------------------------------------------------------===#
# MultiMem
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
fn _get_multimem_ld_reduce_asm[
    dtype: DType,
    *,
    count: Int,
    reduction: ReduceOp,
    scope: Scope,
    consistency: Consistency,
    accum_type: DType,
    output_width: Int,
]() -> String:
    """Generates the assembly instruction string for multimem load-reduce operations.

    This internal function constructs the appropriate NVIDIA PTX assembly instruction
    string for performing vectorized load-reduce operations using the multimem feature
    available on SM90+ GPUs.

    Parameters:
        dtype: Data dtype for the operation (must be a floating point type).
        count: Vector size for PTX (corresponds to .v2, .v4, .v8 qualifiers, or no .v for scalar).
        reduction: Type of reduction operation to perform.
        scope: Memory scope for the operation.
        consistency: Memory consistency model to use.
        accum_type: Data dtype used for accumulation during reduction. Defaults to
            float32 for float16/bfloat16 inputs and matches input dtype for float32.
        output_width: Number of elements packed into a single output register (e.g. bf16x2).

    Returns:
        A string literal containing the PTX assembly instruction.

    Constraints:
        - Only supported on SM90+ GPUs.
        - Type must be a floating point type.
        - Total bit width (count * output_width * size_of[dtype] * 8) must be 32, 64, or 128 bits.
    """
    comptime assert (
        _is_sm_9x_or_newer()
    ), "multimem is only supported on SM90+ GPUs"
    comptime assert (
        dtype.is_floating_point()
    ), "multimem requires floating point type"
    comptime assert consistency in (
        Consistency.WEAK,
        Consistency.RELAXED,
        Consistency.ACQUIRE,
    ), "multimem.ld_reduce consistency must be in {weak, relaxed, acquire}"
    comptime total_bits = count * output_width * size_of[dtype]() * 8
    comptime assert total_bits in (
        32,
        64,
        128,
    ), "total bit width must be 32, 64, or 128 bits"
    comptime assert (
        dtype != DType.float64 or count == 1
    ), "float64 requires count=1 (no .vec qualifier allowed)"

    comptime ss = ".global"
    comptime vec = ".v" + _int_to_str[count]() if count > 1 else ""
    comptime op = "." + reduction.mnemonic()
    comptime dtype_mnemonic = "." + _get_type_mnemonic[dtype]() + (
        "x" + _int_to_str[output_width]() if output_width > 1 else ""
    )
    comptime accum = (
        ".acc::" + _get_type_mnemonic[accum_type]()
    ) if accum_type != dtype else ""
    comptime asm = "multimem.ld_reduce." + consistency.mnemonic() + "." + scope.mnemonic() + ss + op + accum + vec + dtype_mnemonic
    return asm


@always_inline("nodebug")
fn multimem_ld_reduce[
    dtype: DType,
    *,
    count: Int,
    reduction: ReduceOp,
    scope: Scope,
    consistency: Consistency,
    accum_type: DType = get_accum_type[dtype](),
    output_width: Int = 1,
](
    addr: UnsafePointer[
        mut=False, Scalar[dtype], address_space = AddressSpace.GLOBAL
    ],
) -> StaticTuple[SIMD[dtype, output_width], count]:
    """Performs a vectorized load-reduce operation using NVIDIA's multimem feature.

    This function loads multiple values from global memory and performs a reduction
    operation across them in a single instruction. It utilizes NVIDIA's multimem
    feature available on SM90+ GPUs for improved performance.

    Parameters:
        dtype: Data dtype for the operation (must be a floating point type).
        count: Vector size for PTX (corresponds to .v2, .v4, .v8 qualifiers, or no .v for scalar).
        reduction: Type of reduction operation to perform.
        scope: Memory scope for the operation.
        consistency: Memory consistency model to use.
        accum_type: Data dtype used for accumulation. Defaults to a wider dtype than input
                   (e.g. float32 for float16 inputs) to maintain precision during reduction.
        output_width: Number of elements packed into a single output register (e.g. bf16x2).

    Args:
        addr: Pointer to global memory where data will be loaded from.

    Returns:
        A StaticTuple containing 'count' SIMD vectors of width 'output_width'
        holding the results of the load-reduce operation.

    Constraints:
        - Only supported on SM90+ GPUs.
        - Total bit width (count * output_width * size_of[dtype] * 8) must be 32, 64, or 128 bits.
        - Type must be a floating point type.
        - float64 requires count=1 (no .vec qualifier allowed).
    """
    comptime total_bits = count * output_width * size_of[dtype]() * 8
    comptime assert total_bits in (
        32,
        64,
        128,
    ), "total bit width must be 32, 64, or 128 bits"
    comptime assert (
        dtype != DType.float64 or count == 1
    ), "float64 requires count=1 (no .vec qualifier allowed)"

    comptime asm = _get_multimem_ld_reduce_asm[
        dtype,
        count=count,
        reduction=reduction,
        scope=scope,
        consistency=consistency,
        accum_type=accum_type,
        output_width=output_width,
    ]()

    comptime if count == 1:
        var r = inlined_assembly[
            asm + " {$0}, [$1];",
            SIMD[dtype, output_width],
            constraints="=r,l,~{memory}",
            has_side_effect=True,
        ](addr.bitcast[NoneType]())
        return StaticTuple[SIMD[dtype, output_width], count](r)
    elif count == 2:
        var r = inlined_assembly[
            asm + " {$0,$1}, [$2];",
            _RegisterPackType[
                SIMD[dtype, output_width], SIMD[dtype, output_width]
            ],
            constraints="=r,=r,l,~{memory}",
            has_side_effect=True,
        ](addr.bitcast[NoneType]())
        return StaticTuple[SIMD[dtype, output_width], count](r[0], r[1])
    elif count == 4:
        var r = inlined_assembly[
            asm + " {$0,$1,$2,$3}, [$4];",
            _RegisterPackType[
                SIMD[dtype, output_width],
                SIMD[dtype, output_width],
                SIMD[dtype, output_width],
                SIMD[dtype, output_width],
            ],
            constraints="=r,=r,=r,=r,l,~{memory}",
            has_side_effect=True,
        ](addr.bitcast[NoneType]())

        return StaticTuple[SIMD[dtype, output_width], count](
            r[0], r[1], r[2], r[3]
        )
    elif count == 8:
        var r = inlined_assembly[
            asm + " {$0,$1,$2,$3,$4,$5,$6,$7}, [$8];",
            _RegisterPackType[
                SIMD[dtype, output_width],
                SIMD[dtype, output_width],
                SIMD[dtype, output_width],
                SIMD[dtype, output_width],
                SIMD[dtype, output_width],
                SIMD[dtype, output_width],
                SIMD[dtype, output_width],
                SIMD[dtype, output_width],
            ],
            constraints="=r,=r,=r,=r,=r,=r,=r,=r,l,~{memory}",
            has_side_effect=True,
        ](addr.bitcast[NoneType]())

        return StaticTuple[SIMD[dtype, output_width], count](
            r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]
        )

    return StaticTuple[SIMD[dtype, output_width], count]()


@always_inline("nodebug")
fn multimem_ld_reduce[
    dtype: DType,
    *,
    simd_width: Int,
    reduction: ReduceOp,
    scope: Scope,
    consistency: Consistency,
    accum_type: DType = get_accum_type[dtype](),
](
    addr: UnsafePointer[
        mut=False, Scalar[dtype], address_space = AddressSpace.GLOBAL
    ],
) -> SIMD[dtype, simd_width]:
    """Simplified multimem_ld_reduce that automatically calculates optimal packing.

    This wrapper automatically determines the optimal output_width and count
    parameters based on the requested simd_width and data type, using 32-bit
    word packing for efficiency.

    Parameters:
        dtype: Data type of the elements to load (must be a floating point type).
        simd_width: Total number of elements to process (must be 1, 2, 4, or 8).
        reduction: Type of reduction operation to perform.
        scope: Memory scope for the operation.
        consistency: Memory consistency model to use.
        accum_type: Data type used for accumulation.

    Args:
        addr: Pointer to global memory where data will be loaded from.

    Returns:
        A SIMD vector containing simd_width elements with the reduction results.

    Constraints:
        - Only supported on SM90+ GPUs.
        - simd_width must be 1, 2, 4, or 8.
        - Total bit width (count * output_width * size_of[dtype] * 8) must be 32, 64, or 128 bits.
        - Type must be a floating point type.
        - float64 requires count=1 (no .vec qualifier allowed).
    """
    comptime output_width = 4 // size_of[dtype]()
    comptime count = simd_width // output_width
    comptime assert simd_width in (
        1,
        2,
        4,
        8,
    ), "simd_width must be 1, 2, 4, or 8"
    comptime total_bits = count * output_width * size_of[dtype]() * 8
    comptime assert total_bits in (
        32,
        64,
        128,
    ), "total bit width must be 32, 64, or 128 bits"
    comptime assert (
        dtype != DType.float64 or count == 1
    ), "float64 requires count=1 (no .vec qualifier allowed)"

    var results = multimem_ld_reduce[
        dtype,
        count=count,
        reduction=reduction,
        scope=scope,
        consistency=consistency,
        accum_type=accum_type,
        output_width=output_width,
    ](addr)

    # Pack results into a single SIMD vector
    var result = SIMD[dtype, simd_width]()

    comptime for i in range(count):
        comptime for j in range(output_width):
            result[i * output_width + j] = results[i][j]
    return result


@always_inline("nodebug")
fn _get_multimem_st_asm[
    dtype: DType,
    *,
    count: Int,
    scope: Scope,
    consistency: Consistency,
    width: Int = 1,
]() -> String:
    comptime assert (
        _is_sm_9x_or_newer()
    ), "multimem is only supported on SM90+ GPUs"
    comptime assert (
        dtype.is_floating_point()
    ), "multimem requires floating point type"
    comptime assert consistency in (
        Consistency.WEAK,
        Consistency.RELAXED,
        Consistency.RELEASE,
    ), "multimem.st consistency must be in {weak, relaxed, release}"
    comptime total_bits = count * width * size_of[dtype]() * 8
    comptime assert total_bits in (
        32,
        64,
        128,
    ), "total bit width must be 32, 64, or 128 bits"
    comptime assert (
        dtype != DType.float64 or count == 1
    ), "float64 requires count=1 (no .vec qualifier allowed)"

    comptime ss = ".global"
    comptime vec = ".v" + _int_to_str[count]() if count > 1 else ""
    comptime dtype_mnemonic = "." + _get_type_mnemonic[dtype]() + (
        "x" + _int_to_str[width]() if width > 1 else ""
    )
    comptime asm = "multimem.st." + consistency.mnemonic() + "." + scope.mnemonic() + ss + vec + dtype_mnemonic
    return asm


@always_inline("nodebug")
fn multimem_st[
    dtype: DType,
    *,
    count: Int,
    scope: Scope,
    consistency: Consistency,
    width: Int = 1,
](
    addr: UnsafePointer[
        mut=True, Scalar[dtype], address_space = AddressSpace.GLOBAL
    ],
    values: StaticTuple[SIMD[dtype, width], count],
) -> None:
    """Stages an inline multimem.st instruction.

    This operation performs a store to all memory locations pointed to by the
    multimem address using the specified memory consistency model and scope.

    Parameters:
        dtype: The data type of elements to store (must be a floating point type).
        count: Vector size for PTX (corresponds to .v2, .v4, .v8 qualifiers, or no .v for scalar).
        scope: Memory scope for visibility of the store operation
            (CTA/Cluster/GPU/System).
        consistency: Memory consistency semantics (weak/relaxed/release).
        width: Vector width modifier for packed data types (default 1).

    Args:
        addr: Multimem address in global address space pointing to multiple
            locations.
        values: Packed SIMD values to store, with count matching the template
            parameter.

    Notes:

    - Requires SM90+ GPU architecture (PTX ISA 8.1+).
    - The address must be a valid multimem address.
    - Supported dtype-width combinations must total 32/64/128 bits.
    - Default memory semantics: weak consistency (when not specified).
    - Vector stores (.v2/.v4/.v8) require matching total size constraints.
    - float64 requires count=1 (no .vec qualifier allowed).

    Example:

    ```mojo
    from gpu.memory.memory import *

    # Store 2 float32 values to multimem address.
    multimem_st[DType.float32, count=2, scope=Scope.CTA, consistency=Consistency.RELAXED](
        addr, StaticTuple[DType.float32, 2](val1, val2)
    )

    # Vector store of 4 float16x2 values.
    multimem_st[DType.float16, count=4, scope=Scope.CLUSTER, consistency=Consistency.RELEASE, width=2](
        addr, StaticTuple[DType.float16, 4](vec1, vec2, vec3, vec4)
    )
    ```

    See Also:
        [PTX ISA Documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-multimem-ld-reduce-multimem-st-multimem-red).
    """
    comptime total_bits = count * width * size_of[dtype]() * 8
    comptime assert total_bits in (
        32,
        64,
        128,
    ), "total bit width must be 32, 64, or 128 bits"
    comptime assert (
        dtype != DType.float64 or count == 1
    ), "float64 requires count=1 (no .vec qualifier allowed)"

    comptime asm = _get_multimem_st_asm[
        dtype,
        count=count,
        scope=scope,
        consistency=consistency,
        width=width,
    ]()

    comptime if count == 1:
        inlined_assembly[
            asm + " [$0], {$1};",
            NoneType,
            constraints="l,r,~{memory}",
            has_side_effect=True,
        ](addr.bitcast[NoneType](), values[0])
    elif count == 2:
        inlined_assembly[
            asm + " [$0], {$1,$2};",
            NoneType,
            constraints="l,r,r,~{memory}",
            has_side_effect=True,
        ](addr.bitcast[NoneType](), values[0], values[1])
    elif count == 4:
        inlined_assembly[
            asm + " [$0], {$1,$2,$3,$4};",
            NoneType,
            constraints="l,r,r,r,r,~{memory}",
            has_side_effect=True,
        ](addr.bitcast[NoneType](), values[0], values[1], values[2], values[3])
    elif count == 8:
        inlined_assembly[
            asm + " [$0], {$1,$2,$3,$4,$5,$6,$7,$8};",
            NoneType,
            constraints="l,r,r,r,r,r,r,r,r,~{memory}",
            has_side_effect=True,
        ](
            addr.bitcast[NoneType](),
            values[0],
            values[1],
            values[2],
            values[3],
            values[4],
            values[5],
            values[6],
            values[7],
        )


fn multimem_st[
    dtype: DType,
    *,
    simd_width: Int,
    scope: Scope,
    consistency: Consistency,
](
    addr: UnsafePointer[
        mut=True, Scalar[dtype], address_space = AddressSpace.GLOBAL
    ],
    value: SIMD[dtype, simd_width],
):
    """Simplified multimem_st that automatically calculates optimal packing.

    This wrapper automatically determines the optimal width and count
    parameters based on the requested simd_width and data type, using 32-bit
    word packing for efficiency.

    Parameters:
        dtype: Data type of the elements to store (must be a floating point type).
        simd_width: Total number of elements to store (must be 1, 2, 4, or 8).
        scope: Memory scope for the operation.
        consistency: Memory consistency model to use.

    Args:
        addr: Multimem address in global address space pointing to multiple
            locations.
        value: SIMD vector containing simd_width elements to store.

    Constraints:
        - Only supported on SM90+ GPUs.
        - dtype must be 4 bytes or smaller (float32, float16, bfloat16, etc.).
          For float64, use the explicit width/count overload.
        - simd_width must be 1, 2, 4, or 8.
        - Total bit width (count * width * size_of[dtype] * 8) must be 32, 64, or 128 bits.
        - Type must be a floating point type.
    """
    comptime assert (
        _is_sm_9x_or_newer()
    ), "multimem is only supported on SM90+ GPUs"
    comptime assert size_of[dtype]() <= 4, (
        "dtype must be 4 bytes or smaller (use explicit width/count overload"
        " for float64)"
    )
    comptime width = 4 // size_of[dtype]()
    comptime count = simd_width // width
    comptime assert simd_width in (
        1,
        2,
        4,
        8,
    ), "simd_width must be 1, 2, 4, or 8"
    comptime total_bits = count * width * size_of[dtype]() * 8
    comptime assert total_bits in (
        32,
        64,
        128,
    ), "total bit width must be 32, 64, or 128 bits"

    # Unpack SIMD vector into StaticTuple format
    var values = StaticTuple[SIMD[dtype, width], count]()

    comptime for i in range(count):
        values[i] = value.slice[width, offset = i * width]()

    multimem_st[
        dtype,
        count=count,
        scope=scope,
        consistency=consistency,
        width=width,
    ](addr, values)


# ===-----------------------------------------------------------------------===#
# Utilities
# ===-----------------------------------------------------------------------===#


fn _get_type_mnemonic[dtype: DType]() -> StaticString:
    """Returns the mnemonic string representation for a given DType.

    This internal utility function converts floating point DTypes into their
    corresponding string mnemonics used in GPU assembly instructions.
    """
    if dtype == DType.float32:
        return "f32"
    elif dtype == DType.float16:
        return "f16"
    elif dtype == DType.bfloat16:
        return "bf16"
    elif dtype == DType.float64:
        return "f64"
    elif dtype == DType.float8_e4m3fn:
        return "e4m3"
    elif dtype == DType.float8_e5m2:
        return "e5m2"

    return "unknown dtype mnemonic"


fn _int_to_str[val: Int]() -> StaticString:
    """Converts an integer value to a static string."""
    return get_static_string[String(val)]()
