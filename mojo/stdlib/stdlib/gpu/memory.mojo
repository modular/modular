# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from collections.string import StaticString
from collections.optional import OptionalReg
from collections.string.string_slice import _get_kgen_string, get_static_string
from sys import (
    alignof,
    bitwidthof,
    is_amd_gpu,
    is_gpu,
    is_nvidia_gpu,
    llvm_intrinsic,
    sizeof,
)
from sys._assembly import inlined_assembly
from sys.info import _is_sm_9x_or_newer
from sys.intrinsics import _RegisterPackType

from builtin.dtype import _uint_type_of_width
from memory import UnsafePointer
from memory.pointer import AddressSpace as _AddressSpace
from memory.pointer import _GPUAddressSpace
from memory.pointer import _GPUAddressSpace as GPUAddressSpace
from memory.unsafe import bitcast

from utils import IndexList, StaticTuple
from utils.numerics import get_accum_type

from ._utils import to_i16, to_i32, to_i64, to_llvm_ptr, to_llvm_shared_mem_ptr
from .intrinsics import Scope

# ===-----------------------------------------------------------------------===#
# AddressSpace
# ===-----------------------------------------------------------------------===#

alias AddressSpace = _GPUAddressSpace

# ===-----------------------------------------------------------------------===#
# CacheOperation
# ===-----------------------------------------------------------------------===#


@fieldwise_init
@register_passable("trivial")
struct CacheOperation:
    """Represents different GPU cache operation policies.

    This struct defines various caching behaviors for GPU memory operations,
    controlling how data is cached and evicted at different cache levels.
    The policies affect performance and memory coherency.
    """

    var _value: Int

    alias ALWAYS = Self(0)
    """Cache at all levels. This will be accessed again.

    Best for data that will be frequently reused across multiple threads.
    Provides fastest subsequent access but uses the most cache space.
    """

    alias GLOBAL = Self(1)
    """Cache at global level.

    Caches data only in the L2 cache, bypassing L1.
    Good for data shared between different thread blocks.
    """

    alias STREAMING = Self(2)
    """Streaming, this is likely to be accessed once.

    Optimizes for streaming access patterns where data is only read once.
    May bypass certain cache levels for better throughput.
    """

    alias LAST_USE = Self(3)
    """Indicates the cache line will not be used again.

    Hints to the cache that this data can be evicted after this access.
    Helps optimize cache utilization.
    """

    alias VOLATILE = Self(4)
    """Don't cache, and fetch again.

    Forces reads/writes to bypass cache and go directly to memory.
    Useful for memory-mapped I/O or when cache coherency is required.
    """

    alias WRITE_BACK = Self(5)
    """Write back at all coherent levels.

    Updates all cache levels and eventually writes to memory.
    Most efficient for multiple writes to same location.
    """

    alias WRITE_THROUGH = Self(6)
    """Write through to system memory.

    Immediately writes updates to memory while updating cache.
    Provides stronger consistency but lower performance than write-back.
    """

    fn __eq__(self, other: Self) -> Bool:
        """Tests if two CacheOperation instances are equal.

        Args:
            other: The CacheOperation to compare against.

        Returns:
            True if the operations are equal, False otherwise.
        """
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        """Tests if two CacheOperation instances are not equal.

        Args:
            other: The CacheOperation to compare against.

        Returns:
            True if the operations are not equal, False otherwise.
        """
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        """Tests if two CacheOperation instances are identical.

        Args:
            other: The CacheOperation to compare against.

        Returns:
            True if the operations are identical, False otherwise.
        """
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        """Tests if two CacheOperation instances are not identical.

        Args:
            other: The CacheOperation to compare against.

        Returns:
            True if the operations are not identical, False otherwise.
        """
        return self != other

    @always_inline
    fn mnemonic(self) -> StaticString:
        """Returns the PTX mnemonic string for this cache operation.

        Converts the cache operation into its corresponding PTX assembly
        mnemonic string used in GPU instructions.

        Returns:
            A string literal containing the PTX mnemonic for this operation.
        """
        if self is Self.ALWAYS:
            return "ca"
        if self is Self.GLOBAL:
            return "cg"
        if self is Self.STREAMING:
            return "cs"
        if self is Self.LAST_USE:
            return "lu"
        if self is Self.VOLATILE:
            return "cv"
        if self is Self.WRITE_BACK:
            return "wb"
        if self is Self.WRITE_THROUGH:
            return "wt"

        return "unknown cache operation"


# ===-----------------------------------------------------------------------===#
# CacheEviction
# ===-----------------------------------------------------------------------===#


@fieldwise_init
@register_passable("trivial")
struct CacheEviction:
    """Represents cache eviction policies for GPU memory operations.

    This struct defines different cache eviction priorities that control how data is
    evicted from cache when space is needed. The policies affect cache utilization
    and performance by controlling which data gets evicted first.
    """

    var _value: Int

    alias EVICT_NORMAL = Self(0)
    """Default cache eviction priority.

    Data cached with normal priority follows standard cache replacement policies.
    This is the default behavior and suitable for most general-purpose data access
    patterns where no special caching requirements exist.
    """

    alias EVICT_FIRST = Self(1)
    """Highest eviction priority - data will be evicted first.

    Data cached with this priority is marked as the first candidate for eviction
    when cache space is needed. This is optimal for:
    - Streaming data that will not be reused
    - Single-pass algorithms
    - Data with low temporal locality
    """

    alias EVICT_LAST = Self(2)
    """Lowest eviction priority - data will be evicted last.

    Data cached with this priority remains in cache until all higher priority data
    is evicted. Best used for:
    - Frequently accessed data
    - Data needed across multiple kernel launches
    - Critical data structures that benefit from cache persistence
    """

    alias EVICT_UNCHANGED = Self(3)
    """Preserves existing cache eviction priority.

    When this policy is used:
    - Existing cache entries maintain their current eviction priority
    - No changes are made to the cache replacement order
    - Useful for operations that should not affect caching behavior
    """

    alias NO_ALLOCATE = Self(4)
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

    fn __ne__(self, other: Self) -> Bool:
        """Tests if two CacheEviction instances are not equal.

        Args:
            other: The CacheEviction to compare against.

        Returns:
            True if the eviction policies are not equal, False otherwise.
        """
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        """Tests if two CacheEviction instances are identical.

        Args:
            other: The CacheEviction to compare against.

        Returns:
            True if the eviction policies are identical, False otherwise.
        """
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        """Tests if two CacheEviction instances are not identical.

        Args:
            other: The CacheEviction to compare against.

        Returns:
            True if the eviction policies are not identical, False otherwise.
        """
        return self != other

    @always_inline
    fn mnemonic(self) -> StaticString:
        """Returns the string mnemonic for this cache eviction policy.

        Converts the cache eviction policy into its corresponding string
        representation used in GPU instructions and debugging.

        Returns:
            A string literal containing the mnemonic for this eviction policy.
        """
        if self is Self.EVICT_NORMAL:
            return "evict_normal"
        if self is Self.EVICT_FIRST:
            return "evict_first"
        if self is Self.EVICT_LAST:
            return "evict_last"
        if self is Self.EVICT_UNCHANGED:
            return "evict_unchanged"
        if self is Self.NO_ALLOCATE:
            return "no_allocate"
        return "unknown cache eviction"


# ===-----------------------------------------------------------------------===#
# Fill
# ===-----------------------------------------------------------------------===#


@fieldwise_init
@register_passable("trivial")
struct Fill:
    """Represents memory fill patterns for GPU memory operations.

    This struct defines different fill patterns that can be used when allocating or
    initializing GPU memory. The patterns control how memory is initialized, which
    can be important for debugging and performance optimization.
    """

    var _value: Int

    alias NONE = Self(0)
    """No fill pattern - memory is left uninitialized."""

    alias ZERO = Self(1)
    """Fill memory with zeros."""

    alias NAN = Self(2)
    """Fill memory with NaN values. Useful for debugging floating point computations."""

    fn __eq__(self, other: Self) -> Bool:
        """Tests if two Fill instances have the same fill pattern.

        Args:
            other: The Fill instance to compare against.

        Returns:
            True if the fill patterns are equal, False otherwise.
        """
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        """Tests if two Fill instances have different fill patterns.

        Args:
            other: The Fill instance to compare against.

        Returns:
            True if the fill patterns are different, False otherwise.
        """
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        """Tests if two Fill instances are identical.

        Args:
            other: The Fill instance to compare against.

        Returns:
            True if the fill patterns are identical, False otherwise.
        """
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        """Tests if two Fill instances are not identical.

        Args:
            other: The Fill instance to compare against.

        Returns:
            True if the fill patterns are not identical, False otherwise.
        """
        return self != other

    @always_inline
    fn __str__(self) -> String:
        """Returns a string representation of the fill pattern.

        Converts the fill pattern into a human-readable string for debugging
        and display purposes.

        Returns:
            A string describing the fill pattern.
        """
        if self is Self.NONE:
            return "none"
        if self is Self.ZERO:
            return "zero"
        if self is Self.NAN:
            return "nan"
        return "unknown fill"


# ===-----------------------------------------------------------------------===#
# Consistency
# ===-----------------------------------------------------------------------===#


@fieldwise_init
@register_passable("trivial")
struct Consistency(Copyable, EqualityComparable, Movable):
    """Represents memory consistency models for GPU memory operations.

    This struct defines different memory consistency levels that control how memory
    operations are ordered and synchronized between threads. The consistency model
    affects both performance and correctness of parallel algorithms.
    """

    var _value: Int

    alias WEAK = Self(0)
    """Weakest consistency model with minimal ordering guarantees.

    Provides maximum flexibility for hardware/compiler optimizations but requires
    careful synchronization by the programmer."""

    alias RELAXED = Self(1)
    """Relaxed consistency with basic ordering guarantees.

    Provides some ordering guarantees while still allowing optimizations.
    Suitable for operations that don't require strict ordering."""

    alias ACQUIRE = Self(2)
    """Acquire consistency for synchronization operations.

    Ensures all subsequent memory operations are ordered after this operation.
    Used in producer-consumer patterns."""

    alias RELEASE = Self(3)
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

    fn __ne__(self, other: Self) -> Bool:
        """Tests if two Consistency instances are not equal.

        Args:
            other: The Consistency instance to compare against.

        Returns:
            True if the consistency levels are different, False otherwise.
        """
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        """Tests if two Consistency instances are identical.

        Args:
            other: The Consistency instance to compare against.

        Returns:
            True if the consistency levels are identical, False otherwise.
        """
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        """Tests if two Consistency instances are not identical.

        Args:
            other: The Consistency instance to compare against.

        Returns:
            True if the consistency levels are not identical, False otherwise.
        """
        return self != other

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
        if self is Self.WEAK:
            return "weak"
        if self is Self.RELAXED:
            return "relaxed"
        if self is Self.ACQUIRE:
            return "acquire"
        if self is Self.RELEASE:
            return "release"

        return "unknown consistency"


# ===-----------------------------------------------------------------------===#
# ReduceOp
# ===-----------------------------------------------------------------------===#


@fieldwise_init
@register_passable("trivial")
struct ReduceOp:
    """Represents reduction operations for parallel reduction algorithms.

    This struct defines different reduction operations that can be performed
    across multiple threads in parallel. These operations are commonly used
    in parallel reduction algorithms on GPUs.
    """

    var _value: Int

    alias ADD = Self(0)
    """Addition reduction operation.

    Combines values by adding them together."""

    alias MIN = Self(1)
    """Minimum reduction operation.

    Finds the minimum value across all inputs."""

    alias MAX = Self(2)
    """Maximum reduction operation.

    Finds the maximum value across all inputs."""

    alias AND = Self(3)
    """Bitwise AND reduction operation.

    Performs bitwise AND across all inputs."""

    alias OR = Self(4)
    """Bitwise OR reduction operation.

    Performs bitwise OR across all inputs."""

    alias XOR = Self(5)
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

    fn __ne__(self, other: Self) -> Bool:
        """Tests if two ReduceOp instances are not equal.

        Args:
            other: The ReduceOp instance to compare against.

        Returns:
            True if the reduction operations are different, False otherwise.
        """
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        """Tests if two ReduceOp instances are identical.

        Args:
            other: The ReduceOp instance to compare against.

        Returns:
            True if the reduction operations are identical, False otherwise.
        """
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        """Tests if two ReduceOp instances are not identical.

        Args:
            other: The ReduceOp instance to compare against.

        Returns:
            True if the reduction operations are not identical, False otherwise.
        """
        return self != other

    @always_inline
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
        if self is Self.ADD:
            return "add"
        if self is Self.MIN:
            return "min"
        if self is Self.MAX:
            return "max"
        if self is Self.AND:
            return "and"
        if self is Self.OR:
            return "or"
        if self is Self.XOR:
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

    @parameter
    if eviction_policy is CacheEviction.EVICT_NORMAL:
        return 0
    elif eviction_policy is CacheEviction.EVICT_LAST:
        return inlined_assembly[
            "createpolicy.fractional.L2::evict_last.b64 $0;",
            UInt64,
            constraints="=l",
        ]()
    else:
        constrained[
            eviction_policy is CacheEviction.EVICT_FIRST,
            "invalid eviction policy, only support normal, first, and last",
        ]()
        return inlined_assembly[
            "createpolicy.fractional.L2::evict_first.b64 $0;",
            UInt64,
            constraints="=l",
        ]()


@always_inline("nodebug")
fn async_copy[
    type: DType, //,
    size: Int,
    *,
    fill: OptionalReg[Scalar[type]] = None,
    bypass_L1_16B: Bool = True,
    l2_prefetch: OptionalReg[Int] = None,
    eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
](
    src: UnsafePointer[Scalar[type], address_space = AddressSpace.GLOBAL],
    dst: UnsafePointer[Scalar[type], address_space = AddressSpace.SHARED],
    src_size: Int32 = 0,
    predicate: Bool = False,
):
    """Asynchronously copies data from global memory to shared memory.

    This function provides a high-performance asynchronous memory copy operation with
    configurable caching behavior, prefetching, and fill values. It maps directly to
    the PTX cp.async instruction on NVIDIA GPUs.

    Parameters:
        type: The data type to copy (e.g. float32, int32).
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
        - Fill value only supported for types <= 32 bits.
        - Size must be 4, 8, or 16 bytes.
        - Cannot enable both L2 prefetch and L1 bypass.
        - L2 prefetch size must be 64, 128, or 256 bytes.
    """
    constrained[
        not fill or sizeof[type]() <= sizeof[Int32](),
        "if the fill value is specified, then the type must be 32bit or less",
    ]()
    constrained[size in (4, 8, 16)]()
    constrained[
        not (l2_prefetch.__bool__() == bypass_L1_16B == True),
        "both enable l2 prefetching and l1 bypass cannot be True",
    ]()
    constrained[
        not l2_prefetch or l2_prefetch.value() in (64, 128, 256),
        "the l2 prefetch size must be in bounds",
    ]()

    @parameter
    if is_amd_gpu():
        # Use sync load and stores for now
        # TODO(KERN-1249): add async memcopy to AMD
        alias n_scalars = size // sizeof[type]()
        var n_src_scalars = src_size // sizeof[type]()

        @parameter
        if fill:
            for i in range(n_src_scalars):
                dst.store(i, src.load(i))
            for i in range(n_src_scalars, n_scalars):
                dst.store(i, fill.value())
        else:

            @parameter
            for i in range(n_scalars):
                dst.store(i, src.load(i))
        return
    # Cache always: cache data in L1 first, then copy to shared memory.
    # Cache global: bypass L1 cache
    # We always do the latter.
    alias cache_op = CacheOperation.GLOBAL.mnemonic() if (
        bypass_L1_16B and size == 16
    ) else CacheOperation.ALWAYS.mnemonic()
    alias access_size = _int_to_str[size]()

    alias cache_hint = ".L2::cache_hint" if eviction_policy is not CacheEviction.EVICT_NORMAL else StaticString(
        ""
    )
    var cache_policy = _mark_eviction[eviction_policy]()

    alias l2_prefetch_substr = ".L2::" + _int_to_str[
        l2_prefetch.value()
    ]() + "B" if l2_prefetch else ""

    alias cp_async_asm = "cp.async." + cache_op + ".shared.global" + cache_hint + l2_prefetch_substr

    @parameter
    if Bool(fill) and Bool(fill.value() == 0):
        debug_assert(
            not predicate, "Predicate bit has to be set False for zero fill."
        )

        alias args_with_fill = " [$0], [$1], $2, $3"
        alias asm = cp_async_asm + args_with_fill

        @parameter
        if eviction_policy is CacheEviction.EVICT_NORMAL:
            inlined_assembly[asm + ";", NoneType, constraints="r,l,n,r"](
                Int32(Int(dst)), src, Int32(size), Int32(src_size)
            )
        else:
            inlined_assembly[asm + ", $4;", NoneType, constraints="r,l,n,r,l"](
                Int32(Int(dst)), src, Int32(size), Int32(src_size), cache_policy
            )
    elif fill:
        constrained[
            size == 16, "Non zero filling is supported only for 16B access."
        ]()

        # Pack filling values into 4B registers.
        @always_inline
        fn _i32_repr[fill: Scalar[type]]() -> Int32:
            @parameter
            if sizeof[type]() == 1:
                return bitcast[DType.int32, 1](
                    SIMD[type, 4](fill, fill, fill, fill)
                )
            elif sizeof[type]() == 2:
                return bitcast[DType.int32, 1](SIMD[type, 2](fill, fill))
            elif sizeof[type]() == 4:
                return bitcast[DType.int32](fill)

            return 0

        var fill_val = _i32_repr[fill.value()]()
        alias header_asm = "{\n.reg .pred p;\nsetp.ne.b32 p, $0, 0;\n"
        alias footer_asm = "@!p st.shared.v4.b32 [$1], {$4, $5, $6, $7};\n}\n"
        alias args_with_fill = " [$1], [$2], $3"
        alias copy_asm = header_asm + "@p " + cp_async_asm + args_with_fill

        @parameter
        if eviction_policy is CacheEviction.EVICT_NORMAL:
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

        alias args = " [$0], [$1], $2"
        alias asm = cp_async_asm + args

        @parameter
        if eviction_policy is CacheEviction.EVICT_NORMAL:
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

    @parameter
    if is_nvidia_gpu():
        llvm_intrinsic["llvm.nvvm.cp.async.commit.group", NoneType]()


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

    @parameter
    if is_nvidia_gpu():
        llvm_intrinsic["llvm.nvvm.cp.async.wait.group", NoneType](n)


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

    @parameter
    if is_nvidia_gpu():
        llvm_intrinsic["llvm.nvvm.cp.async.wait.all", NoneType]()


@always_inline
fn external_memory[
    type: AnyTrivialRegType,
    *,
    address_space: _AddressSpace,
    alignment: Int,
    name: StaticString = "extern_ptr_syml",
]() -> UnsafePointer[type, address_space=address_space, alignment=alignment]:
    """Gets a pointer to dynamically allocated external memory.

    This function returns a pointer to external memory that can be used for dynamic
    shared memory allocations in GPU kernels. The memory is allocated in the specified
    address space with the given alignment requirements.

    Parameters:
        type: The type of elements stored in the memory. Must be a trivial register type.
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
    var extern_ptr_symbol = UnsafePointer[
        StaticTuple[type, 0], address_space=address_space, alignment=alignment
    ](
        __mlir_op.`pop.extern_ptr_symbol`[
            _type = UnsafePointer[
                StaticTuple[type, 0],
                address_space=address_space,
                alignment=alignment,
            ]._mlir_type,
            name = _get_kgen_string[name](),
            alignment = alignment.value,
        ]()
    )
    return extern_ptr_symbol.bitcast[type]()


# ===-----------------------------------------------------------------------===#
# TMA
# ===-----------------------------------------------------------------------===#


@always_inline
fn fence_proxy_tensormap_generic_sys_acquire[
    type: AnyType,
](
    ptr: UnsafePointer[type, address_space = GPUAddressSpace.GENERIC, **_],
    size: Int32,
):
    """Acquires a system-wide memory fence for tensor map operations.

    This function establishes a memory fence that ensures proper synchronization
    between tensor map operations and system memory. It guarantees that all previous
    memory operations are completed before subsequent tensor map accesses.

    Parameters:
        type: The data type of the tensor map object being synchronized.

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
fn tma_store_fence():
    """Establishes a memory fence for shared memory stores in TMA operations.

    This function creates a memory barrier that ensures all previous shared memory
    stores are completed before subsequent TMA (Tensor Memory Access) store operations
    begin. This is crucial for maintaining memory consistency in tensor operations.

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


@always_inline
fn cp_async_bulk_tensor_shared_cluster_global[
    dst_type: AnyType,  # Type of the destination memory
    mbr_type: AnyType,  # Type of the memory barrier
    rank: Int,  # Dimensionality of the tensor (1, 2, or 3)
    /,
    *,
    cta_group: Int = 1,
](
    dst_mem: UnsafePointer[dst_type, address_space = GPUAddressSpace.SHARED],
    tma_descriptor: UnsafePointer[NoneType],
    mem_bar: UnsafePointer[mbr_type, address_space = GPUAddressSpace.SHARED],
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
    constrained[rank <= 3, "Expecting rank-1 or rank-2 tensors"]()

    constrained[cta_group in (1, 2), "cta_group must be 1 or 2"]()
    alias tma_asm = String(
        "cp.async.bulk.tensor.",
        rank,
        "d",
        ".cta_group::",
        cta_group,
        ".shared::cluster.global.mbarrier::complete_tx::bytes",
    )

    @parameter
    if rank == 3:

        @parameter
        if cta_group == 1:
            __mlir_op.`nvvm.cp.async.bulk.tensor.shared.cluster.global`[
                _properties = __mlir_attr.`{operandSegmentSizes = array<i32: 1,1,3,1,0,0,0,0>}`
            ](
                to_llvm_shared_mem_ptr(dst_mem),
                to_llvm_ptr(tma_descriptor),
                to_i32(coords[0]),
                to_i32(coords[1]),
                to_i32(coords[2]),
                to_llvm_shared_mem_ptr(mem_bar),
            )
        else:
            inlined_assembly[
                tma_asm + " [$0], [$1, {$3, $4, $5}], [$2];",
                NoneType,
                constraints="r,l,r,r,r,r",
            ](
                Int32(Int(dst_mem)),
                tma_descriptor,
                Int32(Int(mem_bar)),
                Int32(coords[0]),
                Int32(coords[1]),
                Int32(coords[2]),
            )
    elif rank == 2:

        @parameter
        if cta_group == 1:
            __mlir_op.`nvvm.cp.async.bulk.tensor.shared.cluster.global`[
                _properties = __mlir_attr.`{operandSegmentSizes = array<i32: 1,1,2,1,0,0,0,0>}`
            ](
                to_llvm_shared_mem_ptr(dst_mem),
                to_llvm_ptr(tma_descriptor),
                to_i32(coords[0]),
                to_i32(coords[1]),
                to_llvm_shared_mem_ptr(mem_bar),
            )
        else:
            inlined_assembly[
                tma_asm + " [$0], [$1, {$3, $4}], [$2];",
                NoneType,
                constraints="r,l,r,r,r",
            ](
                Int32(Int(dst_mem)),
                tma_descriptor,
                Int32(Int(mem_bar)),
                Int32(coords[0]),
                Int32(coords[1]),
            )
    else:

        @parameter
        if cta_group == 1:
            __mlir_op.`nvvm.cp.async.bulk.tensor.shared.cluster.global`[
                _properties = __mlir_attr.`{operandSegmentSizes = array<i32: 1,1,1,1,0,0,0,0>}`
            ](
                to_llvm_shared_mem_ptr(dst_mem),
                to_llvm_ptr(tma_descriptor),
                to_i32(coords[0]),
                to_llvm_shared_mem_ptr(mem_bar),
            )
        else:
            inlined_assembly[
                tma_asm + " [$0], [$1, {$3}], [$2];",
                NoneType,
                constraints="r,l,r,r",
            ](
                Int32(Int(dst_mem)),
                tma_descriptor,
                Int32(Int(mem_bar)),
                Int32(coords[0]),
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
    dst_mem: UnsafePointer[dst_type, address_space = GPUAddressSpace.SHARED],
    tma_descriptor: UnsafePointer[NoneType],
    mem_bar: UnsafePointer[mbr_type, address_space = GPUAddressSpace.SHARED],
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
    constrained[rank == 1 or rank == 2, "Expecting rank-1 or rank-2 tensors"]()

    constrained[cta_group in (1, 2), "cta_group must be 1 or 2"]()
    alias tma_asm = String(
        "cp.async.bulk.tensor.",
        rank,
        "d",
        ".cta_group::",
        cta_group,
        ".shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster",
    )

    @parameter
    if rank == 2:

        @parameter
        if cta_group == 1:
            __mlir_op.`nvvm.cp.async.bulk.tensor.shared.cluster.global`[
                _properties = __mlir_attr.`{operandSegmentSizes = array<i32: 1,1,2,1,0,1,0,0>}`
            ](
                to_llvm_shared_mem_ptr(dst_mem),
                to_llvm_ptr(tma_descriptor),
                to_i32(coords[0]),
                to_i32(coords[1]),
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
                Int32(Int(mem_bar)),
                multicast_mask,
                Int32(coords[0]),
                Int32(coords[1]),
            )
    else:

        @parameter
        if cta_group == 1:
            __mlir_op.`nvvm.cp.async.bulk.tensor.shared.cluster.global`[
                _properties = __mlir_attr.`{operandSegmentSizes = array<i32: 1,1,1,1,0,1,0,0>}`
            ](
                to_llvm_shared_mem_ptr(dst_mem),
                to_llvm_ptr(tma_descriptor),
                to_i32(coords[0]),
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
                Int32(Int(mem_bar)),
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
    src_mem: UnsafePointer[src_type, address_space = GPUAddressSpace.SHARED],
    tma_descriptor: UnsafePointer[NoneType],
    coords: IndexList[rank],
):
    """Initiates an asynchronous copy operation to transfer tensor data from shared CTA
    memory to global memory using NVIDIA's Tensor Memory Access (TMA) mechanism.

    This function provides an efficient way to write data back from shared memory to global
    memory using TMA. It supports both rank-1 and rank-2 tensors and allows control over
    cache eviction policy.

    Parameters:
        src_type: The data type of the source tensor elements.
        rank: The dimensionality of the tensor (must be 1 or 2).
        eviction_policy: Optional cache eviction policy that controls how the data is handled
                        in the cache hierarchy. Defaults to EVICT_NORMAL.

    Args:
        src_mem: Pointer to the source data in shared memory that will be copied to global
                memory. Must be properly aligned according to TMA requirements.
        tma_descriptor: Pointer to the TMA descriptor containing metadata about tensor layout
                       and memory access patterns.
        coords: Coordinates specifying which tile of the tensor to copy. For rank-1 tensors,
               this is a single coordinate. For rank-2 tensors, this contains both row and
               column coordinates.

    Notes:

    - This operation is asynchronous - use appropriate memory barriers to ensure completion.
    - Only supports rank-1 and rank-2 tensors.
    - Requires NVIDIA GPU with TMA support.
    - The source memory must be properly aligned for TMA operations.
    - The TMA descriptor must be properly initialized before use.
    """
    constrained[rank == 1 or rank == 2, "Expecting rank-1 or rank-2 tensors"]()

    alias cache_hint: Bool = eviction_policy is not CacheEviction.EVICT_NORMAL

    @parameter
    if rank == 2:
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
    src_mem: UnsafePointer[src_type, address_space = GPUAddressSpace.SHARED],
    tma_descriptor: UnsafePointer[NoneType],
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
        reduction_kind: The type of reduction operation to perform. Supported operations are:
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
    constrained[rank == 1 or rank == 2, "Expecting rank-1 or rank-2 tensors"]()
    alias cache_hint: Bool = eviction_policy is not CacheEviction.EVICT_NORMAL

    @parameter
    if rank == 2:
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
    type: DType, //,
    width: Int = 1,
    *,
    read_only: Bool = False,
    prefetch_size: OptionalReg[Int] = None,
    cache_policy: CacheOperation = CacheOperation.ALWAYS,
    eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    alignment: Int = alignof[Scalar[type]]() if is_gpu() else 1,
](ptr: UnsafePointer[Scalar[type]]) -> SIMD[type, width]:
    """Internal implementation of vectorized memory loads from global memory.

    This function provides low-level control over cache behavior and memory access patterns
    for loading data from global memory into vector registers.

    Parameters:
        type: The data type to load.
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
    constrained[
        ptr.address_space == _GPUAddressSpace.GENERIC,
        "must be global address space",
    ]()
    constrained[type.is_numeric(), "type must be numeric"]()

    @parameter
    if is_amd_gpu():
        # TODO: KERN-1230
        constrained[read_only == False]()
        return ptr.load[width=width]()

    @parameter
    if prefetch_size:
        constrained[prefetch_size.value() in (64, 128, 256)]()

    alias bytes_to_load = sizeof[type]() * width
    alias type_bitwidth = bitwidthof[type]()

    @parameter
    if bytes_to_load < sizeof[DType.uint32]():
        return ptr.load[width=width, alignment=alignment]()

    @parameter
    if type.is_floating_point() or type.is_signed():
        return bitcast[type, width](
            _load_impl[
                width=width,
                prefetch_size=prefetch_size,
                cache_policy=cache_policy,
                eviction_policy=eviction_policy,
                alignment=alignment,
            ](ptr.bitcast[Scalar[_uint_type_of_width[type_bitwidth]()]]())
        )

    @parameter
    if (
        type_bitwidth <= 16
        and sizeof[DType.uint32]() <= bytes_to_load < sizeof[DType.uint64]()
    ):
        return bitcast[type, width](
            _load_impl[
                width = (bytes_to_load // sizeof[DType.uint32]()),
                prefetch_size=prefetch_size,
                cache_policy=cache_policy,
                eviction_policy=eviction_policy,
                alignment=alignment,
            ](ptr.bitcast[UInt32]())
        )

    alias type_mnemonic = "u" + _int_to_str[type_bitwidth]()
    alias cache_policy_mnemonic = cache_policy.mnemonic()
    alias eviction_policy_mnemonic = (
        ".L1::" + eviction_policy.mnemonic()
    ) if eviction_policy != CacheEviction.EVICT_NORMAL else ""
    alias pretch_size_mnemonic = (
        ".L2::" + _int_to_str[prefetch_size.value()]() + "B"
    ) if prefetch_size else ""
    alias cache_operation = ".nc" if read_only else ""

    alias cache_policy_inst = (
        "" if cache_policy
        is CacheOperation.ALWAYS else ("." + cache_policy_mnemonic)
    )
    alias v_width = ("" if width == 1 else ".v" + _int_to_str[width]())

    alias instruction_name = "ld.global" + cache_policy_inst + cache_operation + eviction_policy_mnemonic + pretch_size_mnemonic + v_width + "." + type_mnemonic

    var res = SIMD[type, width]()

    @parameter
    if width == 1:
        var tmp = inlined_assembly[
            "ld.global " + cache_policy_inst + cache_operation + " $0, [$2];",
            Scalar[type],
            constraints="=r,l,r",
            has_side_effect=True,
        ](ptr.bitcast[NoneType](), res[0])
        return SIMD[type, width](tmp)
    elif width == 2:
        var tmp = inlined_assembly[
            instruction_name + " {$0, $1}, [$2];",
            _RegisterPackType[Scalar[type], Scalar[type]],
            constraints="=r,=r,l,r,r",
            has_side_effect=True,
        ](ptr.bitcast[NoneType](), res[0], res[1])
        return SIMD[type, width](tmp[0], tmp[1])
    elif width == 4:
        var tmp = inlined_assembly[
            instruction_name + " {$0, $1, $2, $3}, [$4];",
            _RegisterPackType[
                Scalar[type], Scalar[type], Scalar[type], Scalar[type]
            ],
            constraints="=r,=r,=r,=r,l,r,r,r,r",
            has_side_effect=True,
        ](ptr.bitcast[NoneType](), res[0], res[1], res[2], res[3])
        return SIMD[type, width](tmp[0], tmp[1], tmp[2], tmp[3])

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
    return rebind[SIMD[type, width]](lhs.join(rhs))


@always_inline
fn load[
    type: DType, //,
    width: Int = 1,
    *,
    read_only: Bool = False,
    prefetch_size: OptionalReg[Int] = None,
    cache_policy: CacheOperation = CacheOperation.ALWAYS,
    eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    alignment: Int = alignof[Scalar[type]]() if is_nvidia_gpu() else 1,
](ptr: UnsafePointer[Scalar[type]]) -> SIMD[type, width]:
    """Loads data from global memory into a SIMD vector.

    Provides a high-level interface for vectorized memory loads with configurable
    cache behavior and memory access patterns.

    Parameters:
        type: The data type to load.
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
    type: DType, //,
    width: Int = 1,
    *,
    read_only: Bool = False,
    prefetch_size: OptionalReg[Int] = None,
    cache_policy: CacheOperation = CacheOperation.ALWAYS,
    eviction_policy: CacheEviction = CacheEviction.EVICT_NORMAL,
    alignment: Int = alignof[Scalar[type]]() if is_nvidia_gpu() else 1,
](ptr: UnsafePointer[Scalar[type]], offset: OffsetType) -> SIMD[type, width]:
    """Loads data from global memory with an offset into a SIMD vector.

    Provides a high-level interface for vectorized memory loads with configurable
    cache behavior and memory access patterns, supporting offset-based addressing.

    Parameters:
        OffsetType: Type of the offset value.
        type: The data type to load.
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
    type: DType,
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
        type: Data type for the operation (float32, float16, or bfloat16).
        count: Number of elements to load and reduce (2 or 4).
        reduction: Type of reduction operation to perform.
        scope: Memory scope for the operation.
        consistency: Memory consistency model to use.
        accum_type: Data type used for accumulation during reduction. Defaults to
            float32 for float16/bfloat16 inputs and matches input type for float32.
        output_width: Width of each output SIMD vector (default 1).

    Returns:
        A string literal containing the PTX assembly instruction.

    Constraints:
        - Only supported on SM90+ GPUs.
        - Type must be float32, float16, or bfloat16.
        - Count must be 2 or 4.
    """
    constrained[
        _is_sm_9x_or_newer(), "multimem is only supported on SM90+ GPUs"
    ]()
    constrained[type.is_floating_point(), "type must be floating point"]()
    constrained[
        type in (DType.float32, DType.float16, DType.bfloat16),
        "type must be float32, float16, or bfloat16",
    ]()
    constrained[
        consistency
        in (Consistency.WEAK, Consistency.RELAXED, Consistency.ACQUIRE),
        "multimem.ld_reduce consistency must be in {weak, relaxed, acquire}",
    ]()

    alias ss = ".global"
    alias vec = ".v" + _int_to_str[count]()
    alias op = "." + reduction.mnemonic()
    alias type_mnemonic = "." + _get_type_mnemonic[type]() + (
        "x" + _int_to_str[output_width]() if output_width > 1 else ""
    )
    alias accum = (
        ".acc::" + _get_type_mnemonic[accum_type]()
    ) if accum_type is not type else ""
    alias asm = "multimem.ld_reduce." + consistency.mnemonic() + "." + scope.mnemonic() + ss + op + accum + vec + type_mnemonic
    return asm


@always_inline("nodebug")
fn multimem_ld_reduce[
    type: DType,
    *,
    count: Int,
    reduction: ReduceOp,
    scope: Scope,
    consistency: Consistency,
    accum_type: DType = get_accum_type[type](),
    output_width: Int = 1,
](
    addr: UnsafePointer[Scalar[type], address_space = AddressSpace.GLOBAL],
) -> StaticTuple[SIMD[accum_type, output_width], count]:
    """Performs a vectorized load-reduce operation using NVIDIA's multimem feature.

    This function loads multiple values from global memory and performs a reduction
    operation across them in a single instruction. It utilizes NVIDIA's multimem
    feature available on SM90+ GPUs for improved performance.

    Parameters:
        type: Data type for the operation (float32, float16, or bfloat16).
        count: Number of elements to load and reduce (2 or 4).
        reduction: Type of reduction operation to perform.
        scope: Memory scope for the operation.
        consistency: Memory consistency model to use.
        accum_type: Data type used for accumulation. Defaults to a wider type than input
                   (e.g. float32 for float16 inputs) to maintain precision during reduction.
        output_width: Width of each output SIMD vector (default 1).

    Args:
        addr: Pointer to global memory where data will be loaded from.

    Returns:
        A StaticTuple containing 'count' SIMD vectors of width 'output_width'
        holding the results of the load-reduce operation.

    Constraints:
        - Only supported on SM90+ GPUs.
        - Count must be 2 or 4.
        - Type must be float32, float16, or bfloat16.
    """
    constrained[count in (2, 4), "count must be 2 or 4"]()

    alias asm = _get_multimem_ld_reduce_asm[
        type,
        count=count,
        reduction=reduction,
        scope=scope,
        consistency=consistency,
        accum_type=accum_type,
        output_width=output_width,
    ]()

    @parameter
    if count == 2:
        var r = inlined_assembly[
            asm + " {$0,$1}, [$2];",
            _RegisterPackType[
                SIMD[accum_type, output_width], SIMD[accum_type, output_width]
            ],
            constraints="=r,=r,l,~{memory}",
            has_side_effect=True,
        ](addr.bitcast[NoneType]())
        return StaticTuple[SIMD[accum_type, output_width], count](r[0], r[1])

    @parameter
    if count == 4:
        var r = inlined_assembly[
            asm + " {$0,$1,$2,$3}, [$4];",
            _RegisterPackType[
                SIMD[accum_type, output_width],
                SIMD[accum_type, output_width],
                SIMD[accum_type, output_width],
                SIMD[accum_type, output_width],
            ],
            constraints="=r,=r,=r,=r,l,~{memory}",
            has_side_effect=True,
        ](addr.bitcast[NoneType]())

        return StaticTuple[SIMD[accum_type, output_width], count](
            r[0], r[1], r[2], r[3]
        )

    return StaticTuple[SIMD[accum_type, output_width], count]()


@always_inline("nodebug")
fn _get_multimem_st_asm[
    type: DType,
    *,
    count: Int,
    scope: Scope,
    consistency: Consistency,
    width: Int = 1,
]() -> String:
    constrained[
        _is_sm_9x_or_newer(), "multimem is only supported on SM90+ GPUs"
    ]()
    constrained[type.is_floating_point(), "type must be floating point"]()
    constrained[
        type in (DType.float32, DType.float16, DType.bfloat16),
        "type must be float32, float16, or bfloat16",
    ]()
    constrained[
        consistency
        in (Consistency.WEAK, Consistency.RELAXED, Consistency.RELEASE),
        "multimem.st consistency must be in {weak, relaxed, release}",
    ]()

    alias ss = ".global"
    alias vec = ".v" + _int_to_str[count]()
    alias type_mnemonic = "." + _get_type_mnemonic[type]() + (
        "x" + _int_to_str[width]() if width > 1 else ""
    )
    alias asm = "multimem.st." + consistency.mnemonic() + "." + scope.mnemonic() + ss + vec + type_mnemonic
    return asm


@always_inline("nodebug")
fn multimem_st[
    type: DType,
    *,
    count: Int,
    scope: Scope,
    consistency: Consistency,
    width: Int = 1,
](
    addr: UnsafePointer[Scalar[type], address_space = AddressSpace.GLOBAL],
    values: StaticTuple[SIMD[type, width], count],
) -> None:
    """Stages an inline multimem.st instruction.

    This operation performs a store to all memory locations pointed to by the
    multimem address using the specified memory consistency model and scope.

    Parameters:
        type: The data type of elements to store (must be float16, bfloat16, or
            float32).
        count: Number of vector elements per store operation (2 or 4).
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
    - Supported type-width combinations must total 32/64/128 bits.
    - Default memory semantics: weak consistency (when not specified).
    - Vector stores (.v2/.v4) require matching total size constraints.

    Example:

    ```mojo
    from gpu.memory import *

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
    constrained[count in (2, 4), "count must be 2 or 4"]()

    alias asm = _get_multimem_st_asm[
        type,
        count=count,
        scope=scope,
        consistency=consistency,
        width=width,
    ]()

    @parameter
    if count == 2:
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


# ===-----------------------------------------------------------------------===#
# Utilities
# ===-----------------------------------------------------------------------===#


fn _get_type_mnemonic[type: DType]() -> StaticString:
    """Returns the mnemonic string representation for a given DType.

    This internal utility function converts floating point DTypes into their
    corresponding string mnemonics used in GPU assembly instructions.
    """
    if type is DType.float32:
        return "f32"
    elif type is DType.float16:
        return "f16"
    elif type is DType.bfloat16:
        return "bf16"
    if type is DType.float64:
        return "f64"

    return "unknown type mnemonic"


fn _int_to_str[val: Int]() -> StaticString:
    """Converts an integer value to a static string."""
    return get_static_string[String(val)]()
