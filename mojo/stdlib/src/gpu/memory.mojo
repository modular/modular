# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes NVIDIA GPUs memory operations."""

from collections import Optional
from memory import UnsafePointer
from sys._assembly import inlined_assembly
from memory.reference import _GPUAddressSpace
from memory.unsafe import bitcast
from utils import StaticIntTuple

# ===----------------------------------------------------------------------===#
# AddressSpace
# ===----------------------------------------------------------------------===#

alias AddressSpace = _GPUAddressSpace

# ===----------------------------------------------------------------------===#
# cp.async
# ===----------------------------------------------------------------------===#


@always_inline
fn async_copy[
    type: AnyType, //,
    size: Int,
    *,
    bypass_L1_16B: Bool = True,
    l2_prefetch: Optional[Int] = None,
](
    src: UnsafePointer[type, AddressSpace.GLOBAL],
    dst: UnsafePointer[type, AddressSpace.SHARED],
):
    """Asynchronously copy `size` amount of bytes from src global memory address
    to shared memory `dst` address.

    Parameters:
        type: The pointer type.
        size: Number of bytes to copy.
        bypass_L1_16B: Bypass the L1 cache for 16 bypes copy.
        l2_prefetch: Enable L2 prefetching and specify the size.

    Args:
        src: Global memory pointer.
        dst: Shared memory pointer.
    """
    # TODO: Constrained on device capability.
    constrained[size in (4, 8, 16)]()
    constrained[
        not (l2_prefetch.__bool__() == bypass_L1_16B == True),
        "both enable l2 prefetching and l1 bypass cannot be True",
    ]()
    constrained[
        not l2_prefetch or l2_prefetch.value() in (64, 128, 256),
        "the l2 prefetch size must be in bounds",
    ]()

    alias cache_op = CacheOperation.GLOBAL.mnemonic() if (
        bypass_L1_16B and size == 16
    ) else CacheOperation.ALWAYS.mnemonic()
    alias access_size = _int_to_str[size]()

    @parameter
    if l2_prefetch:
        alias asm = "cp.async." + cache_op + ".shared.global.L2::" + _int_to_str[
            l2_prefetch.value()
        ]() + "B [$0], [$1], $2;"
        inlined_assembly[asm, NoneType, constraints="r,l,n"](
            Int32(int(dst)), src, Int32(size)
        )
    else:
        alias intrin = "llvm.nvvm.cp.async." + cache_op + ".shared.global." + access_size
        llvm_intrinsic[intrin, NoneType](dst, src)


@always_inline
fn async_copy[
    type: AnyType, //, size: Int, *, bypass_L1_16B: Bool = True
](
    src: UnsafePointer[type, AddressSpace.GLOBAL, *_],
    dst: UnsafePointer[type, AddressSpace.SHARED, *_],
    src_size: Int32,
):
    """Asynchronously copy `size` amount of bytes from src global memory address
    to shared memory `dst` address.

    Parameters:
        type: The pointer type.
        size: Number of bytes to copy.
        bypass_L1_16B: Bypass the L1 cache for 16 bypes copy.

    Args:
        src: Global memory pointer.
        dst: Shared memory pointer.
        src_size: Size of data transferred from src. If `src_size` < `size`,
            the remainder is filled with zero. It's undefined if `src_size`
            is greater.
    """
    # TODO: Constrained on device capability.
    constrained[size == 4 or size == 8 or size == 16]()

    alias cache_op = CacheOperation.GLOBAL.mnemonic() if (
        bypass_L1_16B and size == 16
    ) else CacheOperation.ALWAYS.mnemonic()
    alias access_size = _int_to_str[size]()
    alias intrin = "llvm.nvvm.cp.async." + cache_op + ".shared.global." + access_size + ".s"
    llvm_intrinsic[intrin, NoneType](dst, src, src_size)


@always_inline
fn async_copy_commit_group():
    """Commits all prior initiated but uncommitted cp.async instructions into
    a cp.async-group.
    """
    llvm_intrinsic["llvm.nvvm.cp.async.commit.group", NoneType]()


@always_inline
fn async_copy_wait_group(n: Int32):
    """Wait for the completion of `n` or asynchronous copy operations.

    Args:
        n: The number of pending cp.async-groups.
    """
    llvm_intrinsic["llvm.nvvm.cp.async.wait.group", NoneType](n)


@always_inline
fn async_copy_wait_all():
    """Wait for the completion of all commited cp.async-groups."""
    llvm_intrinsic["llvm.nvvm.cp.async.wait.all", NoneType]()


@always_inline
fn dynamic_shared_memory[
    type: AnyType,
    alignment: Int,
]() -> UnsafePointer[type, _GPUAddressSpace.SHARED]:
    """Gets a pointer to dynamic shared memory.

    Parameters:
        type: The pointer's type.
        alignment: The pointer's address alignment.

    Returns:
        A pointer to dynamic shared memory.
    """
    return __mlir_op.`pop.extern_ptr_symbol`[
        _type = UnsafePointer[type, _GPUAddressSpace.SHARED]._mlir_type,
        alignment = alignment.value,
    ]()


# ===----------------------------------------------------------------------===#
# TMA
# ===----------------------------------------------------------------------===#


@always_inline
fn __to_llvm_shared_mem_ptr[
    type: AnyType
](
    ptr: UnsafePointer[type, GPUAddressSpace.SHARED, *_]
) -> __mlir_type.`!llvm.ptr<3>`:
    """Cast shared memory pointer to LLVMPointer Type.

    Args:
        ptr: Shared memory pointer.

    Returns:
        A pointer of type !llvm.ptr<3>.
    """
    return __mlir_op.`builtin.unrealized_conversion_cast`[
        _type = __mlir_type.`!llvm.ptr<3>`
    ](ptr)


@always_inline
fn __to_llvm_ptr[
    type: AnyType
](ptr: UnsafePointer[type]) -> __mlir_type.`!llvm.ptr`:
    """Cast a pointer to LLVMPointer Type.

    Args:
        ptr: A pointer.

    Returns:
        A pointer of type !llvm.ptr.
    """
    return __mlir_op.`builtin.unrealized_conversion_cast`[
        _type = __mlir_type.`!llvm.ptr`
    ](ptr)


@always_inline
fn __to_i32(val: Int32) -> __mlir_type.i32:
    """Cast Scalar I32 value into MLIR i32.

    Args:
        val: Scalar I32 value.

    Returns:
       Input casted to MLIR i32 value.
    """
    return __mlir_op.`pop.cast_to_builtin`[_type = __mlir_type.`i32`](val.value)


@always_inline
fn cp_async_bulk_tensor_shared_cluster_global[
    dst_type: AnyType, mbr_type: AnyType, rank: Int
](
    dst_mem: UnsafePointer[dst_type, GPUAddressSpace.SHARED],
    tma_descriptor: UnsafePointer[NoneType],
    mem_bar: UnsafePointer[mbr_type, GPUAddressSpace.SHARED],
    coords: StaticIntTuple[rank],
):
    """Initiates an asynchronous copy operation on the tensor data from global
    memory to shared memory.

    Args:
        dst_mem: Pointer to destination shared memory.
        tma_descriptor: Pointer to tensor map descriptor.
        mem_bar: A pointer to shared memory barrier.
        coords: Tile coordinates.
    """
    constrained[rank == 1 or rank == 2, "Expecting rank-1 or rank-2 tensors"]()

    @parameter
    if rank == 2:
        __mlir_op.`nvvm.cp.async.bulk.tensor.shared.cluster.global`[
            _properties = __mlir_attr.`{operandSegmentSizes = array<i32: 1,1,2,1,0,0,0,0>}`
        ](
            __to_llvm_shared_mem_ptr(dst_mem),
            __to_llvm_ptr(tma_descriptor),
            __to_i32(coords[0]),
            __to_i32(coords[1]),
            __to_llvm_shared_mem_ptr(mem_bar),
        )
    else:
        __mlir_op.`nvvm.cp.async.bulk.tensor.shared.cluster.global`[
            _properties = __mlir_attr.`{operandSegmentSizes = array<i32: 1,1,1,1,0,0,0,0>}`
        ](
            __to_llvm_shared_mem_ptr(dst_mem),
            __to_llvm_ptr(tma_descriptor),
            __to_i32(coords[0]),
            __to_llvm_shared_mem_ptr(mem_bar),
        )


# ===----------------------------------------------------------------------===#
# CacheOperation
# ===----------------------------------------------------------------------===#


@value
struct CacheOperation:
    var _value: Int

    alias ALWAYS = Self(0)
    """Cache at all levels. This will be accessed again."""

    alias GLOBAL = Self(1)
    """Cache at global level."""

    alias STREAMING = Self(2)
    """Streaming, this is likely to be accessed once."""

    alias LAST_USE = Self(3)
    """Indicates the cache line will not be used again."""

    alias VOLATILE = Self(4)
    """Don't cache, and fetch again."""

    alias WRITE_BACK = Self(5)
    """Write back at all coherent levels."""

    alias WRITE_THROUGH = Self(6)
    """Write through to system memory."""

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @always_inline
    fn mnemonic(self) -> StringLiteral:
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


# ===----------------------------------------------------------------------===#
# CacheEviction
# ===----------------------------------------------------------------------===#


@value
struct CacheEviction:
    var _value: Int

    alias EVICT_NORMAL = Self(0)
    """Cache data with normal eviction priority."""

    alias EVICT_FIRST = Self(1)
    """Data cached with this priority will be first in the eviction priority
    order and will likely be evicted when cache eviction is required. This
    priority is suitable for streaming data."""

    alias EVICT_LAST = Self(2)
    """Data cached with this priority will be last in the eviction priority
    order and will likely be evicted only after other data with EVICT_NORMAL
    or EVICT_FIRST eviction priotity is already evicted. This priority is
    suitable for data that should remain persistent in cache."""

    alias EVICT_UNCHANGED = Self(3)
    """Do not change eviction priority order as part of this operation."""

    alias NO_ALLOCATE = Self(4)
    """Do not allocate data to cache. This priority is suitable for streaming
    data."""

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __is__(self, other: Self) -> Bool:
        return self == other

    fn __isnot__(self, other: Self) -> Bool:
        return self != other

    @always_inline
    fn mnemonic(self) -> StringLiteral:
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


# ===----------------------------------------------------------------------===#
# Utilities
# ===----------------------------------------------------------------------===#


fn _int_to_str[val: Int]() -> StringLiteral:
    constrained[val in (4, 8, 16, 32, 64, 128)]()

    @parameter
    if val == 4:
        return "4"
    elif val == 8:
        return "8"
    elif val == 16:
        return "16"
    elif val == 32:
        return "32"
    elif val == 64:
        return "64"
    elif val == 128:
        return "128"

    return "Unknown"
