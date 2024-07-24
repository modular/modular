# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes NVIDIA GPUs memory operations."""

from sys.info import alignof, simdwidthof, sizeof, triple_is_nvidia_cuda

from gpu.host._utils import _check_error
from gpu.host.result import Result
from memory import stack_allocation as _generic_stack_allocation
from memory import UnsafePointer
from memory.reference import _GPUAddressSpace
from memory.unsafe import bitcast

# ===----------------------------------------------------------------------===#
# AddressSpace
# ===----------------------------------------------------------------------===#

alias AddressSpace = _GPUAddressSpace

# ===----------------------------------------------------------------------===#
# cp.async
# ===----------------------------------------------------------------------===#


@always_inline
fn async_copy[
    size: Int, type: AnyType, bypass_L1_16B: Bool = True
](
    src: UnsafePointer[type, AddressSpace.GLOBAL],
    dst: UnsafePointer[type, AddressSpace.SHARED],
):
    """Asynchronously copy `size` amount of bytes from src global memory address
    to shared memory `dst` address.

    Parameters:
        size: Number of bytes to copy.
        type: The pointer type.
        bypass_L1_16B: Bypass the L1 cache for 16 bypes copy.

    Args:
        src: Global memory pointer.
        dst: Shared memory pointer.
    """
    # TODO: Constrained on device capability.
    constrained[size == 4 or size == 8 or size == 16]()

    alias cache_op = "cg" if (bypass_L1_16B and size == 16) else "ca"
    alias access_size = "4" if size == 4 else ("8" if size == 8 else "16")
    alias command = "llvm.nvvm.cp.async." + cache_op + ".shared.global." + access_size
    llvm_intrinsic[command, NoneType](dst, src)


@always_inline
fn async_copy[
    size: Int, type: AnyType, bypass_L1_16B: Bool = True
](
    src: UnsafePointer[type, AddressSpace.GLOBAL],
    dst: UnsafePointer[type, AddressSpace.SHARED],
    src_size: Int32,
):
    """Asynchronously copy `size` amount of bytes from src global memory address
    to shared memory `dst` address.

    Parameters:
        size: Number of bytes to copy.
        type: The pointer type.
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

    alias cache_op = "cg" if (bypass_L1_16B and size == 16) else "ca"
    alias access_size = "4" if size == 4 else ("8" if size == 8 else "16")
    alias command = "llvm.nvvm.cp.async." + cache_op + ".shared.global." + access_size + ".s"
    llvm_intrinsic[command, NoneType](dst, src, src_size)


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
fn cp_async_bulk_tensor_shared_cluser_global[
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
        dst_mem: Pointer to destnation shared memory.
        tma_descriptor: Pointer to tensor map descripotr.
        mem_bar: A pointer to shared memory memory barrier.
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
