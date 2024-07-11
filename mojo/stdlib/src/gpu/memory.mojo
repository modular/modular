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
from memory.reference import _GPUAddressSpace
from memory.unsafe import DTypePointer, bitcast

# ===----------------------------------------------------------------------===#
# AddressSpace
# ===----------------------------------------------------------------------===#

alias AddressSpace = _GPUAddressSpace

# ===----------------------------------------------------------------------===#
# cp.async
# ===----------------------------------------------------------------------===#


@always_inline
fn async_copy[
    size: Int, type: DType
](
    src: DTypePointer[type, AddressSpace.GLOBAL],
    dst: DTypePointer[type, AddressSpace.SHARED],
):
    """Asynchronously copy `size` amount of bytes from src global memory address
    to shared memory `dst` address.

    Parameters:
        size: Number of bytes to copy.
        type: The pointer type.

    Args:
        src: Global memory pointer.
        dst: Shared memory pointer.
    """
    # TODO: Constrained on device capability.
    constrained[size == 4 or size == 8 or size == 16]()

    return async_copy[size](src.address, dst.address)


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
