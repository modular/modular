# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes NVIDIA GPUs memory operations."""

from math import is_power_of_2
from sys.info import alignof, simdwidthof, sizeof, triple_is_nvidia_cuda

from gpu.host._utils import _check_error
from gpu.host.result import Result
from memory import stack_allocation as _generic_stack_allocation
from memory.unsafe import DTypePointer, Pointer, _GPUAddressSpace, bitcast

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
    size: Int, type: AnyRegType
](
    src: Pointer[type, AddressSpace.GLOBAL],
    dst: Pointer[type, AddressSpace.SHARED],
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

    @parameter
    if size == 4:
        llvm_intrinsic["llvm.nvvm.cp.async.ca.shared.global.4", NoneType](
            dst, src
        )
    elif size == 8:
        llvm_intrinsic["llvm.nvvm.cp.async.ca.shared.global.8", NoneType](
            dst, src
        )
    else:
        llvm_intrinsic["llvm.nvvm.cp.async.ca.shared.global.16", NoneType](
            dst, src
        )


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
