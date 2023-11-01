# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes intrinsics for NVIDIA GPUs sync instructions."""

from memory.unsafe import Pointer, DTypePointer, AddressSpace
from .memory import AddressSpace as GPUAddressSpace
from sys import llvm_intrinsic

# ===----------------------------------------------------------------------===#
# barrier
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn barrier():
    """Performs a synchronization barrier on block (equivelent to `__syncthreads`
    in CUDA).
    """
    __mlir_op.`nvvm.barrier0`()


# ===----------------------------------------------------------------------===#
# syncwarp
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn syncwarp(mask: Int = -1):
    """Causes all threads to wait until all lanes specified by the warp mask
    reach the sync warp.

    Args:
      mask: The mask of the warp lanes.
    """
    __mlir_op.`nvvm.bar.warp.sync`(
        __mlir_op.`index.casts`[_type = __mlir_type.i32](mask.value)
    )


# ===----------------------------------------------------------------------===#
# mbarrier
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn _mbarrier_impl[
    type: AnyType, address_space: AddressSpace
](address: Pointer[type, address_space]):
    """Makes the mbarrier object track all prior copy async operations initiated
    by the executing thread.

    Args:
      address: The mbarrier object is at the location.
    """

    @parameter
    if address_space == GPUAddressSpace.SHARED:
        llvm_intrinsic["llvm.nvvm.cp.async.mbarrier.arrive.shared", NoneType](
            address
        )
    elif (
        address_space == GPUAddressSpace.GLOBAL
        or address_space == GPUAddressSpace.GENERIC
    ):
        llvm_intrinsic["llvm.nvvm.cp.async.mbarrier.arrive", NoneType](
            address.address_space_cast[GPUAddressSpace.GENERIC]().address
        )
    else:
        constrained[False, "invalid address space"]()


@always_inline("nodebug")
fn mbarrier[
    type: AnyType, address_space: AddressSpace
](address: Pointer[type, address_space]):
    """Makes the mbarrier object track all prior copy async operations initiated
    by the executing thread.

    Args:
      address: The mbarrier object is at the location.
    """

    _mbarrier_impl(address)


@always_inline("nodebug")
fn mbarrier[
    type: DType, address_space: AddressSpace
](address: DTypePointer[type, address_space]):
    """Makes the mbarrier object track all prior copy async operations initiated
    by the executing thread.

    Args:
      address: The mbarrier object is at the location.
    """

    _mbarrier_impl(address.address)
