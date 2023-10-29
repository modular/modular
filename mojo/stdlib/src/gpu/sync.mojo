# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes intrinsics for NVIDIA GPUs sync instructions."""

from memory.unsafe import Pointer, DTypePointer
from .memory import DevicePointer, DTypeDevicePointer, AddressSpace
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
fn mbarrier[
    type: AnyType, address_space: AddressSpace
](address: DevicePointer[type, address_space]):
    """Makes the mbarrier object track all prior copy async operations initiated
    by the executing thread.

    Args:
      address: The mbarrier object is at the location.
    """

    @parameter
    if address_space == AddressSpace.SHARED:
        llvm_intrinsic["llvm.nvvm.cp.async.mbarrier.arrive.shared", NoneType](
            address
        )
    elif (
        address_space == AddressSpace.GLOBAL
        or address_space == AddressSpace.GENERIC
    ):
        llvm_intrinsic["llvm.nvvm.cp.async.mbarrier.arrive", NoneType](
            llvm_intrinsic[
                "llvm.addrspacecast", __mlir_type[`!kgen.pointer<`, type, `>`]
            ](address.address)
        )
    else:
        constrained[False, "invalid address space"]()


@always_inline("nodebug")
fn mbarrier[type: AnyType](address: Pointer[type]):
    """Makes the mbarrier object track all prior copy async operations initiated
    by the executing thread.

    Args:
      address: The mbarrier object is at the location.
    """

    llvm_intrinsic["llvm.nvvm.cp.async.mbarrier.arrive", NoneType](address)


@always_inline("nodebug")
fn mbarrier[
    type: DType, address_space: AddressSpace
](address: DTypeDevicePointer[type, address_space]):
    """Makes the mbarrier object track all prior copy async operations initiated
    by the executing thread.

    Args:
      address: The mbarrier object is at the location.
    """

    return mbarrier(address._as_scalar_pointer())


@always_inline("nodebug")
fn mbarrier[type: DType](address: DTypePointer[type]):
    """Makes the mbarrier object track all prior copy async operations initiated
    by the executing thread.

    Args:
      address: The mbarrier object is at the location.
    """

    return mbarrier(address._as_scalar_pointer())
