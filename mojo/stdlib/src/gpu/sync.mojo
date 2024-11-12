# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes intrinsics for NVIDIA GPUs sync instructions."""

from sys import llvm_intrinsic
from os import abort

from memory import UnsafePointer
from memory.pointer import AddressSpace

from .memory import AddressSpace as GPUAddressSpace

from ._utils import to_llvm_shared_mem_ptr, to_i32

# ===----------------------------------------------------------------------===#
# barrier
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn barrier():
    """Performs a synchronization barrier on block (equivelent to `__syncthreads`
    in CUDA).
    """

    @parameter
    if is_nvidia_gpu():
        __mlir_op.`nvvm.barrier0`()
    else:
        __mlir_op.`pop.fence`[
            _type=None,
            syncscope = "workgroup".value,
            ordering = __mlir_attr.`#pop<atomic_ordering release>`,
        ]()
        llvm_intrinsic["llvm.amdgcn.s.barrier", NoneType]()
        __mlir_op.`pop.fence`[
            _type=None,
            syncscope = "workgroup".value,
            ordering = __mlir_attr.`#pop<atomic_ordering acquire>`,
        ]()


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

    @parameter
    if is_nvidia_gpu():
        __mlir_op.`nvvm.bar.warp.sync`(
            __mlir_op.`index.casts`[_type = __mlir_type.i32](mask.value)
        )
    else:
        constrained[
            False, "The syncwarp function is not support on AMD GPUs."
        ]()


# ===----------------------------------------------------------------------===#
# mbarrier
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn _mbarrier_impl[
    type: AnyType, address_space: AddressSpace
](address: UnsafePointer[type, address_space, *_]):
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
            address.bitcast[address_space = GPUAddressSpace.GENERIC]().address
        )
    else:
        constrained[False, "invalid address space"]()


@always_inline("nodebug")
fn mbarrier[
    type: AnyType, address_space: AddressSpace
](address: UnsafePointer[type, address_space, *_]):
    """Makes the mbarrier object track all prior copy async operations initiated
    by the executing thread.

    Args:
      address: The mbarrier object is at the location.
    """

    @parameter
    if is_nvidia_gpu():
        _mbarrier_impl(address)
    else:
        constrained[
            False, "The mbarrier function is not supported by AMD GPUs."
        ]()


@always_inline("nodebug")
fn mbarrier_init[
    type: AnyType
](
    shared_mem: UnsafePointer[type, GPUAddressSpace.SHARED, *_],
    num_threads: Int32,
):
    """Initialize shared memory barrier for N number of threads.

    Args:
        shared_mem: Shared memory barrier to initialize.
        num_threads: Number of threads participating.
    """

    @parameter
    if is_nvidia_gpu():
        llvm_intrinsic["llvm.nvvm.mbarrier.init.shared", NoneType](
            shared_mem, num_threads
        )
    else:
        constrained[
            False, "The mbarrier_init function is not supported by AMD GPUs."
        ]()


@always_inline("nodebug")
fn mbarrier_arrive[
    type: AnyType
](shared_mem: UnsafePointer[type, GPUAddressSpace.SHARED, *_]) -> Int:
    """Commits the arrival of thead to a shared memory barrier.

    Args:
        shared_mem: Shared memory barrier.

    Returns:
        An Int64 value representing the state of the memory barrier.
    """

    @parameter
    if is_nvidia_gpu():
        return llvm_intrinsic["llvm.nvvm.mbarrier.arrive.shared", Int](
            shared_mem
        )
    else:
        constrained[
            False, "The mbarrier_arrive function is not supported by AMD GPUs."
        ]()
        return abort[Int]("function not available")


@always_inline("nodebug")
fn mbarrier_test_wait[
    type: AnyType
](
    shared_mem: UnsafePointer[type, GPUAddressSpace.SHARED, *_],
    state: Int,
) -> Bool:
    """Test waiting for the memory barrier.

    Args:
        shared_mem: Shared memory barrier.
        state: Memory barrier arrival state.

    Returns:
        True if all particpating thread arrived to the barrier.
    """

    @parameter
    if is_nvidia_gpu():
        return llvm_intrinsic["llvm.nvvm.mbarrier.test.wait.shared", Bool](
            shared_mem, state
        )
    else:
        constrained[
            False,
            "The mbarrier_test_wait function is not supported by AMD GPUs.",
        ]()
        return abort[Int]("function not available")


@always_inline("nodebug")
fn mbarrier_arrive_expect_tx_shared[
    type: AnyType
](addr: UnsafePointer[type, GPUAddressSpace.SHARED, *_], tx_count: Int32):
    """Performs an expect-tx operation on shared memory barrier.

    This makes the current phase of the mbarrier object to expect and
    track the completion of additional asynchronous transactions.

    Args:
        addr: Shared memory barrier address.
        tx_count: Number of element to the expect-tx operatio.
    """

    @parameter
    if is_nvidia_gpu():
        __mlir_op.`nvvm.mbarrier.arrive.expect_tx.shared`(
            to_llvm_shared_mem_ptr(addr), to_i32(tx_count)
        )
    else:
        constrained[
            False,
            (
                "The mbarrier_arrive_expect_tx_shared function is not supported"
                " by AMD GPUs."
            ),
        ]()


@always_inline("nodebug")
fn mbarrier_try_wait_parity_shared[
    type: AnyType
](
    addr: UnsafePointer[type, GPUAddressSpace.SHARED, *_],
    phase: Int32,
    ticks: Int32,
):
    """Waits for shared memory barrier till the completion of the phase
    or ticks expires.

    Args:
        addr: Shared memory barrier.
        phase: Phase number.
        ticks: Time in nano seconds.
    """

    @parameter
    if is_nvidia_gpu():
        __mlir_op.`nvvm.mbarrier.try_wait.parity.shared`(
            to_llvm_shared_mem_ptr(addr), to_i32(phase), to_i32(ticks)
        )
    else:
        constrained[
            False,
            (
                "The mbarrier_try_wait_arity_shared function is not supported"
                " by AMD GPUs."
            ),
        ]()
