# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes intrinsics for NVIDIA GPUs sync instructions."""

from os import abort
from sys import is_amd_gpu, is_nvidia_gpu, llvm_intrinsic
from sys.info import _is_sm_9x
from sys.param_env import env_get_bool

from memory import UnsafePointer
from memory.pointer import AddressSpace

from ._utils import to_i32, to_llvm_shared_mem_ptr
from .memory import AddressSpace as GPUAddressSpace

# ===-----------------------------------------------------------------------===#
# barrier
# ===-----------------------------------------------------------------------===#

alias _USE_EXPERIMENTAL_AMD_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM = env_get_bool[
    "USE_EXPERIMENTAL_AMD_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM", False
]()


@always_inline("nodebug")
fn barrier():
    """Performs a synchronization barrier on block (equivalent to `__syncthreads`
    in CUDA).
    """

    @parameter
    if is_nvidia_gpu():
        __mlir_op.`nvvm.barrier0`()
    elif _USE_EXPERIMENTAL_AMD_BLOCK_SYNC_LDS_WITHOUT_SYNC_VMEM:
        constrained[is_amd_gpu()]()
        llvm_intrinsic["llvm.amdgcn.s.waitcnt", NoneType](Int32(0xC07F))
        llvm_intrinsic["llvm.amdgcn.s.barrier", NoneType]()
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


@value
@register_passable("trivial")
struct AMDScheduleBarrierMask:
    """Represents different instruction scheduling masks for AMDGPU scheduling instructions.
    These masks control which instructions can be reordered across the barrier.
    """

    var _value: Int32
    # Barrier scheduling control flags
    alias NONE = Self(0)  # No instructions across barrier
    alias ALL_ALU = Self(1 << 0)  # All non-memory, non-side-effect instructions
    alias VALU = Self(1 << 1)  # Vector ALU instructions
    alias SALU = Self(1 << 2)  # Scalar ALU instructions
    alias MFMA = Self(1 << 3)  # Matrix FMA/WMMA instructions
    alias ALL_VMEM = Self(1 << 4)  # All vector memory instructions
    alias VMEM_READ = Self(1 << 5)  # Vector memory read instructions
    alias VMEM_WRITE = Self(1 << 6)  # Vector memory write instructions
    alias ALL_DS = Self(1 << 7)  # All LDS instructions
    alias DS_READ = Self(1 << 8)  # LDS read instructions
    alias DS_WRITE = Self(1 << 9)  # LDS write instructions
    alias TRANS = Self(1 << 10)  # Transcendental instructions

    @implicit
    fn __init__(out self, value: Int):
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __str__(self) -> String:
        if self == Self.NONE:
            return "NONE"
        elif self == Self.ALL_ALU:
            return "ALL_ALU"
        elif self == Self.VALU:
            return "VALU"
        elif self == Self.SALU:
            return "SALU"
        elif self == Self.MFMA:
            return "MFMA"
        elif self == Self.ALL_VMEM:
            return "ALL_VMEM"
        elif self == Self.VMEM_READ:
            return "VMEM_READ"
        elif self == Self.VMEM_WRITE:
            return "VMEM_WRITE"
        elif self == Self.ALL_DS:
            return "ALL_DS"
        elif self == Self.DS_READ:
            return "DS_READ"
        elif self == Self.DS_WRITE:
            return "DS_WRITE"
        elif self == Self.TRANS:
            return "TRANS"
        else:
            return abort[String]("invalid AMDScheduleBarrierMask value")

    fn __int__(self) -> Int:
        return Int(self._value)


@always_inline("nodebug")
fn schedule_barrier(
    mask: AMDScheduleBarrierMask = AMDScheduleBarrierMask.NONE,
):
    """Controls instruction scheduling across the barrier. The mask parameter
    specifies which instruction types can cross the barrier.
    Args:
        mask: Bit mask specifying which instruction types can cross the barrier.
    """

    @parameter
    if is_amd_gpu():
        llvm_intrinsic["llvm.amdgcn.sched.barrier", NoneType](Int32(Int(mask)))
    else:
        constrained[False, "schedule_barrier is only supported on AMDGPU."]()


@always_inline("nodebug")
fn schedule_group_barrier(
    mask: AMDScheduleBarrierMask, size: Int32, sync_id: Int32
):
    """Controls instruction scheduling by creating schedule groups with custom sequence.
    The intrinsic applies to the code that precedes it.

    Args:
        mask: Instruction mask value, same as schedule_barrier masks.
        size: Number of times to repeat the instruction.
        sync_id: Group ID for ordering instructions in sequence within the same group.
    """

    @parameter
    if is_amd_gpu():
        llvm_intrinsic["llvm.amdgcn.sched.group.barrier", NoneType](
            Int32(Int(mask)), size, sync_id
        )
    else:
        constrained[
            False, "schedule_group_barrier is only supported on AMDGPU."
        ]()


# ===-----------------------------------------------------------------------===#
# syncwarp
# ===-----------------------------------------------------------------------===#


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
        # In AMD GPU this is a nop (everything executed in lock-step).
        return


# ===-----------------------------------------------------------------------===#
# mbarrier
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
fn _mbarrier_impl[
    type: AnyType, address_space: AddressSpace
](address: UnsafePointer[type, address_space=address_space, **_]):
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
](address: UnsafePointer[type, address_space=address_space, **_]):
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
            False, "The mbarrier function is not supported on AMD GPUs."
        ]()


@always_inline("nodebug")
fn mbarrier_init[
    type: AnyType
](
    shared_mem: UnsafePointer[
        type, address_space = GPUAddressSpace.SHARED, **_
    ],
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
            False, "The mbarrier_init function is not supported on AMD GPUs."
        ]()


@always_inline("nodebug")
fn mbarrier_arrive[
    type: AnyType
](
    shared_mem: UnsafePointer[type, address_space = GPUAddressSpace.SHARED, **_]
) -> Int:
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
            False, "The mbarrier_arrive function is not supported on AMD GPUs."
        ]()
        return abort[Int]("function not available")


@always_inline("nodebug")
fn mbarrier_test_wait[
    type: AnyType
](
    shared_mem: UnsafePointer[
        type, address_space = GPUAddressSpace.SHARED, **_
    ],
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
            "The mbarrier_test_wait function is not supported on AMD GPUs.",
        ]()
        return abort[Int]("function not available")


@always_inline("nodebug")
fn mbarrier_arrive_expect_tx_shared[
    type: AnyType
](
    addr: UnsafePointer[type, address_space = GPUAddressSpace.SHARED, **_],
    tx_count: Int32,
):
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
                " on AMD GPUs."
            ),
        ]()


@always_inline("nodebug")
fn mbarrier_try_wait_parity_shared[
    type: AnyType
](
    addr: UnsafePointer[type, address_space = GPUAddressSpace.SHARED, **_],
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
                " on AMD GPUs."
            ),
        ]()


@always_inline
fn cp_async_bulk_commit_group():
    """Commits all prior initiated but uncommitted cp.async.bulk instructions into a cp.async.bulk-group.
    """

    @parameter
    if is_nvidia_gpu():
        __mlir_op.`nvvm.cp.async.bulk.commit.group`[_type=None]()
    else:
        constrained[
            False,
            (
                "The cp_async_bulk_commit_group function is not supported"
                " on AMD GPUs."
            ),
        ]()


@always_inline
fn cp_async_bulk_wait_group[n: Int32, read: Bool = True]():
    """Causes the executing thread to wait until only N or fewer of the most recent bulk async-groups
    are pending and all the prior bulk async-groups committed by the executing threads are complete
    When N is 0, the executing thread waits on all the prior bulk async-groups to complete.
    """

    @parameter
    fn get_asm() -> StringLiteral:
        alias base = "llvm.nvvm.cp.async.bulk.wait.group"
        if read:
            return base + ".read"
        return base

    @parameter
    if is_nvidia_gpu():
        llvm_intrinsic[
            get_asm(),
            NoneType,
        ](n)

    else:
        constrained[
            False,
            (
                "The cp_async_bulk_commit_group function is not supported"
                " on AMD GPUs."
            ),
        ]()
