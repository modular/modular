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
"""This module provides GPU synchronization primitives and barriers.

The module includes:
- Block-level synchronization barriers (barrier())
- Warp-level synchronization (syncwarp())
- Memory barriers (mbarrier) for NVIDIA GPUs
- Instruction scheduling controls for AMD GPUs
- Asynchronous copy and bulk transfer synchronization

The synchronization primitives help coordinate execution between threads within
thread blocks and warps, and manage memory consistency across different memory spaces.
"""

from os import abort
from os.atomic import Consistency
from sys import is_amd_gpu, is_nvidia_gpu, llvm_intrinsic
from sys._assembly import inlined_assembly
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
fn named_barrier[num_threads: Int32, id: Int32 = 0]():
    """Performs a named synchronization barrier at the block level.

    This function creates a synchronization point using a specific barrier ID, allowing
    for multiple independent barriers within a thread block. All threads in the block
    must execute this function with the same barrier ID and thread count before any
    thread can proceed past the barrier.

    Parameters:
        num_threads: The number of threads that must reach the barrier before any can proceed.
        id: The barrier identifier (0-16). Default is 0.

    Notes:

        - Only supported on NVIDIA GPUs.
        - Maps directly to the `nvvm.barrier` instruction.
        - Useful for fine-grained synchronization when different subsets of threads
          need to synchronize independently.
        - The barrier ID must not exceed 16.
        - All threads participating in the barrier must specify the same num_threads value.
    """
    constrained[id <= 16, "barrier id should not exceed 16"]()
    constrained[
        is_nvidia_gpu(), "named barrier is only supported by NVIDIA GPUs"
    ]()
    __mlir_op.`nvvm.barrier`[
        _properties = __mlir_attr.`{operandSegmentSizes = array<i32: 1,1>}`
    ](to_i32(id), to_i32(num_threads))


@always_inline("nodebug")
fn barrier():
    """Performs a synchronization barrier at the block level.

    This is equivalent to __syncthreads() in CUDA. All threads in a thread block must
    execute this function before any thread can proceed past the barrier. This ensures
    memory operations before the barrier are visible to all threads after the barrier.
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
            ordering = Consistency.RELEASE.__mlir_attr(),
        ]()
        llvm_intrinsic["llvm.amdgcn.s.barrier", NoneType]()
        __mlir_op.`pop.fence`[
            _type=None,
            syncscope = "workgroup".value,
            ordering = Consistency.ACQUIRE.__mlir_attr(),
        ]()


@fieldwise_init
@register_passable("trivial")
struct AMDScheduleBarrierMask(Intable):
    """Represents different instruction scheduling masks for AMDGPU scheduling instructions.

    These masks control which types of instructions can be reordered across a barrier for
    performance optimization. When used with schedule_barrier(), the mask determines which
    instructions the compiler is allowed to move across the barrier point.
    """

    var _value: Int32
    """Internal value storage for the barrier mask."""

    alias NONE = Self(0)
    """No instructions can cross the barrier. Most restrictive option."""

    alias ALL_ALU = Self(1 << 0)
    """Allows reordering of all arithmetic and logic instructions that don't involve memory operations."""

    alias VALU = Self(1 << 1)
    """Permits reordering of vector arithmetic/logic unit instructions only."""

    alias SALU = Self(1 << 2)
    """Permits reordering of scalar arithmetic/logic unit instructions only."""

    alias MFMA = Self(1 << 3)
    """Allows reordering of matrix multiplication and WMMA instructions."""

    alias ALL_VMEM = Self(1 << 4)
    """Enables reordering of all vector memory operations (reads and writes)."""

    alias VMEM_READ = Self(1 << 5)
    """Allows reordering of vector memory read operations only."""

    alias VMEM_WRITE = Self(1 << 6)
    """Allows reordering of vector memory write operations only."""

    alias ALL_DS = Self(1 << 7)
    """Permits reordering of all Local Data Share (LDS) operations."""

    alias DS_READ = Self(1 << 8)
    """Enables reordering of LDS read operations only."""

    alias DS_WRITE = Self(1 << 9)
    """Enables reordering of LDS write operations only."""

    alias TRANS = Self(1 << 10)
    """Allows reordering of transcendental instructions (sin, cos, exp, etc)."""

    @implicit
    fn __init__(out self, value: Int):
        """Initializes an `AMDScheduleBarrierMask` from an integer value.

        This implicit constructor allows creating a barrier mask directly from an integer,
        which is useful for combining multiple mask flags using bitwise operations.

        Args:
            value: The integer value to use for the barrier mask.
        """
        self._value = value

    fn __eq__(self, other: Self) -> Bool:
        """Compares two `AMDScheduleBarrierMask` instances for equality.

        Args:
            other: The other `AMDScheduleBarrierMask` to compare with.

        Returns:
            True if the masks have the same value, False otherwise.
        """
        return self._value == other._value

    fn __ne__(self, other: Self) -> Bool:
        """Compares two `AMDScheduleBarrierMask` instances for inequality.

        Args:
            other: The other `AMDScheduleBarrierMask` to compare with.

        Returns:
            True if the masks have different values, False otherwise.
        """
        return not (self == other)

    fn __str__(self) -> String:
        """Returns a string representation of the `AMDScheduleBarrierMask`.

        Converts the mask to a human-readable string based on its value.

        Returns:
            A string representation of the mask, or aborts if the value is invalid.
        """
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
        """Converts the `AMDScheduleBarrierMask` to an integer.

        Returns:
            The integer value of the mask, which can be used with low-level APIs.
        """
        return Int(self._value)


@always_inline("nodebug")
fn schedule_barrier(
    mask: AMDScheduleBarrierMask = AMDScheduleBarrierMask.NONE,
):
    """Controls instruction scheduling across a barrier point in AMD GPU code.

    This function creates a scheduling barrier that controls which types of instructions
    can be reordered across it by the compiler. The mask parameter specifies which
    instruction categories (ALU, memory, etc) are allowed to cross the barrier during
    scheduling optimization.

    Args:
        mask: A bit mask of AMDScheduleBarrierMask flags indicating which instruction
            types can be scheduled across this barrier. Default is NONE, meaning no
            instructions can cross.

    Note:
        This function only has an effect on AMD GPUs. On other platforms it will
        raise a compile time error.
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
    """Controls instruction scheduling across a barrier point in AMD GPU code by creating schedule groups.

    This function creates a scheduling barrier that groups instructions into sequences with custom ordering.
    It affects the code that precedes the barrier. The barrier ensures instructions are scheduled according
    to the specified group parameters.

    Args:
        mask: A bit mask of AMDScheduleBarrierMask flags indicating which instruction types can be
            scheduled across this barrier. Similar to schedule_barrier masks.
        size: The number of times to repeat the instruction sequence in the schedule group.
        sync_id: A unique identifier for the group that determines the ordering of instructions
            within the same schedule group.

    Note:
        This function only has an effect on AMD GPUs. On other platforms it will raise a compile time error.
        The sync_id parameter allows creating multiple schedule groups that can be ordered relative to each other.
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
    """Synchronizes threads within a warp using a barrier.

    This function creates a synchronization point where threads in a warp must wait until all
    threads specified by the mask reach this point. On NVIDIA GPUs, it uses warp-level
    synchronization primitives. On AMD GPUs, this is a no-op since threads execute in lock-step.

    Args:
        mask: An integer bitmask specifying which lanes (threads) in the warp should be
            synchronized. Each bit corresponds to a lane, with bit i controlling lane i.
            A value of 1 means the lane participates in the sync, 0 means it does not.
            Default value of -1 (all bits set) synchronizes all lanes.

    Note:
        - On NVIDIA GPUs, this maps to the nvvm.bar.warp.sync intrinsic.
        - On AMD GPUs, this is a no-op since threads execute in lock-step.
        - Threads not participating in the sync must still execute the instruction.
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
    """Internal implementation for making a memory barrier track async operations.

    This is an internal helper function that implements the core memory barrier tracking
    functionality for different address spaces.

    Args:
        address: Pointer to the memory barrier object location.
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
fn async_copy_arrive[
    type: AnyType, address_space: AddressSpace
](address: UnsafePointer[type, address_space=address_space, **_]):
    """Makes a memory barrier track all prior async copy operations from this thread.

    This function ensures that all previously initiated asynchronous copy operations
    from the executing thread are tracked by the memory barrier at the specified location.
    Only supported on NVIDIA GPUs.

    Parameters:
        type: The data type stored at the barrier location.
        address_space: The memory address space where the barrier is located.

    Args:
        address: Pointer to the memory barrier object location.
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
    """Initialize a shared memory barrier for synchronizing multiple threads.

    Sets up a memory barrier in shared memory that will be used to synchronize
    the specified number of threads. Only supported on NVIDIA GPUs.

    Parameters:
        type: The data type stored at the barrier location.

    Args:
        shared_mem: Pointer to shared memory location for the barrier.
        num_threads: Number of threads that will synchronize on this barrier.
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
    """Signal thread arrival at a shared memory barrier.

    Records that the calling thread has reached the barrier synchronization point.
    Only supported on NVIDIA GPUs.

    Parameters:
        type: The data type stored at the barrier location.

    Args:
        shared_mem: Pointer to the shared memory barrier.

    Returns:
        An integer representing the current state of the memory barrier.
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
    """Test if all threads have arrived at the memory barrier.

    Non-blocking check to see if all participating threads have reached the barrier.
    Only supported on NVIDIA GPUs.

    Parameters:
        type: The data type stored at the barrier location.

    Args:
        shared_mem: Pointer to the shared memory barrier.
        state: Expected state of the memory barrier.

    Returns:
        True if all threads have arrived, False otherwise.
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
        return abort[Bool]("function not available")


@always_inline("nodebug")
fn mbarrier_arrive_expect_tx_shared[
    type: AnyType  # The type of the memory barrier
](
    addr: UnsafePointer[type, address_space = GPUAddressSpace.SHARED, **_],
    tx_count: Int32,
):
    """Configure a shared memory barrier to expect additional async transactions.

    Updates the current phase of the memory barrier to track completion of
    additional asynchronous transactions. Only supported on NVIDIA GPUs.

    Parameters:
        type: The type of the memory barrier.

    Args:
        addr: Pointer to the shared memory barrier.
        tx_count: Number of expected transactions to track.
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
    type: AnyType  # The type of the memory barrier
](
    addr: UnsafePointer[type, address_space = GPUAddressSpace.SHARED, **_],
    phase: Int32,
    ticks: Int32,
):
    """Wait for completion of a barrier phase with timeout.

    Waits for the shared memory barrier to complete the specified phase,
    or until the timeout period expires. Only supported on NVIDIA GPUs.

    Parameters:
        type: The type of the memory barrier.

    Args:
        addr: Pointer to the shared memory barrier.
        phase: Phase number to wait for.
        ticks: Timeout period in nanoseconds.
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

    This function commits all previously initiated but uncommitted cp.async.bulk instructions into a
    cp.async.bulk-group. The cp.async.bulk instructions are used for asynchronous bulk memory transfers
    on NVIDIA GPUs.

    The function creates a synchronization point for bulk memory transfers, allowing better control over
    memory movement and synchronization between different stages of computation.

    Note:
        This functionality is only available on NVIDIA GPUs. Attempting to use this function on
        non-NVIDIA GPUs will result in a compile time error.
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
    """Waits for completion of asynchronous bulk memory transfer groups.

    This function causes the executing thread to wait until a specified number of the most recent
    bulk async-groups are pending. It provides synchronization control for bulk memory transfers
    on NVIDIA GPUs.

    Parameters:
        n: The number of most recent bulk async-groups allowed to remain pending. When n=0,
           waits for all prior bulk async-groups to complete.
        read: If True, indicates that subsequent reads to the transferred memory are expected,
              enabling optimizations for read access patterns. Defaults to True.

    Note:
        This functionality is only available on NVIDIA GPUs. Attempting to use this function on
        non-NVIDIA GPUs will result in a compile time error.

    Example:
        ```mojo
        from gpu.sync import cp_async_bulk_wait_group

        # Wait until at most 2 async groups are pending
        cp_async_bulk_wait_group[2]()

        # Wait for all async groups to complete
        cp_async_bulk_wait_group[0]()
        ```
    """

    @parameter
    fn get_asm() -> String:
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
