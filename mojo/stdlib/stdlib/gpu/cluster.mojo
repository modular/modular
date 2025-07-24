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
"""This module provides low-level NVIDIA GPU cluster synchronization primitives for SM90+ architectures.

The module implements thread block cluster operations that enable efficient communication and
synchronization between thread blocks (CTAs) within a cluster on NVIDIA Hopper architecture and newer GPUs.

All functions are constrained to NVIDIA SM90+ GPUs and will raise an error if used on unsupported hardware.

Note: These are low-level primitives that correspond directly to PTX/NVVM instructions and should be used
with careful consideration of the underlying hardware synchronization mechanisms.
"""
from sys import llvm_intrinsic, _RegisterPackType
from sys.info import _is_sm_9x_or_newer
from sys.info import _is_sm_100x_or_newer
from sys._assembly import inlined_assembly
from gpu.memory import _GPUAddressSpace as AddressSpace
from utils.index import IndexList, product

# ===-----------------------------------------------------------------------===#
#  1D ctaid in a cluster
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
fn block_rank_in_cluster() -> UInt32:
    """Returns the unique identifier (rank) for the current thread block within its cluster.

    Returns:
        A unique identifier in the range [0, cluster_size-1] where `cluster_size`
        is the total number of thread blocks in the cluster.

    Note:
        - Only supported on NVIDIA SM90+ GPUs.
        - Maps directly to the `%cluster_ctarank` special register in CUDA PTX.
    """

    constrained[
        _is_sm_9x_or_newer(),
        "block rank identifier is only supported by NVIDIA SM90+ GPUs",
    ]()

    return llvm_intrinsic[
        "llvm.nvvm.read.ptx.sreg.cluster.ctarank",
        UInt32,
        has_side_effect=False,
    ]()


@always_inline("nodebug")
fn elect_one_sync() -> Bool:
    """Elects a single thread within a warp to perform an operation.

    Returns:
        True for the elected thread, False for all other threads in the warp.

    Note:
        - Only supported on NVIDIA SM90+ GPUs.
        - Maps directly to the `elect.sync` instruction in CUDA PTX.
        - Useful for having a single thread perform an operation while
          maintaining warp synchronization.
    """
    constrained[
        _is_sm_9x_or_newer(),
        "elect one sync is only implemented for NVIDIA SM90+ GPUs",
    ]()
    return Bool(__mlir_op.`nvvm.elect.sync`[_type = __mlir_type.`i1`]())


@always_inline("nodebug")
fn elect_one_sync_with_mask(mask: UInt32 = 0xFFFFFFFF) -> Bool:
    """Elects a single thread within a warp to perform an operation.

    Args:
        mask: The mask to use for the election. Defaults to 0xFFFFFFFF.

    Returns:
        True for the elected thread, False for all other threads in the warp.

    Note:
        - Only supported on NVIDIA SM90+ GPUs.
        - Maps directly to the `elect.sync` instruction in CUDA PTX.
        - Useful for having a single thread perform an operation while
          maintaining warp synchronization.
    """
    constrained[
        _is_sm_9x_or_newer(),
        "elect one sync is only implemented for NVIDIA SM90+ GPUs",
    ]()

    alias asm = """{
        .reg .pred P1;
        elect.sync _|P1, $1;
        selp.b32 $0, 1, 0, P1;
        }"""
    var is_elected: UInt32 = inlined_assembly[
        asm, UInt32, has_side_effect=True, constraints="=r,r"
    ](mask)
    return Bool(is_elected)


@always_inline("nodebug")
fn cluster_arrive_relaxed():
    """Signals arrival at a cluster synchronization point with relaxed memory ordering.

    This is a relaxed version of cluster_arrive() that does not enforce memory ordering
    guarantees. It should be used when memory ordering is not required between thread blocks
    in the cluster. Only supported on NVIDIA SM90+ GPUs.
    """
    constrained[
        _is_sm_9x_or_newer(),
        "cluster arrive relaxed is only supported by NVIDIA SM90+ GPUs",
    ]()
    __mlir_op.`nvvm.cluster.arrive.relaxed`[
        _type=None,
        aligned = __mlir_attr.unit,
    ]()


@always_inline("nodebug")
fn cluster_arrive():
    """Signals arrival at a cluster synchronization point with memory ordering guarantees.

    This function ensures all prior memory operations from this thread block are visible to
    other thread blocks in the cluster before proceeding. Only supported on NVIDIA SM90+ GPUs.
    """
    constrained[
        _is_sm_9x_or_newer(),
        "cluster arrive is only supported by NVIDIA SM90+ GPUs",
    ]()
    __mlir_op.`nvvm.cluster.arrive`[
        _type=None,
        aligned = __mlir_attr.unit,
    ]()


@always_inline("nodebug")
fn cluster_wait():
    """Waits for all thread blocks in the cluster to arrive at the synchronization point.

    This function blocks until all thread blocks in the cluster have called cluster_arrive()
    or cluster_arrive_relaxed(). Only supported on NVIDIA SM90+ GPUs.
    """
    constrained[
        _is_sm_9x_or_newer(),
        "cluster wait is only supported by NVIDIA SM90+ GPUs",
    ]()
    __mlir_op.`nvvm.cluster.wait`[
        _type=None,
        aligned = __mlir_attr.unit,
    ]()


@always_inline("nodebug")
fn cluster_sync():
    """Performs a full cluster synchronization with memory ordering guarantees.

    This is a convenience function that combines cluster_arrive() and cluster_wait()
    to provide a full barrier synchronization across all thread blocks in the cluster.
    Ensures memory ordering between thread blocks. Only supported on NVIDIA SM90+ GPUs.
    """
    cluster_arrive()
    cluster_wait()


@always_inline("nodebug")
fn cluster_sync_relaxed():
    """Performs a full cluster synchronization with relaxed memory ordering.

    This is a convenience function that combines cluster_arrive_relaxed() and cluster_wait()
    to provide a barrier synchronization across all thread blocks in the cluster without
    memory ordering guarantees. Only supported on NVIDIA SM90+ GPUs.
    """
    cluster_arrive_relaxed()
    cluster_wait()


@always_inline("nodebug")
fn cluster_sync_acquire():
    """Acquires the cluster sync proxy.

    Only supported on NVIDIA SM90+ GPUs.
    """
    constrained[
        _is_sm_9x_or_newer(),
        "cluster sync acquire is only supported by NVIDIA SM90+ GPUs",
    ]()
    inlined_assembly[
        "fence.proxy.async::generic.acquire.sync_restrict::shared::cluster.cluster;",
        NoneType,
        has_side_effect=True,
        constraints="",
    ]()


@always_inline("nodebug")
fn cluster_sync_release():
    """Release the cluster sync proxy.

    Only supported on NVIDIA SM90+ GPUs."""
    constrained[
        _is_sm_9x_or_newer(),
        "cluster sync release is only supported by NVIDIA SM90+ GPUs",
    ]()
    inlined_assembly[
        "fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster;",
        NoneType,
        has_side_effect=True,
        constraints="",
    ]()


@always_inline("nodebug")
fn clusterlaunchcontrol_query_cancel_is_canceled(
    result: UnsafePointer[UInt128, address_space = AddressSpace.SHARED]
) -> UInt32:
    """Decodes the cancellation request.

    Args:
        result: A pointer to `UInt128` that make up the cancellation request result to decode.

    Returns:
        True if the cancellation request is canceled, False otherwise.

    Only supported on NVIDIA SM100+ GPUs."""
    constrained[
        _is_sm_100x_or_newer(),
        (
            "clusterlaunchcontrol_query_cancel_is_canceled is only supported by"
            "  NVIDIA SM100+ GPUs"
        ),
    ]()

    var ret_val = inlined_assembly[
        """
    {
    .reg .pred p1;
    .reg .b128 clc_result;
    ld.shared.b128 clc_result, [$1];
    clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_result;
    selp.b32 $0, 1, 0, p1;
    }
    """,
        UInt32,
        has_side_effect=True,
        constraints="=r,r",
    ](Int32(Int(result)))
    return ret_val


@always_inline("nodebug")
fn clusterlaunchcontrol_query_cancel_get_first_ctaid[
    id: String
](
    result: UnsafePointer[UInt128, address_space = AddressSpace.SHARED]
) -> UInt32:
    """Decodes the cancellation request.

    Parameters:
        id: The dimension to decode. Must be one of `x`, `y`, `z`.

    Args:
        result: A pointer to `UInt128` that make up the cancellation request result to decode.

    Returns:
        The coordinate of the first CTAID in the canceled cluster.

    Only supported on NVIDIA SM100+ GPUs."""
    constrained[
        _is_sm_100x_or_newer(),
        (
            "clusterlaunchcontrol_query_cancel_get_first_ctaid is only"
            " supported by NVIDIA SM100+ GPUs"
        ),
    ]()
    constrained[
        id == "x" or id == "y" or id == "z",
        "id must be one of `x`, `y`, `z`",
    ]()

    alias asm = (
        """
        {
        .reg .b128 %result;
        ld.shared.b128 %result, [$1];
        clusterlaunchcontrol.query_cancel.get_first_ctaid::"""
        + id
        + """.b32.b128 $0, %result;
        }
        """
    )

    var ret_val = inlined_assembly[
        asm,
        UInt32,
        has_side_effect=True,
        constraints="=r,r",
    ](Int32(Int(result)))
    return ret_val


@always_inline("nodebug")
fn clusterlaunchcontrol_query_cancel_get_first_ctaid_v4(
    result: UnsafePointer[UInt128, address_space = AddressSpace.SHARED],
) -> Tuple[UInt32, UInt32, UInt32]:
    """Decodes the cancellation request.

    Args:
        result: A pointer to `UInt128` that make up the cancellation request result to decode.

    Only supported on NVIDIA SM100+ GPUs."""
    constrained[
        _is_sm_100x_or_newer(),
        (
            "clusterlaunchcontrol_query_cancel_get_first_ctaid_v4 is only"
            " supported by NVIDIA SM100+ GPUs"
        ),
    ]()

    alias asm = """{
        .reg .b128 result;
        ld.shared.b128 result, [$3];
        clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {$0, $1, $2, _}, result;
        }"""

    var coordinates = inlined_assembly[
        asm,
        _RegisterPackType[UInt32, UInt32, UInt32],
        has_side_effect=True,
        constraints="=r,=r,=r,l",
    ](Int32(Int(result)))

    return Tuple[UInt32, UInt32, UInt32](
        coordinates[0],
        coordinates[1],
        coordinates[2],
    )


@always_inline("nodebug")
fn clusterlaunchcontrol_try_cancel[
    multicast: Bool = False
](
    result: UnsafePointer[UInt128, address_space = AddressSpace.SHARED],
    mbar: UnsafePointer[Int64, address_space = AddressSpace.SHARED],
):
    """Requests to atomically cancel the cluster launch if it has not started running yet.

    Args:
        result: A pointer to `UInt128` (16B aligned) that will store the result of the cancellation request.
        mbar: A pointer to an `Int64` (8B aligned) memory barrier state.

    Only supported on NVIDIA SM100+ GPUs."""
    constrained[
        _is_sm_100x_or_newer(),
        (
            "clusterlaunchcontrol_query_cancel_get_first_ctaid_v4 is only"
            " supported by NVIDIA SM100+ GPUs"
        ),
    ]()

    alias asm = (
        """
        clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes"""
        + (".multicast::cluster::all" if multicast else "")
        + """.b128 [$0], [$1];"""
    )

    inlined_assembly[
        asm,
        NoneType,
        has_side_effect=True,
        constraints="r,r",
    ](Int32(Int(result)), Int32(Int(mbar)))


@always_inline("nodebug")
fn cluster_mask_base[
    cluster_shape: IndexList[3],
    axis: Int,
]() -> UInt16:
    """Computes the base mask for a cluster. Base mask in an axis masks
    the first cta in cluster and all ctas along the same axis.
    Example for cluster shape (4, 4, 1), note that cta rank is contiguous
    along the first cluster axis.

         x o o o                       x x x x
         x o o o                       o o o o
         x o o o                       o o o o
         x o o o                       o o o o
    base mask in axis 0          base mask in axis 1


    Parameters:
        cluster_shape: The shape of the cluster.
        axis: The axis to compute the base mask for.

    Returns:
        The base mask for the cluster.

    """
    constrained[
        axis in (0, 1),
        "axis must be one of 0, 1",
    ]()

    constrained[
        product(cluster_shape) <= 16,
        "cluster size must be less than or equal to 16",
    ]()

    @parameter
    if axis == 0:
        return (1 << cluster_shape[0]) - 1

    var mask: UInt16 = 1

    @parameter
    for i in range(cluster_shape[1]):
        mask |= mask << (i * cluster_shape[0])

    return mask
