# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module provides low-level NVIDIA GPU cluster synchronization primitives for SM90+ architectures.

The module implements thread block cluster operations that enable efficient communication and
synchronization between thread blocks (CTAs) within a cluster on NVIDIA Hopper architecture and newer GPUs.

All functions are constrained to NVIDIA SM90+ GPUs and will raise an error if used on unsupported hardware.

Note: These are low-level primitives that correspond directly to PTX/NVVM instructions and should be used
with careful consideration of the underlying hardware synchronization mechanisms.
"""
from sys import is_nvidia_gpu, llvm_intrinsic
from sys.info import _is_sm_9x_or_newer

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
