# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes NVIDIA SM90+ GPUs cluster related operations."""
from sys import is_nvidia_gpu, llvm_intrinsic
from sys.info import _is_sm_9x

# ===-----------------------------------------------------------------------===#
#  1D ctaid in a cluster
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
fn block_rank_in_cluster() -> UInt32:
    """Gets the unique identifier for the current thread block (CTA) in the
    cluster across all dimensions. Equivalent to `%cluster_ctarank` in CUDA."""

    constrained[
        _is_sm_9x(),
        "block rank identifier is only supported by NVIDIA SM90+ GPUs",
    ]()

    return llvm_intrinsic[
        "llvm.nvvm.read.ptx.sreg.cluster.ctarank",
        UInt32,
        has_side_effect=False,
    ]()


@always_inline("nodebug")
fn elect_one_sync() -> Bool:
    constrained[
        _is_sm_9x(),
        "elect one sync is only implemented for NVIDIA SM90+ GPUs",
    ]()
    return Bool(__mlir_op.`nvvm.elect.sync`[_type = __mlir_type.`i1`]())


@always_inline("nodebug")
fn cluster_arrive_relaxed():
    constrained[
        _is_sm_9x(),
        "cluster arrive relaxed is only supported by NVIDIA SM90+ GPUs",
    ]()
    __mlir_op.`nvvm.cluster.arrive.relaxed`[
        _type=None,
        aligned = __mlir_attr.unit,
    ]()


@always_inline("nodebug")
fn cluster_arrive():
    constrained[
        _is_sm_9x(),
        "cluster arrive is only supported by NVIDIA SM90+ GPUs",
    ]()
    __mlir_op.`nvvm.cluster.arrive`[
        _type=None,
        aligned = __mlir_attr.unit,
    ]()


@always_inline("nodebug")
fn cluster_wait():
    constrained[
        _is_sm_9x(),
        "cluster wait is only supported by NVIDIA SM90+ GPUs",
    ]()
    __mlir_op.`nvvm.cluster.wait`[
        _type=None,
        aligned = __mlir_attr.unit,
    ]()


@always_inline("nodebug")
fn cluster_sync():
    cluster_arrive()
    cluster_wait()


@always_inline("nodebug")
fn cluster_sync_relaxed():
    cluster_arrive_relaxed()
    cluster_wait()
