# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes grid-related aliases and functions. Most
of these are generic, a few are specific to NVIDIA GPUs.

You can import these APIs from the `gpu` package. For example:

```mojo
from gpu import block_idx, block_dim, thread_idx
```
"""

from os import abort
from sys.intrinsics import (
    grid_dim as _grid_dim,
    block_idx as _block_idx,
    block_dim as _block_dim,
    thread_idx as _thread_idx,
    global_idx as _global_idx,
    cluster_idx as _cluster_idx,
    cluster_dim as _cluster_dim,
    lane_id as _lane_id,
)

from gpu import WARP_SIZE
from math import fma

# ===-----------------------------------------------------------------------===#
# Aliases
# ===-----------------------------------------------------------------------===#

# Re-defining these aliases so we can attach docstrings to them in the gpu
# package.

alias grid_dim = _grid_dim
"""Provides accessors for getting the `x`, `y`, and `z`
dimensions of a grid."""

alias block_idx = _block_idx
"""Contains the block index in the grid, as `x`, `y`, and `z` values."""

alias block_dim = _block_dim
"""Contains the dimensions of the block as `x`, `y`, and `z` values (for
example, `block_dim.y`)"""

alias thread_idx = _thread_idx
"""Contains the thread index in the block, as `x`, `y`, and `z` values."""

alias global_idx = _global_idx
"""Contains the global offset of the kernel launch, as `x`, `y`, and `z`
values."""

alias cluster_idx = _cluster_idx
"""Contains the cluster index in the grid, as `x`, `y`, and `z` values."""

alias cluster_dim = _cluster_dim
"""Contains the dimensions of the cluster, as `x`, `y`, and `z` values."""

# ===-----------------------------------------------------------------------===#
# lane_id
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
fn lane_id() -> UInt:
    """Returns the lane ID of the current thread.

    Returns:
        The lane ID of the the current thread.
    """
    return _lane_id()


# ===-----------------------------------------------------------------------===#
# sm_id
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
fn sm_id() -> UInt:
    """Returns the Streaming Multiprocessor (SM) ID of the current thread.
    Returns:
        The SM ID of the the current thread.
    """

    @parameter
    if is_nvidia_gpu():
        return UInt(
            Int(
                llvm_intrinsic[
                    "llvm.nvvm.read.ptx.sreg.smid", Int32, has_side_effect=False
                ]().cast[DType.uint32]()
            )
        )
    else:
        constrained[False, "The sm_id function is not supported by AMD GPUs."]()
        return abort[Int]("function not available")


# ===-----------------------------------------------------------------------===#
#  1D ctaid in a cluster
# ===-----------------------------------------------------------------------===#


@always_inline
fn block_rank_in_cluster() -> UInt32:
    """Gets the unique identifier for the current thread block (CTA) in the
    cluster across all dimensions. Equivalent to `%cluster_ctarank` in CUDA."""

    return llvm_intrinsic[
        "llvm.nvvm.read.ptx.sreg.cluster.ctarank",
        UInt32,
        has_side_effect=False,
    ]()
