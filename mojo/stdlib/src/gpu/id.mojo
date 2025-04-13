# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""This module provides GPU thread and block indexing functionality.

It defines aliases and functions for accessing GPU grid, block, thread and cluster
dimensions and indices. These are essential primitives for GPU programming that allow
code to determine its position and dimensions within the GPU execution hierarchy.

Most functionality is architecture-agnostic, with some NVIDIA-specific features clearly marked.
The module is designed to work seamlessly across different GPU architectures while providing
optimal performance through hardware-specific optimizations where applicable."""

from math import fma
from os import abort
from sys.intrinsics import block_dim as _block_dim
from sys.intrinsics import block_id_in_cluster as _block_id_in_cluster
from sys.intrinsics import block_idx as _block_idx
from sys.intrinsics import cluster_dim as _cluster_dim
from sys.intrinsics import cluster_idx as _cluster_idx
from sys.intrinsics import global_idx as _global_idx
from sys.intrinsics import grid_dim as _grid_dim
from sys.intrinsics import lane_id as _lane_id
from sys.intrinsics import thread_idx as _thread_idx

from .globals import WARP_SIZE
from .warp import broadcast

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

alias block_id_in_cluster = _block_id_in_cluster
"""Contains the block id of the threadblock within a cluster, as `x`, `y`, and `z` values."""

# ===-----------------------------------------------------------------------===#
# lane_id
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
fn lane_id() -> UInt:
    """Returns the lane ID of the current thread within its warp.

    The lane ID is a unique identifier for each thread within a warp, ranging from 0 to
    WARP_SIZE-1. This ID is commonly used for warp-level programming and thread
    synchronization within a warp.

    Returns:
        The lane ID (0 to WARP_SIZE-1) of the current thread.
    """
    return _lane_id()


# ===-----------------------------------------------------------------------===#
# warp_id
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
fn warp_id() -> UInt:
    """Returns the warp ID of the current thread within its block.
    The warp ID is a unique identifier for each warp within a block, ranging
    from 0 to BLOCK_SIZE/WARP_SIZE-1. This ID is commonly used for warp-level
    programming and synchronization within a block.

    Returns:
        The warp ID (0 to BLOCK_SIZE/WARP_SIZE-1) of the current thread.
    """
    return broadcast(thread_idx.x // WARP_SIZE)


# ===-----------------------------------------------------------------------===#
# sm_id
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
fn sm_id() -> UInt:
    """Returns the Streaming Multiprocessor (SM) ID of the current thread.

    The SM ID uniquely identifies which physical streaming multiprocessor the thread is
    executing on. This is useful for SM-level optimizations and understanding hardware
    utilization.

    If called on non-NVIDIA GPUs, this function aborts as this functionality
    is only supported on NVIDIA hardware.

    Returns:
        The SM ID of the current thread.
    """

    @parameter
    if is_nvidia_gpu():
        return broadcast(
            UInt(
                Int(
                    llvm_intrinsic[
                        "llvm.nvvm.read.ptx.sreg.smid",
                        Int32,
                        has_side_effect=False,
                    ]().cast[DType.uint32]()
                )
            )
        )
    else:
        constrained[False, "The sm_id function is not supported by AMD GPUs."]()
        return abort[Int]("function not available")
