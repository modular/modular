# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes NVIDIA GPUs id operations."""

from os import abort
from sys.intrinsics import (
    grid_dim,
    block_idx,
    block_dim,
    thread_idx,
    global_idx,
    lane_id,
)

from gpu import WARP_SIZE
from math import fma

# ===-----------------------------------------------------------------------===#
# sm_id
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
fn sm_id() -> UInt:
    """Returns the SM ID of the current thread.
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
