# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes NVIDIA GPUs global constants."""

from sys.info import (
    has_amd_gpu_accelerator,
    has_nvidia_gpu_accelerator,
    is_amd_gpu,
    is_nvidia_gpu,
)

from .host.info import DEFAULT_GPU, DEFAULT_GPU_ARCH

# ===-----------------------------------------------------------------------===#
# WARP_SIZE
# ===-----------------------------------------------------------------------===#


alias WARP_SIZE = _resolve_warp_size()
"""The warp size of the GPU."""


fn _resolve_warp_size() -> Int:
    @parameter
    if is_nvidia_gpu():
        return 32
    elif is_amd_gpu():
        return 64
    elif DEFAULT_GPU_ARCH == "":
        return 0
    else:
        return DEFAULT_GPU.warp_size


# ===-----------------------------------------------------------------------===#
# MAX_THREADS_PER_BLOCK_METADATA
# ===-----------------------------------------------------------------------===#

alias MAX_THREADS_PER_BLOCK_METADATA = _resolve_max_threads_per_block_metadata().value
"""This is metadata tag that is used in conjunction with __llvm_metadata to
give a hint to the compiler about the max threads per block that's used."""


fn _resolve_max_threads_per_block_metadata() -> StringLiteral:
    @parameter
    if is_nvidia_gpu() or has_nvidia_gpu_accelerator():
        return "nvvm.maxntid"
    elif is_amd_gpu() or has_amd_gpu_accelerator():
        return "rocdl.flat_work_group_size"
    else:
        constrained[False, "no acclerator detected"]()
        return ""
