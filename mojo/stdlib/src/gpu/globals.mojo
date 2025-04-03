# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module provides GPU-specific global constants and configuration values.

The module defines hardware-specific constants like warp size and thread block limits
that are used throughout the GPU programming interface. It handles both NVIDIA and AMD
GPU architectures, automatically detecting and configuring the appropriate values based
on the available hardware.

The constants are resolved at compile time based on the target GPU architecture and
are used to optimize code generation and ensure hardware compatibility.
"""

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
"""The number of threads that execute in lockstep within a warp on the GPU.

This constant represents the hardware warp size, which is the number of threads that execute
instructions synchronously as a unit. The value is architecture-dependent:
- 32 threads per warp on NVIDIA GPUs
- 64 threads per warp on AMD GPUs
- 0 if no GPU is detected

The warp size is a fundamental parameter that affects:
- Thread scheduling and execution
- Memory access coalescing
- Synchronization primitives
- Overall performance optimization
"""


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

alias MAX_THREADS_PER_BLOCK_METADATA = _resolve_max_threads_per_block_metadata()
"""This is metadata tag that is used in conjunction with __llvm_metadata to
give a hint to the compiler about the max threads per block that's used."""


fn _resolve_max_threads_per_block_metadata() -> __mlir_type.`!kgen.string`:
    @parameter
    if is_nvidia_gpu() or has_nvidia_gpu_accelerator():
        return "nvvm.maxntid".value
    elif is_amd_gpu() or has_amd_gpu_accelerator():
        return "rocdl.flat_work_group_size".value
    else:
        constrained[False, "no accelerator detected"]()
        return "".value
