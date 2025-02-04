# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes NVIDIA GPUs global constants."""

from sys.info import is_amd_gpu, is_nvidia_gpu

from .host.info import DEFAULT_GPU, DEFAULT_GPU_ARCH

# ===-----------------------------------------------------------------------===#
# Globals
# ===-----------------------------------------------------------------------===#


alias WARP_SIZE = _resolve_warp_size()
"""The warp size of the NVIDIA hardware."""


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
