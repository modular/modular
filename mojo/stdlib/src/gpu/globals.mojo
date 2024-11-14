# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module includes NVIDIA GPUs global constants."""

from sys import warpsize

# ===----------------------------------------------------------------------===#
# Globals
# ===----------------------------------------------------------------------===#


alias WARP_SIZE = 32
"""The warp size of the NVIDIA hardware."""
alias WARP_SIZE_AMD = 64
"""The warp size of the AMD GPU compute hardware."""
