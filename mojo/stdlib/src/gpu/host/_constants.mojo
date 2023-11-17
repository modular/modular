# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements some common constants used throughout the package."""

# ===----------------------------------------------------------------------===#
# Globals
# ===----------------------------------------------------------------------===#


alias CUDA_DRIVER_PATH = "/usr/lib/x86_64-linux-gnu/libcuda.so"

# CUDA Device Attributes, hard coded for A100 chips
# TODO Remove these with #25800
alias CUDA_DEVICE_SM_COUNT = 108
alias CUDA_DEVICE_MAX_REGISTERS_PER_BLOCK = 65536
alias CUDA_DEVICE_MAX_THREADS_PER_SM = 2048
