# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: NVIDIA-GPU
# RUN: %mojo-no-debug %s

from sys import has_nvidia_gpu_accelerator

from testing import assert_true


def test_has_nvidia_gpu_accelerator():
    assert_true(has_nvidia_gpu_accelerator())


def main():
    test_has_nvidia_gpu_accelerator()
