# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: AMD-GPU
# RUN: %mojo-no-debug %s

from sys import has_amd_gpu

from testing import assert_true


def test_has_amd_gpu():
    assert_true(has_amd_gpu())


def main():
    test_has_amd_gpu()
