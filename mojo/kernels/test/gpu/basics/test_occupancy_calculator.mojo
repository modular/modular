# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from gpu.host.info import *
from testing import *


def a100_occupancy_test():
    assert_equal(
        A100.occupancy(threads_per_block=256, registers_per_thread=32), 1
    )
    assert_equal(
        A100.occupancy(threads_per_block=128, registers_per_thread=33), 0.75
    )
    assert_equal(
        A100.occupancy(threads_per_block=256, registers_per_thread=41), 0.625
    )


def main():
    a100_occupancy_test()
