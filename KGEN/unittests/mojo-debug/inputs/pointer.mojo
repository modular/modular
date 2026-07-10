# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from debug_test_utils import keep_alive


def main():
    var int_pointer = alloc[Int]({count = 1}).unsafe_leak()
    int_pointer[0] = 101
    keep_alive(int_pointer)  # breakpoint
