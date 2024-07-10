# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from debug_test_utils import keep_alive


fn main():
    var int_pointer = UnsafePointer[Int].alloc(1)
    int_pointer[0] = 101
    keep_alive(int_pointer)  # breakpoint
