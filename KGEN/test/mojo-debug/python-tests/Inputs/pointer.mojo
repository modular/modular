# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


fn main():
    var int_pointer = Pointer[Int].alloc(1)
    int_pointer[0] = 101
    breakpoint()
