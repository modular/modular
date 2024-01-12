# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


fn main():
    let int_pointer = Pointer[Int].alloc(1)
    int_pointer[0] = 101
    print("end")  # breakpoint
