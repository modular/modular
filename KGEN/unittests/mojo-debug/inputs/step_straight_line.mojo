# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


fn main():
    comptime length = 3

    var vector = List[Int]()  # breakpoint

    vector.append(9)
    vector.append(1)
    vector.append(2)

    var ptr = vector.unsafe_ptr()
