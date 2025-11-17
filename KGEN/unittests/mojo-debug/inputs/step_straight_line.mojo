# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory import LegacyUnsafePointer as UnsafePointer


fn main():
    comptime length = 3

    var vector = List[Int]()  # breakpoint

    vector.append(9)
    vector.append(1)
    vector.append(2)

    var ptr = rebind[UnsafePointer[Int]](vector.unsafe_ptr())
