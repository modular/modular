# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


fn main():
    alias length = 3

    breakpoint()
    var vector = DynamicVector[Int]()

    vector.push_back(9)
    vector.push_back(1)
    vector.push_back(2)

    var ptr = rebind[Pointer[Int]](vector.data)
