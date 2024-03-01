# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


fn main():
    var tuple = StaticTuple[Int16, 4](1, 2, 3, 4)
    var simd = SIMD[DType.int16, 4](1, 2, 3, 4)
    print("bp")  # breakpoint
