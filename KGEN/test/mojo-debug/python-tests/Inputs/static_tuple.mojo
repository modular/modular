# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


fn main():
    var tuple = StaticTuple[4, Int16](1, 2, 3, 4)
    var simd = SIMD[DType.int16, 4](1, 2, 3, 4)
    print("bp")  # breakpoint
