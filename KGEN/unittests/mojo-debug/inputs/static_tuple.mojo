# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from utils import StaticTuple
from debug_test_utils import keep_alive


fn main():
    var tuple = StaticTuple[Int16, 4](1, 2, 3, 4)
    var simd = SIMD[DType.int16, 4](1, 2, 3, 4)
    keep_alive(tuple, simd)  # breakpoint
