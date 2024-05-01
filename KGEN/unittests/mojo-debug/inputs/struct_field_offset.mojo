# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from debug_test_utils import keep_alive


@value
struct MyPair:
    var first: Int8
    var second: Int64


fn main():
    var p = MyPair(42, 3735928559)
    keep_alive(p)  # breakpoint
