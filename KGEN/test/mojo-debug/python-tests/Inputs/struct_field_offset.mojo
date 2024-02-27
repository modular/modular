# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@value
struct MyPair:
    var first: Int8
    var second: Int64


fn main():
    var p = MyPair(42, 3735928559)
    print(p.first)  # breakpoint
