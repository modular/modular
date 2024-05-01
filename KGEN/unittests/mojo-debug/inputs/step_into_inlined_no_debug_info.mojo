# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


fn foo(x: Int) -> Int:
    return x + 1


fn main():
    var x = foo(
        123
    )  # we need this otherwise we don't stop at the breakpoint below
    var y = x.__abs__()  # breakpoint
    print(y)  # expected after step-into
