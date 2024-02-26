# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


from math import abs


fn foo(x: Int) -> Int:
    return x + 1


fn main():
    var x = foo(
        123
    )  # we need this otherwise we don't stop at the breakpoint below
    breakpoint()
    var y = abs(x)
    print(y)  # expected after step-into
