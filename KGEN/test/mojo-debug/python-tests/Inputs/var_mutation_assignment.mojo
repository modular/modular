# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


fn main():
    var i = 5
    var j = 7
    breakpoint()
    print(i)

    i *= 3
    breakpoint()
    print(i)

    j += 6
    breakpoint()
    print(j)

    i -= j
    breakpoint()
    print(i)
