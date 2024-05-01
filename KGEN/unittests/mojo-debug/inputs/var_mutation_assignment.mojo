# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from debug_test_utils import keep_alive


fn main():
    var i = 5
    var j = 7
    print(i)  # breakpoint

    i *= 3
    print(i)  # breakpoint

    j += 6
    print(j)  # breakpoint

    i -= j
    print(i)  # breakpoint
    keep_alive(i, j)
