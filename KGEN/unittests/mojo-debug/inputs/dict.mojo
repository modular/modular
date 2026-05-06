# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from debug_test_utils import keep_alive


def main():
    var d = Dict[String, Int]()
    d["one"] = 1
    d["two"] = 2
    d["three"] = 3
    keep_alive(d)  # breakpoint

    var d2 = Dict[String, Int]()
    d2["x"] = 10
    keep_alive(d2)  # breakpoint

    var d3 = Dict[String, Int]()
    keep_alive(d3)  # breakpoint
