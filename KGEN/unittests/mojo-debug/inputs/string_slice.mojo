# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from debug_test_utils import keep_alive


def main():
    # StaticString is StringSlice[False, StaticConstantOrigin] -- exercises the
    # formatter for a non-empty and an empty slice.
    var s1: StaticString = "static_string"
    var s2: StaticString = ""
    keep_alive(s1, s2)  # breakpoint
