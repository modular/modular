# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from debug_test_utils import keep_alive


fn get_bool() -> Bool:
    return True


fn main():
    var true = True
    var false = False
    var other = get_bool()
    keep_alive(true, false, other)  # breakpoint
