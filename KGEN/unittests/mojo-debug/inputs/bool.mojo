# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from debug_test_utils import keep_alive


def get_bool() -> Bool:
    return True


def main():
    var true = True
    var false = False
    var other = get_bool()
    keep_alive(true, false, other)  # breakpoint
