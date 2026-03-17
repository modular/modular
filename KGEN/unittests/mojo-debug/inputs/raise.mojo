# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


def func(a: Int) raises:
    raise "Exception"  # raises


def main() raises:
    print("will start")  # breakpoint
    func(420)
