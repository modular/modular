# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@always_inline
def func(a: Int) raises:
    raise "Exception"  # raises


def main() raises:
    print("will start")  # breakpoint
    func(5)
