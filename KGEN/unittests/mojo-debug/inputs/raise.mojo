# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@always_inline
fn func(a: Int) raises:
    raise "Exception"  # raises


fn main() raises:
    print("will start")  # breakpoint
    func(420)
