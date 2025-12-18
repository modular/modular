# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@no_inline
fn constrained_method[n: Int]() -> Int where n > 0:
    return n
