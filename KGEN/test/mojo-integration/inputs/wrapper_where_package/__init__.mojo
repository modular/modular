# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from where_package import constrained_method


@no_inline
fn use_constrained_method() -> Int:
    comptime result = constrained_method[2]()
    return result
