# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from where_package import constrained_method


@no_inline
def use_constrained_method() -> Int:
    comptime result = constrained_method[2]()
    return result
