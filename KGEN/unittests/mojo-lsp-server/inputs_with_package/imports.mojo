# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import std.builtin.simd as _simd
import std.collections.string
from .aliases import function, StructWithAlias


def main():
    # Test nested imports.
    if False:
        pass
    else:
        from std.memory import bitcast
