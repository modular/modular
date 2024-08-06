# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import builtin.simd as _simd
import builtin.string
from .aliases import function, StructWithAlias


fn main():
    # Test nested imports.
    if False:
        pass
    else:
        from memory import UnsafePointer
