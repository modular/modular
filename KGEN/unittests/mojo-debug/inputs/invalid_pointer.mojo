# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from debug_test_utils import keep_alive
from std.memory import dealloc


def main():
    var base_alloc = alloc[Float32]({count = 1})
    var base = base_alloc.unsafe_ptr()
    var ptr = base.bitcast[Scalar[DType.invalid]]()
    keep_alive(ptr)  # breakpoint
    dealloc(base_alloc^)
