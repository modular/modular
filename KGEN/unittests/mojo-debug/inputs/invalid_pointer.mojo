# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from debug_test_utils import keep_alive


fn main():
    var base = DTypePointer[DType.float32].alloc(1)
    var ptr = base.bitcast[DType.invalid]()
    keep_alive(ptr)  # breakpoint
    base.free()
