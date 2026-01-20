# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from debug_test_utils import keep_alive


fn main():
    var base = alloc[Float32](1)
    var ptr = base.bitcast[Scalar[DType.invalid]]()
    keep_alive(ptr)  # breakpoint
    base.free()
