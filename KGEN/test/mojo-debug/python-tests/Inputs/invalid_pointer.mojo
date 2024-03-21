# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from debug_test_utils import keep_alive


fn main():
    var ptr = DTypePointer[DType.invalid](100)
    keep_alive(ptr)  # breakpoint
