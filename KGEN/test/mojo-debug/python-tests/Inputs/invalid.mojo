# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import Dict


fn test_key_element() raises:
    var dict = Dict[DType, NoneType]()
    print("bp")  # breakpoint


fn main() raises:
    test_key_element()
