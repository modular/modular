# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory import DTypePointer


@value
struct A:
    var x: DTypePointer[DType.invalid]

    fn __init__(inout self):
        var y = DTypePointer[DType.int8].alloc(1)
        self.x = y.bitcast[DType.invalid]()


fn test_key_element() raises:
    var a = A()
    print("bp")  # breakpoint


fn main() raises:
    test_key_element()
