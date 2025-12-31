# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, *_, **_]


@fieldwise_init
struct A(ImplicitlyCopyable):
    var x: UnsafePointer[Scalar[DType.invalid]]

    fn __init__(out self):
        var y = UnsafePointer[Int8].alloc(1)
        self.x = y.bitcast[Scalar[DType.invalid]]()


fn test_key_element() raises:
    var a = A()
    print("bp")  # breakpoint
    _ = a


fn main() raises:
    test_key_element()
