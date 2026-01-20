# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct A(ImplicitlyCopyable):
    var x: UnsafePointer[Scalar[DType.invalid], MutExternalOrigin]

    fn __init__(out self):
        var y = alloc[Int8](1)
        self.x = y.bitcast[Scalar[DType.invalid]]()


fn test_key_element() raises:
    var a = A()
    print("bp")  # breakpoint
    _ = a


fn main() raises:
    test_key_element()
