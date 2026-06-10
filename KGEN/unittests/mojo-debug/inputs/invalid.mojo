# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct A(ImplicitlyCopyable):
    var x: UnsafePointer[Scalar[DType.invalid], MutUntrackedOrigin]

    def __init__(out self):
        var y = alloc[Int8](1)
        self.x = y.bitcast[Scalar[DType.invalid]]()


def test_key_element() raises:
    var a = A()
    print("bp")  # breakpoint
    _ = a


def main() raises:
    test_key_element()
