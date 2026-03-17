# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct Foo(ImplicitlyCopyable):
    var x: Int
    var y: String
    var z: Int

    def __init__(out self):
        self.x = 123
        self.y = "This is a string"
        self.z = 234


@always_inline
def func(a: Int, b: Foo) raises -> Foo:
    if a == 420:
        raise "some exception"  # breakpoint
    return b


def main() raises:
    print(func(420, Foo()).x)
