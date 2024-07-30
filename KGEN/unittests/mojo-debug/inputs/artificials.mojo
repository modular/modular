# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@value
struct Foo:
    var x: Int
    var y: String
    var z: Int

    fn __init__(inout self):
        self.x = 123
        self.y = "This is a string"
        self.z = 234


@always_inline
fn func(a: Int, b: Foo) raises -> Foo:
    if a == 420:
        raise "some exception"  # breakpoint
    return b


fn main() raises:
    print(func(420, Foo()).x)
