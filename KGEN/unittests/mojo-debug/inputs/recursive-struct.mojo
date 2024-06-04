# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from debug_test_utils import keep_alive
from memory import UnsafePointer


@value
struct Foo:
    var x: Int
    var ptr: UnsafePointer[Foo]


struct Bar:
    var x: Int
    var ptr: UnsafePointer[Bar]

    fn __init__(inout self, x: Int, ptr: UnsafePointer[Bar]):
        self.x = x
        self.ptr = ptr


fn main():
    var f1: Foo = Foo(7, UnsafePointer[Foo]())
    var f2: Foo = Foo(8, UnsafePointer[Foo].address_of(f1))
    print(f2.ptr[].x)

    var b1: Bar = Bar(22, UnsafePointer[Bar]())
    var b2: Bar = Bar(23, UnsafePointer[Bar].address_of(b1))
    print(b2.ptr[].x)  # breakpoint

    keep_alive(f1, f2, b1, b2)
