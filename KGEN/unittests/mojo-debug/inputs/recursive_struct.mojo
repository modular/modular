# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from debug_test_utils import keep_alive


@fieldwise_init
struct Foo:
    var x: Int
    var ptr: UnsafePointer[Foo, MutAnyOrigin]


struct Bar:
    var x: Int
    var ptr: UnsafePointer[Bar, MutAnyOrigin]

    fn __init__(out self, x: Int, ptr: UnsafePointer[Bar, MutAnyOrigin]):
        self.x = x
        self.ptr = ptr


fn main():
    var f1: Foo = Foo(7, {})
    var f2: Foo = Foo(8, UnsafePointer(to=f1))
    print(f2.ptr[].x)

    var b1: Bar = Bar(22, {})
    var b2: Bar = Bar(23, UnsafePointer(to=b1))
    print(b2.ptr[].x)  # breakpoint

    keep_alive(f1, f2, b1, b2)
