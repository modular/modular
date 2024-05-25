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


fn main():
    var f1: Foo = Foo(7, UnsafePointer[Foo]())
    var f2: Foo = Foo(8, UnsafePointer[Foo](f1))
    print(f2.ptr[].x)  # breakpoint

    keep_alive(f1, f2)
