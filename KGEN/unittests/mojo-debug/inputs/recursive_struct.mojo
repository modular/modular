# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from debug_test_utils import keep_alive


@fieldwise_init
struct Foo:
    var x: Int

    @__allow_legacy_any_origin_fields
    var ptr: UnsafePointer[Foo, MutAnyOrigin]


struct Bar:
    var x: Int

    @__allow_legacy_any_origin_fields
    var ptr: UnsafePointer[Bar, MutAnyOrigin]

    def __init__(out self, x: Int, ptr: UnsafePointer[Bar, MutAnyOrigin]):
        self.x = x
        self.ptr = ptr


def main():
    var f1: Foo = Foo(7, UnsafePointer[Foo, MutAnyOrigin].unsafe_dangling())
    var f2: Foo = Foo(8, UnsafePointer(to=f1))
    print(f2.ptr[].x)

    var b1: Bar = Bar(22, UnsafePointer[Bar, MutAnyOrigin].unsafe_dangling())
    var b2: Bar = Bar(23, UnsafePointer(to=b1))
    print(b2.ptr[].x)  # breakpoint

    keep_alive(f1, f2, b1, b2)
