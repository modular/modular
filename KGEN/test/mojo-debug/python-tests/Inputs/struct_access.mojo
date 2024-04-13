# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from debug_test_utils import keep_alive


struct MyPair:
    var first: Int
    var second: Int

    # Make the struct go thru SROA by inlining its init.
    @always_inline("nodebug")
    fn __init__(inout self, first: Int, second: Int):
        self.first = first
        self.second = second


struct MyPairPair:
    var first: MyPair
    var second: MyPair

    @always_inline("nodebug")
    fn __init__(inout self, a: Int, b: Int, c: Int, d: Int):
        self.first = MyPair(a, b)
        self.second = MyPair(c, d)


fn use_address(ptr: Pointer[Int]):
    print(ptr.load())


fn main():
    var p = MyPair(1, 2)
    print(p.first)  # breakpoint
    p.first = 3
    p.second = 4
    print(p.second)  # breakpoint
    use_address(Reference(p.first).get_legacy_pointer())

    var pp = MyPairPair(5, 6, 7, 8)
    print(pp.second.first)  # breakpoint

    keep_alive(p, pp)
