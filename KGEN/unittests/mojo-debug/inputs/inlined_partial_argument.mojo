# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@always_inline
fn modify(inout x: Int):
    x = 42


fn use_ints(x: Int, y: Int):
    pass


@value
@register_passable("trivial")
struct MyPair:
    var x: Int
    var y: Int


fn main():
    var p = MyPair(3, 4)
    modify(p.x)
    use_ints(p.x, p.y)  # breakpoint
