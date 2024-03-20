# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


@register_passable("trivial")
struct Point(CollectionElement):
    var x: Int
    var y: Int

    fn __init__(x: Int, y: Int) -> Self:
        return Point {x: x, y: y}  # breakpoint


fn main():
    var p1 = Point(1, -1)
    var p2 = Point(2, -2)
    print(p1.x, p2.y)
