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
        return Point {x: x, y: y}


fn main():
    var point_vec = DynamicVector[Point](capacity=3)
    var p1 = Point(1, -1)
    var p2 = Point(2, -2)
    var p3 = Point(3, -3)
    point_vec.push_back(p1)
    point_vec.push_back(p2)
    breakpoint()
    point_vec.push_back(p3)

    var int_vec = DynamicVector[Int](capacity=3)
    int_vec.push_back(1)
    int_vec.push_back(2)
    breakpoint()
    int_vec.push_back(3)

    for i in range(0, 100):
        int_vec.push_back(i)
    breakpoint()
    print(int_vec[0])
