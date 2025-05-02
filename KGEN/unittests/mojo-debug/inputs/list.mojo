# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from debug_test_utils import keep_alive


@register_passable("trivial")
struct Point(Copyable, Movable):
    var x: Int
    var y: Int

    fn __init__(out self, x: Int, y: Int):
        self.x = x
        self.y = y


fn main():
    var point_vec = List[Point](capacity=3)
    var p1 = Point(1, -1)
    var p2 = Point(2, -2)
    var p3 = Point(3, -3)
    point_vec.append(p1)
    point_vec.append(p2)
    point_vec.append(p3)
    var value = point_vec[0].x  # breakpoint
    print(value)

    var int_vec = List[Int](capacity=3)
    int_vec.append(1)
    int_vec.append(2)
    int_vec.append(3)
    print(len(int_vec))  # breakpoint

    for i in range(0, 100):
        int_vec.append(i)
    keep_alive(int_vec)  # breakpoint
