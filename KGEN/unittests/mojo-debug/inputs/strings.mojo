# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


fn test(st: String):
    print(st)  # breakpoint


@register_passable("trivial")
struct Point(CollectionElement):
    var x: Int
    var y: Int

    fn __init__(inout self, x: Int, y: Int):
        self.x = x
        self.y = y


fn main():
    var p2 = Point(2, 2)
    var literal = "string_literal"
    var s1 = String("let_string")
    var s2 = String("")
    for i in range(0, 100):
        s2 += str(i)
    var s3 = String()
    test(s2)
    var s4 = Pointer[String].address_of(s2)
    print(literal, s1, s2, s3, s4)  # breakpoint
