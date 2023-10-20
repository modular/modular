# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module implements the dim type."""

from utils.index import Index, StaticIntTuple


@value
@register_passable("trivial")
struct Dim:
    var _value: StaticIntTuple[3]

    @always_inline
    fn __init__(dims: Tuple[Int]) -> Self:
        return Self(dims.get[0, Int]())

    @always_inline
    fn __init__(dims: Tuple[Int, Int]) -> Self:
        return Self(dims.get[0, Int](), dims.get[1, Int]())

    @always_inline
    fn __init__(dims: Tuple[Int, Int, Int]) -> Self:
        return Self(dims.get[0, Int](), dims.get[1, Int](), dims.get[2, Int]())

    @always_inline
    fn __init__(x: Int, y: Int = 1, z: Int = 1) -> Self:
        return Self {_value: Index(x, y, z)}

    @always_inline
    fn __getitem__(self, idx: Int) -> Int:
        return self._value[idx]

    fn __str__(self) -> String:
        var res = String("(x=") + self.x() + ", "
        if self.y() != 1 or self.z() != 1:
            res += String("y=") + self.y()
            if self.z() != 1:
                res += ", z=" + String(self.z())
        res += ")"
        return res

    fn __repr__(self) -> String:
        return self.__str__()

    @always_inline
    fn z(self) -> Int:
        return self[2]

    @always_inline
    fn y(self) -> Int:
        return self[1]

    @always_inline
    fn x(self) -> Int:
        return self[0]
