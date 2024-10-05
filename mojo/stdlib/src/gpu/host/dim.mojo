# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module implements the dim type."""

from utils.index import Index, IndexList


@value
@register_passable("trivial")
struct Dim(Stringable):
    var _value: IndexList[3]

    fn __init__(inout self, dims: (Int,)):
        self = Self(dims[0])

    fn __init__(inout self, dims: (Int, Int)):
        self = Self(dims[0], dims[1])

    fn __init__(inout self, dims: (Int, Int, Int)):
        self = Self(dims[0], dims[1], dims[2])

    fn __init__(inout self, x: Int, y: Int = 1, z: Int = 1):
        self._value = Index(x, y, z)

    fn __getitem__(self, idx: Int) -> Int:
        return self._value[idx]

    @no_inline
    fn __str__(self) -> String:
        var res = String("(x=") + str(self.x()) + ", "
        if self.y() != 1 or self.z() != 1:
            res += String("y=") + str(self.y())
            if self.z() != 1:
                res += ", z=" + str(self.z())
        res += ")"
        return res

    fn __repr__(self) -> String:
        return self.__str__()

    fn z(self) -> Int:
        return self[2]

    fn y(self) -> Int:
        return self[1]

    fn x(self) -> Int:
        return self[0]
