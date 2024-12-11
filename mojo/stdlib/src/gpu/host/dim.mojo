# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""This module implements the dim type."""

from utils.index import Index, IndexList


@value
@register_passable("trivial")
struct Dim(Stringable, Writable):
    var _value: IndexList[3]

    @implicit
    fn __init__(out self, value: IndexList[3]):
        self._value = value

    @implicit
    fn __init__(out self, x: Int):
        self._value = IndexList[3](x, 1, 1)

    fn __init__(out self, x: Int, y: Int):
        self._value = IndexList[3](x, y, 1)

    fn __init__(out self, x: Int, y: Int, z: Int):
        self._value = IndexList[3](x, y, z)

    @implicit
    fn __init__(out self, dims: (Int,)):
        self._value = IndexList[3](dims[0], 1, 1)

    @implicit
    fn __init__(out self, dims: (Int, Int)):
        self._value = IndexList[3](dims[0], dims[1], 1)

    @implicit
    fn __init__(out self, dims: (Int, Int, Int)):
        self._value = IndexList[3](dims[0], dims[1], dims[2])

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

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(repr(self))

    fn z(self) -> Int:
        return self[2]

    fn y(self) -> Int:
        return self[1]

    fn x(self) -> Int:
        return self[0]
