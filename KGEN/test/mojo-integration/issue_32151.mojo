# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# FIXME(#29771): This test leaks memory.
# UNSUPPORTED: asan
# RUN: mojo -O0 %s | FileCheck %s

from utils.variant import Variant
from collections import List


struct MTuple[T: CollectionElement](
    CollectionElement,
    Stringable,
    Formattable,
):
    alias Element = Variant[T, Self]
    var elts: List[Self.Element]

    @always_inline
    fn __init__(inout self: Self):
        self.elts = List[Self.Element]()

    @always_inline
    fn __init__(inout self: Self, value: Self.Element):
        self.elts = List[Self.Element]()
        self.elts.append(value)

    @always_inline
    fn __moveinit__(inout self: Self, owned existing: Self):
        self.elts = existing.elts^

    @always_inline
    fn __copyinit__(inout self: Self, existing: Self):
        self.elts = existing.elts

    @always_inline
    fn cons(self, owned other: Self) -> Self:
        var new = self
        for e in other.elts:
            new.elts.append(e[])
        return new

    @always_inline
    fn __add__(self, owned other: Self) -> Self:
        var new = Self()
        for e in self.elts:
            new.elts.append(e[])
        for e in other.elts:
            new.elts.append(e[])
        return new

    fn __str__(self) -> String:
        return String.format_sequence(self)

    fn format_to(self, inout writer: Formatter):
        writer.write("(")

        for i in range(len(self.elts)):
            if self.elts[i].isa[Int]():
                var value = self.elts[i]
                writer.write(value[Int])
            elif self.elts[i].isa[MTuple[T]]():
                var value = self.elts[i]
                writer.write(value[MTuple[T]])
            else:
                writer.write("?")
            if i < len(self.elts) - 1:
                writer.write(", ")

        writer.write(")")


alias IntTuple = MTuple[Int]


fn main():
    alias tup = IntTuple(IntTuple(3) + IntTuple(4))
    # CHECK: (3, 4)
    print(tup)
    add_print[tup]()


fn add_print[x: IntTuple]():
    alias tup = x + IntTuple(4)
    # CHECK: ((3, 4), 4)
    print(tup)
