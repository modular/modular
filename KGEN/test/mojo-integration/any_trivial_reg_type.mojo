# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo -debug-level full %s 4 5 | FileCheck %s

from collections import OptionalReg
from sys import argv


@register_passable("trivial")
trait Position:
    fn foo(self) -> Self:
        ...


@register_passable("trivial")
struct PositionImpl(Position):
    var x: Int
    var y: Int

    fn __init__(out self, x: Int, y: Int):
        self.x = x
        self.y = y

    @no_inline
    fn foo(self) -> Self:
        print(self.x, self.y)
        return self


fn foo[position_t: Position](x: position_t) -> OptionalReg[position_t]:
    var xx = OptionalReg[position_t](x)
    _ = xx.value().foo()
    return xx


def main():
    # CHECK: 4 5
    pi = PositionImpl(atol(argv()[1]), atol(argv()[2]))
    _ = foo(pi)
