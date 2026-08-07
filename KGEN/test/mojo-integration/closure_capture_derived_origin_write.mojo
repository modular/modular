# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

# Regression test: a closure capturing a pointer whose origin is a field
# projection of `self` must keep that origin mutable.

from std.memory import Pointer


@fieldwise_init
struct Inner(Copyable, Movable):
    var value: Int


struct Outer:
    var inner: Inner

    def __init__(out self):
        self.inner = Inner(1)

    def apply(mut self):
        var p = Pointer(to=self.inner)

        @always_inline
        def closure() {imm}:
            p[].value = 5

        closure()


def main():
    var o = Outer()
    o.apply()
    # CHECK: 5
    print(o.inner.value)
