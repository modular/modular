# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s 3 | FileCheck %s

from sys import argv


trait ATrait(Movable):
    fn my_method(self) -> Int:
        ...


struct AStruct[func: fn (x: Int) unified -> Int](ATrait):
    var myFunc: func

    fn __init__(out self, var x: func):
        self.myFunc = x^

    fn my_method(self) -> Int:
        return self.myFunc(3)


fn takeIt[T: ATrait](impl: T):
    print(impl.my_method())


def main():
    var y: Int = atol(argv()[1])

    fn myclosure(x: Int) unified {var y} -> Int:
        return y + x

    var s = AStruct(myclosure^)
    # CHECK: 6
    takeIt(s)
