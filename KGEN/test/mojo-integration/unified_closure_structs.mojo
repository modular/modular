# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s 3 1 4 | FileCheck %s

from sys import argv


trait ATrait(Movable):
    # In order for a struct that depends on a capturing closure
    # to conform to a trait, all the methods of that trait must be
    # marked as capturing. This is temporary until we remove the capturing
    # effect. Note that the legacy closures are responsible for this restriction.
    # In particular, the following is not supported:
    # trait ATrait(Movable):
    #     fn my_method(self) -> Int:
    #         ...

    # struct ParamStruct[func: fn (x: Int) capturing -> Int](ATrait):
    #     fn my_method(self) -> Int:
    #         return func(2)
    fn my_method(self) capturing -> Int:
        ...


struct AStruct[func: fn (x: Int) unified -> Int](ATrait):
    var myFunc: func

    fn __init__(out self, var x: func):
        self.myFunc = x^

    fn my_method(self) -> Int:
        return self.myFunc(3)


fn takeIt[T: ATrait](impl: T):
    print(impl.my_method())


# COM: Test the capturing effect is propagated through to trait methods
trait DefinesClosure(fn (z: Int) unified -> Int):
    pass


@fieldwise_init
struct DefinesClosureImpl(DefinesClosure):
    var x: Int

    fn __call__(self, z: Int) -> Int:
        return z + self.x


fn takeIt[f: fn (z: Int) unified -> Int](impl: f, y: Int):
    print(impl(y))


def main():
    var y: Int = atol(argv()[1])
    var one = atol(argv()[2])
    var four = atol(argv()[3])

    fn myclosure(x: Int) unified {var y} -> Int:
        return y + x

    var s = AStruct(myclosure^)
    # CHECK: 6
    takeIt(s)
    # CHECK: 5
    var impl = DefinesClosureImpl(one)
    takeIt(impl, four)
