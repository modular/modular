# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s 3 1 4 | FileCheck %s

from std.sys import argv


trait ATrait(Movable):
    # In order for a struct that depends on a capturing closure
    # to conform to a trait, all the methods of that trait must be
    # marked as capturing. This is temporary until we remove the capturing
    # effect. Note that the legacy closures are responsible for this restriction.
    # In particular, the following is not supported:
    # trait ATrait(Movable):
    #     def my_method(self) -> Int:
    #         ...

    # struct ParamStruct[func: def (x: Int) capturing -> Int](ATrait):
    #     def my_method(self) -> Int:
    #         return func(2)
    def my_method(self) capturing -> Int:
        ...


struct AStruct[func: def(x: Int) -> Int](ATrait):
    var myFunc: Self.func

    def __init__(out self, var x: Self.func):
        self.myFunc = x^

    def my_method(self) -> Int:
        return self.myFunc(3)


def takeIt[T: ATrait](impl: T):
    print(impl.my_method())


# COM: Test the capturing effect is propagated through to trait methods
trait DefinesClosure(def(z: Int) -> Int):
    pass


@fieldwise_init
struct DefinesClosureImpl(DefinesClosure):
    var x: Int

    def __call__(self, z: Int) -> Int:
        return z + self.x


def takeIt[f: def(z: Int) -> Int](impl: f, y: Int):
    print(impl(y))


# COM: Ensure closures work when a RegisterPassable struct forwards
# COM: a concrete type argument through a generic closure parameter.
@fieldwise_init
struct RegPassWrapper[U: RegisterPassable & ImplicitlyDestructible](
    RegisterPassable,
):
    var u: Self.U

    def apply_fn[FuncType: def(Self.U) -> Bool](self, func: FuncType) -> Bool:
        return func(self.u)


def testRegisterPassableUnifiedClosureAdaptor():
    def always_true(x: Int) {} -> Bool:
        return True

    var wrapper = RegPassWrapper(5)
    print(wrapper.apply_fn(always_true))


def main() raises:
    var y: Int = atol(argv()[1])
    var one = atol(argv()[2])
    var four = atol(argv()[3])

    def myclosure(x: Int) {var y} -> Int:
        return y + x

    var s = AStruct(myclosure^)
    # CHECK: 6
    takeIt(s)
    # CHECK: 5
    var impl = DefinesClosureImpl(one)
    takeIt(impl, four)

    # COM: Ensure RegisterPassable struct can forward args through closure adaptor
    # CHECK: True
    testRegisterPassableUnifiedClosureAdaptor()
