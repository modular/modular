# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s 8 string | FileCheck %s

from std.sys import argv


def takeItParams[
    UU: Trait, T: def[U: Trait](impl: U) unified -> Int, //
](state: T, x: UU):
    var product2 = state.__call__[UU](x)
    print(product2)


trait Trait(ImplicitlyCopyable):
    def get(self) -> Int:
        ...


@fieldwise_init
struct Impl(Trait):
    var x: Int

    def get(self) -> Int:
        return self.x


@fieldwise_init
struct Impl2(Trait):
    var x: String

    def get(self) -> Int:
        return self.x.__len__()


# COM: Ensure parametric closures are supported
def captureParams[X: Trait, Y: Trait](impl2: X, mut impl3: Y):
    def hasParams[U: Trait](impl: U) unified {read} -> Int:
        return impl.get() + impl2.get() + impl3.get()

    takeItParams(hasParams, impl2)


@fieldwise_init
struct Parameter[*, base: ImplicitlyCopyable & Writable](Copyable):
    var impl: Self.base

    def useIt(self):
        print(self.impl)


def takeIt[f: def() unified -> None](impl: f):
    impl()


def captureIt(p: Parameter[...]):
    @no_inline
    def closure() unified {read p}:
        p.useIt()

    takeIt(closure)


def main() raises:
    var num = atol(argv()[1])
    var str = argv()[2]
    var p1 = Parameter[base=String](str)
    var p2 = Parameter[base=Int](num)
    # CHECK: string
    captureIt(p1)
    # CHECK: 8
    captureIt(p2)

    # COM: Ensure parametric closures are supported
    var x = Impl(num)
    var y = Impl2(str)
    # CHECK: 22
    captureParams(x, y)
