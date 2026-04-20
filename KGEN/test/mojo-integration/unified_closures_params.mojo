# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s 1 4 | FileCheck %s

from std.sys import argv
from std.memory.pointer import AddressSpace, _GPUAddressSpace


def hasOrigin[
    F: def[T: MutOrigin](TypeWithOrigin[T]) unified -> None, //
](f: F):
    f[MutAnyOrigin](TypeWithOrigin[MutAnyOrigin]())


@fieldwise_init
struct TypeWithOrigin[T: MutOrigin](ImplicitlyCopyable, Movable):
    var isMutable: Bool

    def __init__(out self):
        self.isMutable = Self.T.mut


def takeIt[f: ImplicitlyCopyable & def(z: Int) unified -> Int](impl: f, y: Int):
    print(impl(y))


@no_inline
def aThing[f: def(Int) capturing -> Int](y: Int):
    def aClosure(z: Int) unified {var} -> Int:
        return f(y)

    takeIt(aClosure, y)


@no_inline
def itCaptures[THREE: Int](one: Int, four: Int):
    @parameter
    def aParam(z: Int) -> Int:
        return THREE + four + z

    aThing[aParam](one)

    # COM: Ensure nesting in legacy closure does not corrupt the symbol calculation
    comptime if THREE == 3:

        @__copy_capture(one, four)
        @parameter
        def aParam2(zz: Int) -> Int:
            def thing(z: Int) unified {var zz} -> Int:
                return zz

            takeIt(thing, four)
            return one + one

        aThing[aParam2](one)


trait Coordinate:
    def prettyPrint(self):
        ...


@fieldwise_init
struct Cartesian(Coordinate, ImplicitlyCopyable):
    var x: Int
    var y: Int

    def prettyPrint(self):
        print("{", self.x, ",", self.y, "}")


@fieldwise_init
struct Polar[y: Int](Coordinate, ImplicitlyCopyable):
    var x: Int

    def prettyPrint(self):
        print("{", self.x, ",", self.y, "}")


def useDefinesCapturingParamClosure[
    X: Coordinate & ImplicitlyCopyable, C: def() unified -> X
](impl: C):
    var coordinate = impl()
    coordinate.prettyPrint()


def definesCapturingParamClosure[
    X: Coordinate & ImplicitlyCopyable
](something: X, one: Int) raises:
    def closureImpl() unified {var} -> X:
        return something

    # COM: check that concrete types can conform to traits with aliases
    def closureConcreteImpl() unified {var} -> Cartesian:
        return Cartesian(one, one)

    useDefinesCapturingParamClosure[X, type_of(closureImpl)](closureImpl)
    useDefinesCapturingParamClosure[Cartesian, type_of(closureConcreteImpl)](
        closureConcreteImpl
    )


def usesParamRefClosure[
    T: Coordinate & ImplicitlyCopyable,
    C: def[x: Int, Y: Coordinate](xx: T, unused: Y) unified -> Polar[x],
](impl: C, value: T):
    var result = impl[3, Cartesian](value, Cartesian(3, 3))
    result.prettyPrint()


def definesParamRefClosure[T: Coordinate & ImplicitlyCopyable](value: T):
    def closureImpl[
        x: Int, Y: Coordinate
    ](xx: T, unused: Y) unified {var} -> Polar[x]:
        _ = value
        return Polar[x](x)

    usesParamRefClosure[T, type_of(closureImpl)](closureImpl, value)


def takesThin[
    T: ImplicitlyCopyable & Writable, FuncType: def(T) unified
](impl: FuncType, x: T):
    impl(x)


def callTakesThin[T: ImplicitlyCopyable & Writable](x: T):
    def takesItem(item: T) unified {}:
        print(item)

    takesThin[T, type_of(takesItem)](takesItem, x)


def main() raises:
    var one = atol(argv()[1])
    var four = atol(argv()[2])
    # CHECK: 8
    # CHECK: 1
    # CHECK: 2
    itCaptures[3](one, four)

    # Ensure origins are lowered
    # CHECK: True
    def closure[T: MutOrigin](_bar: TypeWithOrigin[T]) unified {read}:
        print(_bar.isMutable)

    hasOrigin(closure)

    # COM: Test rebinds to traits with captures.
    # CHECK: { 1 , 4 }
    # CHECK: { 1 , 1 }
    definesCapturingParamClosure(Cartesian(one, four), one)

    # COM: Test param ref matching: both trait and impl use param types.
    # CHECK: { 3 , 3 }
    definesParamRefClosure(Cartesian(one, four))

    # COM: Test thin closure with captured type parameter.
    # CHECK: 1
    callTakesThin[Int](one)
