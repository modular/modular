# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s 1 4 | FileCheck %s

from sys import argv
from memory.pointer import AddressSpace, _GPUAddressSpace


fn hasOrigin[F: fn[T: MutOrigin] (TypeWithOrigin[T]) unified -> None, //](f: F):
    f[MutAnyOrigin](TypeWithOrigin[MutAnyOrigin]())


@fieldwise_init
struct TypeWithOrigin[T: MutOrigin](ImplicitlyCopyable, Movable):
    var isMutable: Bool

    fn __init__(out self):
        self.isMutable = Self.T.mut


fn takeIt[f: ImplicitlyCopyable & fn (z: Int) unified -> Int](impl: f, y: Int):
    print(impl(y))


@no_inline
fn aThing[f: fn (Int) capturing -> Int](y: Int):
    fn aClosure(z: Int) unified {var} -> Int:
        return f(y)

    takeIt(aClosure, y)


@no_inline
fn itCaptures[THREE: Int](one: Int, four: Int):
    @parameter
    fn aParam(z: Int) -> Int:
        return THREE + four + z

    aThing[aParam](one)

    # COM: Ensure nesting in legacy closure does not corrupt the symbol calculation
    @parameter
    if THREE == 3:

        @__copy_capture(one, four)
        @parameter
        fn aParam2(zz: Int) -> Int:
            fn thing(z: Int) unified {var zz} -> Int:
                return zz

            takeIt(thing, four)
            return one + one

        aThing[aParam2](one)


trait Coordinate:
    fn prettyPrint(self):
        ...


@fieldwise_init
struct Cartesian(Coordinate, ImplicitlyCopyable):
    var x: Int
    var y: Int

    fn prettyPrint(self):
        print("{", self.x, ",", self.y, "}")


@fieldwise_init
struct Polar[y: Int](Coordinate, ImplicitlyCopyable):
    var x: Int

    fn prettyPrint(self):
        print("{", self.x, ",", self.y, "}")


fn useDefinesCapturingParamClosure[
    NOT_X: Coordinate & ImplicitlyCopyable, C: fn () unified -> NOT_X
](impl: C):
    var coordinate = impl()
    coordinate.prettyPrint()


fn definesCapturingParamClosure[
    X: Coordinate & ImplicitlyCopyable
](something: X, one: Int) raises:
    fn closureImpl() unified {var} -> X:
        return something

    # COM: check that concrete types can conform to traits with aliases
    fn closureConcreteImpl() unified {var} -> Cartesian:
        return Cartesian(one, one)

    useDefinesCapturingParamClosure[X, type_of(closureImpl)](closureImpl)
    useDefinesCapturingParamClosure[Cartesian, type_of(closureConcreteImpl)](
        closureConcreteImpl
    )


fn usesParamRefClosure[
    T: Coordinate & ImplicitlyCopyable,
    C: fn[x: Int, Y: Coordinate] (xx: T, unused: Y) unified -> Polar[x],
](impl: C, value: T):
    var typed_value = rebind[C.T](value)
    var result = impl[3, Cartesian](typed_value, Cartesian(3, 3))
    result.prettyPrint()


fn definesParamRefClosure[U: Coordinate & ImplicitlyCopyable](value: U):
    fn closureImpl[
        y: Int, YY: Coordinate
    ](x: U, unused: YY) unified {var} -> Polar[y]:
        _ = value
        return Polar[y](y)

    usesParamRefClosure[U, type_of(closureImpl)](closureImpl, value)


def main():
    var one = atol(argv()[1])
    var four = atol(argv()[2])
    # CHECK: 8
    # CHECK: 1
    # CHECK: 2
    itCaptures[3](one, four)

    # Ensure origins are lowered
    # CHECK: True
    fn closure[T: MutOrigin](_bar: TypeWithOrigin[T]) unified {read}:
        print(_bar.isMutable)

    hasOrigin(closure)

    # COM: Test rebinds to traits with captures.
    # CHECK: { 1 , 4 }
    # CHECK: { 1 , 1 }
    definesCapturingParamClosure(Cartesian(one, four), one)

    # COM: Test param ref matching: both trait and impl use param types.
    # CHECK: { 3 , 3 }
    definesParamRefClosure(Cartesian(one, four))
