# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s 1 4 | FileCheck %s

from std.sys import argv
from std.memory.pointer import AddressSpace, _GPUAddressSpace


def hasOrigin[F: def[T: MutOrigin](TypeWithOrigin[T]) -> None, //](f: F):
    f[MutAnyOrigin](TypeWithOrigin[MutAnyOrigin]())


@fieldwise_init
struct TypeWithOrigin[T: MutOrigin](ImplicitlyCopyable, Movable):
    var isMutable: Bool

    def __init__(out self):
        self.isMutable = Self.T.mut


def takeIt[f: ImplicitlyCopyable & def(z: Int) -> Int](impl: f, y: Int):
    print(impl(y))


@no_inline
def aThing[f: def(Int) capturing -> Int](y: Int):
    def aClosure(z: Int) {var} -> Int:
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
            def thing(z: Int) {var zz} -> Int:
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
    X: Coordinate & ImplicitlyCopyable, C: def() -> X
](impl: C):
    var coordinate = impl()
    coordinate.prettyPrint()


def definesCapturingParamClosure[
    X: Coordinate & ImplicitlyCopyable
](something: X, one: Int) raises:
    def closureImpl() {var} -> X:
        return something

    # COM: check that concrete types can conform to traits with aliases
    def closureConcreteImpl() {var} -> Cartesian:
        return Cartesian(one, one)

    useDefinesCapturingParamClosure[X, type_of(closureImpl)](closureImpl)
    useDefinesCapturingParamClosure[Cartesian, type_of(closureConcreteImpl)](
        closureConcreteImpl
    )


def usesParamRefClosure[
    T: Coordinate & ImplicitlyCopyable,
    C: def[x: Int, Y: Coordinate](xx: T, unused: Y) -> Polar[x],
](impl: C, value: T):
    var result = impl[3, Cartesian](value, Cartesian(3, 3))
    result.prettyPrint()


def definesParamRefClosure[T: Coordinate & ImplicitlyCopyable](value: T):
    def closureImpl[x: Int, Y: Coordinate](xx: T, unused: Y) {var} -> Polar[x]:
        _ = value
        return Polar[x](x)

    usesParamRefClosure[T, type_of(closureImpl)](closureImpl, value)


def takesThin[
    T: ImplicitlyCopyable & Writable, FuncType: def(T) -> None
](impl: FuncType, x: T):
    impl(x)


def callTakesThin[T: ImplicitlyCopyable & Writable](x: T):
    def takesItem(item: T):
        print(item)

    takesThin[T, type_of(takesItem)](takesItem, x)


@fieldwise_init
struct HasParamRank[N: Int](ImplicitlyCopyable):
    comptime rank = Self.N + Self.N

    def printMe(self):
        print(Self.rank)


def consumeHasParamRank[
    c: Int, r: Int, FuncType: def(a: HasParamRank[c]) -> HasParamRank[r]
](impl: FuncType):
    var p = impl(HasParamRank[c]())
    p.printMe()


def closureFromCapturedInt[a: Int, b: Int, c: Int](x: Int):
    def nested(a: HasParamRank[a + b]) {var} -> HasParamRank[b + c]:
        _ = x
        return HasParamRank[b + c]()

    consumeHasParamRank[a + b, b + c, type_of(nested)](nested)


def must_be_read_only_with_origin[
    Mut: Bool, //, o: Origin[mut=Mut], FuncType: def() -> None
](
    impl: FuncType,
    ptr: UnsafePointer[Int, o, address_space=AddressSpace.GENERIC],
):
    impl()


def demo_origin_closure[
    o: Origin[mut=True]
](ptr: UnsafePointer[Int, o, address_space=AddressSpace.GENERIC,],):
    var immut_ptr = ptr.as_immutable()

    def read() {read immut_ptr}:
        print("read only", immut_ptr[0])

    must_be_read_only_with_origin(read, immut_ptr)


def main() raises:
    var one = atol(argv()[1])
    var four = atol(argv()[2])
    # CHECK: 8
    # CHECK: 1
    # CHECK: 2
    itCaptures[3](one, four)

    # Ensure origins are lowered
    # CHECK: True
    def closure[T: MutOrigin](_bar: TypeWithOrigin[T]) {read}:
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

    # COM: Test captured closure with param expressions in sugar-only space.
    # CHECK: 10
    closureFromCapturedInt[1, 2, 3](one)

    # COM: Test closure capture with pointer origins.
    # CHECK: read only 1
    var ptr = UnsafePointer(to=one)
    demo_origin_closure(ptr)
