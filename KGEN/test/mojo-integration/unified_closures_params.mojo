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
