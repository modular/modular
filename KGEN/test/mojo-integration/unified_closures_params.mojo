# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s 1 4 | FileCheck %s

from sys import argv
from memory.pointer import AddressSpace, _GPUAddressSpace


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


def main():
    var one = atol(argv()[1])
    var four = atol(argv()[2])
    # CHECK: 8
    itCaptures[3](one, four)
