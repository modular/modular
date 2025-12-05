# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s AA BB CC foo | FileCheck %s


# COM: Check that the argument is augmented at the definition site.

from sys import argv


struct Mem(ImplicitlyCopyable):
    var str1: String
    var str2: String

    fn __init__(out self, str1: String, str2: String):
        self.str1 = str1
        self.str2 = str2

    fn to_string(self) -> String:
        return self.str1 + self.str2


struct MovableMem(ImplicitlyCopyable):
    var str1: String
    var str2: String

    fn __init__(out self, str1: String, str2: String):
        self.str1 = str1
        self.str2 = str2

    fn __copyinit__(out self, other: Self):
        self.str1 = other.str1
        self.str2 = other.str2
        print("copied mem")

    fn to_string(self) -> String:
        return self.str1 + self.str2


fn takeIt[T: fn () unified -> String](state: T):
    print("captures: ", state())


def main():
    var byCopy: String = String(bytes=argv()[1].as_bytes())

    fn mutateCopy() unified {var byCopy} -> String:
        byCopy += ".v2"
        return byCopy

    # CHECK: captures: AA.v2
    takeIt(mutateCopy)
    # CHECK: AA
    print(byCopy)

    var byMutRef: String = String(bytes=argv()[2].as_bytes())

    fn mutateRef() unified {mut byMutRef} -> String:
        byMutRef += ".v2"
        return byMutRef

    # CHECK: captures: BB.v2
    takeIt(mutateRef)
    # CHECK: BB.v2
    print(byMutRef)

    var byRef: String = String(bytes=argv()[3].as_bytes())

    fn immRef() unified {read byRef} -> String:
        return byRef

    # CHECK: captures: CC
    takeIt(immRef)
    byRef += ".v2"

    # CHECK: CC.v2
    takeIt(immRef)

    # COM: nonmovable types can be captured by copy.
    var x: String = argv()[4]
    var mem = Mem(x, x)

    fn myclosure() unified {var} -> String:
        return mem.to_string()

    # CHECK: captures:  foofoo
    takeIt(myclosure)
    var movableMem = MovableMem(x, x)

    # COM: Copyable closures
    @no_inline
    fn copyMem() unified {var ^}:
        print(movableMem.to_string())

    # CHECK: copied mem
    var copyOfClosure = copyMem
    # CHECK: foofoo
    copyMem()
    # CHECK: foofoo
    copyOfClosure()
