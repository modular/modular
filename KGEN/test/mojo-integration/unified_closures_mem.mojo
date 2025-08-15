# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s AA BB CC | FileCheck %s


# COM: Check that the argument is augmented at the definition site.

from sys import argv


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
