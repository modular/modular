# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from IO import print


struct Struct:
    fn parametric[a: Int](self) -> Int:
        return a

    fn foo(self):
        return


struct ParametricStruct[a: Int]:
    fn foo(self):
        return


fn print10():
    print(10)
