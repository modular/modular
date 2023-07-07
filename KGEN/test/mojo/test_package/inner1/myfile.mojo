# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Int import Int


struct Struct:
    fn parametric[a: Int](self) -> Int:
        return a

    fn foo(self):
        return


struct ParametricStruct[a: Int]:
    fn foo(self):
        return
