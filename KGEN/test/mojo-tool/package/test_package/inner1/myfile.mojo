# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


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
