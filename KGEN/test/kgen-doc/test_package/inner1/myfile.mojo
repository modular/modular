# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #


struct Struct:
    def parametric[a: Int](self) -> Int:
        return a

    def foo(self):
        return


struct ParametricStruct[a: Int]:
    def foo(self):
        return


def print10():
    print(10)
