# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo -debug-level full %s | FileCheck %s

from utils import StringRef


struct S:
    var v: Int

    fn __init__(out self, x: Int):
        print("init", x)
        self.v = x

    fn __del__(owned self):
        print("destroy", self.v)

    fn __copyinit__(out self, existing: Self):
        self.v = existing.v


fn mightThrow() raises:
    return


fn foo(c: Bool):
    var s = S(len("1234"))
    try:
        if c:
            mightThrow()  # destruct 's' if returns
            print(s.v)
    except:
        pass


fn fail(str: StringRef) raises -> S:
    if len(str) > 5:
        raise Error(str)
    return S(len(str))


fn main():
    # CHECK: init 4
    # CHECK: destroy 4
    foo(True)
    # CHECK: exception thrown
    # CHECK-NOT: init 7
    # CHECK-NOT: destroy 7
    try:
        var x = fail("1234567")
    except e:
        print("exception thrown")
