# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo -debug-level full %s | FileCheck %s

from std.collections.string import StringSlice


struct S:
    var v: Int

    @implicit
    def __init__(out self, x: Int):
        print("init", x)
        self.v = x

    def __del__(deinit self):
        print("destroy", self.v)

    def __init__(out self, *, copy: Self):
        self.v = copy.v


def mightThrow() raises:
    return


def foo(c: Bool):
    var s = S("1234".byte_length())
    try:
        if c:
            mightThrow()  # destruct 's' if returns
            print(s.v)
    except:
        pass


def fail(str: StringSlice) raises -> S:
    if str.byte_length() > 5:
        raise Error(str)
    return S(str.byte_length())


def main():
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
