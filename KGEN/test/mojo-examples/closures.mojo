# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo %s | FileCheck %s

from IO import print


fn take_closure_and_print(g: fn (Int) capturing -> Int, x: Int):
    print(g(x))


fn test_take_closure_and_print(x: Int):
    @parameter
    fn h(y: Int) -> Int:
        return x + y

    @parameter
    fn thin(y: Int) -> Int:
        return y + 17

    take_closure_and_print(h, 3)
    take_closure_and_print(thin, 3)


fn main():
    # CHECK: 42
    # CHECK: 20
    test_take_closure_and_print(39)
