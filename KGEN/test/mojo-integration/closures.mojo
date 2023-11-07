# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo -debug-level full %s 2 3 | FileCheck %s
from sys import argv


fn take_closure_and_print(g: fn (Int) capturing -> Int, x: Int):
    print(g(x))


fn test_take_closure_and_print(x: Int, w: Int):
    @parameter
    fn h(y: Int) -> Int:
        return x + y

    @parameter
    fn thin(y: Int) -> Int:
        return y + 17

    take_closure_and_print(h, w)
    take_closure_and_print(thin, w)


fn main():
    try:
        let x = atol(String(argv()[1]))
        let y = atol(String(argv()[2]))
        # CHECK: 5
        # CHECK: 20
        test_take_closure_and_print(x, y)
    except e:
        print(e._message())
