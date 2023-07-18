# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s | FileCheck %s
# RUN: kgen %s %mojo_cpu_build_arch -emit -debug-level=full --O0 -o /dev/null

from IO import print
from Vector import DynamicVector


fn take_closure_and_print(
    g: fn (Int) capturing -> Int,
    x: Int,
):
    print(g(x))


fn test_take_closure_and_print(x: Int):
    var v = DynamicVector[Int](2)
    v.push_back(1)
    v.push_back(2)

    @parameter
    fn FOO(y: Int) -> Int:
        print(v[1])
        return y

    v[1] = 5

    let capture_struct_closure: fn (Int) capturing -> Int = FOO
    let u: Int = 3
    take_closure_and_print(capture_struct_closure, u)
    v._del_old()


fn main():
    let x = 39
    # CHECK: 5
    # CHECK: 3
    test_take_closure_and_print(x)
