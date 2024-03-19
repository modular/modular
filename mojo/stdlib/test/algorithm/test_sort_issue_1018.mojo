# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from random import rand

from algorithm.sort import sort


fn sort_test[D: DType](size: Int, max: Int, name: StringLiteral) raises:
    var p = Pointer[SIMD[D, 1]].alloc(size)
    rand[D](p, size)
    sort[D](p, size)
    for i in range(1, size - 1):
        if p[i] < p[i - 1]:
            print(name, "size:", size, "max:", max, "incorrect sort")
            print("p[", end="")
            print(i - 1, end="")
            print("] =", p.load(i - 1))
            print("p[", end="")
            print(i, end="")
            print("] =", p.load(i))
            print()
            p.free()
            raise "Failed"
    p.free()


fn main():
    try:
        sort_test[DType.int8](300, 3_000, "int8")
        sort_test[DType.float32](3_000, 3_000, "float32")
        sort_test[DType.float64](300_000, 3_000_000_000, "float64")
        # CHECK: Success
        print("Success")
    except e:
        # CHECK-NOT: Failed
        print(e)
