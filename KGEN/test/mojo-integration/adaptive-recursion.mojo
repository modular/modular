# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo -debug-level full %s | FileCheck %s


fn foo[axis: Int](i: Int) -> Int:
    @parameter
    if axis < 0:
        return i
    else:
        return foo[axis - 1](i)


fn main():
    let x = foo[3](42)
    # CHECK: 42
    print(x)
