# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo -debug-level full %s | FileCheck %s


@adaptive
fn foo[axis: Int](i: Int) -> Int:
    constrained[axis < 0]()
    return i


@adaptive
fn foo[axis: Int](i: Int) -> Int:
    constrained[axis >= 0]()
    return foo[axis - 1](i)


fn main():
    let x = foo[3](42)
    # CHECK: 42
    print(x)
