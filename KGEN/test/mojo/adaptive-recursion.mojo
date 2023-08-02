# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s | FileCheck %s
# RUN: %mojo -debug-level full %s | FileCheck %s

from IO import print
from Assert import assert_param


@adaptive
fn foo[axis: Int](i: Int) -> Int:
    assert_param[axis < 0]()
    return i


@adaptive
fn foo[axis: Int](i: Int) -> Int:
    assert_param[axis >= 0]()
    return foo[axis - 1](i)


fn main():
    let x = foo[3](42)
    # CHECK: 42
    print(x)
