# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

from math import exp2


# CHECK-LABEL: test_exp2
fn test_exp2():
    print("== test_exp2")

    # CHECK: 2.0
    print(exp2(Float32(1)))

    # CHECK: 1.148696{{[0-9]+}}
    print(exp2(Float32(0.2)))

    # CHECK: 1.0
    print(exp2(Float32(0)))

    # CHECK: 0.5
    print(exp2(Float32(-1)))

    # CHECK: 4.0
    print(exp2(Float32(2)))


fn main():
    test_exp2()
