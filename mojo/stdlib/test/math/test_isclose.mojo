# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

from math import isclose


# CHECK-LABEL: test_isclose
fn test_isclose():
    print("== test_isclose")

    # CHECK: True
    print(isclose(Float32(2), Float32(2)))

    # CHECK: False
    print(isclose(Float32(2), Float32(3)))

    # CHECK: True
    print(isclose(Float32(2.00000001), Float32(2)))


fn main():
    test_isclose()
