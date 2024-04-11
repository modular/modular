# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

from math import iota, pow


# CHECK-LABEL: test_powi
fn test_powi():
    print("== test_powi")

    var simd_val = iota[DType.float32, 4]() + 1
    # CHECK: [1.0, 32.0, 243.0, 1024.0]
    print(pow[5](simd_val))

    # CHECK: [1.0, 0.03125, 0.004115{{[0-9]+}}, 0.000976{{[0-9]+}}]
    print(pow[-5](simd_val))


fn main():
    test_powi()
