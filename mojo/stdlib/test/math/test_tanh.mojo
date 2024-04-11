# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

from math import iota, tanh


# CHECK-LABEL: test_tanh_float32
fn test_tanh_float32():
    print("== test_tanh_float32")

    var simd_val = 0.5 * iota[DType.float32, 4]()

    # CHECK: [0.0, 0.46211{{[0-9]+}}, 0.76159{{[0-9]+}}, 0.90514{{[0-9]+}}]
    print(tanh(simd_val))

    # CHECK: [0.0, 0.24491{{[0-9]+}}, 0.46211{{[0-9]+}}, 0.63514{{[0-9]+}}]
    print(tanh(0.5 * simd_val))


# CHECK-LABEL: test_tanh_float64
fn test_tanh_float64():
    print("== test_tanh_float64")

    var simd_val = 0.5 * iota[DType.float64, 4]()

    # CHECK: [0.0, 0.46211{{[0-9]+}}, 0.76159{{[0-9]+}}, 0.90514{{[0-9]+}}]
    print(tanh(simd_val))

    # CHECK: [0.0, 0.24491{{[0-9]+}}, 0.46211{{[0-9]+}}, 0.63514{{[0-9]+}}]
    print(tanh(0.5 * simd_val))


fn main():
    test_tanh_float32()
    test_tanh_float64()
