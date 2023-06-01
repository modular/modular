# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo %s | FileCheck %s

from DType import DType
from IO import print
from Math import tanh, iota


# CHECK-LABEL: test_tanh_float32
fn test_tanh_float32():
    print("== test_tanh_float32")

    let simd_val = 0.5 * iota[4, DType.float32]()

    # CHECK: [0.0, 0.46211{{[0-9]+}}, 0.76159{{[0-9]+}}, 0.90514{{[0-9]+}}]
    print(tanh(simd_val))

    # CHECK: [0.0, 0.24491{{[0-9]+}}, 0.46211{{[0-9]+}}, 0.63514{{[0-9]+}}]
    print(tanh(0.5 * simd_val))


# CHECK-LABEL: test_tanh_float64
fn test_tanh_float64():
    print("== test_tanh_float64")

    let simd_val = 0.5 * iota[4, DType.float64]()

    # CHECK: [0.0, 0.46211{{[0-9]+}}, 0.76159{{[0-9]+}}, 0.90514{{[0-9]+}}]
    print(tanh(simd_val))

    # CHECK: [0.0, 0.24491{{[0-9]+}}, 0.46211{{[0-9]+}}, 0.63514{{[0-9]+}}]
    print(tanh(0.5 * simd_val))


fn main():
    test_tanh_float32()
    test_tanh_float64()
