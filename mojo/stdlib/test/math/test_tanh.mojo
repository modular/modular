# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo %s | FileCheck %s

from DType import DType
from IO import print
from Math import tanh, iota


# CHECK-LABEL: test_tanh_f16
fn test_tanh_f16():
    print("== test_tanh_f16")

    let simd_val = 0.5 * iota[4, DType.f16]()

    # CHECK: [0.000000, 0.461914, 0.761719, 0.905762]
    print(tanh(simd_val))

    # CHECK: [0.000000, 0.244873, 0.461914, 0.635742]
    print(tanh(0.5 * simd_val))


# CHECK-LABEL: test_tanh_f32
fn test_tanh_f32():
    print("== test_tanh_f32")

    let simd_val = 0.5 * iota[4, DType.f32]()

    # CHECK: [0.000000, 0.462117, 0.761594, 0.905148]
    print(tanh(simd_val))

    # CHECK: [0.000000, 0.244919, 0.462117, 0.635149]
    print(tanh(0.5 * simd_val))


# CHECK-LABEL: test_tanh_f64
fn test_tanh_f64():
    print("== test_tanh_f64")

    let simd_val = 0.5 * iota[4, DType.f64]()

    # CHECK: [0.000000, 0.462117, 0.761594, 0.905148]
    print(tanh(simd_val))

    # CHECK: [0.000000, 0.244919, 0.462117, 0.635149]
    print(tanh(0.5 * simd_val))


fn main():
    test_tanh_f16()
    test_tanh_f32()
    test_tanh_f64()
