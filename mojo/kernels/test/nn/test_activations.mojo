# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen %s -execute -func='$test_activations::main():index()' | FileCheck %s

from Activations import relu, relu_n1, prelu, gelu, gelu_approximate
from DType import DType
from IO import print
from Math import iota


# CHECK-LABEL: test_relu
fn test_relu():
    print("== test_relu\n")

    let simd_val = iota[4, DType.f32.value]()

    # CHECK: [0.000000, 1.000000, 2.000000, 3.000000]
    print(relu(simd_val))

    # CHECK: [0.000000, 0.000000, 0.000000, 1.000000]
    print(relu(simd_val - 2))

    # CHECK: [0.000000, 0.500000, 1.000000, 1.500000]
    print(relu(0.5 * simd_val))


# CHECK-LABEL: test_relu_n1
fn test_relu_n1():
    print("== test_relu_n1\n")

    let simd_val = iota[4, DType.f32.value]()

    # CHECK: [0.000000, 1.000000, 1.000000, 1.000000]
    print(relu_n1(simd_val))

    # CHECK: [-1.000000, -1.000000, 0.000000, 1.000000]
    print(relu_n1(simd_val - 2))

    # CHECK: [0.000000, 0.500000, 1.000000, 1.000000]
    print(relu_n1(0.5 * simd_val))


# CHECK-LABEL: test_prelu
fn test_prelu():
    print("== test_prelu\n")

    let simd_val = iota[4, DType.f32.value]()

    # CHECK: [0.000000, 1.000000, 2.000000, 3.000000]
    print(prelu(simd_val, 2))

    # CHECK: [-0.200000, -0.100000, 0.000000, 1.000000]
    print(prelu(simd_val - 2, 0.1))

    # CHECK: [0.000000, 0.500000, 1.000000, 1.500000]
    print(prelu(0.5 * simd_val, 0.5))


# CHECK-LABEL: test_gelu
fn test_gelu():
    print("== test_gelu\n")

    let simd_val = 2 - 0.5 * iota[4, DType.f32.value]()

    # CHECK: [1.954500, 1.399789, 0.841345, 0.345731]
    print(gelu(simd_val))

    # CHECK: [0.841345, 0.580029, 0.345731, 0.149677]
    print(gelu(0.5 * simd_val))

    # CHECK: [1.954598, 1.399572, 0.841192, 0.345714]
    print(gelu_approximate(simd_val))

    # CHECK: [0.841192, 0.579961, 0.345714, 0.149675]
    print(gelu_approximate(0.5 * simd_val))


@export
fn main() -> __mlir_type.index:
    test_relu()
    test_relu_n1()
    test_prelu()
    test_gelu()
    return 0
