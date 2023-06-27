# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo %s | FileCheck %s

from Activations import elu, relu, relu_n1, prelu, gelu, gelu_approximate
from DType import DType
from IO import print
from Math import iota

# CHECK-LABEL: test_elu
fn test_elu():
    print("== test_elu")

    let simd_val = iota[DType.float32, 4]()

    # CHECK: [0.0, 1.0, 2.0, 3.0]
    print(elu(simd_val))

    # CHECK: [-0.86466{{[0-9]+}}, -0.63212{{[0-9]+}}, 0.0, 1.0]
    print(elu(simd_val - 2))

    # CHECK: [0.0, 0.5, 1.0, 1.5]
    print(elu(0.5 * simd_val))


# CHECK-LABEL: test_relu
fn test_relu():
    print("== test_relu")

    let simd_val = iota[DType.float32, 4]()

    # CHECK: [0.0, 1.0, 2.0, 3.0]
    print(relu(simd_val))

    # CHECK: [0.0, 0.0, 0.0, 1.0]
    print(relu(simd_val - 2))

    # CHECK: [0.0, 0.5, 1.0, 1.5]
    print(relu(0.5 * simd_val))


# CHECK-LABEL: test_relu_n1
fn test_relu_n1():
    print("== test_relu_n1")

    let simd_val = iota[DType.float32, 4]()

    # CHECK: [0.0, 1.0, 1.0, 1.0]
    print(relu_n1(simd_val))

    # CHECK: [-1.0, -1.0, 0.0, 1.0]
    print(relu_n1(simd_val - 2))

    # CHECK: [0.0, 0.5, 1.0, 1.0]
    print(relu_n1(0.5 * simd_val))


# CHECK-LABEL: test_prelu
fn test_prelu():
    print("== test_prelu")

    let simd_val = iota[DType.float32, 4]()

    # CHECK: [0.0, 1.0, 2.0, 3.0]
    print(prelu(simd_val, 2))

    # CHECK: [-0.2{{[0-9]+}}, -0.1{{[0-9]+}}, 0.0, 1.0]
    print(prelu(simd_val - 2, 0.1))

    # CHECK: [0.0, 0.5, 1.0, 1.5]
    print(prelu(0.5 * simd_val, 0.5))


# CHECK-LABEL: test_gelu_float16
fn test_gelu_float16():
    print("== test_gelu_float16")

    let simd_val = 2 - 0.5 * iota[DType.float16, 4]()

    # There is no difference in the results from MLAS and oneDNN gelu.
    # CHECK: [1.955078125, 1.3994140625, 0.84130859375, 0.345703125]
    print(gelu(simd_val))

    # The results from MLAS gelu is [0.841345, 0.580029, 0.345731, 0.149677].
    # CHECK: [0.84130859375, 0.580078125, 0.345703125, 0.149658203125]
    print(gelu(0.5 * simd_val))

    # CHECK: [1.955078125, 1.3994140625, 0.84130859375, 0.345703125]
    print(gelu_approximate(simd_val))

    # CHECK: [0.84130859375, 0.580078125, 0.345703125, 0.149658203125]
    print(gelu_approximate(0.5 * simd_val))


# CHECK-LABEL: test_gelu_float32
fn test_gelu_float32():
    print("== test_gelu_float32")

    let simd_val = 2 - 0.5 * iota[DType.float32, 4]()

    # There is no difference in the results from MLAS and oneDNN gelu.
    # CHECK: [1.95449{{[0-9]+}}, 1.39978{{[0-9]+}}, 0.84134{{[0-9]+}}, 0.34573{{[0-9]+}}]
    print(gelu(simd_val))

    # The results from MLAS gelu is [0.841345, 0.580029, 0.345731, 0.149677].
    # CHECK: [0.84134{{[0-9]+}}, 0.580029{{[0-9]+}}, 0.34573{{[0-9]+}}, 0.14967{{[0-9]+}}]
    print(gelu(0.5 * simd_val))

    # CHECK: [1.95459{{[0-9]+}}, 1.39957{{[0-9]+}}, 0.84119{{[0-9]+}}, 0.34571{{[0-9]+}}]
    print(gelu_approximate(simd_val))

    # CHECK: [0.84119{{[0-9]+}}, 0.57996{{[0-9]+}}, 0.34571{{[0-9]+}}, 0.14967{{[0-9]+}}]
    print(gelu_approximate(0.5 * simd_val))


# CHECK-LABEL: test_gelu_float64
fn test_gelu_float64():
    print("== test_gelu_float64")

    let simd_val = 2 - 0.5 * iota[DType.float64, 4]()

    # There is no difference in the results from MLAS and oneDNN gelu.
    # CHECK: [1.95449{{[0-9]+}}, 1.39978{{[0-9]+}}, 0.84134{{[0-9]+}}, 0.34573{{[0-9]+}}]
    print(gelu(simd_val))

    # The results from MLAS gelu is [0.841345, 0.580029, 0.345731, 0.149677].
    # CHECK: [0.84134{{[0-9]+}}, 0.580029{{[0-9]+}}, 0.34573{{[0-9]+}}, 0.14967{{[0-9]+}}]
    print(gelu(0.5 * simd_val))

    # CHECK: [1.95459{{[0-9]+}}, 1.39957{{[0-9]+}}, 0.84119{{[0-9]+}}, 0.34571{{[0-9]+}}]
    print(gelu_approximate(simd_val))

    # CHECK: [0.84119{{[0-9]+}}, 0.57996{{[0-9]+}}, 0.34571{{[0-9]+}}, 0.14967{{[0-9]+}}]
    print(gelu_approximate(0.5 * simd_val))


fn main():
    test_elu()
    test_relu()
    test_relu_n1()
    test_prelu()
    test_gelu_float16()
    test_gelu_float32()
    test_gelu_float64()
