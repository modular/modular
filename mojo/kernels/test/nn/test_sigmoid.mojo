# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

from nn.activations import sigmoid, sigmoid_grad


# CHECK-LABEL: test_sigmoid_float32
fn test_sigmoid_float32():
    print("== test_sigmoid_float32")

    # CHECK: 0.5
    print(sigmoid(Float32(0)))

    # CHECK: 0.7310
    print(sigmoid(Float32(1)))

    # CHECK: 0.8807
    print(sigmoid(Float32(2)))

    # CHECK: 0.2689
    print(sigmoid(Float32(-1)))

    # CHECK: 0.1192
    print(sigmoid(Float32(-2)))

    # CHECK: 1.0
    print(sigmoid(Float32(108)))


# CHECK-LABEL: test_sigmoid_float64
fn test_sigmoid_float64():
    print("== test_sigmoid_float64")

    # CHECK: 0.5
    print(sigmoid(Float64(0)))

    # CHECK: 0.7310
    print(sigmoid(Float64(1)))

    # CHECK: 0.8807
    print(sigmoid(Float64(2)))

    # CHECK: 0.2689
    print(sigmoid(Float64(-1)))

    # CHECK: 0.1192
    print(sigmoid(Float64(-2)))

    # CHECK: 1.0
    print(sigmoid(Float64(108)))


# CHECK-LABEL: test_sigmoid_grad_float32
fn test_sigmoid_grad_float32():
    print("== test_sigmoid_grad_float32")

    # CHECK: 0.25
    print(sigmoid_grad(Float32(0)))

    # CHECK: 0.1966
    print(sigmoid_grad(Float32(1)))

    # CHECK: 0.1049
    print(sigmoid_grad(Float32(2)))

    # CHECK: 0.1966
    print(sigmoid_grad(Float32(-1)))

    # CHECK: 0.1049
    print(sigmoid_grad(Float32(-2)))


# CHECK-LABEL: test_sigmoid_grad_float64
fn test_sigmoid_grad_float64():
    print("== test_sigmoid_grad_float64")

    # CHECK: 0.25
    print(sigmoid_grad(Float64(0)))

    # CHECK: 0.1966
    print(sigmoid_grad(Float64(1)))

    # CHECK: 0.1049
    print(sigmoid_grad(Float64(2)))

    # CHECK: 0.1966
    print(sigmoid_grad(Float64(-1)))

    # CHECK: 0.1049
    print(sigmoid_grad(Float64(-2)))


fn main():
    test_sigmoid_float32()
    test_sigmoid_float64()
    test_sigmoid_grad_float32()
    test_sigmoid_grad_float64()
