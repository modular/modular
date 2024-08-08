# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

from layout import Layout, LayoutTensor
from layout.math import sum, max


# CHECK-LABEL: test_reduce_sum
fn test_reduce_sum():
    print("== test_reduce_sum")
    var tensor_4x4 = LayoutTensor[
        DType.float32, Layout.row_major(4, 4)
    ].stack_allocation()
    tensor_4x4.linspace()
    var tensor_4 = LayoutTensor[
        DType.float32, Layout.row_major(4)
    ].stack_allocation()
    # CHECK: 6.0
    # CHECK: 22.0
    # CHECK: 38.0
    # CHECK: 54.0
    sum[axis=1](tensor_4x4, tensor_4)
    print(tensor_4)
    # CHECK: 24.0
    # CHECK: 28.0
    # CHECK: 32.0
    # CHECK: 36.0
    sum[axis=0](tensor_4x4, tensor_4)
    print(tensor_4)


# CHECK-LABEL: test_reduce_max
fn test_reduce_max():
    print("== test_reduce_max")
    var tensor_4x4 = LayoutTensor[
        DType.float32, Layout.row_major(4, 4)
    ].stack_allocation()
    tensor_4x4.linspace()
    var tensor_4 = LayoutTensor[
        DType.float32, Layout.row_major(4)
    ].stack_allocation()
    max[axis=0](tensor_4x4, tensor_4)
    # CHECK: 12.0
    # CHECK: 13.0
    # CHECK: 14.0
    # CHECK: 15.0
    print(tensor_4)

    max[axis=1](tensor_4x4, tensor_4)
    # CHECK: 3.0
    # CHECK: 7.0
    # CHECK: 11.0
    # CHECK: 15.0
    print(tensor_4)


fn main():
    test_reduce_sum()
    test_reduce_max()
