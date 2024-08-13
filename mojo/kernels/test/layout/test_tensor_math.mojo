# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

from layout import Layout, LayoutTensor
from layout.math import exp, sum, max


# CHECK-LABEL: test_reduce_sum
fn test_reduce_sum():
    print("== test_reduce_sum")
    var tensor_4x4 = LayoutTensor[
        DType.float32, Layout.row_major(4, 4)
    ].stack_allocation().linspace()
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
    ].stack_allocation().linspace()
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


# CHECK-LABEL: test_exp
fn test_exp():
    print("== test_exp")
    var tensor_4x4 = LayoutTensor[
        DType.float32, Layout.row_major(4, 4)
    ].stack_allocation().linspace()
    # CHECK: 1.0 2.7182817459106445 7.3890562057495117 20.085536956787109
    # CHECK: 54.598148345947266 148.41316223144531 403.42877197265625 1096.6331787109375
    # CHECK: 2980.9580078125 8103.08349609375 22026.46484375 59874.140625
    # CHECK: 162754.78125 442413.375 1202604.25 3269017.25
    print(exp(tensor_4x4))


# CHECK-LABEL: test_unary_scalar
fn test_unary_scalar():
    print("== test_unary_scalar")
    var tensor_4x4 = LayoutTensor[
        DType.float32, Layout.row_major(4, 4)
    ].stack_allocation().linspace()
    # CHECK: 2.0 3.0 4.0 5.0
    # CHECK: 6.0 7.0 8.0 9.0
    # CHECK: 10.0 11.0 12.0 13.0
    # CHECK: 14.0 15.0 16.0 17.0
    print(tensor_4x4 + 2)
    # CHECK: 0.0 -1.0 -2.0 -3.0
    # CHECK: -4.0 -5.0 -6.0 -7.0
    # CHECK: -8.0 -9.0 -10.0 -11.0
    # CHECK: -12.0 -13.0 -14.0 -15.0
    print(tensor_4x4 - 2)
    # CHECK: 0.0 -10.0 -20.0 -30.0
    # CHECK: -40.0 -50.0 -60.0 -70.0
    # CHECK: -80.0 -90.0 -100.0 -110.0
    # CHECK: -120.0 -130.0 -140.0 -150.0
    print(tensor_4x4 * 10.0)


fn main():
    test_reduce_sum()
    test_reduce_max()
    test_exp()
    test_unary_scalar()
