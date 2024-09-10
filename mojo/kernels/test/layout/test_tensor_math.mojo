# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

from layout import Layout, LayoutTensor, stack_allocation_like
from layout.math import exp, max, sum
from layout.fillers import arange


# CHECK-LABEL: test_reduce_sum
fn test_reduce_sum():
    print("== test_reduce_sum")
    var tensor_4x4 = LayoutTensor[
        DType.float32, Layout.row_major(4, 4)
    ].stack_allocation()
    arange(tensor_4x4)
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
    arange(tensor_4x4)
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


# CHECK-LABEL: test_reduce_res_allocated
fn test_reduce_res_allocated():
    print("== test_reduce_res_allocated")
    var tensor_4x4 = LayoutTensor[
        DType.float32, Layout.row_major(4, 4)
    ].stack_allocation()
    arange(tensor_4x4)
    # CHECK: 12.0
    # CHECK: 13.0
    # CHECK: 14.0
    # CHECK: 15.0
    print(max[axis=0](tensor_4x4))
    # CHECK: 6.0
    # CHECK: 22.0
    # CHECK: 38.0
    # CHECK: 54.0
    print(sum[axis=1](tensor_4x4))


# CHECK-LABEL: test_exp
fn test_exp():
    print("== test_exp")
    var tensor_4x4 = LayoutTensor[
        DType.float32, Layout.row_major(4, 4)
    ].stack_allocation()
    arange(tensor_4x4)
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
    ].stack_allocation()
    arange(tensor_4x4)
    # CHECK: 2.0 3.0 4.0 5.0
    # CHECK: 6.0 7.0 8.0 9.0
    # CHECK: 10.0 11.0 12.0 13.0
    # CHECK: 14.0 15.0 16.0 17.0
    print(tensor_4x4 + 2)
    # CHECK: -2.0 -1.0 0.0 1.0
    # CHECK: 2.0 3.0 4.0 5.0
    # CHECK: 6.0 7.0 8.0 9.0
    # CHECK: 10.0 11.0 12.0 13.0
    print(tensor_4x4 - 2)
    # CHECK: 0.0 10.0 20.0 30.0
    # CHECK: 40.0 50.0 60.0 70.0
    # CHECK: 80.0 90.0 100.0 110.0
    # CHECK: 120.0 130.0 140.0 150.0
    print(tensor_4x4 * 10.0)

    # CHECK: 1.0 2.0 3.0 4.0
    # CHECK: 5.0 6.0 7.0 8.0
    # CHECK: 9.0 10.0 11.0 12.0
    # CHECK: 13.0 14.0 15.0 16.0
    tensor_4x4 += 1
    print(tensor_4x4)


# CHECK-LABLE: test_binary_same_rank
fn test_binary_same_rank():
    print("== test_binary_same_rank")
    var tensor_4x5 = LayoutTensor[
        DType.float32, Layout.row_major(4, 5)
    ].stack_allocation()
    arange(tensor_4x5)
    var tensor_4x5_2 = LayoutTensor[
        DType.float32, Layout.row_major(4, 5)
    ].stack_allocation()
    arange(tensor_4x5_2)
    tensor_4x5_2 = tensor_4x5_2 + 2
    # CHECK: 2.0 4.0 6.0 8.0 10.0
    # CHECK: 12.0 14.0 16.0 18.0 20.0
    # CHECK: 22.0 24.0 26.0 28.0 30.0
    # CHECK: 32.0 34.0 36.0 38.0 40.0
    print(tensor_4x5 + tensor_4x5_2)

    # CHECK: 0.0 0.5 1.0 1.5 2.0
    # CHECK: 2.5 3.0 3.5 4.0 4.5
    # CHECK: 5.0 5.5 6.0 6.5 7.0
    # CHECK: 7.5 8.0 8.5 9.0 9.5
    print(tensor_4x5 / stack_allocation_like(tensor_4x5).fill(2))

    # CHECK: 0.0 2.0 4.0 6.0 8.0
    # CHECK: 10.0 12.0 14.0 16.0 18.0
    # CHECK: 20.0 22.0 24.0 26.0 28.0
    # CHECK: 30.0 32.0 34.0 36.0 38.0
    tensor_4x5 += tensor_4x5
    print(tensor_4x5)


# CHECK-LABEL: test_binary_broadcast_inner
fn test_binary_broadcast_inner():
    print("== test_binary_broadcast_inner")
    var tensor_4x5 = LayoutTensor[
        DType.float32, Layout.row_major(4, 5)
    ].stack_allocation()
    arange(tensor_4x5)
    var tensor_4 = LayoutTensor[
        DType.float32, Layout.row_major(4)
    ].stack_allocation()
    arange(tensor_4)
    tensor_4 = tensor_4 + 1
    # CHECK: -1.0 0.0 1.0 2.0 3.0
    # CHECK: 3.0 4.0 5.0 6.0 7.0
    # CHECK: 7.0 8.0 9.0 10.0 11.0
    # CHECK: 11.0 12.0 13.0 14.0 15.0
    print(tensor_4x5 - tensor_4)

    # CHECK: 0.0 0.5 1.0 1.5 2.0
    # CHECK: 2.5 3.0 3.5 4.0 4.5
    # CHECK: 5.0 5.5 6.0 6.5 7.0
    # CHECK: 7.5 8.0 8.5 9.0 9.5
    print(tensor_4x5 / stack_allocation_like(tensor_4).fill(2))


# CHECK-LABEL: test_softmax_math
fn test_softmax_math():
    print("== test_softmax_math")
    var tensor_5x4 = LayoutTensor[
        DType.float32, Layout.row_major(5, 4)
    ].stack_allocation()
    arange(tensor_5x4)

    var exp_norm = exp(tensor_5x4 - max[axis=1](tensor_5x4))
    var exp_norm_sum = sum[axis=1](exp_norm)
    var soft_max = exp_norm / exp_norm_sum
    # CHECK: 0.032058604061603546 0.087144322693347931 0.23688283562660217 0.64391428232192993
    # CHECK: 0.032058604061603546 0.087144322693347931 0.23688283562660217 0.64391428232192993
    # CHECK: 0.032058604061603546 0.087144322693347931 0.23688283562660217 0.64391428232192993
    # CHECK: 0.032058604061603546 0.087144322693347931 0.23688283562660217 0.64391428232192993
    # CHECK: 0.032058604061603546 0.087144322693347931 0.23688283562660217 0.64391428232192993
    print(soft_max)


# CHECK: test_max_elemntwise
fn test_max_elemntwise():
    print("== test_max_elemntwise")
    var tensor_4x4_a = LayoutTensor[
        DType.float32, Layout.row_major(4, 4)
    ].stack_allocation()
    arange(tensor_4x4_a)

    var tensor_4x4_b = LayoutTensor[
        DType.float32, Layout.row_major(4, 4)
    ].stack_allocation().fill(5)

    # CHECK: 5.0 5.0 5.0 5.0
    # CHECK: 5.0 5.0 6.0 7.0
    # CHECK: 8.0 9.0 10.0 11.0
    # CHECK: 12.0 13.0 14.0 15.0
    print(max(tensor_4x4_a, tensor_4x4_b))


fn main():
    test_reduce_sum()
    test_reduce_max()
    test_reduce_res_allocated()
    test_exp()
    test_unary_scalar()
    test_binary_same_rank()
    test_binary_broadcast_inner()
    test_softmax_math()
    test_max_elemntwise()
