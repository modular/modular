# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo %s | FileCheck %s

from Buffer import Buffer, NDBuffer
from DType import DType
from Reductions import sum, product, max, min, mean, variance
from Int import Int
from Index import StaticIntTuple
from IO import print
from Range import range
from List import create_kgen_list, _get_kgen_list_item


# CHECK-LABEL: test_reductions
fn test_reductions():
    print("== test_reductions\n")

    alias simd_width = 4
    alias size = 100

    # Create a mem of size size
    let vector = Buffer[size, DType.f32].stack_allocation()

    for i in range(size):
        vector[i] = (i + 1).__as_mlir_index()

    # CHECK: 1.000000
    print(min[simd_width](vector))

    # CHECK: 100.000000
    print(max[simd_width](vector))

    # CHECK: 5050.000000
    print(sum[simd_width](vector))


# We use a smaller vector so that we do not overflow
# CHECK-LABEL: test_product
fn test_product():
    print("== test_product\n")

    alias simd_width = 4
    alias size = 10

    # Create a mem of size size
    let vector = Buffer[size, DType.f32].stack_allocation()

    for i in range(size):
        vector[i] = (i + 1).__as_mlir_index()

    # CHECK: 3628800.000000
    print(product[simd_width](vector))


# CHECK-LABEL: test_mean_variance
fn test_mean_variance():
    print("== test_mean_variance\n")

    alias simd_width = 4
    alias size = 100

    # Create a mem of size size
    let vector = Buffer[size, DType.f32].stack_allocation()

    for i in range(size):
        vector[i] = (i + 1).__as_mlir_index()

    # CHECK: 50.500000
    print(mean[simd_width](vector))

    # CHECK: 841.666687
    print(variance[simd_width](vector))


fn test_3d_reductions():
    print("== test_3d_reductions\n")
    alias simd_width = 4

    @always_inline
    fn _test_3d_reductions[
        input_shape: __mlir_type[`!kgen.list<index[3]>`],
        output_shape: __mlir_type[`!kgen.list<index[3]>`],
        reduce_axis: __mlir_type.index,
    ]():
        let input = NDBuffer[3, input_shape, DType.f32].stack_allocation()
        let output = (
            NDBuffer[3, output_shape, DType.f32].stack_allocation().fill(0)
        )
        for i in range(input.size()):
            input.flatten()[i] = i

        sum[
            simd_width,
            3,
            input_shape,
            output_shape,
            DType.f32,
            reduce_axis,
        ](input, output)

        for ii in range(output.size()):
            print(output.flatten()[ii])

    # CHECK: [6.000000]
    # CHECK-NEXT: [22.000000]
    # CHECK-NEXT: [38.000000]
    # CHECK-NEXT: [54.000000]
    _test_3d_reductions[
        create_kgen_list[__mlir_type.index](2, 2, 4),
        create_kgen_list[__mlir_type.index](2, 2, 1),
        2,
    ]()
    # CHECK: [4.000000]
    # CHECK-NEXT: [6.000000]
    # CHECK-NEXT: [8.000000]
    # CHECK-NEXT: [10.000000]
    # CHECK-NEXT: [20.000000]
    # CHECK-NEXT: [22.000000]
    # CHECK-NEXT: [24.000000]
    # CHECK-NEXT: [26.000000]
    _test_3d_reductions[
        create_kgen_list[__mlir_type.index](2, 2, 4),
        create_kgen_list[__mlir_type.index](2, 1, 4),
        1,
    ]()
    # CHECK: [8.000000]
    # CHECK-NEXT: [10.000000]
    # CHECK-NEXT: [12.000000]
    # CHECK-NEXT: [14.000000]
    # CHECK-NEXT: [16.000000]
    # CHECK-NEXT: [18.000000]
    # CHECK-NEXT: [20.000000]
    # CHECK-NEXT: [22.000000]
    _test_3d_reductions[
        create_kgen_list[__mlir_type.index](2, 2, 4),
        create_kgen_list[__mlir_type.index](1, 2, 4),
        0,
    ]()


fn main():
    test_reductions()
    test_product()
    test_mean_variance()
    test_3d_reductions()
