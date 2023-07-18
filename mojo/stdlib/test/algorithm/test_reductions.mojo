# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

from Buffer import Buffer, NDBuffer
from DType import DType
from Index import StaticIntTuple
from IO import print
from List import DimList
from Range import range
from Reductions import (
    all_true,
    any_true,
    max,
    mean,
    min,
    none_true,
    product,
    sum,
    variance,
)


# CHECK-LABEL: test_reductions
fn test_reductions():
    print("== test_reductions")

    alias simd_width = 4
    alias size = 100

    # Create a mem of size size
    let vector = Buffer[size, DType.float32].stack_allocation()

    for i in range(size):
        vector[i] = i + 1

    # CHECK: 1.0
    print(min(vector))

    # CHECK: 100.0
    print(max(vector))

    # CHECK: 5050.0
    print(sum(vector))


# We use a smaller vector so that we do not overflow
# CHECK-LABEL: test_product
fn test_product():
    print("== test_product")

    alias simd_width = 4
    alias size = 10

    # Create a mem of size size
    let vector = Buffer[size, DType.float32].stack_allocation()

    for i in range(size):
        vector[i] = i + 1

    # CHECK: 3628800.0
    print(product(vector))


# CHECK-LABEL: test_mean_variance
fn test_mean_variance():
    print("== test_mean_variance")

    alias simd_width = 4
    alias size = 100

    # Create a mem of size size
    let vector = Buffer[size, DType.float32].stack_allocation()

    for i in range(size):
        vector[i] = i + 1

    # CHECK: 50.5
    print(mean(vector))

    # CHECK: 841.666687
    print(variance(vector, 1))


fn test_3d_reductions():
    print("== test_3d_reductions")
    alias simd_width = 4

    @always_inline
    @parameter
    fn _test_3d_reductions[
        input_shape: DimList,
        output_shape: DimList,
        reduce_axis: Int,
    ]():
        let input = NDBuffer[3, input_shape, DType.float32].stack_allocation()
        let output = NDBuffer[3, output_shape, DType.float32].stack_allocation()
        output.fill(0)

        for i in range(input.size()):
            input.flatten()[i] = i

        sum[
            3,
            input_shape,
            output_shape,
            DType.float32,
            reduce_axis,
        ](input, output)

        for i in range(output.size()):
            print(output.flatten()[i])

    # CHECK: 6.0
    # CHECK-NEXT: 22.0
    # CHECK-NEXT: 38.0
    # CHECK-NEXT: 54.0
    _test_3d_reductions[
        DimList(2, 2, 4),
        DimList(2, 2, 1),
        2,
    ]()
    # CHECK: 4.0
    # CHECK-NEXT: 6.0
    # CHECK-NEXT: 8.0
    # CHECK-NEXT: 10.0
    # CHECK-NEXT: 20.0
    # CHECK-NEXT: 22.0
    # CHECK-NEXT: 24.0
    # CHECK-NEXT: 26.0
    _test_3d_reductions[
        DimList(2, 2, 4),
        DimList(2, 1, 4),
        1,
    ]()
    # CHECK: 8.0
    # CHECK-NEXT: 10.0
    # CHECK-NEXT: 12.0
    # CHECK-NEXT: 14.0
    # CHECK-NEXT: 16.0
    # CHECK-NEXT: 18.0
    # CHECK-NEXT: 20.0
    # CHECK-NEXT: 22.0
    _test_3d_reductions[
        DimList(2, 2, 4),
        DimList(1, 2, 4),
        0,
    ]()


# CHECK-LABEL: test_boolean
fn test_boolean():
    print("== test_boolean")

    alias simd_width = 2
    alias size = 5

    # Create a mem of size size
    let vector = Buffer[size, DType.bool].stack_allocation()
    vector[0] = True.value
    vector[1] = False.value
    vector[2] = False.value
    vector[3] = False.value
    vector[4] = True.value

    # CHECK: False
    print(all_true(vector))

    # CHECK: True
    print(any_true(vector))

    # CHECK: False
    print(none_true(vector))

    ###################################################
    # Check with all the elements set to True
    ###################################################

    for i in range(size):
        vector[i] = True.value

    # CHECK: True
    print(all_true(vector))

    # CHECK: True
    print(any_true(vector))

    # CHECK: False
    print(none_true(vector))

    ###################################################
    # Check with all the elements set to False
    ###################################################

    for i in range(size):
        vector[i] = False.value

    # CHECK: False
    print(all_true(vector))

    # CHECK: False
    print(any_true(vector))

    # CHECK: True
    print(none_true(vector))


fn main():
    test_reductions()
    test_product()
    test_mean_variance()
    test_3d_reductions()
    test_boolean()
