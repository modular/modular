# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: kgen %s -execute -func='$test_broadcast::main():index()' -I %stdlibdir | FileCheck %s

from Broadcast import broadcast
from Buffer import NDBuffer
from DType import DType
from Index import Index
from Int import Int
from IO import print
from List import create_kgen_list
from Memory import memset_zero
from Transpose import _index2D
from Tuple import StaticTuple


fn _index3D(x: Int, y: Int, z: Int) -> StaticTuple[3, __mlir_type.index]:
    return Index(x, y, z).as_tuple()


fn _index5D(
    x: Int, y: Int, z: Int, a: Int, b: Int
) -> StaticTuple[5, __mlir_type.index]:
    return Index(x, y, z, a, b).as_tuple()


# CHECK-LABEL: test_broadcast_same_shape
fn test_broadcast_same_shape():
    print("== test_broadcast_same_shape\n")

    # parameters
    alias input_shape = create_kgen_list[__mlir_type.index](1, 2, 1)
    alias output_shape = create_kgen_list[__mlir_type.index](1, 2, 1)

    # Create a 3D tensor of shape (1, 2, 1), of the form
    # [[[1], [2]]]
    var input = NDBuffer[
        3,
        input_shape,
        DType.index.value,
    ].stack_allocation()
    input.__setitem__(_index3D(0, 0, 0), 1)
    input.__setitem__(_index3D(0, 1, 0), 2)

    # Create a 3D tensor of shape (1, 2, 1)
    var output = NDBuffer[
        3,
        output_shape,
        DType.index.value,
    ].stack_allocation()

    memset_zero[DType.index.value](output.data, 2)

    broadcast[
        3,
        output_shape,
        input_shape,
        DType.index.value,
    ](output, input)
    # output tensor will have the form:
    # [[[1], [2]]]

    # CHECK: 1
    print(input.__getitem__(_index3D(0, 0, 0)))
    # CHECK: 2
    print(input.__getitem__(_index3D(0, 1, 0)))

    # CHECK: 1
    print(output.__getitem__(_index3D(0, 0, 0)))
    # CHECK: 2
    print(output.__getitem__(_index3D(0, 1, 0)))


# CHECK-LABEL: test_broadcast_single_axis
fn test_broadcast_single_axis():
    print("== test_broadcast_single_axis\n")

    # parameters
    alias input_shape = create_kgen_list[__mlir_type.index](1, 2)
    alias output_shape = create_kgen_list[__mlir_type.index](3, 2)

    # Create a 2D tensor of shape (1, 2), of the form
    # [[1, 2]]
    var input = NDBuffer[
        2,
        input_shape,
        DType.index.value,
    ].stack_allocation()

    input.__setitem__(_index2D(0, 0), 1)
    input.__setitem__(_index2D(0, 1), 2)

    # Create a 2D tensor of shape (3, 2)
    var output = NDBuffer[
        2,
        output_shape,
        DType.index.value,
    ].stack_allocation()

    memset_zero(output.data, 6)

    broadcast[
        2,
        output_shape,
        input_shape,
        DType.index.value,
    ](output, input)
    # output tensor will have the form:
    # [[1, 2], [1, 2], [1, 2]]

    # CHECK: 1
    print(input.__getitem__(_index2D(0, 0)))
    # CHECK: 2
    print(input.__getitem__(_index2D(0, 1)))

    # CHECK: 1
    print(output.__getitem__(_index2D(0, 0)))
    # CHECK: 2
    print(output.__getitem__(_index2D(0, 1)))
    # CHECK: 1
    print(output.__getitem__(_index2D(1, 0)))
    # CHECK: 2
    print(output.__getitem__(_index2D(1, 1)))
    # CHECK: 1
    print(output.__getitem__(_index2D(2, 0)))
    # CHECK: 2
    print(output.__getitem__(_index2D(2, 1)))


# CHECK-LABEL: test_broadcast_multi_axes
fn test_broadcast_multi_axes():
    print("== test_broadcast_multi_axes\n")

    # parameters
    alias input_shape = create_kgen_list[__mlir_type.index](1, 2, 1)
    alias output_shape = create_kgen_list[__mlir_type.index](2, 2, 3)

    # Create a 3D tensor of shape (1, 2, 1), of the form
    # [[[1], [2]]]
    var input = NDBuffer[
        3,
        input_shape,
        DType.index.value,
    ].stack_allocation()

    input.__setitem__(_index3D(0, 0, 0), 1)
    input.__setitem__(_index3D(0, 1, 0), 2)

    # Create a 3D tensor of shape (2, 2, 3)
    var output = NDBuffer[
        3,
        output_shape,
        DType.index.value,
    ].stack_allocation()

    memset_zero(output.data, 16)

    broadcast[
        3,
        output_shape,
        input_shape,
        DType.index.value,
    ](output, input)
    # output tensor will have the form:
    # [[[1, 1, 1], [2, 2, 2]],
    #  [[1, 1, 1], [2, 2, 2]]]

    # CHECK: 1
    print(input.__getitem__(_index3D(0, 0, 0)))
    # CHECK: 2
    print(input.__getitem__(_index3D(0, 1, 0)))

    # CHECK: 1
    print(output.__getitem__(_index3D(0, 0, 0)))
    # CHECK: 2
    print(output.__getitem__(_index3D(0, 1, 0)))
    # CHECK: 1
    print(output.__getitem__(_index3D(0, 0, 1)))
    # CHECK: 2
    print(output.__getitem__(_index3D(0, 1, 1)))
    # CHECK: 1
    print(output.__getitem__(_index3D(0, 0, 2)))
    # CHECK: 2
    print(output.__getitem__(_index3D(0, 1, 2)))
    # CHECK: 1
    print(output.__getitem__(_index3D(1, 0, 0)))
    # CHECK: 2
    print(output.__getitem__(_index3D(1, 1, 0)))
    # CHECK: 1
    print(output.__getitem__(_index3D(1, 0, 1)))
    # CHECK: 2
    print(output.__getitem__(_index3D(1, 1, 1)))
    # CHECK: 1
    print(output.__getitem__(_index3D(1, 0, 2)))
    # CHECK: 2
    print(output.__getitem__(_index3D(1, 1, 2)))


fn test_broadcast_multi_axes_nested():

    # parameters
    alias input_shape = create_kgen_list[__mlir_type.index](2, 1, 2, 1, 2)
    alias output_shape = create_kgen_list[__mlir_type.index](2, 2, 2, 2, 2)

    # Create a 5D tensor of shape (2, 1, 2, 1, 2), of the form
    # [[[[[1, 2]], [[3, 4]]]], [[[[5, 6]], [[7, 8]]]]]
    var input = NDBuffer[
        5,
        input_shape,
        DType.index.value,
    ].stack_allocation()

    input.__setitem__(_index5D(0, 0, 0, 0, 0), 1)
    input.__setitem__(_index5D(0, 0, 0, 0, 1), 2)
    input.__setitem__(_index5D(0, 0, 1, 0, 0), 3)
    input.__setitem__(_index5D(0, 0, 1, 0, 1), 4)
    input.__setitem__(_index5D(1, 0, 0, 0, 0), 5)
    input.__setitem__(_index5D(1, 0, 0, 0, 1), 6)
    input.__setitem__(_index5D(1, 0, 1, 0, 0), 7)
    input.__setitem__(_index5D(1, 0, 1, 0, 1), 8)

    # Create a 5D tensor of shape (2, 2, 2, 2, 2)
    var output = NDBuffer[
        5,
        output_shape,
        DType.index.value,
    ].stack_allocation()

    memset_zero(output.data, 32)

    broadcast[
        5,
        output_shape,
        input_shape,
        DType.index.value,
    ](output, input)

    # CHECK: 1
    print(output.__getitem__(_index5D(0, 0, 0, 0, 0)))
    # CHECK: 2
    print(output.__getitem__(_index5D(0, 0, 0, 0, 1)))
    # CHECK: 1
    print(output.__getitem__(_index5D(0, 0, 0, 1, 0)))
    # CHECK: 2
    print(output.__getitem__(_index5D(0, 0, 0, 1, 1)))
    # CHECK: 3
    print(output.__getitem__(_index5D(0, 0, 1, 0, 0)))
    # CHECK: 4
    print(output.__getitem__(_index5D(0, 0, 1, 0, 1)))
    # CHECK: 3
    print(output.__getitem__(_index5D(0, 0, 1, 1, 0)))
    # CHECK: 4
    print(output.__getitem__(_index5D(0, 0, 1, 1, 1)))

    # CHECK: 1
    print(output.__getitem__(_index5D(0, 1, 0, 0, 0)))
    # CHECK: 2
    print(output.__getitem__(_index5D(0, 1, 0, 0, 1)))
    # CHECK: 1
    print(output.__getitem__(_index5D(0, 1, 0, 1, 0)))
    # CHECK: 2
    print(output.__getitem__(_index5D(0, 1, 0, 1, 1)))
    # CHECK: 3
    print(output.__getitem__(_index5D(0, 1, 1, 0, 0)))
    # CHECK: 4
    print(output.__getitem__(_index5D(0, 1, 1, 0, 1)))
    # CHECK: 3
    print(output.__getitem__(_index5D(0, 1, 1, 1, 0)))
    # CHECK: 4
    print(output.__getitem__(_index5D(0, 1, 1, 1, 1)))

    # CHECK: 5
    print(output.__getitem__(_index5D(1, 0, 0, 0, 0)))
    # CHECK: 6
    print(output.__getitem__(_index5D(1, 0, 0, 0, 1)))
    # CHECK: 5
    print(output.__getitem__(_index5D(1, 0, 0, 1, 0)))
    # CHECK: 6
    print(output.__getitem__(_index5D(1, 0, 0, 1, 1)))
    # CHECK: 7
    print(output.__getitem__(_index5D(1, 0, 1, 0, 0)))
    # CHECK: 8
    print(output.__getitem__(_index5D(1, 0, 1, 0, 1)))
    # CHECK: 7
    print(output.__getitem__(_index5D(1, 0, 1, 1, 0)))
    # CHECK: 8
    print(output.__getitem__(_index5D(1, 0, 1, 1, 1)))

    # CHECK: 5
    print(output.__getitem__(_index5D(1, 1, 0, 0, 0)))
    # CHECK: 6
    print(output.__getitem__(_index5D(1, 1, 0, 0, 1)))
    # CHECK: 5
    print(output.__getitem__(_index5D(1, 1, 0, 1, 0)))
    # CHECK: 6
    print(output.__getitem__(_index5D(1, 1, 0, 1, 1)))
    # CHECK: 7
    print(output.__getitem__(_index5D(1, 1, 1, 0, 0)))
    # CHECK: 8
    print(output.__getitem__(_index5D(1, 1, 1, 0, 1)))
    # CHECK: 7
    print(output.__getitem__(_index5D(1, 1, 1, 1, 0)))
    # CHECK: 8
    print(output.__getitem__(_index5D(1, 1, 1, 1, 1)))


@export
fn main() -> __mlir_type.index:
    test_broadcast_same_shape()
    test_broadcast_single_axis()
    test_broadcast_multi_axes()
    test_broadcast_multi_axes_nested()
    return 0
