# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from buffer import NDBuffer
from buffer.dimlist import DimList
from memory import UnsafePointer, stack_allocation
from MOGG import calculate_squeeze_shape, to_buffer

from utils.index import StaticIntTuple


# CHECK-LABEL: test_calculate_squeeze_shape
fn test_calculate_squeeze_shape():
    print("== test_calculate_squeeze_shape")

    var data_matrix = NDBuffer[
        DType.index,
        1,
        DimList(8),
    ].stack_allocation()
    data_matrix[StaticIntTuple[1](0)] = 1
    data_matrix[StaticIntTuple[1](1)] = 10
    data_matrix[StaticIntTuple[1](2)] = 1
    data_matrix[StaticIntTuple[1](3)] = 11
    data_matrix[StaticIntTuple[1](4)] = 1
    data_matrix[StaticIntTuple[1](5)] = 12
    data_matrix[StaticIntTuple[1](6)] = 13
    data_matrix[StaticIntTuple[1](7)] = 1

    # Main thing is indices not sorted.
    var remove_indices = NDBuffer[
        DType.index,
        1,
        DimList(3),
    ].stack_allocation()
    remove_indices[StaticIntTuple[1](0)] = 0
    remove_indices[StaticIntTuple[1](1)] = -1  # same as index 7
    remove_indices[StaticIntTuple[1](2)] = 4

    var final_shape = NDBuffer[
        DType.index,
        1,
        DimList(5),
    ].stack_allocation()

    calculate_squeeze_shape[DType.index, DType.index, False](
        rebind[NDBuffer[DType.index, 1]](data_matrix),
        rebind[NDBuffer[DType.index, 1]](remove_indices),
        rebind[NDBuffer[DType.index, 1]](final_shape),
    )

    # CHECK: 10 1 11 12 13
    print(
        final_shape[0],
        final_shape[1],
        final_shape[2],
        final_shape[3],
        final_shape[4],
    )

    _ = data_matrix
    _ = remove_indices
    _ = final_shape


fn main():
    test_calculate_squeeze_shape()
