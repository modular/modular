# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from memory import stack_allocation
from memory.buffer import NDBuffer
from memory.unsafe import DTypePointer, Pointer
from MOGG import calculate_squeeze_shape, to_buffer

from utils.index import StaticIntTuple
from utils.list import DimList

# CHECK-LABEL: test_calculate_squeeze_shape
fn test_calculate_squeeze_shape():
    print("== test_calculate_squeeze_shape")

    let data_matrix = NDBuffer[
        1,
        DimList(8),
        DType.index,
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
    let remove_indices = NDBuffer[
        1,
        DimList(3),
        DType.index,
    ].stack_allocation()
    remove_indices[StaticIntTuple[1](0)] = 0
    remove_indices[StaticIntTuple[1](1)] = -1  # same as index 7
    remove_indices[StaticIntTuple[1](2)] = 4

    let final_shape = NDBuffer[
        1,
        DimList(5),
        DType.index,
    ].stack_allocation()

    calculate_squeeze_shape[DType.index, DType.index, False](
        rebind[NDBuffer[1, DimList.create_unknown[1](), DType.index]](
            data_matrix
        ),
        rebind[NDBuffer[1, DimList.create_unknown[1](), DType.index]](
            remove_indices
        ),
        rebind[NDBuffer[1, DimList.create_unknown[1](), DType.index]](
            final_shape
        ),
    )

    # CHECK: 10 1 11 12 13
    print(
        final_shape[0],
        final_shape[1],
        final_shape[2],
        final_shape[3],
        final_shape[4],
    )


fn main():
    test_calculate_squeeze_shape()
