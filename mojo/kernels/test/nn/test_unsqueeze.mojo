# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from buffer import NDBuffer
from buffer.dimlist import DimList
from memory import UnsafePointer, stack_allocation
from MOGG import calculate_unsqueeze_shape, to_buffer

from utils.index import IndexList


# CHECK-LABEL: test_calculate_unsqueeze_shape
fn test_calculate_unsqueeze_shape():
    print("== test_calculate_unsqueeze_shape")

    var data_matrix = NDBuffer[
        DType.index,
        1,
        DimList(2),
    ].stack_allocation()
    data_matrix[IndexList[1](0)] = 10
    data_matrix[IndexList[1](1)] = 11

    # Main thing is indices not sorted.
    # Final shape should be [1, 1, 10, 1, 11, 1]
    var padding_indices = NDBuffer[
        DType.index,
        1,
        DimList(4),
    ].stack_allocation()
    padding_indices[IndexList[1](0)] = 3
    padding_indices[IndexList[1](1)] = 1
    padding_indices[IndexList[1](2)] = -6  # same as index 0
    padding_indices[IndexList[1](3)] = 5

    var final_shape = NDBuffer[
        DType.index,
        1,
        DimList(6),
    ].stack_allocation()

    calculate_unsqueeze_shape[DType.index, DType.index, False](
        rebind[NDBuffer[DType.index, 1]](data_matrix),
        rebind[NDBuffer[DType.index, 1]](padding_indices),
        rebind[NDBuffer[DType.index, 1]](final_shape),
    )

    # CHECK: 1 1 10 1 11 1
    print(
        final_shape[0],
        final_shape[1],
        final_shape[2],
        final_shape[3],
        final_shape[4],
        final_shape[5],
    )


fn main():
    test_calculate_unsqueeze_shape()
