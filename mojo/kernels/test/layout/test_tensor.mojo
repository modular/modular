# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

from kernel_utils.int_tuple import IntTuple
from kernel_utils.layout import (
    Layout,
    LayoutList,
    logical_divide,
    print_layout,
    zipped_divide,
)
from kernel_utils.layout_tensor import (
    LayoutTensor,
    NewLayoutTensor,
)
from memory import stack_allocation


fn print_raw_major_tensor[
    layout: Layout, dtype: DType
](tensor: LayoutTensor[layout, dtype]):
    for i in range(tensor.dim[0]()):
        for j in range(tensor.dim[1]()):
            print_no_newline(tensor[i, j], "\t")
        print("")


fn print_tile_tensor[
    layout: Layout, dtype: DType
](tensor: LayoutTensor[layout, dtype]):
    for i in range(tensor.dim[0]()):
        for j in range(tensor.dim[1]()):
            print_no_newline(tensor[i, j], "\t")
        print("")


# CHECK-LABEL: test_basic_tensor_ops
fn test_basic_tensor_ops():
    print("== test_basic_tensor_ops")

    var tensor = NewLayoutTensor[8, 4, DType.float32]()
    tensor.linspace()

    # CHECK: ----original matrix----
    # CHECK: 0.0      1.0     2.0     3.0
    # CHECK: 4.0      5.0     6.0     7.0
    # CHECK: 8.0      9.0     10.0    11.0
    # CHECK: 12.0     13.0    14.0    15.0
    # CHECK: 16.0     17.0    18.0    19.0
    # CHECK: 20.0     21.0    22.0    23.0
    # CHECK: 24.0     25.0    26.0    27.0
    # CHECK: 28.0     29.0    30.0    31.0
    print("----original matrix----")
    print_raw_major_tensor(tensor)

    # CHECK: ----transposed matrix----
    # CHECK: 0.0     4.0     8.0     12.0    16.0    20.0    24.0    28.0
    # CHECK: 1.0     5.0     9.0     13.0    17.0    21.0    25.0    29.0
    # CHECK: 2.0     6.0     10.0    14.0    18.0    22.0    26.0    30.0
    # CHECK: 3.0     7.0     11.0    15.0    19.0    23.0    27.0    31.0
    var transposed_tensor = tensor.transpose()
    print("----transposed matrix----")
    print_raw_major_tensor(transposed_tensor)

    # CHECK: ----tile[ 0 , 0 ]----
    # CHECK: 0.0     1.0
    # CHECK: 4.0     5.0
    # CHECK: ----tile[ 0 , 1 ]----
    # CHECK: 2.0     3.0
    # CHECK: 6.0     7.0
    # CHECK: ----tile[ 1 , 0 ]----
    # CHECK: 8.0     9.0
    # CHECK: 12.0    13.0
    # CHECK: ----tile[ 1 , 1 ]----
    # CHECK: 10.0    11.0
    # CHECK: 14.0    15.0
    # CHECK: ----tile[ 2 , 0 ]----
    # CHECK: 16.0    17.0
    # CHECK: 20.0    21.0
    # CHECK: ----tile[ 2 , 1 ]----
    # CHECK: 18.0    19.0
    # CHECK: 22.0    23.0
    # CHECK: ----tile[ 3 , 0 ]----
    # CHECK: 24.0    25.0
    # CHECK: 28.0    29.0
    # CHECK: ----tile[ 3 , 1 ]----
    # CHECK: 26.0    27.0
    # CHECK: 30.0    31.0
    for tile_i in range(4):
        for tile_j in range(2):
            print("----tile[", tile_i, ",", tile_j, "]----")
            var tile_2x2 = tensor.view[2, 2](tile_i, tile_j)
            print_tile_tensor(tile_2x2)


fn main():
    test_basic_tensor_ops()
