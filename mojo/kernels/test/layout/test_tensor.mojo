# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

from kernel_utils._utils import ManagedLayoutTensor
from kernel_utils.int_tuple import IntTuple
from kernel_utils.layout import Layout, print_layout
from kernel_utils.layout_tensor import LayoutTensor


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

    var managed_tensor = ManagedLayoutTensor[
        Layout(IntTuple(8, 4)), DType.float32
    ]()
    var tensor = managed_tensor.tensor
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
    # CHECK: 0.0     4.0
    # CHECK: 1.0     5.0
    # CHECK: ----tile[ 0 , 1 ]----
    # CHECK: 8.0     12.0
    # CHECK: 9.0     13.0
    # CHECK: ----tile[ 0 , 2 ]----
    # CHECK: 16.0    20.0
    # CHECK: 17.0    21.0
    # CHECK: ----tile[ 0 , 3 ]----
    # CHECK: 24.0    28.0
    # CHECK: 25.0    29.0
    # CHECK: ----tile[ 1 , 0 ]----
    # CHECK: 2.0     6.0
    # CHECK: 3.0     7.0
    # CHECK: ----tile[ 1 , 1 ]----
    # CHECK: 10.0    14.0
    # CHECK: 11.0    15.0
    # CHECK: ----tile[ 1 , 2 ]----
    # CHECK: 18.0    22.0
    # CHECK: 19.0    23.0
    # CHECK: ----tile[ 1 , 3 ]----
    # CHECK: 26.0    30.0
    # CHECK: 27.0    31.0
    for tile_i in range(2):
        for tile_j in range(4):
            print("----tile[", tile_i, ",", tile_j, "]----")
            var tile_2x2 = transposed_tensor.tile[2, 2](tile_i, tile_j)
            print_tile_tensor(tile_2x2)

    _ = managed_tensor ^


# CHECK-LABEL: test_tesnsor_fragments
#   Get fragments of the followig layout
#   TH_(0,0)    TH_(0,1)    TH_(0,0)    TH_(0,1)
#   TH_(1,0)    TH_(1,1)    TH_(1,0)    TH_(1,1)
#   TH_(0,0)    TH_(0,1)    TH_(0,0)    TH_(0,1)
#   TH_(1,0)    TH_(1,1)    TH_(1,0)    TH_(1,1)
#   TH_(0,0)    TH_(0,1)    TH_(0,0)    TH_(0,1)
#   TH_(1,0)    TH_(1,1)    TH_(1,0)    TH_(1,1)
#   TH_(0,0)    TH_(0,1)    TH_(0,0)    TH_(0,1)
#   TH_(1,0)    TH_(1,1)    TH_(1,0)    TH_(1,1)
fn test_tesnsor_fragments():
    print("== test_tesnsor_fragments")

    var managed_tensor = ManagedLayoutTensor[
        Layout(IntTuple(8, 4)), DType.float32
    ]()

    var tensor = managed_tensor.tensor
    tensor.linspace()

    # CHECK: ----fragments-data[ 0 , 0 ]----
    # CHECK: 0.0     2.0
    # CHECK: 8.0     10.0
    # CHECK: 16.0    18.0
    # CHECK: 24.0    26.0
    # CHECK: ----fragments-data[ 0 , 1 ]----
    # CHECK: 1.0     3.0
    # CHECK: 9.0     11.0
    # CHECK: 17.0    19.0
    # CHECK: 25.0    27.0
    # CHECK: ----fragments-data[ 1 , 0 ]----
    # CHECK: 4.0     6.0
    # CHECK: 12.0    14.0
    # CHECK: 20.0    22.0
    # CHECK: 28.0    30.0
    # CHECK: ----fragments-data[ 1 , 1 ]----
    # CHECK: 5.0     7.0
    # CHECK: 13.0    15.0
    # CHECK: 21.0    23.0
    # CHECK: 29.0    31.0
    for th_i in range(2):
        for th_j in range(2):
            print("----fragments-data[", th_i, ",", th_j, "]----")
            var fragment_4x2 = tensor.distribute[Layout(IntTuple(2, 2))](
                th_i, th_j
            )
            print_tile_tensor(fragment_4x2)

    _ = managed_tensor ^


# CHECK-LABEL: test_tensor_tile_and_distribute
fn test_tensor_tile_and_distribute():
    print("== test_tensor_tile_and_distribute")

    var managed_tensor = ManagedLayoutTensor[
        Layout(IntTuple(8, 8)), DType.float32
    ]()

    var tensor = managed_tensor.tensor
    tensor.linspace()

    # CHECK: ----tile-data[ 0 , 0 ]----
    # CHECK: 0.0     1.0     2.0     3.0
    # CHECK: 8.0     9.0     10.0    11.0
    # CHECK: 16.0    17.0    18.0    19.0
    # CHECK: 24.0    25.0    26.0    27.0
    # CHECK: ----fragments-data[ 0 , 0 ]----
    # CHECK: 0.0     2.0
    # CHECK: 16.0    18.0
    # CHECK: ----fragments-data[ 0 , 1 ]----
    # CHECK: 1.0     3.0
    # CHECK: 17.0    19.0
    # CHECK: ----fragments-data[ 1 , 0 ]----
    # CHECK: 8.0     10.0
    # CHECK: 24.0    26.0
    # CHECK: ----fragments-data[ 1 , 1 ]----
    # CHECK: 9.0     11.0
    # CHECK: 25.0    27.0
    # CHECK: ----tile-data[ 0 , 1 ]----
    # CHECK: 4.0     5.0     6.0     7.0
    # CHECK: 12.0    13.0    14.0    15.0
    # CHECK: 20.0    21.0    22.0    23.0
    # CHECK: 28.0    29.0    30.0    31.0
    # CHECK: ----fragments-data[ 0 , 0 ]----
    # CHECK: 4.0     6.0
    # CHECK: 20.0    22.0
    # CHECK: ----fragments-data[ 0 , 1 ]----
    # CHECK: 5.0     7.0
    # CHECK: 21.0    23.0
    # CHECK: ----fragments-data[ 1 , 0 ]----
    # CHECK: 12.0    14.0
    # CHECK: 28.0    30.0
    # CHECK: ----fragments-data[ 1 , 1 ]----
    # CHECK: 13.0    15.0
    # CHECK: 29.0    31.0
    # CHECK: ----tile-data[ 1 , 0 ]----
    # CHECK: 32.0    33.0    34.0    35.0
    # CHECK: 40.0    41.0    42.0    43.0
    # CHECK: 48.0    49.0    50.0    51.0
    # CHECK: 56.0    57.0    58.0    59.0
    # CHECK: ----fragments-data[ 0 , 0 ]----
    # CHECK: 32.0    34.0
    # CHECK: 48.0    50.0
    # CHECK: ----fragments-data[ 0 , 1 ]----
    # CHECK: 33.0    35.0
    # CHECK: 49.0    51.0
    # CHECK: ----fragments-data[ 1 , 0 ]----
    # CHECK: 40.0    42.0
    # CHECK: 56.0    58.0
    # CHECK: ----fragments-data[ 1 , 1 ]----
    # CHECK: 41.0    43.0
    # CHECK: 57.0    59.0
    # CHECK: ----tile-data[ 1 , 1 ]----
    # CHECK: 36.0    37.0    38.0    39.0
    # CHECK: 44.0    45.0    46.0    47.0
    # CHECK: 52.0    53.0    54.0    55.0
    # CHECK: 60.0    61.0    62.0    63.0
    # CHECK: ----fragments-data[ 0 , 0 ]----
    # CHECK: 36.0    38.0
    # CHECK: 52.0    54.0
    # CHECK: ----fragments-data[ 0 , 1 ]----
    # CHECK: 37.0    39.0
    # CHECK: 53.0    55.0
    # CHECK: ----fragments-data[ 1 , 0 ]----
    # CHECK: 44.0    46.0
    # CHECK: 60.0    62.0
    # CHECK: ----fragments-data[ 1 , 1 ]----
    # CHECK: 45.0    47.0
    # CHECK: 61.0    63.0
    for tile_i in range(2):
        for tile_j in range(2):
            var tile_4x4 = tensor.tile[4, 4](tile_i, tile_j)
            print("----tile-data[", tile_i, ",", tile_j, "]----")
            print_tile_tensor(tile_4x4)
            for th_i in range(2):
                for th_j in range(2):
                    var fragment_2x2 = tile_4x4.distribute[IntTuple(2, 2)](
                        th_i, th_j
                    )
                    print("----fragments-data[", th_i, ",", th_j, "]----")
                    print_tile_tensor(fragment_2x2)
    _ = managed_tensor ^


# CHECK-LABEL: test_tensor_tile_and_distribute_custom_layout
fn test_tensor_tile_and_distribute_custom_layout():
    print("== test_tensor_tile_and_distribute_custom_layout")
    var managed_tensor = ManagedLayoutTensor[
        Layout(IntTuple(2, 4)), DType.float32
    ]()
    var tensor = managed_tensor.tensor
    tensor.linspace()
    # CHECK: 0.0   1.0   2.0   3.0
    # CHECK: 4.0   5.0   6.0   7.0
    tensor.print()

    # CHECK: col-major-thread-layout
    # CHECK: ----fragments-data[ 0 , 0 ]----
    # CHECK: 0.0   2.0
    # CHECK: ----fragments-data[ 0 , 1 ]----
    # CHECK: 4.0   6.0
    # CHECK: ----fragments-data[ 1 , 0 ]----
    # CHECK: 1.0   3.0
    # CHECK: ----fragments-data[ 1 , 1 ]----
    # CHECK: 5.0   7.0
    print("col-major-thread-layout")
    for th_i in range(2):
        for th_j in range(2):
            var fragments_1x2 = tensor.distribute[
                Layout(IntTuple(2, 2), IntTuple(2, 1))
            ](th_i, th_j)
            print("----fragments-data[", th_i, ",", th_j, "]----")
            fragments_1x2.print()

    # CHECK: row-major-thread-layout
    # CHECK: ----fragments-data[ 0 , 0 ]----
    # CHECK: 0.0   2.0
    # CHECK: ----fragments-data[ 0 , 1 ]----
    # CHECK: 1.0   3.0
    # CHECK: ----fragments-data[ 1 , 0 ]----
    # CHECK: 4.0   6.0
    # CHECK: ----fragments-data[ 1 , 1 ]----
    # CHECK: 5.0   7.0
    print("row-major-thread-layout")
    for th_i in range(2):
        for th_j in range(2):
            var fragments_1x2 = tensor.distribute[
                Layout(IntTuple(2, 2), IntTuple(1, 2))
            ](th_i, th_j)
            print("----fragments-data[", th_i, ",", th_j, "]----")
            fragments_1x2.print()

    _ = managed_tensor ^


# CHECK-LABEL: test_copy_to_tile_major_layout
fn test_copy_to_tile_major_layout():
    print("== test_copy_to_tile_major_layout")
    var mat_4x4_row_major = LayoutTensor[
        Layout(IntTuple(4, 4), IntTuple(4, 1)), DType.float32
    ].stack_allocation()
    mat_4x4_row_major.linspace()

    # CHECK: (((2, 2), (2, 2)):((1, 8), (2, 4)))
    # CHECK:        0    1    2    3
    # CHECK:     +----+----+----+----+
    # CHECK:  0  |  0 |  2 |  4 |  6 |
    # CHECK:     +----+----+----+----+
    # CHECK:  1  |  1 |  3 |  5 |  7 |
    # CHECK:     +----+----+----+----+
    # CHECK:  2  |  8 | 10 | 12 | 14 |
    # CHECK:     +----+----+----+----+
    # CHECK:  3  |  9 | 11 | 13 | 15 |
    # CHECK:     +----+----+----+----+
    alias tiled_major_layout = Layout(
        IntTuple(IntTuple(2, 2), IntTuple(2, 2)),
        IntTuple(IntTuple(1, 8), IntTuple(2, 4)),
    )
    var mat_4x4_tiled_2x2 = LayoutTensor[
        tiled_major_layout, DType.float32
    ].stack_allocation()
    print_layout(tiled_major_layout)

    mat_4x4_tiled_2x2.copy_from_numa(mat_4x4_row_major)

    # CHECK: mat_4x4_row_major:
    # CHECK: row: 0 data 0.0         1.0     2.0     3.0
    # CHECK: row: 1 data 4.0         5.0     6.0     7.0
    # CHECK: row: 2 data 8.0         9.0     10.0    11.0
    # CHECK: row: 3 data 12.0        13.0    14.0    15.0
    print("mat_4x4_row_major:")
    for i in range(4):
        print_no_newline("row:", i, "data ")
        for j in range(4):
            print_no_newline(mat_4x4_row_major.ptr[i * 4 + j], "\t")
        print("")

    # CHECK: mat_4x4_tiled_2x2:
    # CEHCK: row: 0 data 0.0         4.0     1.0     5.0
    # CEHCK: row: 1 data 2.0         6.0     3.0     7.0
    # CEHCK: row: 2 data 8.0         12.0    9.0     13.0
    # CEHCK: row: 3 data 10.0        14.0    11.0    15.0
    print("mat_4x4_tiled_2x2:")
    for i in range(4):
        print_no_newline("row:", i, "data ")
        for j in range(4):
            print_no_newline(mat_4x4_tiled_2x2.ptr[i * 4 + j], "\t")
        print("")


fn main():
    test_basic_tensor_ops()
    test_tesnsor_fragments()
    test_tensor_tile_and_distribute()
    test_tensor_tile_and_distribute_custom_layout()
    test_copy_to_tile_major_layout()
