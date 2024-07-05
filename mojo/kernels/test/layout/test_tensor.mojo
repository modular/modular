# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from math import ceildiv

from buffer import NDBuffer
from buffer.list import DimList
from layout._utils import ManagedLayoutTensor
from layout.int_tuple import product
from layout.layout import Layout
from layout.layout_tensor import *


fn print_raw_major_tensor[
    layout: Layout, dtype: DType
](tensor: LayoutTensor[dtype, layout]):
    for i in range(tensor.dim[0]()):
        for j in range(tensor.dim[1]()):
            print(tensor[i, j], "\t", end="")
        print("")


fn print_tile_tensor[
    layout: Layout, dtype: DType, mask: Bool
](tensor: LayoutTensor[dtype, layout, masked=mask]):
    for i in range(tensor.dim[0]()):
        for j in range(tensor.dim[1]()):
            print(tensor[i, j], "\t", end="")
        print("")


# Print for shape ((m, n), (p, q)) in a 2D format
fn print_mode2_shape2_tensor[
    layout: Layout, dtype: DType
](tensor: LayoutTensor[dtype, layout]):
    constrained[
        len(layout) == 2
        and len(layout.shape[0]) == 2
        and len(layout.shape[1]) == 2
    ]()
    for i in range(product(layout.shape[0])):
        for j in range(product(layout.shape[1])):
            var idx = layout(IntTuple(i, j))
            print(tensor.ptr[idx], "\t", end="")
        print("")


# CHECK-LABEL: test_basic_tensor_ops
fn test_basic_tensor_ops():
    print("== test_basic_tensor_ops")

    var managed_tensor = ManagedLayoutTensor[
        DType.float32, Layout(IntTuple(8, 4))
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

    print("----1d-tensor-tiles----")
    var tensor_8 = LayoutTensor[DType.float32, Layout(8)].stack_allocation[
        alignment=16
    ]()
    tensor_8.linspace()
    # CHECK: ----tile[ 0 ]----
    # CHECK: 0.0
    # CHECK: 1.0
    # CHECK: ----tile[ 1 ]----
    # CHECK: 2.0
    # CHECK: 3.0
    # CHECK: ----tile[ 2 ]----
    # CHECK: 4.0
    # CHECK: 5.0
    # CHECK: ----tile[ 3 ]----
    # CHECK: 6.0
    # CHECK: 7.0
    for tile_i in range(4):
        print("----tile[", tile_i, "]----")
        var tile = tensor_8.tile[2](tile_i)
        tile.print()

    _ = managed_tensor^


# CHECK-LABEL: test_tesnsor_fragments
#   Get fragments of the followig layout
#   TH_0    TH_2    TH_0    TH_2
#   TH_1    TH_3    TH_1    TH_3
#   TH_0    TH_2    TH_0    TH_2
#   TH_1    TH_3    TH_1    TH_3
#   TH_0    TH_2    TH_0    TH_2
#   TH_1    TH_3    TH_1    TH_3
#   TH_0    TH_2    TH_0    TH_2
#   TH_1    TH_3    TH_1    TH_3
fn test_tesnsor_fragments():
    print("== test_tesnsor_fragments")

    var managed_tensor = ManagedLayoutTensor[
        DType.float32, Layout(IntTuple(8, 4))
    ]()

    var tensor = managed_tensor.tensor
    tensor.linspace()

    # CHECK: ----fragments-data[ 0 ]----
    # CHECK: 0.0     2.0
    # CHECK: 8.0     10.0
    # CHECK: 16.0    18.0
    # CHECK: 24.0    26.0
    # CHECK: ----fragments-data[ 1 ]----
    # CHECK: 4.0     6.0
    # CHECK: 12.0    14.0
    # CHECK: 20.0    22.0
    # CHECK: 28.0    30.0
    # CHECK: ----fragments-data[ 2 ]----
    # CHECK: 1.0     3.0
    # CHECK: 9.0     11.0
    # CHECK: 17.0    19.0
    # CHECK: 25.0    27.0
    # CHECK: ----fragments-data[ 3 ]----
    # CHECK: 5.0     7.0
    # CHECK: 13.0    15.0
    # CHECK: 21.0    23.0
    # CHECK: 29.0    31.0
    for th_i in range(UInt(4)):
        print("----fragments-data[", th_i, "]----")
        var fragment_4x2 = tensor.distribute[Layout(IntTuple(2, 2))](th_i)
        print_tile_tensor(fragment_4x2)

    _ = managed_tensor^


# CHECK-LABEL: test_tensor_tile_and_distribute
fn test_tensor_tile_and_distribute():
    print("== test_tensor_tile_and_distribute")

    var managed_tensor = ManagedLayoutTensor[
        DType.float32, Layout(IntTuple(8, 8))
    ]()

    var tensor = managed_tensor.tensor
    tensor.linspace()

    # CHECK: ----tile-data[ 0 , 0 ]----
    # CHECK: 0.0     1.0     2.0     3.0
    # CHECK: 8.0     9.0     10.0    11.0
    # CHECK: 16.0    17.0    18.0    19.0
    # CHECK: 24.0    25.0    26.0    27.0
    # CHECK: ----fragments-data[ 0 ]----
    # CHECK: 0.0     2.0
    # CHECK: 16.0    18.0
    # CHECK: ----fragments-data[ 1 ]----
    # CHECK: 1.0     3.0
    # CHECK: 17.0    19.0
    # CHECK: ----fragments-data[ 2 ]----
    # CHECK: 8.0     10.0
    # CHECK: 24.0    26.0
    # CHECK: ----fragments-data[ 3 ]----
    # CHECK: 9.0     11.0
    # CHECK: 25.0    27.0
    # CHECK: ----tile-data[ 0 , 1 ]----
    # CHECK: 4.0     5.0     6.0     7.0
    # CHECK: 12.0    13.0    14.0    15.0
    # CHECK: 20.0    21.0    22.0    23.0
    # CHECK: 28.0    29.0    30.0    31.0
    # CHECK: ----fragments-data[ 0 ]----
    # CHECK: 4.0     6.0
    # CHECK: 20.0    22.0
    # CHECK: ----fragments-data[ 1 ]----
    # CHECK: 5.0     7.0
    # CHECK: 21.0    23.0
    # CHECK: ----fragments-data[ 2 ]----
    # CHECK: 12.0    14.0
    # CHECK: 28.0    30.0
    # CHECK: ----fragments-data[ 3 ]----
    # CHECK: 13.0    15.0
    # CHECK: 29.0    31.0
    # CHECK: ----tile-data[ 1 , 0 ]----
    # CHECK: 32.0    33.0    34.0    35.0
    # CHECK: 40.0    41.0    42.0    43.0
    # CHECK: 48.0    49.0    50.0    51.0
    # CHECK: 56.0    57.0    58.0    59.0
    # CHECK: ----fragments-data[ 0 ]----
    # CHECK: 32.0    34.0
    # CHECK: 48.0    50.0
    # CHECK: ----fragments-data[ 1 ]----
    # CHECK: 33.0    35.0
    # CHECK: 49.0    51.0
    # CHECK: ----fragments-data[ 2 ]----
    # CHECK: 40.0    42.0
    # CHECK: 56.0    58.0
    # CHECK: ----fragments-data[ 3 ]----
    # CHECK: 41.0    43.0
    # CHECK: 57.0    59.0
    # CHECK: ----tile-data[ 1 , 1 ]----
    # CHECK: 36.0    37.0    38.0    39.0
    # CHECK: 44.0    45.0    46.0    47.0
    # CHECK: 52.0    53.0    54.0    55.0
    # CHECK: 60.0    61.0    62.0    63.0
    # CHECK: ----fragments-data[ 0 ]----
    # CHECK: 36.0    38.0
    # CHECK: 52.0    54.0
    # CHECK: ----fragments-data[ 1 ]----
    # CHECK: 37.0    39.0
    # CHECK: 53.0    55.0
    # CHECK: ----fragments-data[ 2 ]----
    # CHECK: 44.0    46.0
    # CHECK: 60.0    62.0
    # CHECK: ----fragments-data[ 3 ]----
    # CHECK: 45.0    47.0
    # CHECK: 61.0    63.0
    for tile_i in range(2):
        for tile_j in range(2):
            var tile_4x4 = tensor.tile[4, 4](tile_i, tile_j)
            print("----tile-data[", tile_i, ",", tile_j, "]----")
            print_tile_tensor(tile_4x4)
            for th_i in range(4):
                var fragment_2x2 = tile_4x4.distribute[
                    Layout(IntTuple(2, 2), IntTuple(2, 1))
                ](
                    th_i.value,
                )
                print("----fragments-data[", th_i, "]----")
                print_tile_tensor(fragment_2x2)
    _ = managed_tensor^


# CHECK-LABEL: test_tensor_tile_and_distribute_custom_layout
fn test_tensor_tile_and_distribute_custom_layout():
    print("== test_tensor_tile_and_distribute_custom_layout")
    var managed_tensor = ManagedLayoutTensor[
        DType.float32, Layout(IntTuple(2, 4))
    ]()
    var tensor = managed_tensor.tensor
    tensor.linspace()
    # CHECK: 0.0   1.0   2.0   3.0
    # CHECK: 4.0   5.0   6.0   7.0
    tensor.print()

    # CHECK: row-major-thread-layout
    # CHECK: ----fragments-data[ 0 ]----
    # CHECK: 0.0   2.0
    # CHECK: ----fragments-data[ 1 ]----
    # CHECK: 1.0   3.0
    # CHECK: ----fragments-data[ 2 ]----
    # CHECK: 4.0   6.0
    # CHECK: ----fragments-data[ 3 ]----
    # CHECK: 5.0   7.0
    print("row-major-thread-layout")
    for th_i in range(UInt(4)):
        var fragments_1x2 = tensor.distribute[
            Layout(IntTuple(2, 2), IntTuple(2, 1))
        ](th_i)
        print("----fragments-data[", th_i, "]----")
        fragments_1x2.print()

    # CHECK: col-major-thread-layout
    # CHECK: ----fragments-data[ 0 ]----
    # CHECK: 0.0   2.0
    # CHECK: ----fragments-data[ 1 ]----
    # CHECK: 4.0   6.0
    # CHECK: ----fragments-data[ 2 ]----
    # CHECK: 1.0   3.0
    # CHECK: ----fragments-data[ 3 ]----
    # CHECK: 5.0   7.0
    print("col-major-thread-layout")
    for th_i in range(UInt(4)):
        var fragments_1x2 = tensor.distribute[
            Layout(IntTuple(2, 2), IntTuple(1, 2))
        ](th_i)
        print("----fragments-data[", th_i, "]----")
        fragments_1x2.print()

    _ = managed_tensor^


# CHECK-LABEL: test_copy_to_tile_major_layout
fn test_copy_to_tile_major_layout():
    print("== test_copy_to_tile_major_layout")
    var mat_4x4_row_major = LayoutTensor[
        DType.float32, Layout(IntTuple(4, 4), IntTuple(4, 1))
    ].stack_allocation[alignment=16]()
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
        DType.float32, tiled_major_layout
    ].stack_allocation[alignment=16]()
    print_layout(tiled_major_layout)

    mat_4x4_tiled_2x2.copy_from(mat_4x4_row_major)

    # CHECK: mat_4x4_row_major:
    # CHECK: row: 0 data 0.0         1.0     2.0     3.0
    # CHECK: row: 1 data 4.0         5.0     6.0     7.0
    # CHECK: row: 2 data 8.0         9.0     10.0    11.0
    # CHECK: row: 3 data 12.0        13.0    14.0    15.0
    print("mat_4x4_row_major:")
    for i in range(4):
        print("row:", i, "data ", end="")
        for j in range(4):
            print(mat_4x4_row_major.ptr[i * 4 + j], "\t", end="")
        print("")

    # CHECK: mat_4x4_tiled_2x2:
    # CEHCK: row: 0 data 0.0         4.0     1.0     5.0
    # CEHCK: row: 1 data 2.0         6.0     3.0     7.0
    # CEHCK: row: 2 data 8.0         12.0    9.0     13.0
    # CEHCK: row: 3 data 10.0        14.0    11.0    15.0
    print("mat_4x4_tiled_2x2:")
    for i in range(4):
        print("row:", i, "data ", end="")
        for j in range(4):
            print(mat_4x4_tiled_2x2.ptr[i * 4 + j], "\t", end="")
        print("")


# This test repeats the following layout into the 4x8 matrix resulting:
# TH_0 TH_2 TH_4 TH_6 TH_0 TH_2 TH_4 TH_6
# TH_1 TH_3 TH_5 TH_7 TH_1 TH_3 TH_5 TH_7
# TH_0 TH_2 TH_4 TH_6 TH_0 TH_2 TH_4 TH_6
# TH_1 TH_3 TH_5 TH_7 TH_1 TH_3 TH_5 TH_7
fn test_distribute_tiled_layout():
    print("== test_distribute_tiled_layout")
    var tensor = LayoutTensor[
        DType.float32, Layout(IntTuple(4, 8), IntTuple(8, 1))
    ].stack_allocation[alignment=16]()
    tensor.linspace()
    alias threads_2x4_layout = Layout(
        IntTuple(2, IntTuple(2, 2)), IntTuple(1, IntTuple(2, 4))
    )
    # CHECK: ----fragments-data[ 0 ]----
    # CHECK: 0.0   4.0
    # CHECK: 16.0   20.0
    # CHECK: ----fragments-data[ 1 ]----
    # CHECK: 8.0   12.0
    # CHECK: 24.0   28.0
    # CHECK: ----fragments-data[ 2 ]----
    # CHECK: 1.0   5.0
    # CHECK: 17.0   21.0
    # CHECK: ----fragments-data[ 3 ]----
    # CHECK: 9.0   13.0
    # CHECK: 25.0   29.0
    # CHECK: ----fragments-data[ 4 ]----
    # CHECK: 2.0   6.0
    # CHECK: 18.0   22.0
    # CHECK: ----fragments-data[ 5 ]----
    # CHECK: 10.0   14.0
    # CHECK: 26.0   30.0
    # CHECK: ----fragments-data[ 6 ]----
    # CHECK: 3.0   7.0
    # CHECK: 19.0   23.0
    # CHECK: ----fragments-data[ 7 ]----
    # CHECK: 11.0   15.0
    # CHECK: 27.0   31.0
    for th_i in range(UInt(8)):
        var thread_tile = tensor.distribute[threads_2x4_layout](th_i)
        print("----fragments-data[", th_i, "]----")
        thread_tile.print()


# CHECK-LABEL: test_distribute_with_tile_size
fn test_distribute_with_tile_size():
    print("== test_distribute_with_tile_size")

    var tensor0 = LayoutTensor[
        DType.float32, Layout(IntTuple(16, 8), IntTuple(8, 1))
    ].stack_allocation[alignment=16]()

    tensor0.linspace()

    # Thread layout:
    # TH_0 TH_2
    # TH_1 TH_3
    # TH_4 TH_6
    # TH_5 TH_7
    alias thread_layout = Layout(
        IntTuple(IntTuple(2, 2), 2), IntTuple(IntTuple(1, 4), 2)
    )

    # Each thread should have a rank-2 layout. The first mode is a tile to work
    # on. The second mode is how this tile repeats in the domain. The tiles are
    # distributed as follow:
    # +-----------+-----------+-----------+-----------+
    # | TH_0 TH_0 | TH_2 TH_2 | TH_0 TH_0 | TH_2 TH_2 |
    # | TH_0 TH_0 | TH_2 TH_2 | TH_0 TH_0 | TH_2 TH_2 |
    # |-----------+-----------+-----------+-----------+
    # | TH_1 TH_1 | TH_3 TH_3 | TH_1 TH_1 | TH_3 TH_3 |
    # | TH_1 TH_1 | TH_3 TH_3 | TH_1 TH_1 | TH_3 TH_3 |
    # |-----------+-----------+-----------+-----------+
    # | TH_4 TH_4 | TH_6 TH_6 | TH_4 TH_4 | TH_6 TH_6 |
    # | TH_4 TH_4 | TH_6 TH_6 | TH_4 TH_4 | TH_6 TH_6 |
    # |-----------+-----------+-----------+-----------+
    # | TH_5 TH_5 | TH_7 TH_7 | TH_5 TH_5 | TH_7 TH_7 |
    # | TH_5 TH_5 | TH_7 TH_7 | TH_5 TH_5 | TH_7 TH_7 |
    # |-----------+-----------+-----------+-----------+
    # | TH_0 TH_0 | TH_2 TH_2 | TH_0 TH_0 | TH_2 TH_2 |
    # | TH_0 TH_0 | TH_2 TH_2 | TH_0 TH_0 | TH_2 TH_2 |
    # |-----------+-----------+-----------+-----------+
    # | TH_1 TH_1 | TH_3 TH_3 | TH_1 TH_1 | TH_3 TH_3 |
    # | TH_1 TH_1 | TH_3 TH_3 | TH_1 TH_1 | TH_3 TH_3 |
    # |-----------+-----------+-----------+-----------+
    # | TH_4 TH_4 | TH_6 TH_6 | TH_4 TH_4 | TH_6 TH_6 |
    # | TH_4 TH_4 | TH_6 TH_6 | TH_4 TH_4 | TH_6 TH_6 |
    # |-----------+-----------+-----------+-----------+
    # | TH_5 TH_5 | TH_7 TH_7 | TH_5 TH_5 | TH_7 TH_7 |
    # | TH_5 TH_5 | TH_7 TH_7 | TH_5 TH_5 | TH_7 TH_7 |
    # +-----------+-----------+-----------+-----------+

    # CHECK: ----thread[ 0 ]----
    # CHECK: [0.0, 8.0, 1.0, 9.0] [4.0, 12.0, 5.0, 13.0]
    # CHECK: [64.0, 72.0, 65.0, 73.0] [68.0, 76.0, 69.0, 77.0]
    # CHECK: ----thread[ 1 ]----
    # CHECK: [16.0, 24.0, 17.0, 25.0] [20.0, 28.0, 21.0, 29.0]
    # CHECK: [80.0, 88.0, 81.0, 89.0] [84.0, 92.0, 85.0, 93.0]
    # CHECK: ----thread[ 2 ]----
    # CHECK: [2.0, 10.0, 3.0, 11.0] [6.0, 14.0, 7.0, 15.0]
    # CHECK: [66.0, 74.0, 67.0, 75.0] [70.0, 78.0, 71.0, 79.0]
    # CHECK: ----thread[ 3 ]----
    # CHECK: [18.0, 26.0, 19.0, 27.0] [22.0, 30.0, 23.0, 31.0]
    # CHECK: [82.0, 90.0, 83.0, 91.0] [86.0, 94.0, 87.0, 95.0]
    # CHECK: ----thread[ 4 ]----
    # CHECK: [32.0, 40.0, 33.0, 41.0] [36.0, 44.0, 37.0, 45.0]
    # CHECK: [96.0, 104.0, 97.0, 105.0] [100.0, 108.0, 101.0, 109.0]
    # CHECK: ----thread[ 5 ]----
    # CHECK: [48.0, 56.0, 49.0, 57.0] [52.0, 60.0, 53.0, 61.0]
    # CHECK: [112.0, 120.0, 113.0, 121.0] [116.0, 124.0, 117.0, 125.0]
    # CHECK: ----thread[ 6 ]----
    # CHECK: [34.0, 42.0, 35.0, 43.0] [38.0, 46.0, 39.0, 47.0]
    # CHECK: [98.0, 106.0, 99.0, 107.0] [102.0, 110.0, 103.0, 111.0]
    # CHECK: ----thread[ 7 ]----
    # CHECK: [50.0, 58.0, 51.0, 59.0] [54.0, 62.0, 55.0, 63.0]
    # CHECK: [114.0, 122.0, 115.0, 123.0] [118.0, 126.0, 119.0, 127.0]

    for tid in range(thread_layout.size()):
        print("----thread[", tid, "]----")
        var tile = tensor0.vectorize[2, 2]().distribute[thread_layout](
            tid.value
        )
        tile.print()

    var tensor8x1 = LayoutTensor[
        DType.float32, Layout(IntTuple(8, 1))
    ].stack_allocation[alignment=16]()
    tensor8x1.linspace()

    # +-------------+
    # | TH_0 | TH_2 |
    # | TH_0 | TH_2 |
    # |------+------+
    # | TH_1 | TH_3 |
    # | TH_1 | TH_3 |
    # |------+------+
    # | TH_4 | TH_6 |
    # | TH_4 | TH_6 |
    # |------+------+
    # | TH_5 | TH_7 |
    # | TH_5 | TH_7 |
    # |------+------+

    # Threads 2, 3, 6, 7 should have the same fragments as thread 0, 1, 4, 5.
    # CHECK: ----thread[ 0 ]----
    # CHECK: [0.0, 1.0]
    # CHECK: ----thread[ 1 ]----
    # CHECK: [2.0, 3.0]
    # CHECK: ----thread[ 2 ]----
    # CHECK: [0.0, 1.0]
    # CHECK: ----thread[ 3 ]----
    # CHECK: [2.0, 3.0]
    # CHECK: ----thread[ 4 ]----
    # CHECK: [4.0, 5.0]
    # CHECK: ----thread[ 5 ]----
    # CHECK: [6.0, 7.0]
    # CHECK: ----thread[ 6 ]----
    # CHECK: [4.0, 5.0]
    # CHECK: ----thread[ 7 ]----
    # CHECK: [6.0, 7.0]
    for tid in range(thread_layout.size()):
        print("----thread[", tid, "]----")
        var tile = tensor8x1.vectorize[2, 1]().distribute[
            thread_layout, axis=0
        ](tid.value)
        tile.print()


# CHECK-LABEL: test_vectorize_reads
fn test_vectorize_reads():
    print("== test_vectorize_reads")
    var tensor = LayoutTensor[
        DType.float32, Layout(IntTuple(8, 8), IntTuple(1, 8))
    ].stack_allocation[alignment=16]()
    tensor.linspace()
    var tensor_8_2_vec = tensor.vectorize[1, 4]()
    # CHECK: ((8, 2):(1, 32))
    print(tensor_8_2_vec.layout)
    # CHECK: (4:8)
    print(tensor_8_2_vec.element_layout)
    # CHECK: [0.0, 1.0, 2.0, 3.0] [4.0, 5.0, 6.0, 7.0]
    # CHECK: [8.0, 9.0, 10.0, 11.0] [12.0, 13.0, 14.0, 15.0]
    # CHECK: [16.0, 17.0, 18.0, 19.0] [20.0, 21.0, 22.0, 23.0]
    # CHECK: [24.0, 25.0, 26.0, 27.0] [28.0, 29.0, 30.0, 31.0]
    # CHECK: [32.0, 33.0, 34.0, 35.0] [36.0, 37.0, 38.0, 39.0]
    # CHECK: [40.0, 41.0, 42.0, 43.0] [44.0, 45.0, 46.0, 47.0]
    # CHECK: [48.0, 49.0, 50.0, 51.0] [52.0, 53.0, 54.0, 55.0]
    # CHECK: [56.0, 57.0, 58.0, 59.0] [60.0, 61.0, 62.0, 63.0]
    tensor_8_2_vec.print()

    var tensor_2_8_vec = tensor.vectorize[4, 1]()
    # CHECK: ((2, 8):(4, 8))
    print(tensor_2_8_vec.layout)
    # CHECK: (4:1)
    print(tensor_2_8_vec.element_layout)
    # CHECK: [0.0, 8.0, 16.0, 24.0] [1.0, 9.0, 17.0, 25.0] [2.0, 10.0, 18.0, 26.0] [3.0, 11.0, 19.0, 27.0] [4.0, 12.0, 20.0, 28.0] [5.0, 13.0, 21.0, 29.0] [6.0, 14.0, 22.0, 30.0] [7.0, 15.0, 23.0, 31.0]
    # CHECK: [32.0, 40.0, 48.0, 56.0] [33.0, 41.0, 49.0, 57.0] [34.0, 42.0, 50.0, 58.0] [35.0, 43.0, 51.0, 59.0] [36.0, 44.0, 52.0, 60.0] [37.0, 45.0, 53.0, 61.0] [38.0, 46.0, 54.0, 62.0] [39.0, 47.0, 55.0, 63.0]
    tensor_2_8_vec.print()

    var tensor_2_vec = tensor.vectorize[4, 4]()
    # CHECK: ((2, 2):(4, 32))
    print(tensor_2_vec.layout)
    # CHECK: (4, 4):(1, 8))
    print(tensor_2_vec.element_layout)
    # CHECK: [0.0, 8.0, 16.0, 24.0, 1.0, 9.0, 17.0, 25.0, 2.0, 10.0, 18.0, 26.0, 3.0, 11.0, 19.0, 27.0] [4.0, 12.0, 20.0, 28.0, 5.0, 13.0, 21.0, 29.0, 6.0, 14.0, 22.0, 30.0, 7.0, 15.0, 23.0, 31.0]
    # CHECK: [32.0, 40.0, 48.0, 56.0, 33.0, 41.0, 49.0, 57.0, 34.0, 42.0, 50.0, 58.0, 35.0, 43.0, 51.0, 59.0] [36.0, 44.0, 52.0, 60.0, 37.0, 45.0, 53.0, 61.0, 38.0, 46.0, 54.0, 62.0, 39.0, 47.0, 55.0, 63.0]
    tensor_2_vec.print()


# CHECK-LABEL: test_vectorize_writes
fn test_vectorize_writes():
    print("== test_vectorize_writes")
    var tensor = LayoutTensor[
        DType.float32, Layout(IntTuple(4, 4), IntTuple(1, 4))
    ].stack_allocation[alignment=16]()
    tensor.fill(0)
    var tensor_2x2 = tensor.vectorize[2, 2]()
    tensor_2x2[0, 0] = rebind[tensor_2x2.element_type](
        SIMD[DType.float32, 4](1)
    )
    tensor_2x2[0, 1] = rebind[tensor_2x2.element_type](
        SIMD[DType.float32, 4](2)
    )
    tensor_2x2[1, 0] = rebind[tensor_2x2.element_type](
        SIMD[DType.float32, 4](3)
    )
    tensor_2x2[1, 1] = rebind[tensor_2x2.element_type](
        SIMD[DType.float32, 4](4)
    )
    # CHECK: 1.0 1.0 2.0 2.0
    # CHECK: 1.0 1.0 2.0 2.0
    # CHECK: 3.0 3.0 4.0 4.0
    # CHECK: 3.0 3.0 4.0 4.0
    tensor.print()


# CHECK-LABEL: test_slice
fn test_slice():
    print("==test_slice")
    var tensor = LayoutTensor[
        DType.float32, Layout(IntTuple(4, 4), IntTuple(1, 4))
    ].stack_allocation[alignment=16]()
    tensor.linspace()
    # CHECK: row_slice_sub_column
    # CHECK: 0.0 1.0
    # CHECK: 4.0 5.0
    # CHECK: 8.0 9.0
    # CHECK: 12.0 13.0
    print("row_slice_sub_column")
    tensor.slice[:, :2]().print()
    # CHECK: col_slice_sub_row
    # CHECK: 0.0 1.0 2.0 3.0
    # CHECK: 4.0 5.0 6.0 7.0
    print("col_slice_sub_row")
    tensor.slice[:2, :]().print()
    # sub_slice
    # 5.0 6.0
    # 9.0 10.0
    # 13.0 14.0
    print("sub_slice")
    tensor.slice[1:, 1:3]().print()
    # CHECK: bottom_right
    # CHECK: 10.0 11.0
    # CHECK: 14.0 15.0
    print("bottom_right")
    tensor.slice[2:, 2:]().print()
    # CHECK: top_left
    # CHECK: 0.0 1.0
    # CHECK: 4.0 5.0
    print("top_left")
    tensor.slice[:2, :2]().print()

    print("slice_of_slice")
    # CHECK: slice_of_slice
    # CHECK: 6.0 7.0
    # CHECK: 10.0 11.0
    tensor.slice[1:, 1:]().slice[:2, 1:]().print()


# CHECK-LABEL: test_copy_vectorized
fn test_copy_vectorized():
    print("== test_copy_vectorized")
    var tensor_8_8 = LayoutTensor[
        DType.float32, Layout(IntTuple(8, 8), IntTuple(8, 1))
    ].stack_allocation[alignment=16]()
    tensor_8_8.linspace()
    var vec_8_1 = tensor_8_8.vectorize[1, 4]()
    # CHECK: [0.0, 1.0, 2.0, 3.0] [4.0, 5.0, 6.0, 7.0]
    # CHECK: [8.0, 9.0, 10.0, 11.0] [12.0, 13.0, 14.0, 15.0]
    # CHECK: [16.0, 17.0, 18.0, 19.0] [20.0, 21.0, 22.0, 23.0]
    # CHECK: [24.0, 25.0, 26.0, 27.0] [28.0, 29.0, 30.0, 31.0]
    # CHECK: [32.0, 33.0, 34.0, 35.0] [36.0, 37.0, 38.0, 39.0]
    # CHECK: [40.0, 41.0, 42.0, 43.0] [44.0, 45.0, 46.0, 47.0]
    # CHECK: [48.0, 49.0, 50.0, 51.0] [52.0, 53.0, 54.0, 55.0]
    # CHECK: [56.0, 57.0, 58.0, 59.0] [60.0, 61.0, 62.0, 63.0]
    vec_8_1.print()
    var tensor_8_8_zeros = LayoutTensor[
        DType.float32, Layout(IntTuple(8, 8), IntTuple(8, 1))
    ].stack_allocation[
        alignment = alignof[SIMD[DType.float32, 4]]()
    ]().vectorize[
        1, 4
    ]()
    tensor_8_8_zeros.fill(0)
    tensor_8_8_zeros.copy_from(vec_8_1)
    # CHECK: [0.0, 1.0, 2.0, 3.0] [4.0, 5.0, 6.0, 7.0]
    # CHECK: [8.0, 9.0, 10.0, 11.0] [12.0, 13.0, 14.0, 15.0]
    # CHECK: [16.0, 17.0, 18.0, 19.0] [20.0, 21.0, 22.0, 23.0]
    # CHECK: [24.0, 25.0, 26.0, 27.0] [28.0, 29.0, 30.0, 31.0]
    # CHECK: [32.0, 33.0, 34.0, 35.0] [36.0, 37.0, 38.0, 39.0]
    # CHECK: [40.0, 41.0, 42.0, 43.0] [44.0, 45.0, 46.0, 47.0]
    # CHECK: [48.0, 49.0, 50.0, 51.0] [52.0, 53.0, 54.0, 55.0]
    # CHECK: [56.0, 57.0, 58.0, 59.0] [60.0, 61.0, 62.0, 63.0]
    tensor_8_8_zeros.print()

    var tensor_8_8_zeros_4_1 = stack_allocation_like(tensor_8_8).vectorize[
        4, 1
    ]()
    tensor_8_8_zeros_4_1.fill(0)
    tensor_8_8_zeros_4_1.copy_from(vec_8_1)
    # CHECK: [0.0, 1.0, 2.0, 3.0] [16.0, 17.0, 18.0, 19.0] [32.0, 33.0, 34.0, 35.0] [48.0, 49.0, 50.0, 51.0] [4.0, 5.0, 6.0, 7.0] [20.0, 21.0, 22.0, 23.0] [36.0, 37.0, 38.0, 39.0] [52.0, 53.0, 54.0, 55.0]
    # CHECK: [8.0, 9.0, 10.0, 11.0] [24.0, 25.0, 26.0, 27.0] [40.0, 41.0, 42.0, 43.0] [56.0, 57.0, 58.0, 59.0] [12.0, 13.0, 14.0, 15.0] [28.0, 29.0, 30.0, 31.0] [44.0, 45.0, 46.0, 47.0] [60.0, 61.0, 62.0, 63.0]
    tensor_8_8_zeros_4_1.print()

    var tensor_8_8_zeros_1_4 = stack_allocation_like(tensor_8_8).vectorize[
        1, 4
    ]()
    tensor_8_8_zeros_1_4.fill(0)
    tensor_8_8_zeros_1_4.copy_from(tensor_8_8_zeros_4_1)
    # CHECK: [0.0, 1.0, 2.0, 3.0] [4.0, 5.0, 6.0, 7.0]
    # CHECK: [8.0, 9.0, 10.0, 11.0] [12.0, 13.0, 14.0, 15.0]
    # CHECK: [16.0, 17.0, 18.0, 19.0] [20.0, 21.0, 22.0, 23.0]
    # CHECK: [24.0, 25.0, 26.0, 27.0] [28.0, 29.0, 30.0, 31.0]
    # CHECK: [32.0, 33.0, 34.0, 35.0] [36.0, 37.0, 38.0, 39.0]
    # CHECK: [40.0, 41.0, 42.0, 43.0] [44.0, 45.0, 46.0, 47.0]
    # CHECK: [48.0, 49.0, 50.0, 51.0] [52.0, 53.0, 54.0, 55.0]
    # CHECK: [56.0, 57.0, 58.0, 59.0] [60.0, 61.0, 62.0, 63.0]
    tensor_8_8_zeros_1_4.print()

    var tensor_8_8_zeros_4_4 = LayoutTensor[
        DType.float32, Layout(IntTuple(8, 8), IntTuple(8, 1))
    ].stack_allocation[
        alignment = alignof[SIMD[DType.float32, 4]]()
    ]().vectorize[
        4, 4
    ]()
    tensor_8_8_zeros_4_4.fill(0)
    tensor_8_8_zeros_4_4.copy_from(tensor_8_8.vectorize[4, 4]())
    # CHECK: [0.0, 8.0, 16.0, 24.0, 1.0, 9.0, 17.0, 25.0, 2.0, 10.0, 18.0, 26.0, 3.0, 11.0, 19.0, 27.0] [4.0, 12.0, 20.0, 28.0, 5.0, 13.0, 21.0, 29.0, 6.0, 14.0, 22.0, 30.0, 7.0, 15.0, 23.0, 31.0]
    # CHECK: [32.0, 40.0, 48.0, 56.0, 33.0, 41.0, 49.0, 57.0, 34.0, 42.0, 50.0, 58.0, 35.0, 43.0, 51.0, 59.0] [36.0, 44.0, 52.0, 60.0, 37.0, 45.0, 53.0, 61.0, 38.0, 46.0, 54.0, 62.0, 39.0, 47.0, 55.0, 63.0]
    tensor_8_8_zeros_4_4.print()


# CHECK-LABEL: test_distribute_vectorized
fn test_distribute_vectorized():
    print("== test_distribute_vectorized")
    var tensor_8_8 = LayoutTensor[
        DType.float32, Layout(IntTuple(8, 8), IntTuple(8, 1))
    ].stack_allocation[alignment=16]()
    tensor_8_8.linspace()

    var tensor_8_2xv4 = tensor_8_8.vectorize[1, 4]()
    # CHECK: [0.0, 1.0, 2.0, 3.0] [4.0, 5.0, 6.0, 7.0]
    # CHECK: [8.0, 9.0, 10.0, 11.0] [12.0, 13.0, 14.0, 15.0]
    # CHECK: [16.0, 17.0, 18.0, 19.0] [20.0, 21.0, 22.0, 23.0]
    # CHECK: [24.0, 25.0, 26.0, 27.0] [28.0, 29.0, 30.0, 31.0]
    # CHECK: [32.0, 33.0, 34.0, 35.0] [36.0, 37.0, 38.0, 39.0]
    # CHECK: [40.0, 41.0, 42.0, 43.0] [44.0, 45.0, 46.0, 47.0]
    # CHECK: [48.0, 49.0, 50.0, 51.0] [52.0, 53.0, 54.0, 55.0]
    # CHECK: [56.0, 57.0, 58.0, 59.0] [60.0, 61.0, 62.0, 63.0]
    tensor_8_2xv4.print()

    # CHECK: ----thread[ 0 ]----
    # CHECK: [0.0, 1.0, 2.0, 3.0]
    # CHECK: [16.0, 17.0, 18.0, 19.0]
    # CHECK: [32.0, 33.0, 34.0, 35.0]
    # CHECK: [48.0, 49.0, 50.0, 51.0]
    # CHECK: ----thread[ 1 ]----
    # CHECK: [8.0, 9.0, 10.0, 11.0]
    # CHECK: [24.0, 25.0, 26.0, 27.0]
    # CHECK: [40.0, 41.0, 42.0, 43.0]
    # CHECK: [56.0, 57.0, 58.0, 59.0]
    # CHECK: ----thread[ 2 ]----
    # CHECK: [4.0, 5.0, 6.0, 7.0]
    # CHECK: [20.0, 21.0, 22.0, 23.0]
    # CHECK: [36.0, 37.0, 38.0, 39.0]
    # CHECK: [52.0, 53.0, 54.0, 55.0]
    # CHECK: ----thread[ 3 ]----
    # CHECK: [12.0, 13.0, 14.0, 15.0]
    # CHECK: [28.0, 29.0, 30.0, 31.0]
    # CHECK: [44.0, 45.0, 46.0, 47.0]
    # CHECK: [60.0, 61.0, 62.0, 63.0]
    for tid in range(UInt(4)):
        var fragments = tensor_8_2xv4.distribute[Layout(IntTuple(2, 2))](tid)
        print("----thread[", tid, "]----")
        fragments.print()


fn test_distribute_axis_projection():
    var tensor_4x4 = LayoutTensor[
        DType.float32, Layout(IntTuple(4, 4), IntTuple(4, 1))
    ].stack_allocation[alignment=16]()
    tensor_4x4.linspace()

    # CHECK: th_id 0
    # CHECK: [0.0, 1.0, 2.0, 3.0]
    # CHECK: =====
    # CHECK: th_id 1
    # CHECK: [0.0, 1.0, 2.0, 3.0]
    # CHECK: =====
    # CHECK: th_id 2
    # CHECK: [0.0, 1.0, 2.0, 3.0]
    # CHECK: =====
    # CHECK: th_id 3
    # CHECK: [0.0, 1.0, 2.0, 3.0]
    # CHECK: =====
    # CHECK: th_id 4
    # CHECK: [4.0, 5.0, 6.0, 7.0]
    # CHECK: =====
    # CHECK: th_id 5
    # CHECK: [4.0, 5.0, 6.0, 7.0]
    # CHECK: =====
    # CHECK: th_id 6
    # CHECK: [4.0, 5.0, 6.0, 7.0]
    # CHECK: =====
    # CHECK: th_id 7
    # CHECK: [4.0, 5.0, 6.0, 7.0]
    # CHECK: =====
    # CHECK: th_id 8
    # CHECK: [8.0, 9.0, 10.0, 11.0]
    # CHECK: =====
    # CHECK: th_id 9
    # CHECK: [8.0, 9.0, 10.0, 11.0]
    # CHECK: =====
    # CHECK: th_id 10
    # CHECK: [8.0, 9.0, 10.0, 11.0]
    # CHECK: =====
    # CHECK: th_id 11
    # CHECK: [8.0, 9.0, 10.0, 11.0]
    # CHECK: =====
    # CHECK: th_id 12
    # CHECK: [12.0, 13.0, 14.0, 15.0]
    # CHECK: =====
    # CHECK: th_id 13
    # CHECK: [12.0, 13.0, 14.0, 15.0]
    # CHECK: =====
    # CHECK: th_id 14
    # CHECK: [12.0, 13.0, 14.0, 15.0]
    # CHECK: =====
    # CHECK: th_id 15
    # CHECK: [12.0, 13.0, 14.0, 15.0]
    for th_id in range(UInt(16)):
        print("th_id", th_id)
        var tensor = tensor_4x4.vectorize[1, 4]().distribute[
            Layout.row_major(4, 4), axis=0
        ](th_id)
        tensor.print()
        print("=====")

    # CHECK: th_id 0
    # CHECK: [0.0, 4.0, 8.0, 12.0]
    # CHECK: =====
    # CHECK: th_id 1
    # CHECK: [1.0, 5.0, 9.0, 13.0]
    # CHECK: =====
    # CHECK: th_id 2
    # CHECK: [2.0, 6.0, 10.0, 14.0]
    # CHECK: =====
    # CHECK: th_id 3
    # CHECK: [3.0, 7.0, 11.0, 15.0]
    # CHECK: =====
    # CHECK: th_id 4
    # CHECK: [0.0, 4.0, 8.0, 12.0]
    # CHECK: =====
    # CHECK: th_id 5
    # CHECK: [1.0, 5.0, 9.0, 13.0]
    # CHECK: =====
    # CHECK: th_id 6
    # CHECK: [2.0, 6.0, 10.0, 14.0]
    # CHECK: =====
    # CHECK: th_id 7
    # CHECK: [3.0, 7.0, 11.0, 15.0]
    # CHECK: =====
    # CHECK: th_id 8
    # CHECK: [0.0, 4.0, 8.0, 12.0]
    # CHECK: =====
    # CHECK: th_id 9
    # CHECK: [1.0, 5.0, 9.0, 13.0]
    # CHECK: =====
    # CHECK: th_id 10
    # CHECK: [2.0, 6.0, 10.0, 14.0]
    # CHECK: =====
    # CHECK: th_id 11
    # CHECK: [3.0, 7.0, 11.0, 15.0]
    # CHECK: =====
    # CHECK: th_id 12
    # CHECK: [0.0, 4.0, 8.0, 12.0]
    # CHECK: =====
    # CHECK: th_id 13
    # CHECK: [1.0, 5.0, 9.0, 13.0]
    # CHECK: =====
    # CHECK: th_id 14
    # CHECK: [2.0, 6.0, 10.0, 14.0]
    # CHECK: =====
    # CHECK: th_id 15
    # CHECK: [3.0, 7.0, 11.0, 15.0]
    for th_id in range(UInt(16)):
        print("th_id", th_id)
        var tensor = tensor_4x4.vectorize[4, 1]().distribute[
            Layout.row_major(4, 4), axis=1
        ](th_id)
        tensor.print()
        print("=====")


fn test_split():
    var tensor_4x4 = LayoutTensor[
        DType.float32, Layout(IntTuple(4, 4), IntTuple(4, 1))
    ].stack_allocation[alignment=16]()
    tensor_4x4.linspace()

    var tiles_axis0 = tensor_4x4.split[2]()
    # CHECK: 0.0 1.0 2.0 3.0
    # CHECK: 4.0 5.0 6.0 7.0
    tiles_axis0[0].print()
    # CHECK: 8.0 9.0 10.0 11.0
    # CHECK: 12.0 13.0 14.0 15.0
    tiles_axis0[1].print()

    var tiles_axis1 = tensor_4x4.split[2, axis=1]()
    # CHECK: 0.0 1.0
    # CHECK: 4.0 5.0
    # CHECK: 8.0 9.0
    # CHECK: 12.0 13.0
    tiles_axis1[0].print()
    # CHECK: 2.0 3.0
    # CHECK: 6.0 7.0
    # CHECK: 10.0 11.0
    # CHECK: 14.0 15.0
    tiles_axis1[1].print()

    var tiles_vec2_axis0 = tensor_4x4.vectorize[1, 2]().split[2]()
    # CHECK: [0.0, 1.0] [2.0, 3.0]
    # CHECK: [4.0, 5.0] [6.0, 7.0]
    tiles_vec2_axis0[0].print()
    # CHECK: [8.0, 9.0] [10.0, 11.0]
    # CHECK: [12.0, 13.0] [14.0, 15.0]
    tiles_vec2_axis0[1].print()

    _ = tensor_4x4^


# DISABLED-CHECK-LABEL: test_copy_subtiles_scalars
fn test_copy_subtiles_scalars():
    print("== test_copy_subtiles_scalars")
    var tensor_13x7 = LayoutTensor[
        DType.float32, Layout.row_major(13, 7)
    ].stack_allocation[alignment=16]()
    tensor_13x7.linspace()
    tensor_13x7.print()

    alias tile_m_size = 4
    alias tile_n_size = 2

    # DISABLED-CHECK: ----tile-data[ 0 , 0 ]----
    # DISABLED-CHECK: 0.0 1.0
    # DISABLED-CHECK: 7.0 8.0
    # DISABLED-CHECK: 14.0 15.0
    # DISABLED-CHECK: 21.0 22.0
    # DISABLED-CHECK: ----tile-data[ 0 , 1 ]----
    # DISABLED-CHECK: 2.0 3.0
    # DISABLED-CHECK: 9.0 10.0
    # DISABLED-CHECK: 16.0 17.0
    # DISABLED-CHECK: 23.0 24.0
    # DISABLED-CHECK: ----tile-data[ 0 , 2 ]----
    # DISABLED-CHECK: 4.0 5.0
    # DISABLED-CHECK: 11.0 12.0
    # DISABLED-CHECK: 18.0 19.0
    # DISABLED-CHECK: 25.0 26.0
    # DISABLED-CHECK: ----tile-data[ 0 , 3 ]----
    # DISABLED-CHECK: 6.0 0.0
    # DISABLED-CHECK: 13.0 0.0
    # DISABLED-CHECK: 20.0 0.0
    # DISABLED-CHECK: 27.0 0.0
    # DISABLED-CHECK: ----tile-data[ 1 , 0 ]----
    # DISABLED-CHECK: 28.0 29.0
    # DISABLED-CHECK: 35.0 36.0
    # DISABLED-CHECK: 42.0 43.0
    # DISABLED-CHECK: 49.0 50.0
    # DISABLED-CHECK: ----tile-data[ 1 , 1 ]----
    # DISABLED-CHECK: 30.0 31.0
    # DISABLED-CHECK: 37.0 38.0
    # DISABLED-CHECK: 44.0 45.0
    # DISABLED-CHECK: 51.0 52.0
    # DISABLED-CHECK: ----tile-data[ 1 , 2 ]----
    # DISABLED-CHECK: 32.0 33.0
    # DISABLED-CHECK: 39.0 40.0
    # DISABLED-CHECK: 46.0 47.0
    # DISABLED-CHECK: 53.0 54.0
    # DISABLED-CHECK: ----tile-data[ 1 , 3 ]----
    # DISABLED-CHECK: 34.0 0.0
    # DISABLED-CHECK: 41.0 0.0
    # DISABLED-CHECK: 48.0 0.0
    # DISABLED-CHECK: 55.0 0.0
    # DISABLED-CHECK: ----tile-data[ 2 , 0 ]----
    # DISABLED-CHECK: 56.0 57.0
    # DISABLED-CHECK: 63.0 64.0
    # DISABLED-CHECK: 70.0 71.0
    # DISABLED-CHECK: 77.0 78.0
    # DISABLED-CHECK: ----tile-data[ 2 , 1 ]----
    # DISABLED-CHECK: 58.0 59.0
    # DISABLED-CHECK: 65.0 66.0
    # DISABLED-CHECK: 72.0 73.0
    # DISABLED-CHECK: 79.0 80.0
    # DISABLED-CHECK: ----tile-data[ 2 , 2 ]----
    # DISABLED-CHECK: 60.0 61.0
    # DISABLED-CHECK: 67.0 68.0
    # DISABLED-CHECK: 74.0 75.0
    # DISABLED-CHECK: 81.0 82.0
    # DISABLED-CHECK: ----tile-data[ 2 , 3 ]----
    # DISABLED-CHECK: 62.0 0.0
    # DISABLED-CHECK: 69.0 0.0
    # DISABLED-CHECK: 76.0 0.0
    # DISABLED-CHECK: 83.0 0.0
    # DISABLED-CHECK: ----tile-data[ 3 , 0 ]----
    # DISABLED-CHECK: 84.0 85.0
    # DISABLED-CHECK: 0.0 0.0
    # DISABLED-CHECK: 0.0 0.0
    # DISABLED-CHECK: 0.0 0.0
    # DISABLED-CHECK: ----tile-data[ 3 , 1 ]----
    # DISABLED-CHECK: 86.0 87.0
    # DISABLED-CHECK: 0.0 0.0
    # DISABLED-CHECK: 0.0 0.0
    # DISABLED-CHECK: 0.0 0.0
    # DISABLED-CHECK: ----tile-data[ 3 , 2 ]----
    # DISABLED-CHECK: 88.0 89.0
    # DISABLED-CHECK: 0.0 0.0
    # DISABLED-CHECK: 0.0 0.0
    # DISABLED-CHECK: 0.0 0.0
    # DISABLED-CHECK: ----tile-data[ 3 , 3 ]----
    # DISABLED-CHECK: 90.0 0.0
    # DISABLED-CHECK: 0.0 0.0
    # DISABLED-CHECK: 0.0 0.0
    # DISABLED-CHECK: 0.0 0.0
    for tile_m in range(ceildiv(13, tile_m_size)):
        for tile_n in range(ceildiv(7, tile_n_size)):
            var tile_4x2 = tensor_13x7.tile[tile_m_size, tile_n_size](
                tile_m, tile_n
            )
            print("----tile-data[", tile_m, ",", tile_n, "]----")
            var tile_4x2_cache = LayoutTensor[
                DType.float32, Layout.row_major(tile_m_size, tile_n_size)
            ].stack_allocation[alignment=16]()
            tile_4x2_cache.fill(0)
            tile_4x2_cache.copy_from(tile_4x2)
            tile_4x2_cache.print()


# DISABLED-CHECK-LABEL: test_copy_distributed_subtiles_scalars
fn test_copy_distributed_subtiles_scalars():
    print("== test_copy_distributed_subtiles_scalars")
    var tensor_13x7 = LayoutTensor[
        DType.float32, Layout.row_major(13, 7)
    ].stack_allocation[alignment=16]()
    tensor_13x7.linspace()

    alias tile_m_size = 4
    alias tile_n_size = 4

    # DISABLED-CHECK: ----tile-data[ 0 , 0 ]----
    # DISABLED-CHECK: 0.0 1.0 2.0 3.0
    # DISABLED-CHECK: 7.0 8.0 9.0 10.0
    # DISABLED-CHECK: 14.0 15.0 16.0 17.0
    # DISABLED-CHECK: 21.0 22.0 23.0 24.0
    # DISABLED-CHECK: ----fragments-data[ 0 ]----
    # DISABLED-CHECK: 0.0 2.0
    # DISABLED-CHECK: 14.0 16.0
    # DISABLED-CHECK: ----fragments-data[ 1 ]----
    # DISABLED-CHECK: 1.0 3.0
    # DISABLED-CHECK: 15.0 17.0
    # DISABLED-CHECK: ----fragments-data[ 2 ]----
    # DISABLED-CHECK: 7.0 9.0
    # DISABLED-CHECK: 21.0 23.0
    # DISABLED-CHECK: ----fragments-data[ 3 ]----
    # DISABLED-CHECK: 8.0 10.0
    # DISABLED-CHECK: 22.0 24.0
    # DISABLED-CHECK: ----tile-data[ 0 , 1 ]----
    # DISABLED-CHECK: 4.0 5.0 6.0 0.0
    # DISABLED-CHECK: 11.0 12.0 13.0 0.0
    # DISABLED-CHECK: 18.0 19.0 20.0 0.0
    # DISABLED-CHECK: 25.0 26.0 27.0 0.0
    # DISABLED-CHECK: ----fragments-data[ 0 ]----
    # DISABLED-CHECK: 4.0 6.0
    # DISABLED-CHECK: 18.0 20.0
    # DISABLED-CHECK: ----fragments-data[ 1 ]----
    # DISABLED-CHECK: 5.0 0.0
    # DISABLED-CHECK: 19.0 0.0
    # DISABLED-CHECK: ----fragments-data[ 2 ]----
    # DISABLED-CHECK: 11.0 13.0
    # DISABLED-CHECK: 25.0 27.0
    # DISABLED-CHECK: ----fragments-data[ 3 ]----
    # DISABLED-CHECK: 12.0 0.0
    # DISABLED-CHECK: 26.0 0.0
    # DISABLED-CHECK: ----tile-data[ 1 , 0 ]----
    # DISABLED-CHECK: 28.0 29.0 30.0 31.0
    # DISABLED-CHECK: 35.0 36.0 37.0 38.0
    # DISABLED-CHECK: 42.0 43.0 44.0 45.0
    # DISABLED-CHECK: 49.0 50.0 51.0 52.0
    # DISABLED-CHECK: ----fragments-data[ 0 ]----
    # DISABLED-CHECK: 28.0 30.0
    # DISABLED-CHECK: 42.0 44.0
    # DISABLED-CHECK: ----fragments-data[ 1 ]----
    # DISABLED-CHECK: 29.0 31.0
    # DISABLED-CHECK: 43.0 45.0
    # DISABLED-CHECK: ----fragments-data[ 2 ]----
    # DISABLED-CHECK: 35.0 37.0
    # DISABLED-CHECK: 49.0 51.0
    # DISABLED-CHECK: ----fragments-data[ 3 ]----
    # DISABLED-CHECK: 36.0 38.0
    # DISABLED-CHECK: 50.0 52.0
    # DISABLED-CHECK: ----tile-data[ 1 , 1 ]----
    # DISABLED-CHECK: 32.0 33.0 34.0 0.0
    # DISABLED-CHECK: 39.0 40.0 41.0 0.0
    # DISABLED-CHECK: 46.0 47.0 48.0 0.0
    # DISABLED-CHECK: 53.0 54.0 55.0 0.0
    # DISABLED-CHECK: ----fragments-data[ 0 ]----
    # DISABLED-CHECK: 32.0 34.0
    # DISABLED-CHECK: 46.0 48.0
    # DISABLED-CHECK: ----fragments-data[ 1 ]----
    # DISABLED-CHECK: 33.0 0.0
    # DISABLED-CHECK: 47.0 0.0
    # DISABLED-CHECK: ----fragments-data[ 2 ]----
    # DISABLED-CHECK: 39.0 41.0
    # DISABLED-CHECK: 53.0 55.0
    # DISABLED-CHECK: ----fragments-data[ 3 ]----
    # DISABLED-CHECK: 40.0 0.0
    # DISABLED-CHECK: 54.0 0.0
    # DISABLED-CHECK: ----tile-data[ 2 , 0 ]----
    # DISABLED-CHECK: 56.0 57.0 58.0 59.0
    # DISABLED-CHECK: 63.0 64.0 65.0 66.0
    # DISABLED-CHECK: 70.0 71.0 72.0 73.0
    # DISABLED-CHECK: 77.0 78.0 79.0 80.0
    # DISABLED-CHECK: ----fragments-data[ 0 ]----
    # DISABLED-CHECK: 56.0 58.0
    # DISABLED-CHECK: 70.0 72.0
    # DISABLED-CHECK: ----fragments-data[ 1 ]----
    # DISABLED-CHECK: 57.0 59.0
    # DISABLED-CHECK: 71.0 73.0
    # DISABLED-CHECK: ----fragments-data[ 2 ]----
    # DISABLED-CHECK: 63.0 65.0
    # DISABLED-CHECK: 77.0 79.0
    # DISABLED-CHECK: ----fragments-data[ 3 ]----
    # DISABLED-CHECK: 64.0 66.0
    # DISABLED-CHECK: 78.0 80.0
    # DISABLED-CHECK: ----tile-data[ 2 , 1 ]----
    # DISABLED-CHECK: 60.0 61.0 62.0 0.0
    # DISABLED-CHECK: 67.0 68.0 69.0 0.0
    # DISABLED-CHECK: 74.0 75.0 76.0 0.0
    # DISABLED-CHECK: 81.0 82.0 83.0 0.0
    # DISABLED-CHECK: ----fragments-data[ 0 ]----
    # DISABLED-CHECK: 60.0 62.0
    # DISABLED-CHECK: 74.0 76.0
    # DISABLED-CHECK: ----fragments-data[ 1 ]----
    # DISABLED-CHECK: 61.0 0.0
    # DISABLED-CHECK: 75.0 0.0
    # DISABLED-CHECK: ----fragments-data[ 2 ]----
    # DISABLED-CHECK: 67.0 69.0
    # DISABLED-CHECK: 81.0 83.0
    # DISABLED-CHECK: ----fragments-data[ 3 ]----
    # DISABLED-CHECK: 68.0 0.0
    # DISABLED-CHECK: 82.0 0.0
    # DISABLED-CHECK: ----tile-data[ 3 , 0 ]----
    # DISABLED-CHECK: 84.0 85.0 86.0 87.0
    # DISABLED-CHECK: 0.0 0.0 0.0 0.0
    # DISABLED-CHECK: 0.0 0.0 0.0 0.0
    # DISABLED-CHECK: 0.0 0.0 0.0 0.0
    # DISABLED-CHECK: ----fragments-data[ 0 ]----
    # DISABLED-CHECK: 84.0 86.0
    # DISABLED-CHECK: 0.0 0.0
    # DISABLED-CHECK: ----fragments-data[ 1 ]----
    # DISABLED-CHECK: 85.0 87.0
    # DISABLED-CHECK: 0.0 0.0
    # DISABLED-CHECK: ----fragments-data[ 2 ]----
    # DISABLED-CHECK: 0.0 0.0
    # DISABLED-CHECK: 0.0 0.0
    # DISABLED-CHECK: ----fragments-data[ 3 ]----
    # DISABLED-CHECK: 0.0 0.0
    # DISABLED-CHECK: 0.0 0.0
    # DISABLED-CHECK: ----tile-data[ 3 , 1 ]----
    # DISABLED-CHECK: 88.0 89.0 90.0 0.0
    # DISABLED-CHECK: 0.0 0.0 0.0 0.0
    # DISABLED-CHECK: 0.0 0.0 0.0 0.0
    # DISABLED-CHECK: 0.0 0.0 0.0 0.0
    # DISABLED-CHECK: ----fragments-data[ 0 ]----
    # DISABLED-CHECK: 88.0 90.0
    # DISABLED-CHECK: 0.0 0.0
    # DISABLED-CHECK: ----fragments-data[ 1 ]----
    # DISABLED-CHECK: 89.0 0.0
    # DISABLED-CHECK: 0.0 0.0
    # DISABLED-CHECK: ----fragments-data[ 2 ]----
    # DISABLED-CHECK: 0.0 0.0
    # DISABLED-CHECK: 0.0 0.0
    # DISABLED-CHECK: ----fragments-data[ 3 ]----
    # DISABLED-CHECK: 0.0 0.0
    # DISABLED-CHECK: 0.0 0.0

    for tile_m in range(ceildiv(13, tile_m_size)):
        for tile_n in range(ceildiv(7, tile_n_size)):
            print("----tile-data[", tile_m, ",", tile_n, "]----")
            var tile_4x4 = tensor_13x7.tile[tile_m_size, tile_n_size](
                tile_m, tile_n
            )
            var tile_4x4_cache = LayoutTensor[
                DType.float32, Layout.row_major(tile_m_size, tile_n_size)
            ].stack_allocation[alignment=16]()
            tile_4x4_cache.fill(0)
            tile_4x4_cache.copy_from(tile_4x4)
            tile_4x4_cache.print()

            for th_id in range(UInt(4)):
                print("----fragments-data[", th_id, "]----")
                var tile_2x2 = tile_4x4.distribute[Layout.row_major(2, 2)](
                    th_id
                )
                var tile_2x2_cache = LayoutTensor[
                    DType.float32, Layout.row_major(2, 2)
                ].stack_allocation[alignment=16]()
                tile_2x2_cache.fill(0)
                tile_2x2_cache.copy_from(tile_2x2)
                tile_2x2_cache.print()


fn test_copy_subtiles_scalars_back():
    print("== test_copy_subtiles_scalars_back")

    var tensor_13x7 = LayoutTensor[
        DType.float32, Layout.row_major(13, 7)
    ].stack_allocation[alignment=16]()

    alias tile_m_size = 4
    alias tile_n_size = 4

    tensor_13x7.fill(-1)

    # TODO(#38547) re-enable the checks when the non-deterministic behavior is addressed.
    # CHECK-FIXME: ----tile-data[ 0 , 0 ]----
    # CHECK-FIXME: 0.0 1.0 2.0 3.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: 4.0 5.0 6.0 7.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: 8.0 9.0 10.0 11.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: 12.0 13.0 14.0 15.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: ----tile-data[ 0 , 1 ]----
    # CHECK-FIXME: 0.0 1.0 2.0 3.0 0.0 1.0 2.0
    # CHECK-FIXME: 4.0 5.0 6.0 7.0 4.0 5.0 6.0
    # CHECK-FIXME: 8.0 9.0 10.0 11.0 8.0 9.0 10.0
    # CHECK-FIXME: 12.0 13.0 14.0 15.0 12.0 13.0 14.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: ----tile-data[ 1 , 0 ]----
    # CHECK-FIXME: 0.0 1.0 2.0 3.0 0.0 1.0 2.0
    # CHECK-FIXME: 4.0 5.0 6.0 7.0 4.0 5.0 6.0
    # CHECK-FIXME: 8.0 9.0 10.0 11.0 8.0 9.0 10.0
    # CHECK-FIXME: 12.0 13.0 14.0 15.0 12.0 13.0 14.0
    # CHECK-FIXME: 0.0 1.0 2.0 3.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: 4.0 5.0 6.0 7.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: 8.0 9.0 10.0 11.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: 12.0 13.0 14.0 15.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: ----tile-data[ 1 , 1 ]----
    # CHECK-FIXME: 0.0 1.0 2.0 3.0 0.0 1.0 2.0
    # CHECK-FIXME: 4.0 5.0 6.0 7.0 4.0 5.0 6.0
    # CHECK-FIXME: 8.0 9.0 10.0 11.0 8.0 9.0 10.0
    # CHECK-FIXME: 12.0 13.0 14.0 15.0 12.0 13.0 14.0
    # CHECK-FIXME: 0.0 1.0 2.0 3.0 0.0 1.0 2.0
    # CHECK-FIXME: 4.0 5.0 6.0 7.0 4.0 5.0 6.0
    # CHECK-FIXME: 8.0 9.0 10.0 11.0 8.0 9.0 10.0
    # CHECK-FIXME: 12.0 13.0 14.0 15.0 12.0 13.0 14.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: ----tile-data[ 2 , 0 ]----
    # CHECK-FIXME: 0.0 1.0 2.0 3.0 0.0 1.0 2.0
    # CHECK-FIXME: 4.0 5.0 6.0 7.0 4.0 5.0 6.0
    # CHECK-FIXME: 8.0 9.0 10.0 11.0 8.0 9.0 10.0
    # CHECK-FIXME: 12.0 13.0 14.0 15.0 12.0 13.0 14.0
    # CHECK-FIXME: 0.0 1.0 2.0 3.0 0.0 1.0 2.0
    # CHECK-FIXME: 4.0 5.0 6.0 7.0 4.0 5.0 6.0
    # CHECK-FIXME: 8.0 9.0 10.0 11.0 8.0 9.0 10.0
    # CHECK-FIXME: 12.0 13.0 14.0 15.0 12.0 13.0 14.0
    # CHECK-FIXME: 0.0 1.0 2.0 3.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: 4.0 5.0 6.0 7.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: 8.0 9.0 10.0 11.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: 12.0 13.0 14.0 15.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: ----tile-data[ 2 , 1 ]----
    # CHECK-FIXME: 0.0 1.0 2.0 3.0 0.0 1.0 2.0
    # CHECK-FIXME: 4.0 5.0 6.0 7.0 4.0 5.0 6.0
    # CHECK-FIXME: 8.0 9.0 10.0 11.0 8.0 9.0 10.0
    # CHECK-FIXME: 12.0 13.0 14.0 15.0 12.0 13.0 14.0
    # CHECK-FIXME: 0.0 1.0 2.0 3.0 0.0 1.0 2.0
    # CHECK-FIXME: 4.0 5.0 6.0 7.0 4.0 5.0 6.0
    # CHECK-FIXME: 8.0 9.0 10.0 11.0 8.0 9.0 10.0
    # CHECK-FIXME: 12.0 13.0 14.0 15.0 12.0 13.0 14.0
    # CHECK-FIXME: 0.0 1.0 2.0 3.0 0.0 1.0 2.0
    # CHECK-FIXME: 4.0 5.0 6.0 7.0 4.0 5.0 6.0
    # CHECK-FIXME: 8.0 9.0 10.0 11.0 8.0 9.0 10.0
    # CHECK-FIXME: 12.0 13.0 14.0 15.0 12.0 13.0 14.0
    # CHECK-FIXME: -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: ----tile-data[ 3 , 0 ]----
    # CHECK-FIXME: 0.0 1.0 2.0 3.0 0.0 1.0 2.0
    # CHECK-FIXME: 4.0 5.0 6.0 7.0 4.0 5.0 6.0
    # CHECK-FIXME: 8.0 9.0 10.0 11.0 8.0 9.0 10.0
    # CHECK-FIXME: 12.0 13.0 14.0 15.0 12.0 13.0 14.0
    # CHECK-FIXME: 0.0 1.0 2.0 3.0 0.0 1.0 2.0
    # CHECK-FIXME: 4.0 5.0 6.0 7.0 4.0 5.0 6.0
    # CHECK-FIXME: 8.0 9.0 10.0 11.0 8.0 9.0 10.0
    # CHECK-FIXME: 12.0 13.0 14.0 15.0 12.0 13.0 14.0
    # CHECK-FIXME: 0.0 1.0 2.0 3.0 0.0 1.0 2.0
    # CHECK-FIXME: 4.0 5.0 6.0 7.0 4.0 5.0 6.0
    # CHECK-FIXME: 8.0 9.0 10.0 11.0 8.0 9.0 10.0
    # CHECK-FIXME: 12.0 13.0 14.0 15.0 12.0 13.0 14.0
    # CHECK-FIXME: 0.0 1.0 2.0 3.0 -1.0 -1.0 -1.0
    # CHECK-FIXME: ----tile-data[ 3 , 1 ]----
    # CHECK-FIXME: 0.0 1.0 2.0 3.0 0.0 1.0 2.0
    # CHECK-FIXME: 4.0 5.0 6.0 7.0 4.0 5.0 6.0
    # CHECK-FIXME: 8.0 9.0 10.0 11.0 8.0 9.0 10.0
    # CHECK-FIXME: 12.0 13.0 14.0 15.0 12.0 13.0 14.0
    # CHECK-FIXME: 0.0 1.0 2.0 3.0 0.0 1.0 2.0
    # CHECK-FIXME: 4.0 5.0 6.0 7.0 4.0 5.0 6.0
    # CHECK-FIXME: 8.0 9.0 10.0 11.0 8.0 9.0 10.0
    # CHECK-FIXME: 12.0 13.0 14.0 15.0 12.0 13.0 14.0
    # CHECK-FIXME: 0.0 1.0 2.0 3.0 0.0 1.0 2.0
    # CHECK-FIXME: 4.0 5.0 6.0 7.0 4.0 5.0 6.0
    # CHECK-FIXME: 8.0 9.0 10.0 11.0 8.0 9.0 10.0
    # CHECK-FIXME: 12.0 13.0 14.0 15.0 12.0 13.0 14.0
    # CHECK-FIXME: 0.0 1.0 2.0 3.0 0.0 1.0 2.0

    for tile_m in range(ceildiv(13, tile_m_size)):
        for tile_n in range(ceildiv(7, tile_n_size)):
            print("----tile-data[", tile_m, ",", tile_n, "]----")
            var tensor_4x4 = tensor_13x7.tile[tile_m_size, tile_n_size](
                tile_m, tile_n
            )
            var tile_4x4_cache = LayoutTensor[
                DType.float32, Layout.row_major(tile_m_size, tile_n_size)
            ].stack_allocation[alignment=16]()
            tile_4x4_cache.linspace()
            tensor_4x4.copy_from(tile_4x4_cache)
            tensor_13x7.print()


# CHECK-LABEL: test_slice_with_offsets
fn test_slice_with_offsets():
    print("== test_slice_with_offsets")

    var tensor_4x3x2_row_major = LayoutTensor[
        DType.float32, Layout.row_major(4, 3, 2)
    ].stack_allocation[alignment=16]()

    for i in range(4 * 3 * 2):
        tensor_4x3x2_row_major.ptr[i] = i

    # CHECK: slice-of[0:3,:2,0]
    # CHECK: 0.0 2.0
    # CHECK: 6.0 8.0
    # CHECK: 12.0 14.0
    print("slice-of[0:3,:2,0]")
    tensor_4x3x2_row_major.slice[0:3, 0:2, slice_indices= (0, 1)](
        offsets=(0)
    ).print()

    # CHECK: slice-of-[0:3,:2,1]
    # CHECK: 1.0 3.0
    # CHECK: 7.0 9.0
    # CHECK: 13.0 15.0
    print("slice-of-[0:3,:2,1]")
    tensor_4x3x2_row_major.slice[0:3, 0:2, slice_indices= (0, 1)](
        offsets=(1)
    ).print()

    # CHECK: slice-of-[2,:,:]
    # CHECK: 12.0 13.0
    # CHECK: 14.0 15.0
    # CHECK: 16.0 17.0
    print("slice-of-[2,:,:]")
    tensor_4x3x2_row_major.slice[:, :, slice_indices= (1, 2)](
        offsets=(2)
    ).print()

    print("slice-of-[:,1,:]")
    # CHECK: slice-of-[:,1,:]
    # CHECK: 2.0 3.0
    # CHECK: 8.0 9.0
    # CHECK: 14.0 15.0
    # CHECK: 20.0 21.0
    tensor_4x3x2_row_major.slice[:, :, slice_indices= (0, 2)](
        offsets=(1)
    ).print()

    # CHECK: slice-of-[:,0,0]
    # CHECK: 0.0
    # CHECK: 6.0
    # CHECK: 12.0
    # CHECK: 18.0
    print("slice-of-[:,0,0]")
    tensor_4x3x2_row_major.slice_1d[:, slice_indices= (0)](
        offsets=(0, 0)
    ).print()

    # CHECK: slice-of-[2,:,1]
    # CHECK: 13.0
    # CHECK: 15.0
    # CHECK: 17.0
    print("slice-of-[2,:,1]")
    tensor_4x3x2_row_major.slice_1d[:, slice_indices= (1)](
        offsets=(2, 1)
    ).print()


# CHECK-LABEL: test_layout_tensor_iterator
fn test_layout_tensor_iterator():
    print("== test_layout_tensor_iterator")

    alias size = 64
    alias type = DType.float32

    var ptr = stack_allocation[size, type]()
    for i in range(size):
        ptr[i] = i

    alias layout_2x2_8x1 = Layout(IntTuple(2, 2), IntTuple(8, 1))

    # Non circular iterator.
    # CHECK: 0.0 1.0
    # CHECK: 8.0 9.0
    # CHECK: 4.0 5.0
    # CHECK: 12.0 13.0
    var iter2x2 = LayoutTensorIter[type, layout_2x2_8x1](ptr, size)
    for _ in range(2):
        iter2x2.get().print()
        iter2x2 += 1

    # Non circular iterator with stride
    # CHECK: 0.0 1.0
    # CHECK: 8.0 9.0
    # CHECK: 16.0 17.0
    # CHECK: 24.0 25.0
    iter2x2 = LayoutTensorIter[type, layout_2x2_8x1](ptr, size, stride=16)
    for _ in range(2):
        iter2x2.get().print()
        iter2x2 += 1

    # Non circular iterator with offset and stride
    # CHECK: 4.0 5.0
    # CHECK: 12.0 13.0
    # CHECK: 20.0 21.0
    # CHECK: 28.0 29.0
    iter2x2 = LayoutTensorIter[type, layout_2x2_8x1](
        ptr, size, stride=16, offset=4
    )
    for _ in range(2):
        iter2x2.get().print()
        iter2x2 += 1

    # Circular iterator with offset and stride
    # CHECK: 32.0 33.0
    # CHECK: 40.0 41.0
    # CHECK: 48.0 49.0
    # CHECK: 56.0 57.0
    # CHECK: 0.0 1.0
    # CHECK: 8.0 9.0
    # CHECK: 16.0 17.0
    # CHECK: 24.0 25.0
    var iter2x2_circular = LayoutTensorIter[
        type, layout_2x2_8x1, circular=True
    ](ptr, size, stride=16, offset=32)
    for _ in range(4):
        iter2x2_circular.get().print()
        iter2x2_circular += 1

    # Tiled iterator.
    var tensor = LayoutTensor[type, Layout.row_major(8, 8)](ptr)
    # CHECK: 32.0 33.0
    # CHECK: 40.0 41.0
    # CHECK: 34.0 35.0
    # CHECK: 42.0 43.0
    # CHECK: 36.0 37.0
    # CHECK: 44.0 45.0
    # CHECK: 38.0 39.0
    # CHECK: 46.0 47.0
    var iter = tensor.tiled_iterator[2, 2, axis=1](2, 0)
    for _ in range(4):
        iter.get().print()
        iter += 1
    # CHECK: 38.0 39.0
    # CHECK: 46.0 47.0
    # CHECK: 54.0 55.0
    # CHECK: 62.0 63.0
    iter = tensor.tiled_iterator[2, 2, axis=0](2, 3)
    for _ in range(2):
        iter.get().print()
        iter += 1


# CHECK-LABEL: test_layout_tensor_copy_from_masked_src
fn test_layout_tensor_copy_from_masked_src():
    print("== test_layout_tensor_copy_from_masked_src")

    var tensor_16x16 = LayoutTensor[
        DType.int32, Layout.row_major(16, 16)
    ].stack_allocation[alignment=16]()

    tensor_16x16.fill(0)

    var tensor_15x16 = LayoutTensor[
        DType.int32, Layout.row_major(15, 16)
    ].stack_allocation[alignment=16]()

    tensor_15x16.linspace()
    alias dist_layout = Layout.row_major(8, 4)

    for i in range(dist_layout.size()):
        var dst_frag = tensor_16x16.vectorize[1, 4]().distribute[dist_layout](i)
        var src_frag = tensor_15x16.vectorize[1, 4]().distribute[dist_layout](i)
        dst_frag.copy_from_masked_src(
            src_frag, dst_frag.distance(tensor_16x16.ptr), 15, 16
        )

    # CHECK: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
    # CHECK: 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
    # CHECK: 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
    # CHECK: 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63
    # CHECK: 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79
    # CHECK: 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95
    # CHECK: 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111
    # CHECK: 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127
    # CHECK: 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143
    # CHECK: 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159
    # CHECK: 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175
    # CHECK: 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191
    # CHECK: 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207
    # CHECK: 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223
    # CHECK: 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239
    # CHECK: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    tensor_16x16.print()


# CHECK-LABEL: test_layout_tensor_copy_from_masked_dst
fn test_layout_tensor_copy_from_masked_dst():
    print("== test_layout_tensor_copy_from_masked_dst")

    var tensor_16x16 = LayoutTensor[
        DType.int32, Layout.row_major(16, 16)
    ].stack_allocation[alignment=16]()

    tensor_16x16.linspace()

    var tensor_15x16 = LayoutTensor[
        DType.int32, Layout.row_major(15, 16)
    ].stack_allocation[alignment=16]()

    tensor_15x16.fill(0)

    alias dist_layout = Layout.row_major(8, 4)

    for i in range(dist_layout.size()):
        var dst_frag = tensor_15x16.vectorize[1, 4]().distribute[dist_layout](i)
        var src_frag = tensor_16x16.vectorize[1, 4]().distribute[dist_layout](i)
        dst_frag.copy_from_masked_dst(
            src_frag, dst_frag.distance(tensor_15x16.ptr), 15, 16
        )

    # CHECK: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
    # CHECK: 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
    # CHECK: 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
    # CHECK: 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63
    # CHECK: 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79
    # CHECK: 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95
    # CHECK: 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111
    # CHECK: 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127
    # CHECK: 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143
    # CHECK: 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159
    # CHECK: 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175
    # CHECK: 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191
    # CHECK: 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207
    # CHECK: 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223
    # CHECK: 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239
    tensor_15x16.print()


fn main():
    test_basic_tensor_ops()
    test_tesnsor_fragments()
    test_tensor_tile_and_distribute()
    test_tensor_tile_and_distribute_custom_layout()
    test_copy_to_tile_major_layout()
    test_distribute_tiled_layout()
    test_distribute_with_tile_size()
    test_vectorize_reads()
    test_vectorize_writes()
    test_slice()
    test_copy_vectorized()
    test_distribute_vectorized()
    test_distribute_axis_projection()
    test_split()
    # DISABLE subtiles tests for nw.
    # test_copy_subtiles_scalars()
    # test_copy_distributed_subtiles_scalars()
    # TODO(#38547) re-enable the following test once the non-deterministic behavior is addressed.
    # test_copy_subtiles_scalars_back()
    test_slice_with_offsets()
    test_layout_tensor_iterator()
    test_layout_tensor_copy_from_masked_src()
    test_layout_tensor_copy_from_masked_dst()
