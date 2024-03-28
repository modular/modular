# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

from layout._utils import ManagedLayoutTensor
from layout.int_tuple import product
from layout.layout import Layout
from layout.layout_tensor import *


fn print_raw_major_tensor[
    layout: Layout, dtype: DType
](tensor: LayoutTensor[layout, dtype]):
    for i in range(tensor.dim[0]()):
        for j in range(tensor.dim[1]()):
            print(tensor[i, j], "\t", end="")
        print("")


fn print_tile_tensor[
    layout: Layout, dtype: DType
](tensor: LayoutTensor[layout, dtype]):
    for i in range(tensor.dim[0]()):
        for j in range(tensor.dim[1]()):
            print(tensor[i, j], "\t", end="")
        print("")


# Print for shape ((m, n), (p, q)) in a 2D format
fn print_mode2_shape2_tensor[
    layout: Layout, dtype: DType
](tensor: LayoutTensor[layout, dtype]):
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
        Layout(IntTuple(8, 4)), DType.float32
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
    for th_i in range(4):
        print("----fragments-data[", th_i, "]----")
        var fragment_4x2 = tensor.distribute[Layout(IntTuple(2, 2))](th_i)
        print_tile_tensor(fragment_4x2)

    _ = managed_tensor^


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
                    th_i,
                )
                print("----fragments-data[", th_i, "]----")
                print_tile_tensor(fragment_2x2)
    _ = managed_tensor^


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
    for th_i in range(4):
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
    for th_i in range(4):
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
        Layout(IntTuple(4, 8), IntTuple(8, 1)), DType.float32
    ].stack_allocation()
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
    for th_i in range(8):
        var thread_tile = tensor.distribute[threads_2x4_layout](th_i)
        print("----fragments-data[", th_i, "]----")
        thread_tile.print()


# CHECK-LABEL: test_distribute_with_tile_size
fn test_distribute_with_tile_size():
    print("== test_distribute_with_tile_size")

    var tensor0 = LayoutTensor[
        Layout(IntTuple(16, 8), IntTuple(8, 1)), DType.float32
    ].stack_allocation()

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
    # CHECK: 0.0     64.0    4.0     68.0
    # CHECK: 8.0     72.0    12.0    76.0
    # CHECK: 1.0     65.0    5.0     69.0
    # CHECK: 9.0     73.0    13.0    77.0
    # CHECK: ----thread[ 1 ]----
    # CHECK: 16.0    80.0    20.0    84.0
    # CHECK: 24.0    88.0    28.0    92.0
    # CHECK: 17.0    81.0    21.0    85.0
    # CHECK: 25.0    89.0    29.0    93.0
    # CHECK: ----thread[ 2 ]----
    # CHECK: 2.0     66.0    6.0     70.0
    # CHECK: 10.0    74.0    14.0    78.0
    # CHECK: 3.0     67.0    7.0     71.0
    # CHECK: 11.0    75.0    15.0    79.0
    # CHECK: ----thread[ 3 ]----
    # CHECK: 18.0    82.0    22.0    86.0
    # CHECK: 26.0    90.0    30.0    94.0
    # CHECK: 19.0    83.0    23.0    87.0
    # CHECK: 27.0    91.0    31.0    95.0
    # CHECK: ----thread[ 4 ]----
    # CHECK: 32.0    96.0    36.0    100.0
    # CHECK: 40.0    104.0   44.0    108.0
    # CHECK: 33.0    97.0    37.0    101.0
    # CHECK: 41.0    105.0   45.0    109.0
    # CHECK: ----thread[ 5 ]----
    # CHECK: 48.0    112.0   52.0    116.0
    # CHECK: 56.0    120.0   60.0    124.0
    # CHECK: 49.0    113.0   53.0    117.0
    # CHECK: 57.0    121.0   61.0    125.0
    # CHECK: ----thread[ 6 ]----
    # CHECK: 34.0    98.0    38.0    102.0
    # CHECK: 42.0    106.0   46.0    110.0
    # CHECK: 35.0    99.0    39.0    103.0
    # CHECK: 43.0    107.0   47.0    111.0
    # CHECK: ----thread[ 7 ]----
    # CHECK: 50.0    114.0   54.0    118.0
    # CHECK: 58.0    122.0   62.0    126.0
    # CHECK: 51.0    115.0   55.0    119.0
    # CHECK: 59.0    123.0   63.0    127.0

    for tid in range(thread_layout.size()):
        print("----thread[", tid, "]----")
        var tile = tensor0.distribute[
            thread_layout, tile_size = IntTuple(2, 2)
        ](tid)
        print_mode2_shape2_tensor(tile)

    var tensor1 = LayoutTensor[Layout(8), DType.float32].stack_allocation()
    tensor1.linspace()

    # Threads 2, 3, 6, 7 should have the same fragments as thread 0, 1, 4, 5.
    # CHECK: ----thread[ 0 ]----
    # CHECK: 0.0
    # CHECK: 1.0
    # CHECK: ----thread[ 1 ]----
    # CHECK: 2.0
    # CHECK: 3.0
    # CHECK: ----thread[ 2 ]----
    # CHECK: 0.0
    # CHECK: 1.0
    # CHECK: ----thread[ 3 ]----
    # CHECK: 2.0
    # CHECK: 3.0
    # CHECK: ----thread[ 4 ]----
    # CHECK: 4.0
    # CHECK: 5.0
    # CHECK: ----thread[ 5 ]----
    # CHECK: 6.0
    # CHECK: 7.0
    # CHECK: ----thread[ 6 ]----
    # CHECK: 4.0
    # CHECK: 5.0
    # CHECK: ----thread[ 7 ]----
    # CHECK: 6.0
    # CHECK: 7.0
    for tid in range(thread_layout.size()):
        print("----thread[", tid, "]----")
        var tile = tensor1.distribute[
            thread_layout, tile_size = IntTuple(2), axis=0
        ](tid)
        tile.print()


# CHECK-LABEL: test_vectorize_reads
fn test_vectorize_reads():
    print("== test_vectorize_reads")
    var tensor = LayoutTensor[
        Layout(IntTuple(8, 8), IntTuple(1, 8)), DType.float32
    ].stack_allocation()
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
        Layout(IntTuple(4, 4), IntTuple(1, 4)), DType.float32
    ].stack_allocation()
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
        Layout(IntTuple(4, 4), IntTuple(1, 4)), DType.float32
    ].stack_allocation()
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
        Layout(IntTuple(8, 8), IntTuple(8, 1)), DType.float32
    ].stack_allocation()
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
    var tensor_8_8_zeros = stack_allocation_like(tensor_8_8).vectorize[1, 4]()
    tensor_8_8_zeros.fill(0)
    tensor_8_8_zeros.copy_from_numa(vec_8_1)
    # CHEK: [0.0, 1.0, 2.0, 3.0] [4.0, 5.0, 6.0, 7.0]
    # CHEK: [8.0, 9.0, 10.0, 11.0] [12.0, 13.0, 14.0, 15.0]
    # CHEK: [16.0, 17.0, 18.0, 19.0] [20.0, 21.0, 22.0, 23.0]
    # CHEK: [24.0, 25.0, 26.0, 27.0] [28.0, 29.0, 30.0, 31.0]
    # CHEK: [32.0, 33.0, 34.0, 35.0] [36.0, 37.0, 38.0, 39.0]
    # CHEK: [40.0, 41.0, 42.0, 43.0] [44.0, 45.0, 46.0, 47.0]
    # CHEK: [48.0, 49.0, 50.0, 51.0] [52.0, 53.0, 54.0, 55.0]
    # CHECK: [56.0, 57.0, 58.0, 59.0] [60.0, 61.0, 62.0, 63.0]
    tensor_8_8_zeros.print()

    var tensor_8_8_zeros_4_1 = stack_allocation_like(tensor_8_8).vectorize[
        4, 1
    ]()
    tensor_8_8_zeros_4_1.fill(0)
    tensor_8_8_zeros_4_1.copy_from_numa(vec_8_1)
    # CHECK: [0.0, 1.0, 2.0, 3.0] [16.0, 17.0, 18.0, 19.0] [32.0, 33.0, 34.0, 35.0] [48.0, 49.0, 50.0, 51.0] [4.0, 5.0, 6.0, 7.0] [20.0, 21.0, 22.0, 23.0] [36.0, 37.0, 38.0, 39.0] [52.0, 53.0, 54.0, 55.0]
    # CHECK: [8.0, 9.0, 10.0, 11.0] [24.0, 25.0, 26.0, 27.0] [40.0, 41.0, 42.0, 43.0] [56.0, 57.0, 58.0, 59.0] [12.0, 13.0, 14.0, 15.0] [28.0, 29.0, 30.0, 31.0] [44.0, 45.0, 46.0, 47.0] [60.0, 61.0, 62.0, 63.0]
    tensor_8_8_zeros_4_1.print()

    var tensor_8_8_zeros_1_4 = stack_allocation_like(tensor_8_8).vectorize[
        1, 4
    ]()
    tensor_8_8_zeros_1_4.fill(0)
    tensor_8_8_zeros_1_4.copy_from_numa(tensor_8_8_zeros_4_1)
    # CHECK: [0.0, 1.0, 2.0, 3.0] [4.0, 5.0, 6.0, 7.0]
    # CHECK: [8.0, 9.0, 10.0, 11.0] [12.0, 13.0, 14.0, 15.0]
    # CHECK: [16.0, 17.0, 18.0, 19.0] [20.0, 21.0, 22.0, 23.0]
    # CHECK: [24.0, 25.0, 26.0, 27.0] [28.0, 29.0, 30.0, 31.0]
    # CHECK: [32.0, 33.0, 34.0, 35.0] [36.0, 37.0, 38.0, 39.0]
    # CHECK: [40.0, 41.0, 42.0, 43.0] [44.0, 45.0, 46.0, 47.0]
    # CHECK: [48.0, 49.0, 50.0, 51.0] [52.0, 53.0, 54.0, 55.0]
    # CHECK: [56.0, 57.0, 58.0, 59.0] [60.0, 61.0, 62.0, 63.0]
    tensor_8_8_zeros_1_4.print()


# CHECK-LABEL: test_distribute_vectorized
fn test_distribute_vectorized():
    print("== test_distribute_vectorized")
    var tensor_8_8 = LayoutTensor[
        Layout(IntTuple(8, 8), IntTuple(8, 1)), DType.float32
    ].stack_allocation()
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
    for tid in range(4):
        var fragments = tensor_8_2xv4.distribute[Layout(IntTuple(2, 2))](tid)
        print("----thread[", tid, "]----")
        fragments.print()


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
