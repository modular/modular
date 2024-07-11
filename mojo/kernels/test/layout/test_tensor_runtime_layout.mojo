# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from layout import LayoutTensor, Layout, RuntimeLayout, RuntimeTuple
from layout.int_tuple import UNKNOWN_VALUE


#  CHECK-LABEL: test_fill_and_print
def test_fill_and_print():
    print("== test_fill_and_print")

    alias layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    var dynamic_layout = RuntimeLayout[layout](
        RuntimeTuple[layout.shape](4, 8), RuntimeTuple[layout.stride](8, 1)
    )

    var storage = DTypePointer[DType.float32].alloc(dynamic_layout.size())

    var tensor = LayoutTensor[DType.float32, layout](storage, dynamic_layout)

    tensor.linspace()

    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
    # CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
    # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    tensor.print()

    storage.free()


#  CHECK-LABEL: test_set_and_get_items
def test_set_and_get_items():
    print("== test_set_and_get_items")

    alias layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    var dynamic_layout = RuntimeLayout[layout](
        RuntimeTuple[layout.shape](4, 4), RuntimeTuple[layout.stride](4, 1)
    )

    var storage = DTypePointer[DType.float32].alloc(dynamic_layout.size())

    var tensor = LayoutTensor[DType.float32, layout](storage, dynamic_layout)

    for i in range(4):
        for j in range(4):
            tensor[i, j] = i * 4 + j + 2

    # CHECK: 2.0 3.0 4.0 5.0
    # CHECK: 6.0 7.0 8.0 9.0
    # CHECK: 10.0 11.0 12.0 13.0
    # CHECK: 14.0 15.0 16.0 17.0
    tensor.print()

    storage.free()


#  CHECK-LABEL: test_tile
def test_tile():
    print("== test_tile")

    alias layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    var dynamic_layout = RuntimeLayout[layout](
        RuntimeTuple[layout.shape](4, 4), RuntimeTuple[layout.stride](4, 1)
    )

    var storage = DTypePointer[DType.float32].alloc(dynamic_layout.size())

    var tensor = LayoutTensor[DType.float32, layout](storage, dynamic_layout)
    tensor.linspace()

    # CHECK: ((2, 2):(-1, 1))
    print(tensor.tile[2, 2](0, 0).layout)

    # CHECK: ((2, 2):(4, 1))
    print(tensor.tile[2, 2](0, 0).runtime_layout)

    # CHECK: ----tile-data[ 0 , 0 ]----
    # CHECK: 0.0 1.0
    # CHECK: 4.0 5.0
    # CHECK: ----tile-data[ 0 , 1 ]----
    # CHECK: 2.0 3.0
    # CHECK: 6.0 7.0
    # CHECK: ----tile-data[ 1 , 0 ]----
    # CHECK: 8.0 9.0
    # CHECK: 12.0 13.0
    # CHECK: ----tile-data[ 1 , 1 ]----
    # CHECK: 10.0 11.0
    # CHECK: 14.0 15.0
    for tile_i in range(2):
        for tile_j in range(2):
            print("----tile-data[", tile_i, ",", tile_j, "]----")
            var tile_2x2 = tensor.tile[2, 2](tile_i, tile_j)
            tile_2x2.print()

    storage.free()


fn test_tile_and_distribute():
    print("== test_tile_and_distribute")

    alias layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    var dynamic_layout = RuntimeLayout[layout](
        RuntimeTuple[layout.shape](8, 8), RuntimeTuple[layout.stride](8, 1)
    )

    var storage = DTypePointer[DType.float32].alloc(dynamic_layout.size())

    var tensor = LayoutTensor[DType.float32, layout](storage, dynamic_layout)
    tensor.linspace()

    # ---tile-data[ 0 , 0 ]----
    # 0.0 1.0 2.0 3.0
    # 8.0 9.0 10.0 11.0
    # 16.0 17.0 18.0 19.0
    # 24.0 25.0 26.0 27.0
    # ----fragments-data[ 0 ]----
    # 0.0 2.0
    # 16.0 18.0
    # ----fragments-data[ 1 ]----
    # 1.0 3.0
    # 17.0 19.0
    # ----fragments-data[ 2 ]----
    # 8.0 10.0
    # 24.0 26.0
    # ----fragments-data[ 3 ]----
    # 9.0 11.0
    # 25.0 27.0
    # ----tile-data[ 0 , 1 ]----
    # 4.0 5.0 6.0 7.0
    # 12.0 13.0 14.0 15.0
    # 20.0 21.0 22.0 23.0
    # 28.0 29.0 30.0 31.0
    # ----fragments-data[ 0 ]----
    # 4.0 6.0
    # 20.0 22.0
    # ----fragments-data[ 1 ]----
    # 5.0 7.0
    # 21.0 23.0
    # ----fragments-data[ 2 ]----
    # 12.0 14.0
    # 28.0 30.0
    # ----fragments-data[ 3 ]----
    # 13.0 15.0
    # 29.0 31.0
    # ----tile-data[ 1 , 0 ]----
    # 32.0 33.0 34.0 35.0
    # 40.0 41.0 42.0 43.0
    # 48.0 49.0 50.0 51.0
    # 56.0 57.0 58.0 59.0
    # ----fragments-data[ 0 ]----
    # 32.0 34.0
    # 48.0 50.0
    # ----fragments-data[ 1 ]----
    # 33.0 35.0
    # 49.0 51.0
    # ----fragments-data[ 2 ]----
    # 40.0 42.0
    # 56.0 58.0
    # ----fragments-data[ 3 ]----
    # 41.0 43.0
    # 57.0 59.0
    # ----tile-data[ 1 , 1 ]----
    # 36.0 37.0 38.0 39.0
    # 44.0 45.0 46.0 47.0
    # 52.0 53.0 54.0 55.0
    # 60.0 61.0 62.0 63.0
    # ----fragments-data[ 0 ]----
    # 36.0 38.0
    # 52.0 54.0
    # ----fragments-data[ 1 ]----
    # 37.0 39.0
    # 53.0 55.0
    # ----fragments-data[ 2 ]----
    # 44.0 46.0
    # 60.0 62.0
    # ----fragments-data[ 3 ]----
    # 45.0 47.0
    # 61.0 63.0
    for tile_i in range(2):
        for tile_j in range(2):
            print("----tile-data[", tile_i, ",", tile_j, "]----")
            var tile_4x4 = tensor.tile[4, 4](tile_i, tile_j)
            tile_4x4.print()
            for th_i in range(4):
                var tile_2x2 = tile_4x4.distribute[Layout.row_major(2, 2)](th_i)
                print("----fragments-data[", th_i, "]----")
                print(tile_2x2[0, 0], tile_2x2[0, 1])
                print(tile_2x2[1, 0], tile_2x2[1, 1])


def main():
    test_fill_and_print()
    test_set_and_get_items()
    test_tile()
    test_tile_and_distribute()
