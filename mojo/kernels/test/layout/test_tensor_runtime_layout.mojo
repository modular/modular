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


def main():
    test_fill_and_print()
    test_set_and_get_items()
    test_tile()
