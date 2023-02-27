# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: lit %s | FileCheck %s

from Int import Int
from Functional import tile
from List import VariadicList
from IO import print
from Index import Index

# Helper workgroup function to test dynamic workgroup tiling.
@always_inline
fn print_number_dynamic(data_idx: Int, tile_size: Int):
    # Print out the range of workload that this launched instance is
    #  processing, in (begin, end).
    print(Index(data_idx, data_idx + tile_size))


# Helper workgroup function to test static workgroup tiling.
@always_inline
fn print_number_static[tile_size: Int](data_idx: Int):
    print_number_dynamic(data_idx, tile_size)


# CHECK-LABEL: test_static_tile
fn test_static_tile():
    print("test_static_tile\n")
    # CHECK: (0, 4)
    # CHECK: (4, 6)
    tile[print_number_static, VariadicList[Int](4, 3, 2, 1)](0, 6)
    # CHECK: (0, 4)
    # CHECK: (4, 8)
    tile[print_number_static, VariadicList[Int](4, 3, 2, 1)](0, 8)
    # CHECK: (1, 5)
    # CHECK: (5, 6)
    tile[print_number_static, VariadicList[Int](4, 3, 2, 1)](1, 6)


# CHECK-LABEL: test_dynamic_tile
fn test_dynamic_tile():
    print("test_dynamic_tile\n")
    # CHECK: (1, 4)
    # CHECK: (4, 5)
    tile[print_number_dynamic](1, 5, VariadicList[Int](3, 2))
    # CHECK: (0, 4)
    # CHECK: (4, 5)
    # CHECK: (5, 6)
    tile[print_number_dynamic](0, 6, VariadicList[Int](4, 1))
    # CHECK: (2, 7)
    # CHECK: (7, 12)
    # CHECK: (12, 15)
    # CHECK: (15, 16)
    tile[print_number_dynamic](2, 16, VariadicList[Int](5, 3))


fn main():
    test_static_tile()
    test_dynamic_tile()
