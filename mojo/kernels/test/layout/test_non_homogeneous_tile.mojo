# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from layout import LayoutTensor, Layout, RuntimeLayout, IntTuple
from layout.layout import UNKNOWN_VALUE
from utils import IndexList
from math import ceildiv
from layout.fillers import arange
from layout._utils import ManagedLayoutTensor


# CHECK-LABEL: test_single_unknown_tile
def test_single_unknown_tile():
    print("== test_single_unknown_tile")

    alias type = DType.float32
    alias M0 = 6
    alias N0 = 6
    alias BM0 = 4
    alias BN0 = 3
    alias layout0 = Layout.row_major(UNKNOWN_VALUE, N0)
    var runtimelayout0 = RuntimeLayout[layout0].row_major(IndexList[2](M0, N0))
    var tensorUxN = ManagedLayoutTensor[
        type, layout0, __experimental_non_homogeneous_tile=True
    ](runtimelayout0)
    arange(tensorUxN.tensor, 0, 0.5)
    # CHECK: ----check axis 0 ----
    # CHECK: ----tile[ 0 , 0 ]----
    # CHECK: ((-1, 3):(6, 1))
    # CHECK: ((4, 3):(6, 1))
    # CHECK: 0.0 0.5 1.0
    # CHECK: 3.0 3.5 4.0
    # CHECK: 6.0 6.5 7.0
    # CHECK: 9.0 9.5 10.0
    # CHECK: ----tile[ 0 , 1 ]----
    # CHECK: ((-1, 3):(6, 1))
    # CHECK: ((4, 3):(6, 1))
    # CHECK: 1.5 2.0 2.5
    # CHECK: 4.5 5.0 5.5
    # CHECK: 7.5 8.0 8.5
    # CHECK: 10.5 11.0 11.5
    # CHECK: ----tile[ 1 , 0 ]----
    # CHECK: ((-1, 3):(6, 1))
    # CHECK: ((2, 3):(6, 1))
    # CHECK: 12.0 12.5 13.0
    # CHECK: 15.0 15.5 16.0
    # CHECK: ----tile[ 1 , 1 ]----
    # CHECK: ((-1, 3):(6, 1))
    # CHECK: ((2, 3):(6, 1))
    # CHECK: 13.5 14.0 14.5
    # CHECK: 16.5 17.0 17.5
    print("----check axis 0 ----")
    for i in range(M0 // BM0 + 1):
        for j in range(N0 // BN0):
            var unknown_tile = tensorUxN.tensor.tile[BM0, BN0](i, j)
            print("----tile[", i, ",", j, "]----")
            print(unknown_tile.layout)
            print(unknown_tile.runtime_layout)
            print(unknown_tile)

    alias M1 = 6
    alias N1 = 6
    alias BM1 = 3
    alias BN1 = 4

    alias layout1 = Layout.row_major(M1, UNKNOWN_VALUE)
    var runtimelayout1 = RuntimeLayout[layout1].row_major(IndexList[2](M1, N1))
    var tensorMxU = ManagedLayoutTensor[
        type, layout1, __experimental_non_homogeneous_tile=True
    ](runtimelayout1)
    arange(tensorMxU.tensor, 0, 0.5)
    # CHECK: ----check axis 1 ----
    # CHECK: ----tile[ 0 , 0 ]----
    # CHECK: ((3, -1):(-1, 1))
    # CHECK: ((3, 4):(6, 1))
    # CHECK: 0.0 0.5 1.0 1.5
    # CHECK: 3.0 3.5 4.0 4.5
    # CHECK: 6.0 6.5 7.0 7.5
    # CHECK: ----tile[ 0 , 1 ]----
    # CHECK: ((3, -1):(-1, 1))
    # CHECK: ((3, 2):(6, 1))
    # CHECK: 2.0 2.5
    # CHECK: 5.0 5.5
    # CHECK: 8.0 8.5
    # CHECK: ----tile[ 1 , 0 ]----
    # CHECK: ((3, -1):(-1, 1))
    # CHECK: ((3, 4):(6, 1))
    # CHECK: 9.0 9.5 10.0 10.5
    # CHECK: 12.0 12.5 13.0 13.5
    # CHECK: 15.0 15.5 16.0 16.5
    # CHECK: ----tile[ 1 , 1 ]----
    # CHECK: ((3, -1):(-1, 1))
    # CHECK: ((3, 2):(6, 1))
    # CHECK: 11.0 11.5
    # CHECK: 14.0 14.5
    # CHECK: 17.0 17.5
    print("----check axis 1 ----")
    for i in range(M1 // BM1):
        for j in range(N1 // BN1 + 1):
            var unknown_tile = tensorMxU.tensor.tile[BM1, BN1](i, j)
            print("----tile[", i, ",", j, "]----")
            print(unknown_tile.layout)
            print(unknown_tile.runtime_layout)
            print(unknown_tile)


def test_non_homogeneous_copy_from():
    print("== test_non_homogeneous_copy_from")

    alias type = DType.float32
    alias M = 6
    alias N = 6

    alias layout = Layout.row_major(UNKNOWN_VALUE, N)
    alias layout2x2 = Layout.row_major(2, 2)

    var src_runtimelayout = RuntimeLayout[layout].row_major(IndexList[2](M, N))
    var tensorUxN = ManagedLayoutTensor[
        type, layout, __experimental_non_homogeneous_tile=True
    ](src_runtimelayout)
    arange(tensorUxN.tensor, 0, 0.5)

    # CHECK: ----original tensor----
    # CHECK: 0.0 0.5 1.0 1.5 2.0 2.5
    # CHECK: 3.0 3.5 4.0 4.5 5.0 5.5
    # CHECK: 6.0 6.5 7.0 7.5 8.0 8.5
    # CHECK: 9.0 9.5 10.0 10.5 11.0 11.5
    # CHECK: 12.0 12.5 13.0 13.5 14.0 14.5
    # CHECK: 15.0 15.5 16.0 16.5 17.0 17.5
    print("----original tensor----")
    print(tensorUxN.tensor)

    var tensor2x2 = LayoutTensor[
        type, layout2x2, __experimental_non_homogeneous_tile=True
    ].stack_allocation().fill(0)
    # CHECK: ----check copy ----
    # CHECK: unknown tile
    # CHECK: ((3, 4):(6, 1))
    # CHECK: 0.0 0.5 1.0 1.5
    # CHECK: 3.0 3.5 4.0 4.5
    # CHECK: 6.0 6.5 7.0 7.5
    # CHECK: tensor 2x2
    # CHECK: 0.0 0.5
    # CHECK: 3.0 3.5
    # CHECK: unknown tile
    # CHECK: ((3, 4):(6, 1))
    # CHECK: 2.0 2.5
    # CHECK: 5.0 5.5
    # CHECK: 8.0 8.5
    # CHECK: tensor 2x2
    # CHECK: 2.0 2.5
    # CHECK: 5.0 5.5
    # CHECK: unknown tile
    # CHECK: ((3, 4):(6, 1))
    # CHECK: 9.0 9.5 10.0 10.5
    # CHECK: 12.0 12.5 13.0 13.5
    # CHECK: 15.0 15.5 16.0 16.5
    # CHECK: tensor 2x2
    # CHECK: 9.0 9.5
    # CHECK: 12.0 12.5
    # CHECK: unknown tile
    # CHECK: ((3, 4):(6, 1))
    # CHECK: 11.0 11.5
    # CHECK: 14.0 14.5
    # CHECK: 17.0 17.5
    # CHECK: tensor 2x2
    # CHECK: 11.0 11.5
    # CHECK: 14.0 14.5
    print("----check copy ----")
    for i in range(M // 3):
        for j in range(ceildiv(N, 4)):
            var unknown3x4 = tensorUxN.tensor.tile[3, 4](i, j)
            tensor2x2.copy_from(unknown3x4)
            print("unknown tile")
            print(unknown3x4.runtime_layout)
            print(unknown3x4)
            print("tensor 2x2")
            print(tensor2x2)
    # CHECK: ----check vectorized copy ----
    # CHECK: unknown tile
    # CHECK: ((3, 4):(6, 1))
    # CHECK: 0.0 0.5 1.0 1.5
    # CHECK: 3.0 3.5 4.0 4.5
    # CHECK: 6.0 6.5 7.0 7.5
    # CHECK: tensor 2x2
    # CHECK: 0.0 0.5
    # CHECK: 3.0 3.5
    # CHECK: unknown tile
    # CHECK: ((3, 4):(6, 1))
    # CHECK: 2.0 2.5
    # CHECK: 5.0 5.5
    # CHECK: 8.0 8.5
    # CHECK: tensor 2x2
    # CHECK: 2.0 2.5
    # CHECK: 5.0 5.5
    # CHECK: unknown tile
    # CHECK: ((3, 4):(6, 1))
    # CHECK: 9.0 9.5 10.0 10.5
    # CHECK: 12.0 12.5 13.0 13.5
    # CHECK: 15.0 15.5 16.0 16.5
    # CHECK: tensor 2x2
    # CHECK: 9.0 9.5
    # CHECK: 12.0 12.5
    # CHECK: unknown tile
    # CHECK: ((3, 4):(6, 1))
    # CHECK: 11.0 11.5
    # CHECK: 14.0 14.5
    # CHECK: 17.0 17.5
    # CHECK: tensor 2x2
    # CHECK: 11.0 11.5
    # CHECK: 14.0 14.5
    print("----check vectorized copy ----")
    for i in range(M // 3):
        for j in range(ceildiv(N, 4)):
            var unknown3x4 = tensorUxN.tensor.tile[3, 4](i, j)
            var tensor2x2_vec = tensor2x2.vectorize[1, 2]()
            tensor2x2_vec.copy_from(unknown3x4.vectorize[1, 2]())
            print("unknown tile")
            print(unknown3x4.runtime_layout)
            print(unknown3x4)
            print("tensor 2x2")
            print(tensor2x2)

    alias layoutMxU = Layout.row_major(M, UNKNOWN_VALUE)
    var src_unknown = RuntimeLayout[layoutMxU].row_major(IndexList[2](M, N))
    var tensorMxU = ManagedLayoutTensor[
        type, layoutMxU, __experimental_non_homogeneous_tile=True
    ](src_unknown)
    arange(tensorMxU.tensor, 0, 0.5)
    # CHECK: ----check unknown layout copy ----
    # CHECK: unknown tile
    # CHECK: ((3, 4):(6, 1))
    # CHECK: 0.0 0.5 1.0 1.5
    # CHECK: 3.0 3.5 4.0 4.5
    # CHECK: 6.0 6.5 7.0 7.5
    # CHECK: tensor 2x2
    # CHECK: 0.0 0.5
    # CHECK: 3.0 3.5
    # CHECK: unknown tile
    # CHECK: ((3, 2):(6, 1))
    # CHECK: 2.0 2.5
    # CHECK: 5.0 5.5
    # CHECK: 8.0 8.5
    # CHECK: tensor 2x2
    # CHECK: 2.0 2.5
    # CHECK: 5.0 5.5
    # CHECK: unknown tile
    # CHECK: ((3, 4):(6, 1))
    # CHECK: 9.0 9.5 10.0 10.5
    # CHECK: 12.0 12.5 13.0 13.5
    # CHECK: 15.0 15.5 16.0 16.5
    # CHECK: tensor 2x2
    # CHECK: 9.0 9.5
    # CHECK: 12.0 12.5
    # CHECK: unknown tile
    # CHECK: ((3, 2):(6, 1))
    # CHECK: 11.0 11.5
    # CHECK: 14.0 14.5
    # CHECK: 17.0 17.5
    # CHECK: tensor 2x2
    # CHECK: 11.0 11.5
    # CHECK: 14.0 14.5
    print("----check unknown layout copy ----")
    for i in range(M // 3):
        for j in range(ceildiv(N, 4)):
            var unknown3x4 = tensorMxU.tensor.tile[3, 4](i, j)
            tensor2x2.copy_from(unknown3x4)
            print("unknown tile")
            print(unknown3x4.runtime_layout)
            print(unknown3x4)
            print("tensor 2x2")
            print(tensor2x2)
    # CHECK: ----check unknown layout vectorized copy ----
    # CHECK: unknown tile
    # CHECK: ((4, 4):(6, 1))
    # CHECK: 0.0 0.5 1.0 1.5
    # CHECK: 3.0 3.5 4.0 4.5
    # CHECK: 6.0 6.5 7.0 7.5
    # CHECK: 9.0 9.5 10.0 10.5
    # CHECK: tensor 2x2
    # CHECK: 0.0 0.5
    # CHECK: 3.0 3.5
    # CHECK: unknown tile
    # CHECK: ((4, 2):(6, 1))
    # CHECK: 2.0 2.5
    # CHECK: 5.0 5.5
    # CHECK: 8.0 8.5
    # CHECK: 11.0 11.5
    # CHECK: tensor 2x2
    # CHECK: 2.0 2.5
    # CHECK: 5.0 5.5
    print("----check unknown layout vectorized copy ----")
    for i in range(M // 4):
        for j in range(ceildiv(N, 4)):
            var unknown4x4 = tensorMxU.tensor.tile[4, 4](i, j)
            var tensor2x2_vec = tensor2x2.vectorize[2, 1]()
            tensor2x2_vec.copy_from(unknown4x4.vectorize[2, 1]())
            print("unknown tile")
            print(unknown4x4.runtime_layout)
            print(unknown4x4)
            print("tensor 2x2")
            print(tensor2x2)


def test_non_homogeneous_distribute():
    print("== test_non_homogeneous_distribute")

    alias type = DType.float32
    alias M = 8
    alias N = 6

    alias layoutMxU = Layout.row_major(M, UNKNOWN_VALUE)

    var src_runtimelayout = RuntimeLayout[layoutMxU].row_major(
        IndexList[2](M, N)
    )
    var tensorMxU = ManagedLayoutTensor[
        type, layoutMxU, __experimental_non_homogeneous_tile=True
    ](src_runtimelayout)
    arange(tensorMxU.tensor, 0, 0.5)

    # CHECK: ----tile-data[ 0 , 0 ]----
    # CHECK: 0.0 0.5 1.0 1.5
    # CHECK: 3.0 3.5 4.0 4.5
    # CHECK: 6.0 6.5 7.0 7.5
    # CHECK: 9.0 9.5 10.0 10.5
    # CHECK: ----fragments-data[ 0 ]----
    # CHECK: 0.0 1.0
    # CHECK: 6.0 7.0
    # CHECK: ----fragments-data[ 1 ]----
    # CHECK: 0.5 1.5
    # CHECK: 6.5 7.5
    # CHECK: ----fragments-data[ 2 ]----
    # CHECK: 3.0 4.0
    # CHECK: 9.0 10.0
    # CHECK: ----fragments-data[ 3 ]----
    # CHECK: 3.5 4.5
    # CHECK: 9.5 10.5
    # CHECK: ----tile-data[ 0 , 1 ]----
    # CHECK: 2.0 2.5
    # CHECK: 5.0 5.5
    # CHECK: 8.0 8.5
    # CHECK: 11.0 11.5
    # CHECK: ----fragments-data[ 0 ]----
    # CHECK: 2.0
    # CHECK: 8.0
    # CHECK: ----fragments-data[ 1 ]----
    # CHECK: 2.5
    # CHECK: 8.5
    # CHECK: ----fragments-data[ 2 ]----
    # CHECK: 5.0
    # CHECK: 11.0
    # CHECK: ----fragments-data[ 3 ]----
    # CHECK: 5.5
    # CHECK: 11.5
    # CHECK: ----tile-data[ 1 , 0 ]----
    # CHECK: 12.0 12.5 13.0 13.5
    # CHECK: 15.0 15.5 16.0 16.5
    # CHECK: 18.0 18.5 19.0 19.5
    # CHECK: 21.0 21.5 22.0 22.5
    # CHECK: ----fragments-data[ 0 ]----
    # CHECK: 12.0 13.0
    # CHECK: 18.0 19.0
    # CHECK: ----fragments-data[ 1 ]----
    # CHECK: 12.5 13.5
    # CHECK: 18.5 19.5
    # CHECK: ----fragments-data[ 2 ]----
    # CHECK: 15.0 16.0
    # CHECK: 21.0 22.0
    # CHECK: ----fragments-data[ 3 ]----
    # CHECK: 15.5 16.5
    # CHECK: 21.5 22.5
    # CHECK: ----tile-data[ 1 , 1 ]----
    # CHECK: 14.0 14.5
    # CHECK: 17.0 17.5
    # CHECK: 20.0 20.5
    # CHECK: 23.0 23.5
    # CHECK: ----fragments-data[ 0 ]----
    # CHECK: 14.0
    # CHECK: 20.0
    # CHECK: ----fragments-data[ 1 ]----
    # CHECK: 14.5
    # CHECK: 20.5
    # CHECK: ----fragments-data[ 2 ]----
    # CHECK: 17.0
    # CHECK: 23.0
    # CHECK: ----fragments-data[ 3 ]----
    # CHECK: 17.5
    # CHECK: 23.5
    for i in range(ceildiv(M, 4)):
        for j in range(ceildiv(M, 4)):
            var unknown4x4 = tensorMxU.tensor.tile[4, 4](i, j)
            print("----tile-data[", i, ",", j, "]----")
            print(unknown4x4)
            for th_i in range(4):
                var tile_2x2 = unknown4x4.distribute[Layout.row_major(2, 2)](
                    th_i
                )
                print("----fragments-data[", th_i, "]----")
                print(tile_2x2)

    alias simd_size = 2
    alias layout8xU = Layout.row_major(8, UNKNOWN_VALUE)
    alias thread_layout8xU = Layout.row_major(8 // simd_size, 2)

    var tensor8xU = ManagedLayoutTensor[
        type,
        layout8xU,
        __experimental_non_homogeneous_tile=True,
    ](RuntimeLayout[layout8xU].row_major(IndexList[2](8, 7)))
    arange(tensor8xU.tensor)

    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0
    # CHECK: 7.0 8.0 9.0 10.0 11.0 12.0 13.0
    # CHECK: 14.0 15.0 16.0 17.0 18.0 19.0 20.0
    # CHECK: 21.0 22.0 23.0 24.0 25.0 26.0 27.0
    # CHECK: 28.0 29.0 30.0 31.0 32.0 33.0 34.0
    # CHECK: 35.0 36.0 37.0 38.0 39.0 40.0 41.0
    # CHECK: 42.0 43.0 44.0 45.0 46.0 47.0 48.0
    # CHECK: 49.0 50.0 51.0 52.0 53.0 54.0 55.0
    # CHECK: ----thread[ 0 ]----
    # CHECK: [0.0, 7.0] [2.0, 9.0] [4.0, 11.0] [6.0, 13.0]
    # CHECK: ----thread[ 1 ]----
    # CHECK: [1.0, 8.0] [3.0, 10.0] [5.0, 12.0]
    # CHECK: ----thread[ 2 ]----
    # CHECK: [14.0, 21.0] [16.0, 23.0] [18.0, 25.0] [20.0, 27.0]
    # CHECK: ----thread[ 3 ]----
    # CHECK: [15.0, 22.0] [17.0, 24.0] [19.0, 26.0]
    # CHECK: ----thread[ 4 ]----
    # CHECK: [28.0, 35.0] [30.0, 37.0] [32.0, 39.0] [34.0, 41.0]
    # CHECK: ----thread[ 5 ]----
    # CHECK: [29.0, 36.0] [31.0, 38.0] [33.0, 40.0]
    # CHECK: ----thread[ 6 ]----
    # CHECK: [42.0, 49.0] [44.0, 51.0] [46.0, 53.0] [48.0, 55.0]
    # CHECK: ----thread[ 7 ]----
    # CHECK: [43.0, 50.0] [45.0, 52.0] [47.0, 54.0]
    print(tensor8xU.tensor)
    for tid in range(thread_layout8xU.size()):
        print("----thread[", tid, "]----")
        var tile = tensor8xU.tensor.vectorize[simd_size, 1]().distribute[
            thread_layout8xU
        ](tid)
        print(tile)

    alias layoutUx8 = Layout.row_major(UNKNOWN_VALUE, 8)
    alias thread_layoutUx8 = Layout.row_major(2, 8 // simd_size)

    var tensorUx8 = ManagedLayoutTensor[
        type,
        layoutUx8,
        __experimental_non_homogeneous_tile=True,
    ](RuntimeLayout[layoutUx8].row_major(IndexList[2](7, 8)))
    arange(tensorUx8.tensor)

    # CHECK: 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0
    # CHECK: 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0
    # CHECK: 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0
    # CHECK: 24.0 25.0 26.0 27.0 28.0 29.0 30.0 31.0
    # CHECK: 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0
    # CHECK: 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0
    # CHECK: 48.0 49.0 50.0 51.0 52.0 53.0 54.0 55.0
    # CHECK: ----thread[ 0 ]----
    # CHECK: [0.0, 1.0]
    # CHECK: [16.0, 17.0]
    # CHECK: [32.0, 33.0]
    # CHECK: [48.0, 49.0]
    # CHECK: ----thread[ 1 ]----
    # CHECK: [2.0, 3.0]
    # CHECK: [18.0, 19.0]
    # CHECK: [34.0, 35.0]
    # CHECK: [50.0, 51.0]
    # CHECK: ----thread[ 2 ]----
    # CHECK: [4.0, 5.0]
    # CHECK: [20.0, 21.0]
    # CHECK: [36.0, 37.0]
    # CHECK: [52.0, 53.0]
    # CHECK: ----thread[ 3 ]----
    # CHECK: [6.0, 7.0]
    # CHECK: [22.0, 23.0]
    # CHECK: [38.0, 39.0]
    # CHECK: [54.0, 55.0]
    # CHECK: ----thread[ 4 ]----
    # CHECK: [8.0, 9.0]
    # CHECK: [24.0, 25.0]
    # CHECK: [40.0, 41.0]
    # CHECK: ----thread[ 5 ]----
    # CHECK: [10.0, 11.0]
    # CHECK: [26.0, 27.0]
    # CHECK: [42.0, 43.0]
    # CHECK: ----thread[ 6 ]----
    # CHECK: [12.0, 13.0]
    # CHECK: [28.0, 29.0]
    # CHECK: [44.0, 45.0]
    # CHECK: ----thread[ 7 ]----
    # CHECK: [14.0, 15.0]
    # CHECK: [30.0, 31.0]
    # CHECK: [46.0, 47.0]
    print(tensorUx8.tensor)
    for tid in range(thread_layoutUx8.size()):
        print("----thread[", tid, "]----")
        var tile = tensorUx8.tensor.vectorize[1, simd_size]().distribute[
            thread_layoutUx8
        ](tid)
        print(tile)


def test_non_homogeneous_tiled_iterator():
    print("== test_non_homogeneous_tiled_iterator")

    alias type = DType.float32
    alias M = 8
    alias N = 7

    alias layoutMxU = Layout.row_major(M, UNKNOWN_VALUE)

    var src_runtimelayout = RuntimeLayout[layoutMxU].row_major(
        IndexList[2](M, N)
    )
    var tensorMxU = ManagedLayoutTensor[
        type, layoutMxU, __experimental_non_homogeneous_tile=True
    ](src_runtimelayout)
    arange(tensorMxU.tensor, 0, 0.5)

    # CHECK: 0.0 0.5 1.0 1.5 2.0 2.5 3.0
    # CHECK: 3.5 4.0 4.5 5.0 5.5 6.0 6.5
    # CHECK: 7.0 7.5 8.0 8.5 9.0 9.5 10.0
    # CHECK: 10.5 11.0 11.5 12.0 12.5 13.0 13.5
    # CHECK: 14.0 14.5 15.0 15.5 16.0 16.5 17.0
    # CHECK: 17.5 18.0 18.5 19.0 19.5 20.0 20.5
    # CHECK: 21.0 21.5 22.0 22.5 23.0 23.5 24.0
    # CHECK: 24.5 25.0 25.5 26.0 26.5 27.0 27.5
    # CHECK: ----coord[ 0 , 0 ]----
    # CHECK: 0.0 0.5 1.0 1.5
    # CHECK: 3.5 4.0 4.5 5.0
    # CHECK: 7.0 7.5 8.0 8.5
    # CHECK: 10.5 11.0 11.5 12.0
    # CHECK: ----coord[ 0 , 1 ]----
    # CHECK: 2.0 2.5 3.0
    # CHECK: 5.5 6.0 6.5
    # CHECK: 9.0 9.5 10.0
    # CHECK: 12.5 13.0 13.5
    # CHECK: ----coord[ 1 , 0 ]----
    # CHECK: 14.0 14.5 15.0 15.5
    # CHECK: 17.5 18.0 18.5 19.0
    # CHECK: 21.0 21.5 22.0 22.5
    # CHECK: 24.5 25.0 25.5 26.0
    # CHECK: ----coord[ 1 , 1 ]----
    # CHECK: 16.0 16.5 17.0
    # CHECK: 19.5 20.0 20.5
    # CHECK: 23.0 23.5 24.0
    # CHECK: 26.5 27.0 27.5
    print(tensorMxU.tensor)
    for i in range(ceildiv(M, 4)):
        var iter = tensorMxU.tensor.tiled_iterator[4, 4, axis=1](i, 0)
        for j in range(ceildiv(N, 4)):
            var unknown_tile = iter[]
            print("----coord[", i, ",", j, "]----")
            print(unknown_tile)
            iter = iter.next()

    alias M1 = 7
    alias N1 = 8

    alias layoutUxN = Layout.row_major(UNKNOWN_VALUE, N1)

    var runtime_layoutUxN = RuntimeLayout[layoutUxN].row_major(
        IndexList[2](M1, N1)
    )
    var tensorUxN = ManagedLayoutTensor[
        type, layoutUxN, __experimental_non_homogeneous_tile=True
    ](runtime_layoutUxN)
    arange(tensorUxN.tensor, 0, 0.5)

    # CHECK: 0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5
    # CHECK: 4.0 4.5 5.0 5.5 6.0 6.5 7.0 7.5
    # CHECK: 8.0 8.5 9.0 9.5 10.0 10.5 11.0 11.5
    # CHECK: 12.0 12.5 13.0 13.5 14.0 14.5 15.0 15.5
    # CHECK: 16.0 16.5 17.0 17.5 18.0 18.5 19.0 19.5
    # CHECK: 20.0 20.5 21.0 21.5 22.0 22.5 23.0 23.5
    # CHECK: 24.0 24.5 25.0 25.5 26.0 26.5 27.0 27.5
    # CHECK: ----coord[ 0 , 0 ]----
    # CHECK: 0.0 0.5 1.0 1.5
    # CHECK: 4.0 4.5 5.0 5.5
    # CHECK: 8.0 8.5 9.0 9.5
    # CHECK: 12.0 12.5 13.0 13.5
    # CHECK: ----coord[ 1 , 0 ]----
    # CHECK: 16.0 16.5 17.0 17.5
    # CHECK: 20.0 20.5 21.0 21.5
    # CHECK: 24.0 24.5 25.0 25.5
    # CHECK: ----coord[ 0 , 1 ]----
    # CHECK: 2.0 2.5 3.0 3.5
    # CHECK: 6.0 6.5 7.0 7.5
    # CHECK: 10.0 10.5 11.0 11.5
    # CHECK: 14.0 14.5 15.0 15.5
    # CHECK: ----coord[ 1 , 1 ]----
    # CHECK: 18.0 18.5 19.0 19.5
    # CHECK: 22.0 22.5 23.0 23.5
    # CHECK: 26.0 26.5 27.0 27.5
    print(tensorUxN.tensor)
    for j in range(ceildiv(N1, 4)):
        var iter = tensorUxN.tensor.tiled_iterator[4, 4, axis=0](0, j)
        for i in range(ceildiv(M1, 4)):
            var unknown_tile = iter[]
            print("----coord[", i, ",", j, "]----")
            print(unknown_tile)
            iter = iter.next()

    var iter1 = tensorUxN.tensor.tiled_iterator[4, 4, axis=0](0, 0)
    iter1 += 1
    # CHECK: ----coord[ 1 , 0 ]----
    # CHECK: 16.0 16.5 17.0 17.5
    # CHECK: 20.0 20.5 21.0 21.5
    # CHECK: 24.0 24.5 25.0 25.5
    print("----coord[ 1 , 0 ]----")
    print(iter1[])

    var iter2 = tensorMxU.tensor.tiled_iterator[4, 4, axis=1](0, 0)
    iter2 += 1
    # CHECK: ----coord[ 0 , 1 ]----
    # CHECK: 2.0 2.5 3.0
    # CHECK: 5.5 6.0 6.5
    # CHECK: 9.0 9.5 10.0
    # CHECK: 12.5 13.0 13.5
    print("----coord[ 0 , 1 ]----")
    print(iter2[])

    var iter3 = tensorMxU.tensor.tiled_iterator[4, 4, axis=1](0, 0)
    iter3._incr()
    # CHECK: ----coord[ 0 , 1 ]----
    # CHECK: 2.0 2.5 3.0
    # CHECK: 5.5 6.0 6.5
    # CHECK: 9.0 9.5 10.0
    # CHECK: 12.5 13.0 13.5
    print("----coord[ 0 , 1 ]----")
    print(iter3[])

    _ = tensorMxU^
    _ = tensorUxN^


def main():
    test_single_unknown_tile()
    test_non_homogeneous_copy_from()
    test_non_homogeneous_distribute()
    test_non_homogeneous_tiled_iterator()
