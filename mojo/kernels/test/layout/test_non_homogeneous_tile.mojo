# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s | FileCheck %s

from layout import LayoutTensor, Layout, RuntimeLayout
from layout.layout import UNKNOWN_VALUE
from utils import StaticIntTuple
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
    var runtimelayout0 = RuntimeLayout[layout0].row_major(
        StaticIntTuple[2](M0, N0)
    )
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
    var runtimelayout1 = RuntimeLayout[layout1].row_major(
        StaticIntTuple[2](M1, N1)
    )
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

    var src_runtimelayout = RuntimeLayout[layout].row_major(
        StaticIntTuple[2](M, N)
    )
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
    var src_unknown = RuntimeLayout[layoutMxU].row_major(
        StaticIntTuple[2](M, N)
    )
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


def main():
    test_single_unknown_tile()
    test_non_homogeneous_copy_from()
