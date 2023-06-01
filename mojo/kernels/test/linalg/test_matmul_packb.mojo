# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo %s | FileCheck %s

from DType import DType
from Buffer import NDBuffer
from Index import Index
from List import DimList
from Matmul import PackMatrixCols
from SIMD import SIMD
from TargetInfo import dtype_simd_width
from IO import print

alias type = DType.float32
alias simd_size: Int = dtype_simd_width[DType.float32]()
alias pack_inner_size: Int = 4
alias tile_inner_size: Int = pack_inner_size * simd_size
alias width = 2 * tile_inner_size

alias N: Int = 128
alias K: Int = 128
alias kc = 128


@export
fn pack_b(
    packed_b: NDBuffer[
        3, DimList(width // tile_inner_size, K, tile_inner_size), type
    ],
    b: NDBuffer[2, DimList(K, N), type],
):
    PackMatrixCols[
        DimList(K, N),
        DimList(width // tile_inner_size, K, tile_inner_size),
        type,
        simd_size,
        tile_inner_size,
    ].run(
        packed_b,
        b,
        Index(0, 0),
        Index(kc, width),
        Index(K, N),
    )


fn test_pack_b():
    let packed_b = NDBuffer[
        3, DimList(width // tile_inner_size, K, tile_inner_size), type
    ].aligned_stack_allocation[64]()
    packed_b.fill(1)
    let b = NDBuffer[2, DimList(K, N), type].aligned_stack_allocation[64]()
    b.fill(1)
    pack_b(packed_b, b)

    # CHECK: 1.0
    print(packed_b[0, 0, 0])


fn main():
    test_pack_b()
