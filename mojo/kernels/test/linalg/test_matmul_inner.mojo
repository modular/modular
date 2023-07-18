# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo %s | FileCheck %s

from Matmul import (
    MatmulInnerLoopBPacked,
    GemmShape,
    MatmulConfig,
)
from MatmulUtils import get_pack_data_size
from Buffer import NDBuffer
from TargetInfo import simdwidthof
from Index import Index
from DType import DType
from List import DimList
from IO import print

alias a_type = DType.float32
alias b_type = DType.float32
alias c_type = DType.float32


alias simd_size: Int = simdwidthof[c_type]()
alias a_row_size: Int = 6
alias pack_inner_size: Int = 4
alias prefetch_b_distance_k: Int = 4

alias M: Int = 64
alias N: Int = 64
alias K: Int = 64

alias tile_inner_size: Int = pack_inner_size * simd_size


@export(ABI="C")
fn matmul_inner_loop(
    c: NDBuffer[2, DimList(M, N), c_type],
    a: NDBuffer[2, DimList(M, K), a_type],
    b_packed: NDBuffer[
        3,
        DimList(
            N // tile_inner_size,
            K,
            tile_inner_size,
        ),
        b_type,
    ],
):

    MatmulInnerLoopBPacked[
        DimList(M, K),
        DimList(M, N),
        DimList(
            N // tile_inner_size,
            K,
            tile_inner_size,
        ),
        a_type,
        b_type,
        c_type,
        simd_size,
        a_row_size,
        pack_inner_size * simd_size,
        True,  # skip bound check
        prefetch_b_distance_k,
        False,  # critical_stride
    ].run(
        c,
        a,
        b_packed,
        # Below are configurations for outer loops, just
        #  use the trivial numbers for now.
        GemmShape(0, 0, 0),  # Tile offset.
        GemmShape(M, N, K),  # Global tile dimension.
        Index(N, K),  # Local tile dimension.
    )


# CHECK-LABEL: test_micro_kernel
fn test_micro_kernel():
    print("== test_micro_kernel")

    let a = NDBuffer[2, DimList(M, K), a_type].aligned_stack_allocation[128]()
    a.fill(1)

    let b_packed = NDBuffer[
        3,
        DimList(
            N // tile_inner_size,
            K,
            tile_inner_size,
        ),
        b_type,
    ].aligned_stack_allocation[128]()
    b_packed.fill(1)

    let c = NDBuffer[2, DimList(M, N), c_type].aligned_stack_allocation[128]()
    c.fill(0)

    matmul_inner_loop(c, a, b_packed)

    # CHECK: 64.0
    print(c[0, 0])


fn main():
    test_micro_kernel()
