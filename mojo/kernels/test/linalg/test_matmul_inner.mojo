# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo %s | FileCheck %s

from Matmul import (
    MatmulInnerLoopBPacked,
    GemmShape,
    MatmulConfig,
    get_pack_data_size,
)
from Buffer import NDBuffer
from TargetInfo import dtype_simd_width
from Index import Index
from DType import DType
from List import create_dim_list, DimList
from IO import print
from Int import Int


alias simd_size: Int = dtype_simd_width[DType.f32]()
alias a_row_size: Int = 5
alias pack_inner_size: Int = 4
alias prefetch_b_distance_k: Int = 4

alias M: Int = 64
alias N: Int = 64
alias K: Int = 64

alias type = DType.f32

alias tile_inner_size: Int = pack_inner_size * simd_size


@export
fn matmul_inner_loop(
    c: NDBuffer[2, create_dim_list(M, N), type],
    a: NDBuffer[2, create_dim_list(M, K), type],
    b_packed: NDBuffer[
        3,
        create_dim_list(
            N // tile_inner_size,
            K,
            tile_inner_size,
        ),
        type,
    ],
):

    MatmulInnerLoopBPacked[
        create_dim_list(M, K),
        create_dim_list(M, N),
        create_dim_list(
            N // tile_inner_size,
            K,
            tile_inner_size,
        ),
        type,
        type,
        simd_size,
        a_row_size,
        pack_inner_size * simd_size,
        True,  # skip bound check
        prefetch_b_distance_k,
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
    print("== test_micro_kernel\n")

    let a = (
        NDBuffer[2, create_dim_list(M, K), type]
        .aligned_stack_allocation[128]()
        .fill(1)
    )

    let b_packed = (
        NDBuffer[
            3,
            create_dim_list(
                N // tile_inner_size,
                K,
                tile_inner_size,
            ),
            type,
        ]
        .aligned_stack_allocation[128]()
        .fill(1)
    )

    let c = (
        NDBuffer[2, create_dim_list(M, N), type]
        .aligned_stack_allocation[128]()
        .fill(0)
    )

    matmul_inner_loop(c, a, b_packed)

    # CHECK: [64.000000]
    print(c[0, 0])


fn main():
    test_micro_kernel()
