# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: asan
# RUN: %mojo -debug-level full %s | FileCheck %s

from sys.info import simdwidthof

from Matmul import GemmShape, MatmulConfig, MatmulInnerLoopBPacked
from MatmulUtils import (
    get_matmul_kernel_shape,
    get_matmul_prefetch_b_distance_k,
)
from memory.buffer import NDBuffer

from utils.index import Index
from utils.list import DimList

alias a_type = DType.float32
alias b_type = DType.float32
alias c_type = DType.float32


alias simd_size: Int = simdwidthof[c_type]()
alias kernel_shape = get_matmul_kernel_shape[a_type, b_type, c_type, False]()
alias a_row_size = kernel_shape.a_row_size
alias pack_inner_size = kernel_shape.pack_inner_size
alias tile_inner_size: Int = pack_inner_size * simd_size

alias prefetch_b_distance_k: Int = get_matmul_prefetch_b_distance_k()

alias M: Int = 64
alias N: Int = 64
alias K: Int = 64


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
        False,  # saturated_vnni
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
