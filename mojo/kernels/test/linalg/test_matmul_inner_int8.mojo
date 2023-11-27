# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from math import align_up
from sys.info import simdwidthof

from Matmul import GemmShape, MatmulConfig, MatmulInnerLoopBPacked
from MatmulUtils import (
    get_matmul_a_row_size,
    get_matmul_pack_inner_size,
    get_matmul_prefetch_b_distance_k,
    get_matmul_arch_factor,
    use_vnni_fn,
    use_i8mm_fn,
)
from memory.buffer import NDBuffer

from utils.index import Index
from utils.list import DimList

alias a_type = DType.uint8
alias b_type = DType.int8
alias c_type = DType.int32

alias simd_size: Int = simdwidthof[c_type]()
alias a_row_size: Int = get_matmul_a_row_size[False]()
alias pack_inner_size: Int = get_matmul_pack_inner_size[False]()
alias prefetch_b_distance_k: Int = get_matmul_prefetch_b_distance_k()

alias use_vnni = use_vnni_fn[a_type, b_type, c_type]()
alias use_i8mm = use_i8mm_fn[a_type, b_type, c_type]()
alias factor = get_matmul_arch_factor[use_vnni, use_i8mm]()

alias M: Int = 64
alias N: Int = 64
alias K: Int = 256
alias KH = align_up(K, factor)

alias tile_inner_size: Int = pack_inner_size * simd_size


@export(ABI="C")
fn matmul_inner_loop(
    c: NDBuffer[2, DimList(M, N), c_type],
    a: NDBuffer[2, DimList(M, K), a_type],
    b_packed: NDBuffer[
        3,
        DimList(
            N // tile_inner_size,
            KH // factor,
            factor * tile_inner_size,
        ),
        b_type,
    ],
):
    MatmulInnerLoopBPacked[
        DimList(M, K),
        DimList(M, N),
        DimList(
            N // tile_inner_size,
            KH // factor,
            factor * tile_inner_size,
        ),
        a_type,
        b_type,
        c_type,
        simd_size,
        a_row_size,
        pack_inner_size * simd_size,
        True,  # skip bound check
        prefetch_b_distance_k,
        True,  # saturated_vnni
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
            KH // factor,
            factor * tile_inner_size,
        ),
        b_type,
    ].aligned_stack_allocation[128]()
    b_packed.fill(1)

    let c = NDBuffer[2, DimList(M, N), c_type].aligned_stack_allocation[128]()
    c.fill(0)

    matmul_inner_loop(c, a, b_packed)

    var val = c[0, 0]
    # CHECK: 256
    if not (use_i8mm or use_vnni):
        val = 256
    print(c[0, 0])


fn main():
    test_micro_kernel()
