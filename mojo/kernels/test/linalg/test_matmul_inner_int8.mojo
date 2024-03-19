# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from math import align_up
from sys.info import simdwidthof

from buffer import NDBuffer
from Matmul import GemmShape, MatmulConfig, MatmulInnerLoopBPacked
from MatmulUtils import (
    get_matmul_arch_factor,
    get_matmul_kernel_shape,
    get_matmul_prefetch_b_distance_k,
    use_i8mm_fn,
    use_vnni_fn,
)

from utils.index import Index
from buffer.list import DimList

alias a_type = DType.uint8
alias b_type = DType.int8
alias c_type = DType.int32

alias simd_size: Int = simdwidthof[c_type]()
alias kernel_shape = get_matmul_kernel_shape[a_type, b_type, c_type, False]()
alias a_row_size = kernel_shape.a_row_size
alias pack_inner_size = kernel_shape.pack_inner_size
alias tile_inner_size: Int = pack_inner_size * simd_size

alias prefetch_b_distance_k: Int = get_matmul_prefetch_b_distance_k()

alias use_vnni = use_vnni_fn[a_type, b_type, c_type]()
alias use_i8mm = use_i8mm_fn[a_type, b_type, c_type]()
alias factor = get_matmul_arch_factor[use_vnni, use_i8mm]()

alias M: Int = 64
alias N: Int = 64
alias K: Int = 256
alias NP = align_up(N, tile_inner_size)
alias KH = align_up(K, factor)


@export(ABI="C")
fn matmul_inner_loop(
    c: NDBuffer[c_type, 2, DimList(M, N)],
    a: NDBuffer[a_type, 2, DimList(M, K)],
    b_packed: NDBuffer[
        b_type,
        3,
        DimList(
            NP // tile_inner_size,
            KH // factor,
            factor * tile_inner_size,
        ),
    ],
):
    MatmulInnerLoopBPacked[
        DimList(M, K),
        DimList(M, N),
        DimList(
            NP // tile_inner_size,
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

    var a = NDBuffer[a_type, 2, DimList(M, K)].aligned_stack_allocation[128]()
    a.fill(1)

    var b_packed = NDBuffer[
        b_type,
        3,
        DimList(
            NP // tile_inner_size,
            KH // factor,
            factor * tile_inner_size,
        ),
    ].aligned_stack_allocation[128]()
    b_packed.fill(1)

    var c = NDBuffer[c_type, 2, DimList(M, N)].aligned_stack_allocation[128]()
    c.fill(0)

    matmul_inner_loop(c, a, b_packed)

    var val = c[0, 0]
    # CHECK: 256
    if not (use_i8mm or use_vnni):
        val = 256
    print(c[0, 0])


fn main():
    test_micro_kernel()
