# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# UNSUPPORTED: asan
# RUN: %mojo -debug-level full %s | FileCheck %s

from sys.info import has_neon, simdwidthof

from Matmul import GemmShape, MatmulConfig, MatmulInnerLoopBPacked
from MatmulUtils import (
    get_matmul_kernel_shape,
    get_matmul_prefetch_b_distance_k,
)
from buffer import NDBuffer

from utils.index import Index
from utils.list import DimList

from math import div_ceil

alias prefetch_b_distance_k: Int = get_matmul_prefetch_b_distance_k()

alias M: Int = 64
alias N: Int = 64
alias K: Int = 64


fn matmul_inner_loop[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    tile_inner_size: Int,
    simd_size: Int,
    a_row_size: Int,
    pack_inner_size: Int,
](
    c: NDBuffer[c_type, 2, DimList(M, N)],
    a: NDBuffer[a_type, 2, DimList(M, K)],
    b_packed: NDBuffer[
        b_type,
        3,
        DimList(
            div_ceil(N, tile_inner_size),
            K,
            tile_inner_size,
        ),
    ],
):
    MatmulInnerLoopBPacked[
        DimList(M, K),
        DimList(M, N),
        DimList(
            div_ceil(N, tile_inner_size),
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
fn test_micro_kernel[
    a_type: DType,
    b_type: DType,
    c_type: DType,
]():
    print("== test_micro_kernel")

    alias simd_size: Int = simdwidthof[c_type]()
    alias kernel_shape = get_matmul_kernel_shape[
        a_type, b_type, c_type, False
    ]()
    alias a_row_size = kernel_shape.a_row_size
    alias pack_inner_size = kernel_shape.pack_inner_size
    alias tile_inner_size: Int = pack_inner_size * simd_size

    var a = NDBuffer[a_type, 2, DimList(M, K)].aligned_stack_allocation[128]()
    a.fill(1)

    var b_packed = NDBuffer[
        b_type,
        3,
        DimList(
            div_ceil(N, tile_inner_size),
            K,
            tile_inner_size,
        ),
    ].aligned_stack_allocation[128]()
    b_packed.fill(1)

    var c = NDBuffer[c_type, 2, DimList(M, N)].aligned_stack_allocation[128]()
    c.fill(0)

    matmul_inner_loop[
        a_type,
        b_type,
        c_type,
        tile_inner_size,
        simd_size,
        a_row_size,
        pack_inner_size,
    ](c, a, b_packed)

    # CHECK: 64.0
    print(c[0, 0])


@export(ABI="C")
fn main():
    test_micro_kernel[DType.float32, DType.float32, DType.float32]()

    # TODO(30525): Re-enable after we resolve llvm lowering issues.
    @parameter
    if not has_neon():
        test_micro_kernel[DType.bfloat16, DType.bfloat16, DType.bfloat16]()
        test_micro_kernel[DType.bfloat16, DType.bfloat16, DType.float32]()
