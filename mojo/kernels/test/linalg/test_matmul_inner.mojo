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
from TargetInfo import simd_width, sizeof
from Index import Index
from DType import DType
from List import create_dim_list, DimList
from IO import print
from Int import Int


alias simd_size: Int = simd_width[__mlir_type.f32]()
alias a_row_size: Int = 5
alias pack_inner_size: Int = 4
alias prefetch_b_distance_k: Int = 4

alias tile_inner_size = pack_inner_size * simd_size


fn get_matmul_config[gemm_dimension: GemmShape]() -> MatmulConfig:
    """Utility function to extract matmul configuration parameters for exported"""
    return MatmulConfig {
        shape_a: create_dim_list(gemm_dimension.M, gemm_dimension.K),
        shape_b: DimList[2].create_unknown(),
        shape_c: create_dim_list(gemm_dimension.M, gemm_dimension.N),
        packed_shape: create_dim_list(
            gemm_dimension.N // tile_inner_size,
            gemm_dimension.K,
            tile_inner_size,
        ),
        simd_size: simd_size.__as_mlir_index(),
        a_row_size: a_row_size.__as_mlir_index(),
        pack_inner_size: (pack_inner_size * simd_size).__as_mlir_index(),
        pack_data_size: get_pack_data_size().__as_mlir_index(),
        prefetch_b_distance_k: prefetch_b_distance_k,
    }


fn matmul_inner_loop[gemm_dimension: GemmShape, type: DType]():
    alias config = get_matmul_config[gemm_dimension]()

    let a = NDBuffer[2, config.shape_a, type].stack_allocation().fill(1)

    let b_packed = (
        NDBuffer[3, config.packed_shape, type]
        .aligned_stack_allocation[128]()
        .fill(1)
    )

    let c = NDBuffer[2, config.shape_c, type].aligned_stack_allocation[128]()

    MatmulInnerLoopBPacked[
        config.shape_a,
        config.shape_c,
        config.packed_shape,
        type,
        type,
        config.simd_size,
        config.a_row_size,
        config.pack_inner_size,
        True,  # skip bound check
        prefetch_b_distance_k,
    ].run(
        c,
        a,
        b_packed,
        # Below are configurations for outer loops, just
        #  use the trivial numbers for now.
        GemmShape(0, 0, 0),  # Tile offset.
        gemm_dimension,  # Global tile dimension.
        Index(gemm_dimension.N, gemm_dimension.K),  # Local tile dimension.
    )

    # CHECK: [64.000000]
    print(c[0, 0])


fn main():
    matmul_inner_loop[GemmShape(64, 64, 64), DType.f32]()
