# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Buffer import NDBuffer, Buffer
from DType import DType
from Functional import unroll
from List import Dim, DimList
from SIMD import SIMD
from Index import Index
from Range import range


fn gemv[
    simd_width: Int,
    out_size: Dim,
    lhs_shape: DimList,
    rhs_size: Dim,
    type: DType,
](
    out: Buffer[out_size, type],
    lhs: NDBuffer[2, lhs_shape, type],
    rhs: Buffer[rhs_size, type],
):
    let m: Int = lhs.dim[0]()
    let n: Int = lhs.dim[1]()

    alias col_unroll_factor = 1
    alias row_block_size = 8
    alias col_block_size = simd_width * col_unroll_factor
    let vector_end_col = (n // col_block_size) * col_block_size
    let vector_end_row = (m // row_block_size) * row_block_size

    let accums = NDBuffer[
        2,
        DimList(row_block_size, col_block_size),
        type,
    ].aligned_stack_allocation[64]()

    for row_idx in range(0, vector_end_row, row_block_size):
        accums.zero()

        for col_idx in range(0, vector_end_col, col_block_size):

            @always_inline
            @parameter
            fn _do_accum[idx: Int]():
                let accum_idx = Index(idx, 0)
                let accum = accums.simd_load[col_block_size](accum_idx)
                let row_chunk = lhs.simd_load[col_block_size](
                    Index(row_idx + idx, col_idx)
                )
                let col_chunk = rhs.simd_load[col_block_size](col_idx)
                accums.simd_store[col_block_size](
                    accum_idx, row_chunk.fma(col_chunk, accum)
                )

            unroll[row_block_size, _do_accum]()

        let scalar_accums = Buffer[
            row_block_size,
            type,
        ].aligned_stack_allocation[64]()
        scalar_accums.zero()

        for col_idx in range(vector_end_col, n):

            @unroll
            for idx in range(row_block_size):
                let row = lhs.simd_load[1](Index(row_idx + idx, col_idx))
                let col = rhs[col_idx]
                scalar_accums[idx] += row * col

        @always_inline
        @parameter
        fn body[idx: Int]():
            let accum_idx = Index(idx, 0)
            let curr_accum = accums.simd_load[col_block_size](accum_idx)
            scalar_accums[idx] = scalar_accums[idx] + curr_accum.reduce_add()

        unroll[row_block_size, body]()

        out.simd_store[row_block_size](
            row_idx, scalar_accums.simd_load[row_block_size](0)
        )

    for row_idx in range(vector_end_row, m):
        var simd_accum = SIMD[type, col_block_size](0)
        for col_idx in range(0, vector_end_col, col_block_size):
            let row_chunk = lhs.simd_load[col_block_size](
                Index(row_idx, col_idx)
            )
            let col_chunk = rhs.simd_load[col_block_size](col_idx)
            simd_accum = row_chunk.fma(col_chunk, simd_accum)
        out[row_idx] = simd_accum.reduce_add()

        for col_idx in range(vector_end_col, n):
            let row = lhs.simd_load[1](Index(row_idx, col_idx))
            let col = rhs.simd_load[1](col_idx)
            out[row_idx] = out[row_idx] + row * col
