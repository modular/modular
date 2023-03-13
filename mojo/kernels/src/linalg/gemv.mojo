# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Buffer import NDBuffer, Buffer
from DType import DType
from Functional import unroll
from Int import Int
from List import create_kgen_list_unknown, create_kgen_list
from Range import range
from SIMD import SIMD
from TargetInfo import simd_width, sizeof
from IO import print
from Index import Index


fn gemv[
    simd_width: __mlir_type.index,
    out_size: __mlir_type.index,
    lhs_shape: __mlir_type[`!kgen.list<index[2]>`],
    rhs_size: __mlir_type.index,
    type: DType,
](
    out: Buffer[out_size, type],
    lhs: NDBuffer[2, lhs_shape, type],
    rhs: Buffer[rhs_size, type],
):
    let m: Int = lhs.dim[0]()
    let n: Int = lhs.dim[1]()

    alias col_unroll_factor = 1  # parameter
    alias row_block_size = 8  # parameter
    alias col_block_size = simd_width * col_unroll_factor
    let vector_end_col = (n // col_block_size) * col_block_size
    let vector_end_row = (m // row_block_size) * row_block_size

    # Allocate row_block_size accumulator values. This is a matrix of row_block_size x col_block_size
    let accums = NDBuffer[
        2,
        create_kgen_list[__mlir_type.index](row_block_size, col_block_size),
        type,
    ].aligned_stack_allocation[64]()

    var row_idx: Int = 0
    var col_idx: Int = 0
    while row_idx < vector_end_row:

        @always_inline
        fn _set_zero[idx: Int]():
            let zero = SIMD[col_block_size, type](0)
            accums.simd_store[col_block_size](Index(idx, 0), zero)

        unroll[row_block_size, _set_zero]()

        col_idx = 0
        while col_idx < vector_end_col:

            @always_inline
            fn _do_accum[idx: Int]():
                # Row `idx`
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

            col_idx += col_block_size

        # Allocate row_block_size scalar accumulator values.
        let scalar_accums = Buffer[
            row_block_size,
            type,
        ].aligned_stack_allocation[64]()
        for ii in range(row_block_size):
            let accum_idx = Index(ii, 0)
            let curr_accum = accums.simd_load[col_block_size](accum_idx)
            scalar_accums[ii] = curr_accum.reduce_add()

        # Store the results
        out.simd_store[row_block_size](
            row_idx, scalar_accums.simd_load[row_block_size](0)
        )

        row_idx += row_block_size

    row_idx = vector_end_row
    while row_idx < m:
        col_idx = 0
        var simd_accum = SIMD[col_block_size, type](0)
        while col_idx < vector_end_col:

            let row_chunk = lhs.simd_load[col_block_size](
                Index(row_idx, col_idx)
            )
            let col_chunk = rhs.simd_load[col_block_size](col_idx)
            simd_accum = row_chunk.fma(col_chunk, simd_accum)

            col_idx += col_block_size
        out[row_idx] = simd_accum.reduce_add()
        row_idx += 1
