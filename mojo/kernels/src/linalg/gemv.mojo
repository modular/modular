# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from algorithm import unroll, async_parallelize
from math import align_down
from memory.buffer import Buffer, NDBuffer

from utils.index import Index
from utils.list import Dim, DimList


fn dot[
    lhs_shape: DimList, rhs_size: Dim, simd_width: Int, type: DType
](
    row_idx: Int,
    vector_end_col: Int,
    lhs: NDBuffer[2, lhs_shape, type],
    rhs: Buffer[rhs_size, type],
) -> SIMD[type, 1]:
    var simd_accum = SIMD[type, simd_width](0)
    for col_idx in range(0, vector_end_col, simd_width):
        let row_chunk = lhs.simd_load[simd_width](Index(row_idx, col_idx))
        let col_chunk = rhs.simd_load[simd_width](col_idx)
        simd_accum = row_chunk.fma(col_chunk, simd_accum)
    return simd_accum.reduce_add()


# Parallelized version of Gemv


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
    out_chain: OutputChainPtr,
):
    let m: Int = lhs.dim[0]()
    let n: Int = lhs.dim[1]()

    alias col_unroll_factor = 8
    alias row_block_size = 32
    alias col_block_size = simd_width * col_unroll_factor
    let vector_end_col = align_down(n, col_block_size)
    let vector_end_row = align_down(m, row_block_size)

    @always_inline
    @parameter
    fn process_row_block(row_block_idx: Int):
        let row_idx = row_block_idx * row_block_size

        alias simd_alignment = 64
        let accums = Buffer[
            row_block_size,
            type,
        ].aligned_stack_allocation[simd_alignment]()
        accums.zero()

        @unroll
        for idx in range(row_block_size):
            accums[idx] = dot[lhs_shape, rhs_size, col_block_size, type](
                idx, vector_end_col, lhs, rhs
            )

        @unroll
        for idx in range(row_block_size):
            for col_idx in range(vector_end_col, n):
                let row = lhs.simd_load[1](Index(row_idx + idx, col_idx))
                let col = rhs[col_idx]
                accums[idx] += row * col

        out.simd_store[row_block_size](
            row_idx, accums.simd_load[row_block_size](0)
        )

    let row_block_count = vector_end_row // row_block_size
    if out_chain:
        async_parallelize[process_row_block](out_chain, row_block_count)
    else:
        for row_block_idx in range(row_block_count):
            process_row_block(row_block_idx)

    for row_idx in range(vector_end_row, m):
        var out_val = dot[lhs_shape, rhs_size, col_block_size, type](
            row_idx, vector_end_col, lhs, rhs
        )
        for col_idx in range(vector_end_col, n):
            let row = lhs.simd_load[1](Index(row_idx, col_idx))
            let col = rhs.simd_load[1](col_idx)
            out_val += row * col
        out[row_idx] = out_val


fn trivial_gemv[
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
    for m in range(lhs.dim[0]()):
        var out_row_val: SIMD[type, 1] = 0
        for n in range(lhs.dim[1]()):
            let row = lhs.simd_load[1](Index(m, n))
            let col = rhs.simd_load[1](n)
            out_row_val += row * col
        out[m] = out_row_val


fn orig_gemv[
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

            @unroll
            for idx in range(row_block_size):
                let accum_idx = Index(idx, 0)
                let accum = accums.simd_load[col_block_size](accum_idx)
                let row_chunk = lhs.simd_load[col_block_size](
                    Index(row_idx + idx, col_idx)
                )
                let col_chunk = rhs.simd_load[col_block_size](col_idx)
                accums.simd_store[col_block_size](
                    accum_idx, row_chunk.fma(col_chunk, accum)
                )

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

        @unroll
        for idx in range(row_block_size):
            let accum_idx = Index(idx, 0)
            let curr_accum = accums.simd_load[col_block_size](accum_idx)
            scalar_accums[idx] = scalar_accums[idx] + curr_accum.reduce_add()

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
