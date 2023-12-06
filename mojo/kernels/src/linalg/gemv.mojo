# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from algorithm import unroll, async_parallelize
from math import align_down
from memory.buffer import Buffer, NDBuffer
from algorithm.reduction import _reduce_generator

from utils.index import Index
from utils.list import Dim, DimList
from MatmulUtils import elementwise_lambda_fn_sig_type

# Parallelized version of Gemv


fn gemv[
    parallelize: Bool,
    c_size: Dim,
    c_type: DType,
    a_shape: DimList,
    a_type: DType,
    b_size: Dim,
    b_type: DType,
](
    c_buf: Buffer[c_size, c_type],
    a_buf: NDBuffer[2, a_shape, a_type],
    b_buf: Buffer[b_size, b_type],
    out_chain: OutputChainPtr = OutputChainPtr(),
):
    @parameter
    fn null_lambda[
        val_type: DType, width: Int
    ](out_coords: StaticIntTuple[2], out_val: SIMD[val_type, width]):
        pass

    gemv[
        parallelize,
        c_size,
        c_type,
        a_shape,
        a_type,
        b_size,
        b_type,
        False,
        null_lambda,
    ](c_buf, a_buf, b_buf, out_chain)


fn gemv[
    parallelize: Bool,
    c_size: Dim,
    c_type: DType,
    a_shape: DimList,
    a_type: DType,
    b_size: Dim,
    b_type: DType,
    elementwise_epilogue_enabled: Bool,
    elementwise_lambda_fn: elementwise_lambda_fn_sig_type,
](
    c_buf: Buffer[c_size, c_type],
    a_buf: NDBuffer[2, a_shape, a_type],
    b_buf: Buffer[b_size, b_type],
    out_chain: OutputChainPtr = OutputChainPtr(),
):
    alias simd_width = simdwidthof[c_type]()

    let M = a_buf.dim[0]()
    let K = a_buf.dim[1]()

    @always_inline
    @parameter
    fn input_fn[
        type: DType, width: Int, rank: Int
    ](idx: StaticIntTuple[rank]) -> SIMD[type, width]:
        return (
            a_buf.simd_load[width](Index(idx[0], idx[1])).cast[type]()
            * b_buf.simd_load[width](idx[1]).cast[type]()
        ).cast[type]()

    @always_inline
    @parameter
    fn output_fn[
        out_type: DType, width: Int, r: Int
    ](idx: StaticIntTuple[r], value: SIMD[out_type, width]):
        @parameter
        if elementwise_epilogue_enabled:
            elementwise_lambda_fn[out_type, width](Index(idx[0], 0), value)
        else:
            c_buf.simd_store[width](idx[0], value.cast[c_type]())

    @always_inline
    @parameter
    fn reduce_impl[
        ty: DType, width: Int
    ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
        return v1 + v2

    _reduce_generator[
        c_type,
        2,
        # single_thread_blocking_override,
        not parallelize,
        input_fn,
        output_fn,
        reduce_impl,
    ](StaticIntTuple[2](M, K), 0, 1, out_chain)


fn trivial_gemv[
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
