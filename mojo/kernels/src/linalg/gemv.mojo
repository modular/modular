# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from memory.buffer import Buffer, NDBuffer
from algorithm.reduction import _reduce_generator

from utils.index import Index
from utils.list import Dim, DimList
from MatmulUtils import elementwise_lambda_fn_sig_type

from runtime.llcl import OutputChainPtr

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


@always_inline
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
        out_type: DType, width: Int, rank: Int
    ](idx: StaticIntTuple[rank], value: SIMD[out_type, width]):
        @parameter
        if elementwise_epilogue_enabled:

            @unroll
            for i in range(width):
                elementwise_lambda_fn[out_type, 1](
                    Index(idx[0] + i, 0), value[i]
                )
        else:
            c_buf.simd_store[width](idx[0], value.cast[c_type]())

    @always_inline
    @parameter
    fn reduce_impl[
        ty: DType, width: Int
    ](v1: SIMD[ty, width], v2: SIMD[ty, width]) -> SIMD[ty, width]:
        return v1 + v2

    try:
        _reduce_generator[
            input_fn,
            output_fn,
            reduce_impl,
            single_thread_blocking_override = not parallelize,
        ](
            Index(M, K),
            init=Scalar[c_type](0),
            reduce_dim=1,
            out_chain=out_chain,
        )
    except e:
        trap(e)


fn naive_gemv[
    c_size: Dim,
    a_shape: DimList,
    b_size: Dim,
    type: DType,
](
    c_buf: Buffer[c_size, type],
    a_buf: NDBuffer[2, a_shape, type],
    b_buf: Buffer[b_size, type],
):
    let M = a_buf.dim[0]()
    let K = a_buf.dim[1]()

    c_buf.zero()
    for k in range(K):
        let b_val = b_buf[k]
        for m in range(M):
            let a_val = a_buf[m, k]
            c_buf[m] += a_val * b_val
