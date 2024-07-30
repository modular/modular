# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import OptionalReg as Optional

from algorithm.reduction import _reduce_generator
from buffer import Buffer, NDBuffer
from buffer.dimlist import Dim, DimList

from utils import StaticIntTuple
from utils.index import Index

from .utils import elementwise_epilogue_type

# Parallelized version of Gemv


@always_inline
fn gemv[
    parallelize: Bool,
    c_size: Dim,
    c_type: DType,
    a_shape: DimList,
    a_type: DType,
    b_size: Dim,
    b_type: DType,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c_buf: Buffer[c_type, c_size],
    a_buf: NDBuffer[a_type, 2, a_shape],
    b_buf: Buffer[b_type, b_size],
):
    alias simd_width = simdwidthof[c_type]()

    var M = a_buf.dim[0]()
    var K = a_buf.dim[1]()

    @always_inline
    @parameter
    fn input_fn[
        type: DType, width: Int, rank: Int
    ](idx: StaticIntTuple[rank]) -> SIMD[type, width]:
        return (
            a_buf.load[width=width]((idx[0], idx[1])).cast[type]()
            * b_buf.load[width=width](idx[1]).cast[type]()
        ).cast[type]()

    @always_inline
    @parameter
    fn output_fn[
        out_type: DType, width: Int, rank: Int
    ](idx: StaticIntTuple[rank], value: SIMD[out_type, width]):
        @parameter
        if elementwise_lambda_fn:
            alias func = elementwise_lambda_fn.value()

            @parameter
            for i in range(width):
                func[out_type, 1]((idx[0] + i, 0), value[i])
        else:
            c_buf.store[width=width](idx[0], value.cast[c_type]())

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
        )

    except e:
        abort(e)


fn naive_gemv[
    c_size: Dim,
    a_shape: DimList,
    b_size: Dim,
    type: DType,
](
    c_buf: Buffer[type, c_size],
    a_buf: NDBuffer[type, 2, a_shape],
    b_buf: Buffer[type, b_size],
):
    var M = a_buf.dim[0]()
    var K = a_buf.dim[1]()

    c_buf.zero()
    for k in range(K):
        var b_val = b_buf[k]
        for m in range(M):
            var a_val = a_buf[m, k]
            c_buf[m] += a_val * b_val
