# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from Buffer import Buffer, NDBuffer
from DType import DType
from Functional import vectorize_unroll
from List import Dim, DimList
from Range import range
from Functional import vectorize
from SIMD import SIMD
from Index import StaticIntTuple
from StaticTuple import StaticTuple
from TypeUtilities import rebind
from Reductions import mean, variance
from Math import sqrt


fn layer_norm[
    simd_width: Int,
    type: DType,
    input_fn: fn[mytype: DType, width: Int] (Int, Int) capturing -> SIMD[
        mytype, width
    ],
    shape: DimList,
    inner_dim: DimList,
](
    out_buf: NDBuffer[2, shape, type],
    gamma_buf: NDBuffer[1, inner_dim, type],
    beta_buf: NDBuffer[1, inner_dim, type],
    eps: SIMD[type, 1],
):
    """Computes layernorm(elementwise_fn(x)) across the last dimension of x, where layernorm is
    defined as $(x-mean(x))/(sqrt(var(x)+eps)*gamma + beta$.

    Currently performs 4 passes over the input data, which is NOT efficient, but
    this will be reduced to 2 in the future by fusing the add, mean, and variance
    loops using Welford's algorithm.

    Parameters:
        simd_width: The vector width for the computation.
        input_fn: Function called to generate an input value.
        shape: The x and out buffers' shape.
        type: The x and out buffers' elements dtype.

    Args:
        out_buf: The output buffer.
        gamma: The gamma value to use in the layernorm calculation.
        beta: The beta value to use in the layernorm calculation.
        eps: The eps value to use in the layernorm calculation.
    """

    let m = out_buf.dim[0]()
    let n = out_buf.dim[1]()  # contiguous

    for i in range(m):
        let start_coord = StaticIntTuple[2](i, 0)
        let out_slice = Buffer[shape.at[1](), type](
            out_buf._offset(start_coord), n
        )

        fn input_gen_wrapper[simd_width: Int](idx: Int):
            out_slice.simd_store[simd_width](
                idx, input_fn[type, simd_width](idx, i)
            )

        vectorize[simd_width, input_gen_wrapper](n)

        let mean_val = mean[simd_width](out_slice)
        let var_val = variance[simd_width](
            out_slice, mean_val, 0
        )  # use biased estimator

        let norm_factor = 1 / sqrt(var_val + eps)

        fn _normalize[simd_width: Int](idx: Int):
            let out_val = out_slice.simd_load[simd_width](idx)
            let norm_val = (
                out_val - mean_val
            ) * norm_factor * gamma_buf.simd_load[simd_width](
                idx
            ) + beta_buf.simd_load[
                simd_width
            ](
                idx
            )
            out_slice.simd_store(idx, norm_val)

        vectorize[simd_width, _normalize](n)
