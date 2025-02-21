# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from math import ceildiv, isqrt
from sys import simdwidthof

from algorithm import mean, variance
from buffer import NDBuffer
from buffer.dimlist import DimList
from memory import UnsafePointer
from nn.normalization import *
from testing import assert_almost_equal

from utils.index import Index, IndexList


fn run_layer_norm_cpu[
    type: DType, rank: Int
](shape: IndexList[rank], rtol: Scalar[type] = 0.01) raises:
    var cols = shape[rank - 1]
    var rows = shape.flattened_length() // cols

    var input_ptr = UnsafePointer[Scalar[type]].alloc(rows * cols)
    var output_ptr = UnsafePointer[Scalar[type]].alloc(rows * cols)
    var gamma_ptr = UnsafePointer[Scalar[type]].alloc(cols)
    var beta_ptr = UnsafePointer[Scalar[type]].alloc(cols)

    for i in range(rows * cols):
        var val = Scalar[type](i)
        input_ptr[i] = val

    for i in range(cols):
        gamma_ptr[i] = ((i + cols) / cols).cast[type]()
        beta_ptr[i] = (i / cols).cast[type]()

    var param_shape = IndexList[1](cols)

    var input_buf = NDBuffer[type, rank](input_ptr, shape)
    var output_buf = NDBuffer[type, rank](output_ptr, shape)
    var gamma = NDBuffer[type, 1](gamma_ptr, param_shape)
    var beta = NDBuffer[type, 1](beta_ptr, param_shape)
    var epsilon = Scalar[type](0.0001)

    @__copy_capture(input_buf)
    @always_inline
    @parameter
    fn input_fn[
        width: Int, _rank: Int
    ](idx: IndexList[_rank]) -> SIMD[type, width]:
        return input_buf.load[width=width](rebind[IndexList[rank]](idx))

    @__copy_capture(gamma)
    @always_inline
    @parameter
    fn gamma_fn[
        width: Int, rank: Int
    ](idx: IndexList[rank]) -> SIMD[type, width]:
        return gamma.load[width=width](idx[0])

    layer_norm_cpu[input_fn, gamma_fn](shape, beta, epsilon, output_buf)

    for r in range(rows):
        var vec = NDBuffer[type, 1](input_ptr + r * cols, cols)
        var mean_ref = mean(vec)
        var var_ref = variance(vec, correction=0)
        var norm_factor_ref = isqrt(var_ref + epsilon)
        for c in range(cols):
            var idx = r * cols + c
            var val = (
                (input_ptr[idx] - mean_ref) * norm_factor_ref
            ) * gamma_ptr[c] + beta_ptr[c]
            assert_almost_equal(val, output_ptr[idx], rtol=rtol)

    input_ptr.free()
    output_ptr.free()
    gamma_ptr.free()
    beta_ptr.free()


def main():
    run_layer_norm_cpu[DType.float32](Index(3, 5))
    run_layer_norm_cpu[DType.float32](Index(3, 8))
    run_layer_norm_cpu[DType.float32](Index(7, 33))
    run_layer_norm_cpu[DType.float32](Index(1, 1024))
    run_layer_norm_cpu[DType.float32](Index(1, 8192))

    # variable rank
    run_layer_norm_cpu[DType.float32](Index(0))
    run_layer_norm_cpu[DType.float32](Index(5))
    run_layer_norm_cpu[DType.float32](Index(3, 4, 10, 20, 8))
    run_layer_norm_cpu[DType.float32](Index(1, 5, 6, 10, 128))
