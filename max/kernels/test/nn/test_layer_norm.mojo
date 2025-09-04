# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from math import isqrt

from layout.math import mean, variance
from layout import (
    Layout,
    LayoutTensor,
    RuntimeLayout,
    RuntimeTuple,
    UNKNOWN_VALUE,
)
from layout.int_tuple import fill_like
from nn.normalization import *
from testing import assert_almost_equal

from utils.index import Index, IndexList

alias layout_1d = Layout.row_major(UNKNOWN_VALUE)


fn run_layer_norm_cpu[
    dtype: DType, rank: Int
](shape: IndexList[rank], rtol: Float64 = 0.01) raises:
    alias layout = Layout.row_major[rank]()
    var cols = shape[rank - 1]
    var rows = shape.flattened_length() // cols

    var input_ptr = UnsafePointer[Scalar[dtype]].alloc(rows * cols)
    var output_ptr = UnsafePointer[Scalar[dtype]].alloc(rows * cols)
    var gamma_ptr = UnsafePointer[Scalar[dtype]].alloc(cols)
    var beta_ptr = UnsafePointer[Scalar[dtype]].alloc(cols)

    for i in range(rows * cols):
        var val = Scalar[dtype](i)
        input_ptr[i] = val

    for i in range(cols):
        gamma_ptr[i] = ((i + cols) / cols).cast[dtype]()
        beta_ptr[i] = (i / cols).cast[dtype]()

    var param_shape = IndexList[1](cols)

    var input_buf = LayoutTensor[dtype, layout](
        input_ptr, RuntimeLayout[layout].row_major(shape)
    )
    var output_buf = LayoutTensor[dtype, layout](
        output_ptr, RuntimeLayout[layout].row_major(shape)
    )
    var gamma = LayoutTensor[dtype, layout_1d](
        gamma_ptr, RuntimeLayout[layout_1d].row_major(param_shape)
    )
    var beta = LayoutTensor[dtype, layout_1d](
        beta_ptr, RuntimeLayout[layout_1d].row_major(param_shape)
    )
    var epsilon = Scalar[dtype](0.0001)

    @__copy_capture(input_buf)
    @always_inline
    @parameter
    fn input_fn[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
        var idx = input_buf.runtime_layout(
            RuntimeTuple[fill_like(input_buf.layout.shape, UNKNOWN_VALUE)](
                coords
            )
        )
        return input_buf.ptr.load[width=width](idx)

    @__copy_capture(gamma)
    @always_inline
    @parameter
    fn gamma_fn[
        width: Int, rank: Int
    ](coords: IndexList[rank]) -> SIMD[dtype, width]:
        var idx = gamma.runtime_layout(
            RuntimeTuple[IntTuple(UNKNOWN_VALUE)](IndexList[1](coords[0]))
        )
        return gamma.ptr.load[width=width](idx)

    @__copy_capture(output_buf)
    @always_inline
    @parameter
    fn output_fn[
        width: Int, _rank: Int, alignment: Int
    ](coords: IndexList[_rank], val: SIMD[dtype, width]):
        var idx = output_buf.runtime_layout(
            RuntimeTuple[fill_like(output_buf.layout.shape, UNKNOWN_VALUE)](
                coords
            )
        )
        output_buf.ptr.store[width=width, alignment=alignment](
            idx, rebind[SIMD[dtype, width]](val)
        )

    layer_norm_cpu[input_fn, gamma_fn, output_fn](shape, beta, epsilon)

    for r in range(rows):
        var vec = LayoutTensor[dtype, layout_1d](
            input_ptr + r * cols,
            RuntimeLayout[layout_1d].row_major(IndexList[1](cols)),
        )
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
    print("0")
    run_layer_norm_cpu[DType.float32](Index(3, 5))
    print("1")
    run_layer_norm_cpu[DType.float32](Index(3, 8))
    print("2")
    run_layer_norm_cpu[DType.float32](Index(7, 33))
    print("3")
    run_layer_norm_cpu[DType.float32](Index(1, 1024))
    print("4")
    run_layer_norm_cpu[DType.float32](Index(1, 8192))

    # variable rank
    print("5")
    run_layer_norm_cpu[DType.float32](Index(0))
    print("6")
    run_layer_norm_cpu[DType.float32](Index(5))
    print("7")
    run_layer_norm_cpu[DType.float32](Index(3, 4, 10, 20, 8))
    print("8")
    run_layer_norm_cpu[DType.float32](Index(1, 5, 6, 10, 128))
