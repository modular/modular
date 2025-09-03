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

from random import random_float64
from sys import env_get_dtype

from benchmark import Bench, BenchConfig, Bencher, BenchId
from gpu.host import DeviceContext
from internal_utils import env_get_shape, int_list_to_tuple
from layout import (
    Layout,
    LayoutTensor,
    RuntimeLayout,
    RuntimeTuple,
    UNKNOWN_VALUE,
)
from layout.int_tuple import fill_like
from nn.normalization import layer_norm_gpu, rms_norm_gpu

from utils.index import IndexList


fn bench_layer_norm_gpu[
    dtype: DType, rank: Int
](
    ctx: DeviceContext,
    mut b: Bench,
    fn_name: String,
    shape: IndexList[rank],
) raises:
    var cols = shape[rank - 1]
    var rows = shape.flattened_length() // cols

    var data_h = UnsafePointer[Scalar[dtype]].alloc(rows * cols)
    var res = UnsafePointer[Scalar[dtype]].alloc(rows * cols)
    var gamma_h = UnsafePointer[Scalar[dtype]].alloc(cols)
    var beta_h = UnsafePointer[Scalar[dtype]].alloc(cols)

    for i in range(rows * cols):
        var val = Scalar[dtype](random_float64(0, 100).cast[dtype]())
        data_h[i] = val

    for i in range(cols):
        gamma_h[i] = ((i + cols) / cols).cast[dtype]()
        beta_h[i] = (i / cols).cast[dtype]()

    var data_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var gamma_d = ctx.enqueue_create_buffer[dtype](cols)
    var beta_d = ctx.enqueue_create_buffer[dtype](cols)

    var param_shape = IndexList[1](cols)

    alias layout = Layout.row_major[rank]()
    alias layout_1d = Layout.row_major(UNKNOWN_VALUE)
    var data_buf = LayoutTensor[dtype, layout](
        data_d.unsafe_ptr(), RuntimeLayout[layout].row_major(shape)
    )
    var gamma = LayoutTensor[dtype, layout_1d](
        gamma_d.unsafe_ptr(), RuntimeLayout[layout_1d].row_major(param_shape)
    )
    var beta = LayoutTensor[dtype, layout_1d](
        beta_d.unsafe_ptr(), RuntimeLayout[layout_1d].row_major(param_shape)
    )
    var epsilon = Scalar[dtype]()

    ctx.enqueue_copy(data_d, data_h)
    ctx.enqueue_copy(gamma_d, gamma_h)
    ctx.enqueue_copy(beta_d, beta_h)

    @__copy_capture(data_buf)
    @always_inline
    @parameter
    fn input_fn[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
        var idx = data_buf.runtime_layout(
            RuntimeTuple[fill_like(data_buf.layout.shape, UNKNOWN_VALUE)](
                coords
            )
        )

        return data_buf.ptr.load[width=width](idx)

    @__copy_capture(gamma)
    @always_inline
    @parameter
    fn gamma_fn[
        width: Int, rank: Int
    ](coords: IndexList[rank]) -> SIMD[dtype, width]:
        var idx = gamma.runtime_layout(
            RuntimeTuple[fill_like(gamma.layout.shape, UNKNOWN_VALUE)](
                coords[0]
            )
        )

        return gamma.ptr.load[width=width](idx)

    @always_inline
    @__copy_capture(beta)
    @parameter
    fn output_fn[
        width: Int, rank_: Int, alignment: Int
    ](coords: IndexList[rank_], val: SIMD[dtype, width]) -> None:
        var idx = data_buf.runtime_layout(
            RuntimeTuple[fill_like(data_buf.layout.shape, UNKNOWN_VALUE)](
                coords
            )
        )

        data_buf.ptr.store[width=width, alignment=alignment](idx, val)

    @always_inline
    @__copy_capture(shape, beta, epsilon, data_buf)
    @parameter
    fn bench_fn(mut b: Bencher) raises:
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            layer_norm_gpu[input_fn, gamma_fn, output_fn](
                shape, beta, epsilon, ctx=ctx
            )

        b.iter_custom[kernel_launch](ctx)

    b.bench_function[bench_fn](
        BenchId("layer_norm", input_id=String(fn_name, dtype, shape, sep="/"))
    )

    ctx.synchronize()

    _ = data_d
    _ = gamma_d
    _ = beta_d

    data_h.free()
    res.free()
    gamma_h.free()
    beta_h.free()


fn bench_rms_norm_gpu[
    dtype: DType, rank: Int
](
    ctx: DeviceContext,
    mut b: Bench,
    fn_name: String,
    shape: IndexList[rank],
) raises:
    var cols = shape[rank - 1]
    var rows = shape.flattened_length() // cols

    var data_h = UnsafePointer[Scalar[dtype]].alloc(rows * cols)
    var res = UnsafePointer[Scalar[dtype]].alloc(rows * cols)
    var gamma_h = UnsafePointer[Scalar[dtype]].alloc(cols)

    for i in range(rows * cols):
        var val = Scalar[dtype](random_float64(0, 100).cast[dtype]())
        data_h[i] = val

    for i in range(cols):
        gamma_h[i] = ((i + cols) / cols).cast[dtype]()

    var data_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var gamma_d = ctx.enqueue_create_buffer[dtype](cols)

    var param_shape = IndexList[1](cols)

    alias layout = Layout.row_major[rank]()
    alias layout_1d = Layout.row_major(UNKNOWN_VALUE)
    var data_buf = LayoutTensor[dtype, layout](
        data_d.unsafe_ptr(), RuntimeLayout[layout].row_major(shape)
    )
    var gamma = LayoutTensor[dtype, layout_1d](
        gamma_d.unsafe_ptr(), RuntimeLayout[layout_1d].row_major(param_shape)
    )
    var epsilon = Scalar[dtype](0.001)
    var weight_offset = Scalar[dtype](0.0)

    ctx.enqueue_copy(data_d, data_h)
    ctx.enqueue_copy(gamma_d, gamma_h)

    @__copy_capture(data_buf)
    @always_inline
    @parameter
    fn input_fn[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
        var idx = data_buf.runtime_layout(
            RuntimeTuple[fill_like(data_buf.layout.shape, UNKNOWN_VALUE)](
                coords
            )
        )

        return data_buf.ptr.load[width=width](idx)

    @always_inline
    @__copy_capture(data_buf)
    @parameter
    fn identity_output_fn[
        width: Int, alignment: Int
    ](coords: IndexList[rank], val: SIMD[dtype, width]) -> None:
        var idx = data_buf.runtime_layout(
            RuntimeTuple[fill_like(data_buf.layout.shape, UNKNOWN_VALUE)](
                coords
            )
        )
        data_buf.ptr.store[width=width, alignment=alignment](idx, val)

    @always_inline
    @__copy_capture(shape, gamma, epsilon, weight_offset)
    @parameter
    fn bench_fn(mut b: Bencher) raises:
        @parameter
        @always_inline
        fn kernel_launch(ctx: DeviceContext) raises:
            rms_norm_gpu[
                input_fn, identity_output_fn, multiply_before_cast=True
            ](shape, gamma, epsilon, weight_offset, ctx)

        b.iter_custom[kernel_launch](ctx)

    b.bench_function[bench_fn](
        BenchId("rms_norm", input_id=String(fn_name, "/", dtype, "/", shape)),
    )

    ctx.synchronize()

    _ = data_d
    _ = gamma_d

    data_h.free()
    res.free()
    gamma_h.free()


def main():
    alias dtype = env_get_dtype["dtype", DType.bfloat16]()
    alias shape = int_list_to_tuple[env_get_shape["shape", "256x256"]()]()

    var m = Bench(BenchConfig(num_repetitions=1))
    with DeviceContext() as ctx:

        @parameter
        if len(shape) == 2:
            bench_layer_norm_gpu[dtype](ctx, m, "layer_norm_gpu", shape)
        elif len(shape) == 3:
            bench_rms_norm_gpu[dtype](ctx, m, "rms_norm_gpu", shape)

    m.dump_report()
