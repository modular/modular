# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
from layout._coord import Coord, Idx, coord_to_index_list
from layout._layout import row_major
from layout._tile_tensor import TileTensor
from nn.normalization import layer_norm_gpu, rms_norm_gpu

from utils.index import Index, IndexList


fn bench_layer_norm_gpu[
    rank: Int, //, dtype: DType, shape: IndexList[rank]
](ctx: DeviceContext, mut b: Bench, fn_name: String) raises:
    comptime cols = shape[rank - 1]
    comptime rows = shape.flattened_length() // cols

    var data_h = alloc[Scalar[dtype]](rows * cols)
    var res = alloc[Scalar[dtype]](rows * cols)
    var gamma_h = alloc[Scalar[dtype]](cols)
    var beta_h = alloc[Scalar[dtype]](cols)

    for i in range(rows * cols):
        var val = Scalar[dtype](random_float64(0, 100).cast[dtype]())
        data_h[i] = val

    for i in range(cols):
        gamma_h[i] = (Float64(i + cols) / Float64(cols)).cast[dtype]()
        beta_h[i] = (Float64(i) / Float64(cols)).cast[dtype]()

    var data_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var gamma_d = ctx.enqueue_create_buffer[dtype](cols)
    var beta_d = ctx.enqueue_create_buffer[dtype](cols)

    var param_shape = Index(cols)

    var data_buf = TileTensor(data_d, row_major(Coord(shape)))
    var gamma = TileTensor(gamma_d, row_major(Coord(param_shape)))
    var beta = TileTensor(beta_d, row_major(Coord(param_shape)))
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
        var idx = data_buf.layout(Coord(coords))

        return data_buf.ptr.load[width=width](idx)

    @__copy_capture(gamma)
    @always_inline
    @parameter
    fn gamma_fn[
        width: Int, rank: Int
    ](coords: IndexList[rank]) -> SIMD[dtype, width]:
        var idx = gamma.layout(Idx(coords[0]))

        return gamma.ptr.load[width=width](idx)

    @always_inline
    @__copy_capture(beta)
    @parameter
    fn output_fn[
        width: Int, rank_: Int, alignment: Int
    ](coords: IndexList[rank_], val: SIMD[dtype, width]) -> None:
        var idx = data_buf.layout(Coord(coords))

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
    rank: Int, //, dtype: DType, shape: IndexList[rank]
](ctx: DeviceContext, mut b: Bench, fn_name: String) raises:
    comptime cols = shape[rank - 1]
    comptime rows = shape.flattened_length() // cols

    var data_h = alloc[Scalar[dtype]](rows * cols)
    var res = alloc[Scalar[dtype]](rows * cols)
    var gamma_h = alloc[Scalar[dtype]](cols)

    for i in range(rows * cols):
        var val = Scalar[dtype](random_float64(0, 100).cast[dtype]())
        data_h[i] = val

    for i in range(cols):
        gamma_h[i] = (Float64(i + cols) / Float64(cols)).cast[dtype]()

    var data_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var gamma_d = ctx.enqueue_create_buffer[dtype](cols)

    var param_shape = Index(cols)

    var data_buf = TileTensor(data_d, row_major(Coord(shape)))
    var gamma = TileTensor(gamma_d, row_major(Coord(param_shape)))
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
        var idx = data_buf.layout(Coord(coords))

        return data_buf.ptr.load[width=width](idx)

    @always_inline
    @__copy_capture(data_buf)
    @parameter
    fn identity_output_fn[
        width: Int, alignment: Int
    ](coords: IndexList[rank], val: SIMD[dtype, width]) -> None:
        var idx = data_buf.layout(Coord(coords))
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
    comptime dtype = env_get_dtype["dtype", DType.bfloat16]()
    comptime shape = int_list_to_tuple[env_get_shape["shape", "256x256"]()]()

    var m = Bench(BenchConfig(num_repetitions=1))
    with DeviceContext() as ctx:
        comptime if len(shape) == 2:
            bench_layer_norm_gpu[dtype, shape](ctx, m, "layer_norm_gpu")
        elif len(shape) == 3:
            bench_rms_norm_gpu[dtype, shape](ctx, m, "rms_norm_gpu")

    m.dump_report()
