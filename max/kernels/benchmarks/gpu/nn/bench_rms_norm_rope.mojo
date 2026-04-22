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

from std.random import random_float64
from std.sys import get_defined_dtype

from std.benchmark import Bench, BenchConfig, Bencher, BenchId
from std.gpu.host import DeviceContext
from internal_utils import get_defined_shape, int_list_to_tuple
from layout import Coord, Idx, TileTensor, row_major
from nn.normalization import rms_norm_rope_gpu

from std.utils.index import Index, IndexList


def bench_rms_norm_rope_gpu[
    rank: Int, //, dtype: DType, shape: IndexList[rank]
](ctx: DeviceContext, mut b: Bench, fn_name: String) raises:
    comptime cols = shape[rank - 1]
    comptime rows = shape.flattened_length() // cols

    var data_h = alloc[Scalar[dtype]](rows * cols)
    var gamma_h = alloc[Scalar[dtype]](cols)
    var cos_h = alloc[Scalar[dtype]](rows * cols)
    var sin_h = alloc[Scalar[dtype]](rows * cols)

    for i in range(rows * cols):
        data_h[i] = Scalar[dtype](random_float64(-1, 1).cast[dtype]())
        cos_h[i] = Scalar[dtype](random_float64(-1, 1).cast[dtype]())
        sin_h[i] = Scalar[dtype](random_float64(-1, 1).cast[dtype]())

    for i in range(cols):
        gamma_h[i] = (Float64(i + cols) / Float64(cols)).cast[dtype]()

    var data_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var gamma_d = ctx.enqueue_create_buffer[dtype](cols)
    var cos_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var sin_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var output_d = ctx.enqueue_create_buffer[dtype](rows * cols)

    var param_shape = Index(cols)

    var data_buf = TileTensor(data_d, row_major(Coord(shape)))
    var output_buf = TileTensor(output_d, row_major(Coord(shape)))
    var gamma = TileTensor(gamma_d, row_major(Coord(param_shape)))
    var cos_vals = TileTensor(cos_d, row_major(Coord(shape)))
    var sin_vals = TileTensor(sin_d, row_major(Coord(shape)))
    var epsilon = Scalar[dtype](0.001)
    var weight_offset = Scalar[dtype](0.0)

    ctx.enqueue_copy(data_d, data_h)
    ctx.enqueue_copy(gamma_d, gamma_h)
    ctx.enqueue_copy(cos_d, cos_h)
    ctx.enqueue_copy(sin_d, sin_h)

    @__copy_capture(data_buf)
    @always_inline
    @parameter
    def input_fn[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
        var idx = data_buf.layout(Coord(coords))
        return data_buf.raw_load[width=width](idx)

    @__copy_capture(cos_vals)
    @always_inline
    @parameter
    def cos_fn[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
        var idx = cos_vals.layout(Coord(coords))
        return cos_vals.raw_load[width=width](idx)

    @__copy_capture(sin_vals)
    @always_inline
    @parameter
    def sin_fn[
        width: Int, _rank: Int
    ](coords: IndexList[_rank]) -> SIMD[dtype, width]:
        var idx = sin_vals.layout(Coord(coords))
        return sin_vals.raw_load[width=width](idx)

    @always_inline
    @__copy_capture(output_buf)
    @parameter
    def output_fn[
        width: Int, alignment: Int
    ](coords: IndexList[rank], val: SIMD[dtype, width]) -> None:
        var idx = output_buf.layout(Coord(coords))
        output_buf.raw_store[width=width, alignment=alignment](idx, val)

    @always_inline
    @__copy_capture(shape, gamma, epsilon, weight_offset, cos_vals, sin_vals)
    @parameter
    def bench_fn(mut b: Bencher) raises:
        @parameter
        @always_inline
        def kernel_launch(ctx: DeviceContext) raises:
            rms_norm_rope_gpu[
                input_fn, cos_fn, sin_fn, output_fn, multiply_before_cast=False
            ](shape, gamma, epsilon, weight_offset, cos_vals, sin_vals, ctx)

        b.iter_custom[kernel_launch](ctx)

    b.bench_function[bench_fn](
        BenchId(
            "rms_norm_rope",
            input_id=String(fn_name, "/", dtype, "/", shape),
        ),
    )

    ctx.synchronize()

    _ = data_d
    _ = gamma_d
    _ = cos_d
    _ = sin_d
    _ = output_d

    data_h.free()
    gamma_h.free()
    cos_h.free()
    sin_h.free()


def main() raises:
    comptime dtype = get_defined_dtype["dtype", DType.bfloat16]()
    comptime shape = int_list_to_tuple[
        get_defined_shape["shape", "32x2048x12x128"]()
    ]()

    var m = Bench(BenchConfig(num_repetitions=1))
    with DeviceContext() as ctx:
        bench_rms_norm_rope_gpu[dtype, shape](ctx, m, "rms_norm_rope_gpu")

    m.dump_report()
