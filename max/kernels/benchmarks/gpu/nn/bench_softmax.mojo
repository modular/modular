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
from std.sys import (
    is_defined,
    get_defined_dtype,
    get_defined_string,
    simd_width_of,
)

from std.benchmark import Bench, BenchConfig, Bencher, BenchId
from std.gpu.host import DeviceContext
from internal_utils import get_defined_shape, int_list_to_tuple
from layout import Coord, TileTensor, row_major
from nn.softmax import softmax, softmax_with_temperature

from std.utils.index import IndexList


def bench_softmax_gpu[
    rank: Int, //, dtype: DType, shape: IndexList[rank]
](ctx: DeviceContext, mut b: Bench, fn_name: String) raises:
    comptime cols = shape[rank - 1]
    comptime total = shape.flattened_length()

    var data_h = alloc[Scalar[dtype]](total)

    for i in range(total):
        data_h[i] = Scalar[dtype](random_float64(-1, 1).cast[dtype]())

    var data_d = ctx.enqueue_create_buffer[dtype](total)
    var out_d = ctx.enqueue_create_buffer[dtype](total)

    var data_buf = TileTensor(data_d, row_major(Coord(shape)))
    var out_buf = TileTensor(out_d, row_major(Coord(shape)))

    ctx.enqueue_copy(data_d, data_h)

    @always_inline
    @__copy_capture(shape, data_buf, out_buf)
    @parameter
    def bench_fn(mut b: Bencher) raises:
        @parameter
        @always_inline
        def kernel_launch(ctx: DeviceContext) raises:
            softmax[dtype, simd_width_of[dtype](), rank](
                data_buf, out_buf, rank - 1
            )

        b.iter_custom[kernel_launch](ctx)

    b.bench_function[bench_fn](
        BenchId("softmax", input_id=String(fn_name, "/", dtype, "/", shape))
    )

    ctx.synchronize()

    _ = data_d
    _ = out_d

    data_h.free()


def bench_softmax_with_temperature_gpu[
    dtype: DType, shape: IndexList
](
    ctx: DeviceContext,
    mut b: Bench,
    fn_name: String,
    temperature: Float32,
) raises:
    comptime total = shape.flattened_length()
    comptime rows = shape[0]

    var data_h = alloc[Scalar[dtype]](total)
    for i in range(total):
        data_h[i] = Scalar[dtype](random_float64(-1, 1).cast[dtype]())

    var data_d = ctx.enqueue_create_buffer[dtype](total)
    var out_d = ctx.enqueue_create_buffer[dtype](total)

    var data_buf = TileTensor(data_d, row_major(Coord(shape)))
    var out_buf = TileTensor(out_d, row_major(Coord(shape)))

    ctx.enqueue_copy(data_d, data_h)

    var temp = temperature

    @always_inline
    @__copy_capture(data_buf, out_buf, temp)
    @parameter
    def bench_fn(mut b: Bencher) raises:
        @parameter
        @always_inline
        def kernel_launch(ctx: DeviceContext) raises:
            softmax_with_temperature(ctx, data_buf, out_buf, temp)

        b.iter_custom[kernel_launch](ctx)

    b.bench_function[bench_fn](
        BenchId(
            "softmax_with_temperature",
            input_id=String(fn_name, "/", dtype, "/", shape, "/T=", temp),
        )
    )

    ctx.synchronize()

    _ = data_d
    _ = out_d

    data_h.free()


def main() raises:
    comptime dtype = get_defined_dtype["dtype", DType.bfloat16]()
    comptime shape = int_list_to_tuple[
        get_defined_shape["shape", "256x256"]()
    ]()
    var m = Bench(BenchConfig(num_repetitions=1))
    with DeviceContext() as ctx:
        comptime if is_defined["temperature"]():
            var temperature = Float32(
                atof(get_defined_string["temperature", "1.0"]())
            )
            bench_softmax_with_temperature_gpu[dtype, shape](
                ctx, m, "softmax_with_temperature_gpu", temperature
            )
        elif len(shape) == 3:
            bench_softmax_gpu[dtype, shape](ctx, m, "softmax_gpu")

    m.dump_report()
