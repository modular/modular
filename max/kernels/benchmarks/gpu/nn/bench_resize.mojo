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

"""Benchmarks the GPU resize kernels (nearest, linear and bicubic) on NCHW tensors.

Run through kbench with bench_resize.yaml, or directly, for example:

    mojo -D op=nearest -D dtype=bfloat16 -D in_shape=1x3x224x224 \
        -D out_shape=1x3x448x448 bench_resize.mojo
"""

from std.sys import get_defined_dtype, get_defined_string, size_of

from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.gpu.host import DeviceContext
from internal_utils import get_defined_shape, int_list_to_tuple
from layout import Coord, TileTensor, row_major
from nn.bicubic import resize_bicubic
from nn.resize import (
    CoordinateTransformationMode,
    RoundMode,
    resize_linear_gpu,
    resize_nearest_neighbor,
)

from std.utils.index import IndexList


def bench_resize_gpu[
    in_rank: Int,
    out_rank: Int,
    //,
    dtype: DType,
    op: StaticString,
    in_shape: IndexList[in_rank],
    out_shape: IndexList[out_rank],
](ctx: DeviceContext, mut b: Bench) raises:
    comptime assert in_rank == out_rank, "input and output ranks must match"

    var input_size = in_shape.flattened_length()
    var output_size = out_shape.flattened_length()

    var in_device = ctx.enqueue_create_buffer[dtype](input_size)
    var out_device = ctx.enqueue_create_buffer[dtype](output_size)
    ctx.enqueue_memset(in_device, 1)
    ctx.synchronize()

    var input = TileTensor(in_device, row_major(Coord(in_shape)))
    var output = TileTensor(out_device, row_major(Coord(out_shape)))

    @inline(.always)
    def bench_fn(mut b: Bencher) raises {mut output, imm input, imm ctx}:
        @inline(.always)
        def kernel_launch(ctx: DeviceContext) raises {mut output, imm input}:
            comptime if op == "nearest":
                resize_nearest_neighbor[
                    CoordinateTransformationMode.HalfPixel,
                    RoundMode.HalfDown,
                    target="gpu",
                ](input, output, ctx)
            elif op == "linear":
                resize_linear_gpu[
                    CoordinateTransformationMode.HalfPixel, False
                ](input, output, ctx)
            else:
                resize_bicubic[target="gpu"](output, input, ctx)

        bencher_iter_custom(b, kernel_launch, ctx)

    # Lower bound on memory traffic: read every input element once and write
    # every output element once.
    var total_bytes = (input_size + output_size) * size_of[dtype]()

    b.bench_function(
        bench_fn,
        BenchId(
            "resize_" + String(op),
            input_id=String(dtype, "/", in_shape, "->", out_shape),
        ),
        [ThroughputMeasure(BenchMetric.bytes, total_bytes)],
    )

    ctx.synchronize()

    _ = in_device
    _ = out_device


def main() raises:
    comptime dtype = get_defined_dtype["dtype", .float32]()
    comptime op = get_defined_string["op", "nearest"]()
    comptime in_shape = int_list_to_tuple[
        get_defined_shape["in_shape", "1x3x224x224"]()
    ]()
    comptime out_shape = int_list_to_tuple[
        get_defined_shape["out_shape", "1x3x448x448"]()
    ]()

    var m = Bench(BenchConfig(num_repetitions=1))
    with DeviceContext() as ctx:
        bench_resize_gpu[dtype, op, in_shape, out_shape](ctx, m)

    m.dump_report()
