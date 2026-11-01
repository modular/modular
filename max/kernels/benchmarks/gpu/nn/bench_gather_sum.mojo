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

from std.sys import get_defined_dtype, get_defined_int, size_of

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
from internal_utils import get_defined_shape
from layout import Coord, TileTensor, row_major
from nn.gather_scatter import gather_sum

from std.utils.index import IndexList


def bench_gather_sum_gpu[
    dtype: DType,
    batch: Int,
    multi_hot: Int,
    table_size: Int,
    embedding_dim: Int,
](ctx: DeviceContext, mut b: Bench) raises:
    var input_device = ctx.enqueue_create_buffer[dtype](
        table_size * embedding_dim
    )
    var output_device = ctx.enqueue_create_buffer[dtype](batch * embedding_dim)
    # Multi-hot indices in [0, table_size), one row per batch entry.
    var indices_host = ctx.enqueue_create_host_buffer[.int32](batch * multi_hot)
    for i in range(batch * multi_hot):
        indices_host[i] = Int32(i % table_size)
    var indices_device = ctx.enqueue_create_buffer[.int32](batch * multi_hot)
    ctx.enqueue_copy(indices_device, indices_host)
    ctx.enqueue_memset(input_device, 1)
    ctx.synchronize()

    var input = TileTensor(
        input_device, row_major(Coord(table_size, embedding_dim))
    )
    var output = TileTensor(
        output_device, row_major(Coord(batch, embedding_dim))
    )
    var indices = TileTensor(indices_device, row_major(Coord(batch, multi_hot)))

    @inline(.always)
    def bench_fn(
        mut b: Bencher,
    ) raises {mut output, imm input, imm indices, imm ctx}:
        @inline(.always)
        def kernel_launch(
            ctx: DeviceContext,
        ) raises {mut output, imm input, imm indices,}:
            gather_sum[target="gpu"](output, input, indices, ctx)

        bencher_iter_custom(b, kernel_launch, ctx)

    # Lower bound on memory traffic: every gathered row is read once and
    # every output element is written once; the int32 indices ride along.
    var total_bytes = (
        batch * multi_hot * (embedding_dim * size_of[dtype]() + 4)
        + batch * embedding_dim * size_of[dtype]()
    )

    b.bench_function(
        bench_fn,
        BenchId(
            "gather_sum",
            input_id=String(
                dtype,
                "/",
                batch,
                "x",
                multi_hot,
                "@",
                table_size,
                "x",
                embedding_dim,
            ),
        ),
        [ThroughputMeasure(BenchMetric.bytes, total_bytes)],
    )

    ctx.synchronize()

    _ = input_device
    _ = output_device
    _ = indices_device


def main() raises:
    comptime dtype = get_defined_dtype["dtype", .float32]()
    comptime table_size = get_defined_int["table_size", 100000]()
    comptime embedding_dim = get_defined_int["embedding_dim", 64]()
    comptime batch = get_defined_int["batch", 1024]()
    comptime multi_hot = get_defined_int["multi_hot", 100]()

    var m = Bench(BenchConfig(num_repetitions=1))
    with DeviceContext() as ctx:
        bench_gather_sum_gpu[
            dtype, batch, multi_hot, table_size, embedding_dim
        ](ctx, m)
