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
"""Benchmark candidate expansion and sparse-column remapping."""

from std.benchmark import Bench, BenchConfig, Bencher, BenchId
from max.benchmark import bencher_iter_custom
from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from nn.indexer_candidates import (
    indexer_expand_candidates,
    indexer_remap_sparse_columns,
)


def bench_candidates[
    ratio: Int
](ctx: DeviceContext, mut bench: Bench, rows: Int) raises:
    comptime candidates = 2048
    comptime size = candidates * ratio
    comptime block_kv = 8
    comptime selected = 512
    var input = ctx.enqueue_create_host_buffer[.int32](rows * candidates)
    var columns = ctx.enqueue_create_host_buffer[.int32](rows * selected)
    for row in range(rows):
        for i in range(candidates):
            input[row * candidates + i] = Int32((i * 37 + 2026) % candidates)
        for i in range(selected):
            columns[row * selected + i] = Int32(
                (i * 13 + 2026) % (size * block_kv)
            )
    var d_input = ctx.enqueue_create_buffer[.int32](len(input))
    var d_columns = ctx.enqueue_create_buffer[.int32](len(columns))
    var d_starts = ctx.enqueue_create_buffer[.int32](rows)
    var d_stops = ctx.enqueue_create_buffer[.int32](rows)
    var d_blocks = ctx.enqueue_create_buffer[.int32](rows * size)
    var d_ends = ctx.enqueue_create_buffer[.int32](rows)
    var d_output = ctx.enqueue_create_buffer[.int32](rows * selected)
    ctx.enqueue_copy(dst_buf=d_input, src_buf=input)
    ctx.enqueue_copy(dst_buf=d_columns, src_buf=columns)
    d_starts.enqueue_fill(0)
    d_stops.enqueue_fill(Int32(size * block_kv))
    # Constructing a mutable TileTensor inside the launch closure below
    # would give it an immutable origin, so build the output tensors first.
    var blocks = TileTensor(d_blocks, row_major(rows, size))
    var ends = TileTensor(d_ends, row_major(rows, 1))
    var output = TileTensor(d_output, row_major(rows, selected))
    var blocks_imm = blocks.as_imm()

    def launch(c: DeviceContext) raises {imm}:
        indexer_expand_candidates[candidates, ratio, block_kv](
            TileTensor(d_input, row_major(rows, candidates)),
            TileTensor(d_starts, row_major(rows, 1)),
            TileTensor(d_stops, row_major(rows, 1)),
            blocks,
            ends,
            c,
        )
        indexer_remap_sparse_columns[block_kv](
            TileTensor(d_columns, row_major(rows, selected)),
            blocks_imm,
            TileTensor(d_starts, row_major(rows, 1)),
            output,
            selected,
            size * block_kv,
            c,
        )

    def measure(mut b: Bencher) raises {imm}:
        bencher_iter_custom(b, launch, ctx)

    bench.bench_function(
        measure,
        BenchId(
            "indexer_candidates",
            input_id=String("ratio=", ratio, "/rows=", rows),
        ),
    )
    ctx.synchronize()


def main() raises:
    var ctx = DeviceContext()
    var bench = Bench(BenchConfig(max_iters=10000))
    comptime for ratio in range(1, 3):
        for rows in [1, 8, 64, 65, 256]:
            bench_candidates[ratio](ctx, bench, rows)
    bench.dump_report()
