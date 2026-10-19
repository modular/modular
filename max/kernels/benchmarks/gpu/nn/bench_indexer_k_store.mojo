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
"""Benchmark paged index-key production for compression ratios one and two."""

from std.benchmark import Bench, BenchConfig, Bencher, BenchId
from std.math import ceildiv
from max.benchmark import bencher_iter_custom
from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
from nn.indexer_k_store import indexer_k_norm_rope_store


def bench_store[
    fp4: Bool, ratio: Int
](ctx: DeviceContext, mut bench: Bench, count: Int) raises:
    comptime page_size = 64
    comptime row_bytes = 68 if fp4 else 132
    var keys = ctx.enqueue_create_host_buffer[.bfloat16](count * 128)
    var positions = ctx.enqueue_create_host_buffer[.int64](count)
    var slots = ctx.enqueue_create_host_buffer[.int64](count)
    var frequencies = ctx.enqueue_create_host_buffer[.float32](
        count * ratio * 64
    )
    for row in range(count):
        positions[row] = Int64(row * ratio + ratio - 1)
        slots[row] = Int64(row)
        for d in range(128):
            keys[row * 128 + d] = (
                Float32((row * 17 + d * 13) % 97 - 48) / 16
            ).cast[.bfloat16]()
    for position in range(count * ratio):
        for d in range(32):
            frequencies[position * 64 + d] = 0.6
            frequencies[position * 64 + 32 + d] = 0.8
    var d_keys = ctx.enqueue_create_buffer[.bfloat16](len(keys))
    var d_positions = ctx.enqueue_create_buffer[.int64](count)
    var d_slots = ctx.enqueue_create_buffer[.int64](count)
    var d_frequencies = ctx.enqueue_create_buffer[.float32](len(frequencies))
    var d_weights = ctx.enqueue_create_buffer[.bfloat16](128)
    var d_cache = ctx.enqueue_create_buffer[.uint8](
        ceildiv(count, page_size) * page_size * row_bytes
    )
    ctx.enqueue_copy(dst_buf=d_keys, src_buf=keys)
    ctx.enqueue_copy(dst_buf=d_positions, src_buf=positions)
    ctx.enqueue_copy(dst_buf=d_slots, src_buf=slots)
    ctx.enqueue_copy(dst_buf=d_frequencies, src_buf=frequencies)
    d_weights.enqueue_fill(1)
    var pages = ceildiv(count, page_size)
    # Built outside the `{imm}` launch closure: the DeviceBuffer constructor
    # binds `ref[Self.origin]`, so a capture inside the closure would give the
    # cache tensor an immutable origin.
    var cache = TileTensor(d_cache, row_major(pages, page_size * row_bytes))

    def launch(c: DeviceContext) raises {imm}:
        indexer_k_norm_rope_store[fp4, ratio](
            TileTensor(d_keys, row_major(count, 128)),
            TileTensor(d_positions, row_major(count)),
            TileTensor(d_frequencies, row_major(count * ratio, 64)),
            TileTensor(d_weights, row_major(128)),
            TileTensor(d_slots, row_major(count)),
            cache,
            page_size,
            1e-20,
            c,
        )

    def measure(mut b: Bencher) raises {imm}:
        bencher_iter_custom(b, launch, ctx)

    bench.bench_function(
        measure,
        BenchId(
            "indexer_k_store",
            input_id=String("fp4=", fp4, "/ratio=", ratio, "/tokens=", count),
        ),
    )
    ctx.synchronize()


def main() raises:
    var ctx = DeviceContext()
    var bench = Bench(BenchConfig(max_iters=10000))
    comptime for mode in range(2):
        comptime for ratio in range(1, 3):
            for count in [1, 8, 64, 65, 256, 1024]:
                bench_store[Bool(mode), ratio](ctx, bench, count)
    bench.dump_report()
