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
"""Benchmarks the mHC stream write-back `mhc_post` on cold buffers.

`mhc_post` switches to non-temporal loads and stores from one token per CU on
AMD (256 on MI355X) and past `8 * SMs` tokens on NVIDIA (1184 on B200). Each
run times both forced arms by default (`--arms=01`) so the threshold can be
checked at any token count; the yaml sweeps both sides of it.
"""

from std.sys import get_defined_dtype, size_of

from internal_utils import CacheBustingBuffer, arg_parse
from internal_utils._cache_busting import CACHE_BUST_BYTES
from internal_utils._utils import InitializationType
from layout import TileTensor, row_major
from max.benchmark import bencher_iter_custom
from max.gpu.host import DeviceContext
from nn.mhc import mhc_post
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)


def bench_mhc_post[
    dtype: DType, hc: Int, hidden: Int
](ctx: DeviceContext, mut m: Bench, num_rows: Int, non_temporal: Bool) raises:
    var n_streams = num_rows * hc * hidden
    var n_y = num_rows * hidden
    var n_post = num_rows * hc
    var n_comb = num_rows * hc * hc
    comptime align = 16 // size_of[dtype]()
    # One window of the largest tensor can exceed the default budget, which
    # would collapse the buffer to a single, cache-warm window.
    var budget = max(CACHE_BUST_BYTES, 2 * n_streams * size_of[dtype]())

    var cb_streams = CacheBustingBuffer[dtype](
        n_streams, align, ctx, budget_bytes=budget
    )
    var cb_out = CacheBustingBuffer[dtype](
        n_streams, align, ctx, budget_bytes=budget
    )
    var cb_y = CacheBustingBuffer[dtype](n_y, align, ctx)
    var cb_post = CacheBustingBuffer[.float32](n_post, 4, ctx)
    var cb_comb = CacheBustingBuffer[.float32](n_comb, 4, ctx)
    cb_streams.init_on_device(InitializationType.uniform_distribution, ctx)
    cb_y.init_on_device(InitializationType.uniform_distribution, ctx)
    cb_post.init_on_device(InitializationType.uniform_distribution, ctx)
    cb_comb.init_on_device(InitializationType.uniform_distribution, ctx)

    var streams_shape = row_major((num_rows, hc, hidden))

    def launch(
        ctx: DeviceContext, iteration: Int
    ) raises {
        mut cb_streams, mut cb_out, mut cb_y, mut cb_post, mut cb_comb, imm
    }:
        mhc_post[hc_mult=hc, hidden_size=hidden, target="gpu"](
            TileTensor(cb_out.offset_ptr(iteration), streams_shape),
            TileTensor(
                cb_streams.offset_ptr(iteration), streams_shape
            ).as_imm(),
            TileTensor(
                cb_y.offset_ptr(iteration), row_major((num_rows, hidden))
            ).as_imm(),
            TileTensor(
                cb_post.offset_ptr(iteration), row_major((num_rows, hc))
            ).as_imm(),
            TileTensor(
                cb_comb.offset_ptr(iteration), row_major((num_rows, hc, hc))
            ).as_imm(),
            ctx,
            force_non_temporal=non_temporal,
        )

    @inline(.always)
    def bench_func(mut b: Bencher) raises {imm}:
        bencher_iter_custom(b, launch, ctx)

    var num_bytes = (2 * n_streams + n_y) * size_of[dtype]() + (
        n_post + n_comb
    ) * size_of[DType.float32]()
    m.bench_function(
        bench_func,
        BenchId(
            "mhc_post",
            input_id=String(
                "/T=",
                num_rows,
                "/hc=",
                hc,
                "/d=",
                hidden,
                "/nt=",
                Int(non_temporal),
            ),
        ),
        [ThroughputMeasure(BenchMetric.bytes, num_bytes)],
    )
    ctx.synchronize()

    _ = cb_streams
    _ = cb_out
    _ = cb_y
    _ = cb_post
    _ = cb_comb


def main() raises:
    var num_rows = Int(arg_parse("num_rows", 16384))
    # One digit per timed arm, in order: 0 forces non-temporal accesses off,
    # 1 forces them on. "00" or "11" measures the run-to-run noise floor.
    var arms = arg_parse("arms", "01")
    comptime dtype = get_defined_dtype["dtype", .bfloat16]()

    var m = Bench()
    with DeviceContext() as ctx:
        for arm in arms.codepoint_slices():
            var nt = String(arm)
            if nt != "0" and nt != "1":
                raise Error("--arms takes digits 0 and 1, got ", arms)
            bench_mhc_post[dtype, hc=4, hidden=4096](
                ctx, m, num_rows, non_temporal=nt == "1"
            )
    m.dump_report()
