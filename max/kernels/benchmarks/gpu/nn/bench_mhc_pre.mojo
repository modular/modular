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
"""Benchmarks the fused mHC pre site `mhc_pre` on cold buffers.

`mhc_pre` splits K across `KS` blocks per 16 tokens when there are too few
tokens to fill the GPU, with a second launch that combines the slices.
`--force_k_splits` pins `KS` (0 keeps the default), and each result's name
carries the launch config `mhc_pre_launch_config` returned, so a change to
the heuristic shows up in the output. Bytes moved count the streams and
`hc_fn` read once plus the outputs; the K-split workspace is not counted.
"""

from std.sys import get_defined_bool, get_defined_dtype, get_defined_int
from std.sys import size_of
from std.utils.numerics import isfinite

from internal_utils import CacheBustingBuffer, arg_parse
from internal_utils._cache_busting import CACHE_BUST_BYTES
from internal_utils._utils import InitializationType
from layout import Idx, TileTensor, row_major
from max.benchmark import bencher_iter_custom
from max.gpu.host import DeviceContext
from nn.mhc import mhc_pre, mhc_pre_launch_config
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)


def bench_mhc_pre[
    fn_dtype: DType, hidden: Int, has_norm: Bool
](ctx: DeviceContext, mut m: Bench, num_rows: Int, force_k_splits: Int) raises:
    comptime hc = 4
    comptime mix_width = 2 * hc + hc * hc
    comptime sinkhorn_iters = 20
    comptime rms_eps = Float32(1e-5)
    comptime hc_eps = Float32(1e-6)
    comptime norm_eps = Float32(1e-5)
    comptime K = hc * hidden
    comptime gamma_size = hidden if has_norm else 1
    var config = mhc_pre_launch_config(num_rows, hidden, force_k_splits)
    var k_splits = config[1]
    if force_k_splits != 0 and k_splits != force_k_splits:
        raise Error(
            "mhc_pre_launch_config returned KS=",
            k_splits,
            " for forced KS=",
            force_k_splits,
        )
    if config[2] != (1 if k_splits == 1 else 2):
        raise Error(
            "mhc_pre_launch_config returned ",
            config[2],
            " launches at KS=",
            k_splits,
        )

    var n_streams = num_rows * K
    var n_collapsed = num_rows * hidden
    var n_gate = num_rows * hc
    var n_comb = num_rows * hc * hc
    # One window of the streams can exceed the default budget, which would
    # collapse the buffer to a single, cache-warm window.
    var budget = max(
        CACHE_BUST_BYTES, 2 * n_streams * size_of[DType.bfloat16]()
    )

    var cb_streams = CacheBustingBuffer[.bfloat16](
        n_streams, 8, ctx, budget_bytes=budget
    )
    var cb_fn = CacheBustingBuffer[fn_dtype](
        mix_width * K, 16 // size_of[fn_dtype](), ctx
    )
    var cb_gamma = CacheBustingBuffer[.bfloat16](gamma_size, 8, ctx)
    var cb_collapsed = CacheBustingBuffer[.bfloat16](n_collapsed, 8, ctx)
    var cb_post = CacheBustingBuffer[.float32](n_gate, 4, ctx)
    var cb_pre = CacheBustingBuffer[.float32](n_gate, 4, ctx)
    var cb_comb = CacheBustingBuffer[.float32](n_comb, 4, ctx)
    cb_streams.init_on_device(InitializationType.uniform_distribution, ctx)
    cb_fn.init_on_device(InitializationType.uniform_distribution, ctx)
    cb_gamma.init_on_device(InitializationType.uniform_distribution, ctx)

    var base_h = ctx.enqueue_create_host_buffer[.float32](mix_width)
    var scale_h = ctx.enqueue_create_host_buffer[.float32](3)
    for i in range(mix_width):
        base_h[i] = Float32(i % 3) * 0.25 - 0.25
    # Uniform [0, 1) streams and weights put each projection near K / 4 times
    # the RMS scale; this brings the gate inputs back to O(1) so the gates,
    # and Sinkhorn's exp, stay finite.
    for i in range(3):
        scale_h[i] = Float32(2.0 / Float64(K))
    var base_d = ctx.enqueue_create_buffer[.float32](mix_width)
    var scale_d = ctx.enqueue_create_buffer[.float32](3)
    ctx.enqueue_copy(base_d, base_h)
    ctx.enqueue_copy(scale_d, scale_h)

    def launch(
        ctx: DeviceContext, iteration: Int
    ) raises {
        mut cb_streams,
        mut cb_fn,
        mut cb_gamma,
        mut cb_collapsed,
        mut cb_post,
        mut cb_pre,
        mut cb_comb,
        imm,
    }:
        mhc_pre[
            hc_mult=hc,
            hidden_size=hidden,
            hc_sinkhorn_iters=sinkhorn_iters,
            has_norm=has_norm,
            target="gpu",
        ](
            TileTensor(
                cb_post.offset_ptr(iteration), row_major((num_rows, Idx[hc]))
            ),
            TileTensor(
                cb_comb.offset_ptr(iteration),
                row_major((num_rows, Idx[hc], Idx[hc])),
            ),
            TileTensor(
                cb_collapsed.offset_ptr(iteration),
                row_major((num_rows, Idx[hidden])),
            ),
            TileTensor(
                cb_pre.offset_ptr(iteration), row_major((num_rows, Idx[hc]))
            ),
            TileTensor(
                cb_streams.offset_ptr(iteration),
                row_major((num_rows, Idx[hc], Idx[hidden])),
            ).as_imm(),
            TileTensor(
                cb_fn.offset_ptr(iteration), row_major((Idx[mix_width], Idx[K]))
            ).as_imm(),
            TileTensor(base_d, row_major((Idx[mix_width],))).as_imm(),
            TileTensor(scale_d, row_major((Idx[3],))).as_imm(),
            TileTensor(
                cb_gamma.offset_ptr(iteration), row_major((Idx[gamma_size],))
            ).as_imm(),
            rms_eps,
            hc_eps,
            norm_eps,
            ctx,
            force_k_splits=force_k_splits,
        )

    # Timing a NaN-filled run would not time the gates the model sees.
    launch(ctx, 0)
    var post_h = ctx.enqueue_create_host_buffer[.float32](n_gate)
    var comb_h = ctx.enqueue_create_host_buffer[.float32](n_comb)
    var post_view = cb_post.device_buffer().create_sub_buffer[.float32](
        0, n_gate
    )
    var comb_view = cb_comb.device_buffer().create_sub_buffer[.float32](
        0, n_comb
    )
    ctx.enqueue_copy(post_h, post_view)
    ctx.enqueue_copy(comb_h, comb_view)
    ctx.synchronize()
    for i in range(n_gate):
        if not isfinite(post_h[i]):
            raise Error("mhc_pre post is not finite at ", i)
    for i in range(n_comb):
        if not isfinite(comb_h[i]):
            raise Error("mhc_pre comb is not finite at ", i)

    @inline(.always)
    def bench_func(mut b: Bencher) raises {imm}:
        bencher_iter_custom(b, launch, ctx)

    var num_bytes = (
        (n_streams + n_collapsed + (gamma_size if has_norm else 0))
        * size_of[DType.bfloat16]()
        + mix_width * K * size_of[fn_dtype]()
        + (2 * n_gate + n_comb) * size_of[DType.float32]()
    )
    m.bench_function(
        bench_func,
        BenchId(
            "mhc_pre",
            input_id=String(
                "/T=",
                num_rows,
                "/d=",
                hidden,
                "/fn=",
                fn_dtype,
                "/norm=",
                Int(has_norm),
                "/MB=",
                config[0],
                "/KS=",
                k_splits,
                "/launches=",
                config[2],
            ),
        ),
        [ThroughputMeasure(BenchMetric.bytes, num_bytes)],
    )
    ctx.synchronize()

    _ = post_view
    _ = comb_view
    _ = base_d
    _ = scale_d
    _ = cb_streams
    _ = cb_fn
    _ = cb_gamma
    _ = cb_collapsed
    _ = cb_post
    _ = cb_pre
    _ = cb_comb


def main() raises:
    var num_rows = Int(arg_parse("num_rows", 8))
    var force_k_splits = Int(arg_parse("force_k_splits", 0))
    comptime fn_dtype = get_defined_dtype["fn_dtype", .float32]()
    comptime hidden = get_defined_int["hidden", 4096]()
    comptime has_norm = get_defined_bool["has_norm", True]()

    var m = Bench()
    with DeviceContext() as ctx:
        bench_mhc_pre[fn_dtype, hidden, has_norm](
            ctx, m, num_rows, force_k_splits
        )
    m.dump_report()
