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
"""Benchmarks the rejection sampler's residual draw.

Speculative decoding recovers a rejected draft token from the residual
`max(p - q, 0)`, where `p` is the target's truncated distribution and `q` the
proposal's, one row per (request, draft position). The two arms are the two
ways to sample it:

- `fuse=0` forms the residual in HBM with an elementwise pass and samples it,
  which is what a graph of `relu(p - q)` -> `gumbel_argmax_from_probs` does.
  Both of its kernels exist on `main`, so this arm is the pre-change graph.
- `fuse=1` runs `gumbel_sampling_fused_gpu[subtract=True]`, which reads `p`
  and `q` and forms the residual in registers.

Both draw the same token from the same seed; only the traffic and the launch
count differ. `rows` is the flattened `batch * num_steps` row count the verify
step samples, so the per-candidate cost is the time divided by `rows`.
"""

from std.sys import align_of, get_defined_int

from max.benchmark import bencher_iter_custom
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    pdl_launch_attributes,
)
from layout import (
    DefaultEngine,
    TensorEngine,
    TensorLayout,
    TileTensor,
    row_major,
)
from nn.topk import gumbel_sampling_fused_gpu

from std.benchmark import Bench, BenchConfig, Bencher, BenchId
from std.math import max
from std.sys.info import simd_width_of


@__name(t"bench_write_residual_{dtype}")
def _write_residual_kernel[
    dtype: DType,
    LayoutType: TensorLayout,
    Engine: TensorEngine = DefaultEngine[element_width=1],
    OutEngine: TensorEngine = DefaultEngine[element_width=1],
](
    probs: TileTensor[dtype, LayoutType, ImmutAnyOrigin, Engine=Engine],
    proposal: TileTensor[dtype, LayoutType, ImmutAnyOrigin, Engine=Engine],
    residual: TileTensor[dtype, LayoutType, MutAnyOrigin, Engine=OutEngine],
    d: Int32,
):
    """Materializes `max(p - q, 0)`, one block per row."""
    comptime width = simd_width_of[dtype]()
    comptime align = align_of[SIMD[dtype, width]]()

    var row = block_idx.x
    var block_size = Int(block_dim.x)
    var chunks = Int(d) // width
    for i in range(Int(thread_idx.x), chunks, block_size):
        var p = probs.load[width=width, alignment=align, invariant=False](
            (row, i * width)
        ).cast[.float32]()
        var q = proposal.load[width=width, alignment=align, invariant=False](
            (row, i * width)
        ).cast[.float32]()
        residual.store((row, i * width), max(p - q, Float32(0.0)).cast[dtype]())


def _bench_materialized[
    dtype: DType
](rows: Int, vocab: Int, ctx: DeviceContext[], mut b: Bench) raises:
    """Forms the residual in HBM, then samples it: two launches."""
    comptime block_size = ctx.default_device_info.max_thread_block_size

    var probs_buf = ctx.enqueue_create_buffer[dtype](rows * vocab)
    var proposal_buf = ctx.enqueue_create_buffer[dtype](rows * vocab)
    var residual_buf = ctx.enqueue_create_buffer[dtype](rows * vocab)
    var out_buf = ctx.enqueue_create_buffer[.int64](rows)
    var seed_buf = ctx.enqueue_create_buffer[.uint64](rows)
    # The proposal carries less mass than the target at every element, so the
    # residual is positive everywhere and neither arm short-circuits.
    var target_mass = Float32(2.0) / Float32(vocab)
    var proposal_mass = Float32(0.5) / Float32(vocab)
    probs_buf.enqueue_fill(Scalar[dtype](target_mass))
    proposal_buf.enqueue_fill(Scalar[dtype](proposal_mass))
    seed_buf.enqueue_fill(UInt64(42))
    ctx.synchronize()

    var probs = (
        TileTensor(probs_buf, row_major(rows, vocab))
        .as_unsafe_any_origin()
        .as_imm()
    )
    var proposal = (
        TileTensor(proposal_buf, row_major(rows, vocab))
        .as_unsafe_any_origin()
        .as_imm()
    )
    var residual = TileTensor(residual_buf, row_major(rows, vocab))
    var out = TileTensor(out_buf, row_major(rows))
    var seed = (
        TileTensor(seed_buf, row_major(rows)).as_unsafe_any_origin().as_imm()
    )

    comptime materialize = _write_residual_kernel[
        dtype, probs.LayoutType, Engine=probs.Engine, OutEngine=residual.Engine
    ]

    @inline(.always)
    def bench_fn(
        mut bb: Bencher,
    ) raises {
        mut residual,
        mut out,
        imm probs,
        imm proposal,
        imm seed,
        imm ctx,
        imm rows,
        imm vocab,
    }:
        @inline(.always)
        def launch(
            dctx: DeviceContext[],
        ) raises {
            mut residual,
            mut out,
            imm probs,
            imm proposal,
            imm seed,
            imm rows,
            imm vocab,
        }:
            dctx.enqueue_function[materialize](
                probs,
                proposal,
                residual,
                Int32(vocab),
                grid_dim=rows,
                block_dim=block_size,
                attributes=pdl_launch_attributes(PDLLevel.ON),
            )
            gumbel_sampling_fused_gpu[from_probs=True](
                dctx, residual.as_imm(), out, seed=seed
            )

        bencher_iter_custom(bb, launch, ctx)

    b.bench_function(
        bench_fn,
        BenchId(
            "residual_gumbel", input_id=String("rows=", rows, "/vocab=", vocab)
        ),
    )
    ctx.synchronize()

    _ = probs_buf^
    _ = proposal_buf^
    _ = residual_buf^
    _ = out_buf^
    _ = seed_buf^


def _bench_fused[
    dtype: DType
](rows: Int, vocab: Int, ctx: DeviceContext[], mut b: Bench) raises:
    """Forms the residual in registers: one launch."""
    var probs_buf = ctx.enqueue_create_buffer[dtype](rows * vocab)
    var proposal_buf = ctx.enqueue_create_buffer[dtype](rows * vocab)
    var fallback_buf = ctx.enqueue_create_buffer[.int64](rows)
    var out_buf = ctx.enqueue_create_buffer[.int64](rows)
    var seed_buf = ctx.enqueue_create_buffer[.uint64](rows)
    var target_mass = Float32(2.0) / Float32(vocab)
    var proposal_mass = Float32(0.5) / Float32(vocab)
    probs_buf.enqueue_fill(Scalar[dtype](target_mass))
    proposal_buf.enqueue_fill(Scalar[dtype](proposal_mass))
    # Every row carries a proposal distribution, so no row takes the one-hot
    # branch; the operand is there because the op takes it unconditionally.
    fallback_buf.enqueue_fill(Int64(-1))
    seed_buf.enqueue_fill(UInt64(42))
    ctx.synchronize()

    var probs = (
        TileTensor(probs_buf, row_major(rows, vocab))
        .as_unsafe_any_origin()
        .as_imm()
    )
    var proposal = (
        TileTensor(proposal_buf, row_major(rows, vocab))
        .as_unsafe_any_origin()
        .as_imm()
    )
    var fallback = (
        TileTensor(fallback_buf, row_major(rows))
        .as_unsafe_any_origin()
        .as_imm()
    )
    var out = TileTensor(out_buf, row_major(rows))
    var seed = (
        TileTensor(seed_buf, row_major(rows)).as_unsafe_any_origin().as_imm()
    )

    @inline(.always)
    def bench_fn(
        mut bb: Bencher,
    ) raises {
        mut out, imm probs, imm proposal, imm fallback, imm seed, imm ctx
    }:
        @inline(.always)
        def launch(
            dctx: DeviceContext[],
        ) raises {mut out, imm probs, imm proposal, imm fallback, imm seed}:
            gumbel_sampling_fused_gpu[from_probs=False, subtract=True](
                dctx,
                probs,
                out,
                seed=seed,
                sub=proposal,
                fallback=fallback,
            )

        bencher_iter_custom(bb, launch, ctx)

    b.bench_function(
        bench_fn,
        BenchId(
            "residual_gumbel", input_id=String("rows=", rows, "/vocab=", vocab)
        ),
    )
    ctx.synchronize()

    _ = probs_buf^
    _ = proposal_buf^
    _ = fallback_buf^
    _ = out_buf^
    _ = seed_buf^


def main() raises:
    # Runtime, not `comptime`: kbench builds one binary per bench file and
    # passes these through the environment, so freezing them here would run
    # every shape at the default.
    var rows = get_defined_int["rows", 8]()
    var vocab = get_defined_int["vocab", 128256]()
    var fuse = get_defined_int["fuse", 1]()

    var m = Bench(BenchConfig(num_repetitions=1))
    m.config.max_iters = 200
    m.config.show_progress = False
    with DeviceContext() as ctx:
        if fuse == 1:
            _bench_fused[DType.float32](rows, vocab, ctx, m)
        else:
            _bench_materialized[DType.float32](rows, vocab, ctx, m)

    # kbench reads the measurements off this report; a bench that skips it
    # records zeros.
    print()
    m.dump_report()
