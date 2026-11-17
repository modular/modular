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
"""Router cost at GLM-5.3-Flash's gate shape on B200.

The router is `[M, 4096] bf16 x [288, 4096]^T bf16`, accumulated in float32,
then sigmoid, the correction bias and a top-8, once per MoE layer. Under
expert parallelism a device routes its own rows: 2, 6, 24, 36 and 48 at MTP 5
with 1, 4, 16, 24 and 32 requests. `--rows=1,2,...` picks the row counts, up
to 128, the gate GEMV op's window.

Arms, each a whole router call: `prod` is the `moe.router.gate.mixed.gemv`
op's GEMV (`router_gate_mixed_gemv`), then `single_group_router` on
`sigmoid(logits)`, as the graph fuses the sigmoid into its input; `fused` is
`sigmoid_gemv_single_group_router` with float32 scores; `copy_bf16` reads and
writes the weight bytes once. Each call reads its weight from its own window of
a cache-busting buffer, as a decode step's MoE layers evict each other. A timed
graph replays `CALLS` calls, so host launch cost drops out. The arms run in
interleaved rounds, in alternating order, and the median round is reported.
Then `fused`'s expert sets and weights are compared with `prod`'s.
"""

from std.math import ceildiv
from std.random import random_float64, seed
from std.sys import get_defined_int

from internal_utils import arg_parse
from internal_utils._cache_busting import CacheBustingBuffer
from layout import Coord, Idx, TileTensor, row_major
from linalg.gemv import router_gate_mixed_gemv
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import (
    DeviceBuffer,
    DeviceContext,
    DeviceGraph,
    DeviceGraphBuilder,
)
from nn.activations import sigmoid
from nn.moe import sigmoid_gemv_single_group_router, single_group_router

comptime N = 288
comptime K = 4096
comptime TOP_K = 8
comptime SCALE = Float32(2.5)
comptime CALLS = get_defined_int["CALLS", 96]()
comptime REPLAYS = get_defined_int["REPLAYS", 20]()
comptime ROUNDS = get_defined_int["ROUNDS", 5]()


def copy_bf16(
    src: MutPointer[Scalar[DType.bfloat16], MutUntrackedOrigin],
    dst: MutPointer[Scalar[DType.bfloat16], MutUntrackedOrigin],
    n: Int32,
):
    var i = (block_idx.x * block_dim.x + thread_idx.x) * 8
    if i < Int(n):
        dst.unsafe_store(i, src.unsafe_load[width=8](i))


def median(var xs: List[Float64]) -> Float64:
    sort(xs)
    return xs[len(xs) // 2]


def graph_us[
    F: def(DeviceContext, Int) raises -> None
](ctx: DeviceContext, f: F) raises -> Float64:
    """Microseconds a call, `CALLS` calls replayed back to back."""

    def build(mut builder: DeviceGraphBuilder) raises {imm}:
        var rec = builder.recording_context()
        for call in range(CALLS):
            f(rec, call)

    var graph = DeviceGraph.create(ctx, build)

    def replay() raises {imm}:
        graph.replay()

    for _ in range(2):
        replay()
    ctx.synchronize()
    return (
        Float64(ctx.execution_time(replay, REPLAYS))
        / Float64(REPLAYS * CALLS)
        / 1e3
    )


struct Outputs(Movable):
    """One arm's expert ids and weights."""

    var idx: DeviceBuffer[DType.int32]
    var w: DeviceBuffer[DType.float32]

    def __init__(out self, ctx: DeviceContext, m: Int) raises:
        self.idx = ctx.enqueue_create_buffer[DType.int32](m * TOP_K)
        self.w = ctx.enqueue_create_buffer[DType.float32](m * TOP_K)


def compare(m: Int, expected: Outputs, got: Outputs) raises:
    """Prints how many rows pick a different expert set than `prod`, and the
    largest weight difference over the rows that pick the same set."""
    var rows_differ = 0
    var max_diff = Float32(0)
    with expected.idx.map_to_host() as ri, expected.w.map_to_host() as rw:
        with got.idx.map_to_host() as gi, got.w.map_to_host() as gw:
            for t in range(m):
                var matched = 0
                var diff = Float32(0)
                for r in range(t * TOP_K, (t + 1) * TOP_K):
                    for q in range(t * TOP_K, (t + 1) * TOP_K):
                        if gi[q] == ri[r]:
                            matched += 1
                            diff = max(diff, abs(gw[q] - rw[r]))
                if matched < TOP_K:
                    rows_differ += 1
                else:
                    max_diff = max(max_diff, diff)
    print(
        t"m={m} check fused: rows_differ={rows_differ}/{m}"
        t" max_weight_diff={max_diff}"
    )


def bench_rows(
    ctx: DeviceContext,
    ring: CacheBustingBuffer[DType.bfloat16],
    copy_dst: DeviceBuffer[DType.bfloat16],
    m: Int,
) raises:
    """Times every arm at `m` rows, then checks `fused` against `prod`."""
    var act = ctx.enqueue_create_buffer[DType.bfloat16](m * K)
    var bias = ctx.enqueue_create_buffer[DType.float32](N)
    with act.map_to_host() as a, bias.map_to_host() as b:
        seed(100 + m)
        for i in range(m * K):
            a[i] = random_float64(min=-1.0, max=1.0).cast[DType.bfloat16]()
        for i in range(N):
            b[i] = random_float64(min=-0.1, max=0.1).cast[DType.float32]()
    # Not an arm's output buffer: the top-k's score closure reads it while the
    # top-k writes the outputs.
    var logits = ctx.enqueue_create_buffer[DType.float32](m * N)
    var prod_out = Outputs(ctx, m)
    var fused_out = Outputs(ctx, m)

    var a_tt = TileTensor(act, row_major(m, Idx[K])).as_imm()
    var bias_tt = TileTensor(bias, row_major(Idx[N])).as_imm()
    var c_tt = TileTensor(logits, row_major(m, Idx[N]))
    var logits_ro = c_tt.as_imm()
    var prod_idx = TileTensor(prod_out.idx, row_major(m, Idx[TOP_K]))
    var prod_w = TileTensor(prod_out.w, row_major(m, Idx[TOP_K]))
    var fused_idx = TileTensor(fused_out.idx, row_major(m, Idx[TOP_K]))
    var fused_w = TileTensor(fused_out.w, row_major(m, Idx[TOP_K]))

    def load_scores[
        width: Int
    ](coords: Coord) {var logits_ro} -> SIMD[DType.float32, width]:
        return sigmoid(logits_ro.load[width=width](coords))

    def prod(
        c: DeviceContext, call: Int
    ) raises {
        imm, ring, a_tt, c_tt, prod_idx, prod_w, logits_ro, bias_tt, load_scores
    }:
        var w = TileTensor(ring.offset_ptr(call), row_major(Idx[N], Idx[K]))
        router_gate_mixed_gemv[DType.bfloat16, N](
            c_tt, a_tt, w.as_imm(), m, N, K, c
        )
        single_group_router[N, TOP_K, True](
            prod_idx, prod_w, logits_ro.layout, load_scores, bias_tt, SCALE, c
        )

    def fused(
        c: DeviceContext, call: Int
    ) raises {imm, ring, a_tt, bias_tt, fused_idx, fused_w}:
        var w = TileTensor(ring.offset_ptr(call), row_major(Idx[N], Idx[K]))
        sigmoid_gemv_single_group_router[N, TOP_K, True](
            fused_idx, fused_w, a_tt, w.as_imm(), bias_tt, SCALE, c
        )

    def copy_arm(c: DeviceContext, call: Int) raises {imm, ring, copy_dst}:
        c.enqueue_function[copy_bf16](
            ring.offset_ptr(call),
            copy_dst.unsafe_ptr(),
            Int32(N * K),
            grid_dim=ceildiv(N * K, 8 * 256),
            block_dim=256,
        )

    var prod_us = List[Float64]()
    var fused_us = List[Float64]()
    var copy_us = List[Float64]()
    for r in range(ROUNDS):
        for j in range(3):
            var arm = j if r % 2 == 0 else 2 - j
            if arm == 0:
                prod_us.append(graph_us(ctx, prod))
            elif arm == 1:
                fused_us.append(graph_us(ctx, fused))
            else:
                copy_us.append(graph_us(ctx, copy_arm))
    for r in range(ROUNDS):
        print(
            t"m={m} round={r} prod={prod_us[r]} fused={fused_us[r]}"
            t" copy_bf16={copy_us[r]}"
        )
    var prod_med = median(prod_us^)
    var fused_med = median(fused_us^)
    print(
        t"m={m} median prod={prod_med} fused={fused_med}"
        t" copy_bf16={median(copy_us^)} fused-prod={fused_med - prod_med}"
    )

    prod(ctx, 0)
    fused(ctx, 0)
    ctx.synchronize()
    compare(m, prod_out, fused_out)


def main() raises:
    var rows_arg = arg_parse("rows", "1,2,3,4,6,24,36,48")
    var rows = List[Int]()
    for part in rows_arg.split(","):
        rows.append(Int(part))
    with DeviceContext() as ctx:
        var ring = CacheBustingBuffer[DType.bfloat16](N * K, ctx)
        var host = ctx.enqueue_create_host_buffer[DType.bfloat16](N * K)
        ctx.synchronize()
        seed(7)
        for i in range(N * K):
            host[i] = random_float64(min=-0.05, max=0.05).cast[DType.bfloat16]()
        for call in range(CALLS):
            var window = ring.device_buffer().create_sub_buffer[DType.bfloat16](
                ring.offset(call), N * K
            )
            ctx.enqueue_copy(window, host)
        var copy_dst = ctx.enqueue_create_buffer[DType.bfloat16](N * K)
        for m in rows:
            if m < 1 or m > 128:
                raise Error(t"--rows takes 1 to 128 rows, not {m}")
            bench_rows(ctx, ring, copy_dst, m)
