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
"""Router cost at MiMo-V2.6-Flash's gate shape on B200.

The router is `[M, 4096] bf16 x [256, 4096]^T fp32`, once per MoE layer and 47
times a decode step, with `M = 64` at c8 (8 requests x 8 verify positions) and
`M = 8` at c1. Its weight is the whole cost, and a GEMV re-reads it once per row
block, so the launch's output tile decides the traffic.

Arms:

- `gemv_split_k (tm, tn) exact`: each band of the launcher's
  `_SM100_ROUTER_GATE_BANDS` table with its row guard off -- what the launcher
  runs at a row count the band's `tile_m` divides. A guard-less launch at any
  other row count reads and writes past the row buffers, so that arm is only
  timed where it is in bounds.
- `gemv_split_k (tm, tn) guard`: the same band with the guard on, i.e. what the
  launcher's fallback runs, and what the guard costs a band that would
  otherwise take a `tile_m` the row count does not divide.
- `router_gate_mixed_gemv`: the launcher itself, i.e. the band its table picks
  at this row count -- the arm production runs.
- `topk only`: the shipped `single_group_router`, the router family's second
  launch.
- `split-K router`: `sigmoid_gemv_single_group_router`, which computes the gate
  inside the top-k kernel from 16 hidden-dimension splits and so fuses the
  gate, its sigmoid, the correction bias and the top-k into one launch. It
  changes the summation order of every dot product, so it is a cost reference
  for a redesign and not a drop-in replacement for the bit-exact path.
- `copy`: the stream floor, one read of the same weight bytes and nothing else.

Every arm runs over `NUM_LAYERS` distinct weight copies, so each launch reads a
weight the previous launches evicted, the way a decode step's 47 layer weights
evict each other. A `marker` kernel precedes each arm so an nsys trace splits
per arm. The printed time is host wall clock, which for these arms is device
time: 940 launches queue far ahead of the device, and a trace's per-arm span
(first kernel start to last end) matches the printed number to within a few
percent. Per-kernel durations in that trace do not: consecutive launches
overlap through PDL, so a kernel's recorded span includes its predecessor's
tail.

`main` also times the `[128, 6144]` gate (MiniMax-M3's, and every other shape)
at the three row counts no row's `tile_m` divides -- 9, 17 and 25 -- where the
launcher takes its `(4, 1)` or `(4, 2)` band with the row guard on. Those rows
keep the guard because of how much faster they are there than the guarded
`(8, 2)` band a count-above-every-row would fall to.
"""

from std.math import ceildiv
from std.random import random_float64, seed
from std.sys import get_defined_int
from std.time import monotonic

from layout import Idx, TileTensor, row_major
from linalg.gemv import (
    _SM100_ROUTER_GATE_BANDS,
    _sm100_router_gate_band,
    gemv_split_k,
    router_gate_mixed_gemv,
)
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    pdl_launch_attributes,
)
from nn.moe import (
    sigmoid_gemv_single_group_router,
    single_group_router,
)

comptime N = 256
comptime K = 4096
comptime TOP_K = 8
comptime NUM_LAYERS = get_defined_int["NUM_LAYERS", 47]()
comptime REPS = get_defined_int["REPS", 20]()

# MiniMax-M3's gate, i.e. the shaped rows every other gate shape shares.
comptime MINIMAX_N = 128
comptime MINIMAX_K = 6144


def marker(x: MutPointer[Scalar[DType.float32], MutUntrackedOrigin]):
    x.unsafe_store(0, Float32(0))


def copy_f32(
    src: MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
    dst: MutPointer[Scalar[DType.float32], MutUntrackedOrigin],
    n: Int32,
):
    var i = (block_idx.x * block_dim.x + thread_idx.x) * 4
    if i < Int(n):
        dst.unsafe_store(i, src.unsafe_load[width=4](i))


struct Operands[static_N: Int, static_K: Int](Movable):
    """One row count's tensors: per-layer weights, activation and outputs."""

    var weights: List[DeviceBuffer[DType.float32]]
    var act: DeviceBuffer[DType.bfloat16]
    var logits: DeviceBuffer[DType.float32]
    var idx: DeviceBuffer[DType.int32]
    var scores: DeviceBuffer[DType.float32]
    var bias: DeviceBuffer[DType.float32]
    var copy_dst: DeviceBuffer[DType.float32]
    var host_w: HostBuffer[DType.float32]

    def __init__(out self, ctx: DeviceContext, m: Int) raises:
        self.weights = List[DeviceBuffer[DType.float32]]()
        self.act = ctx.enqueue_create_buffer[DType.bfloat16](m * Self.static_K)
        self.logits = ctx.enqueue_create_buffer[DType.float32](
            m * Self.static_N
        )
        self.idx = ctx.enqueue_create_buffer[DType.int32](m * TOP_K)
        self.scores = ctx.enqueue_create_buffer[DType.float32](m * TOP_K)
        self.bias = ctx.enqueue_create_buffer[DType.float32](Self.static_N)
        self.copy_dst = ctx.enqueue_create_buffer[DType.float32](
            Self.static_N * Self.static_K
        )
        self.host_w = ctx.enqueue_create_host_buffer[DType.float32](
            Self.static_N * Self.static_K
        )

        seed(7)
        var a_host = ctx.enqueue_create_host_buffer[DType.bfloat16](
            m * Self.static_K
        )
        for i in range(m * Self.static_K):
            a_host[i] = random_float64(min=-1.0, max=1.0).cast[DType.bfloat16]()
        ctx.enqueue_copy(self.act, a_host)
        for layer in range(NUM_LAYERS):
            for i in range(Self.static_N * Self.static_K):
                self.host_w[i] = random_float64(min=-1.0, max=1.0).cast[
                    DType.float32
                ]()
            self.weights.append(
                ctx.enqueue_create_buffer[DType.float32](
                    Self.static_N * Self.static_K
                )
            )
            ctx.enqueue_copy(self.weights[layer], self.host_w)
        ctx.synchronize()


def time_launches[
    LaunchFn: def(Int) raises -> None
](
    ctx: DeviceContext,
    marker_buf: DeviceBuffer[DType.float32],
    label: String,
    launch: LaunchFn,
) raises:
    """Marks the trace, warms the arm, then times `NUM_LAYERS * REPS` launches.
    """
    ctx.enqueue_function[marker](
        marker_buf.unsafe_ptr(), grid_dim=1, block_dim=1
    )
    for _ in range(REPS):
        for layer in range(NUM_LAYERS):
            launch(layer)
    ctx.synchronize()
    var start = monotonic()
    for _ in range(REPS):
        for layer in range(NUM_LAYERS):
            launch(layer)
    ctx.synchronize()
    var us = Float64(monotonic() - start) / 1e3 / Float64(NUM_LAYERS * REPS)
    print(label, us, "us (host, per launch)")


def band_arm[
    m: Int,
    static_N: Int,
    static_K: Int,
    tile_m: Int,
    tile_n: Int,
    num_threads: Int,
    unroll: Int,
    guard_m: Bool,
](
    ctx: DeviceContext,
    ops: Operands[static_N, static_K],
    marker_buf: DeviceBuffer[DType.float32],
    label: String,
) raises:
    """Times one `gemv_split_k` band over the layer ring."""
    var a_tt = TileTensor(ops.act, row_major(m, Idx[static_K])).as_imm()
    var c_tt = TileTensor(ops.logits, row_major(m, Idx[static_N]))
    # The per-layer weight tensor is built inside the timed closure, so its
    # layout and engine come from the type alone.
    comptime weight_type = type_of(
        TileTensor(ops.weights[0], row_major(Idx[static_N], Idx[static_K]))
    )

    comptime kernel = gemv_split_k[
        DType.float32,
        DType.bfloat16,
        DType.float32,
        type_of(c_tt).LayoutType,
        type_of(a_tt).LayoutType,
        weight_type.LayoutType,
        type_of(c_tt).Engine,
        type_of(a_tt).Engine,
        weight_type.Engine,
        simd_width=4,
        tile_m=tile_m,
        tile_n=tile_n,
        num_threads=num_threads,
        unroll_factor=unroll,
        weight_non_temporal=False,
        check_bounds_m=guard_m,
        check_bounds_n=static_N % tile_n != 0,
        pdl_level=PDLLevel.OVERLAP_AT_BEGINNING,
    ]

    def launch(layer: Int) raises {imm, ctx, ops, a_tt, c_tt}:
        var b_tt = TileTensor(
            ops.weights[layer], row_major(Idx[static_N], Idx[static_K])
        ).as_imm()
        ctx.enqueue_function[kernel](
            c_tt,
            a_tt,
            b_tt,
            Int32(m),
            Int32(static_N),
            Int32(static_K),
            grid_dim=(ceildiv(m, tile_m), ceildiv(static_N, tile_n)),
            block_dim=num_threads,
            attributes=pdl_launch_attributes(PDLLevel.OVERLAP_AT_BEGINNING),
        )

    time_launches(ctx, marker_buf, label, launch)


def band_label[band_idx: Int]() -> String:
    """Returns the label of a band in the launcher's own table."""
    comptime band = _SM100_ROUTER_GATE_BANDS[band_idx]
    return (
        "gemv_split_k ("
        + String(band[0])
        + ", "
        + String(band[1])
        + ") u"
        + String(band[3])
    )


def run[m: Int](ctx: DeviceContext) raises:
    """Times every arm at one MiMo-V2.6-Flash row count."""
    var ops = Operands[N, K](ctx, m)
    var marker_buf = ctx.enqueue_create_buffer[DType.float32](1)
    var a_tt = TileTensor(ops.act, row_major(m, Idx[K])).as_imm()
    var c_tt = TileTensor(ops.logits, row_major(m, Idx[N]))
    var idx_tt = TileTensor(ops.idx, row_major(m, Idx[TOP_K]))
    var scores_tt = TileTensor(ops.scores, row_major(m, Idx[TOP_K]))
    var bias_tt = TileTensor(ops.bias, row_major(Idx[N])).as_imm()
    var logits_tt = TileTensor(ops.logits, row_major(m, Idx[N])).as_imm()

    def label_for(name: String) -> String:
        return "m=" + String(m) + " " + name

    comptime wide = N == 256 and K == 4096
    var launcher_band = _sm100_router_gate_band[wide](m)
    comptime for i in range(len(_SM100_ROUTER_GATE_BANDS)):
        comptime band = _SM100_ROUTER_GATE_BANDS[i]
        comptime label = band_label[i]()
        if launcher_band == i:
            print("   launcher picks", label)
        # The launcher's own guard choice: none when the band's `tile_m`
        # divides `m`. A guard-less launch at any other row count reads and
        # writes past the row buffers, so it is only timed where it is in
        # bounds.
        comptime if m % band[0] == 0:
            band_arm[m, N, K, band[0], band[1], band[2], band[3], False](
                ctx, ops, marker_buf, label_for(label + " exact")
            )
        band_arm[m, N, K, band[0], band[1], band[2], band[3], True](
            ctx, ops, marker_buf, label_for(label + " guard")
        )

    def launcher_arm(layer: Int) raises {imm, ctx, ops, a_tt, c_tt}:
        var b_tt = TileTensor(
            ops.weights[layer],
            row_major(Idx[N], Idx[K]),
        ).as_imm()
        router_gate_mixed_gemv[DType.float32, N](c_tt, a_tt, b_tt, m, N, K, ctx)

    time_launches(
        ctx, marker_buf, label_for("router_gate_mixed_gemv"), launcher_arm
    )

    def topk_arm(
        layer: Int,
    ) raises {imm, ctx, idx_tt, scores_tt, logits_tt, bias_tt}:
        _ = layer
        single_group_router[N, TOP_K, True](
            idx_tt, scores_tt, logits_tt, bias_tt, Float32(1.0), ctx
        )

    time_launches(ctx, marker_buf, label_for("topk only            "), topk_arm)

    def split_k_arm(
        layer: Int,
    ) raises {imm, ctx, idx_tt, scores_tt, a_tt, bias_tt, ops}:
        var gw_tt = TileTensor(
            ops.weights[layer], row_major(Idx[N], Idx[K])
        ).as_imm()
        sigmoid_gemv_single_group_router[N, TOP_K, True](
            idx_tt, scores_tt, a_tt, gw_tt, bias_tt, Float32(1.0), ctx
        )

    time_launches(
        ctx, marker_buf, label_for("split-K router       "), split_k_arm
    )

    def copy_arm(layer: Int) raises {imm, ctx, ops}:
        ctx.enqueue_function[copy_f32](
            ops.weights[layer].unsafe_ptr(),
            ops.copy_dst.unsafe_ptr(),
            Int32(N * K),
            grid_dim=ceildiv(N * K, 4 * 256),
            block_dim=256,
        )

    time_launches(ctx, marker_buf, label_for("copy floor (4 MiB)   "), copy_arm)


def run_guarded_rows(ctx: DeviceContext) raises:
    """Times the guarded bands the `[128, 6144]` gate takes at 9, 17 and 25 rows.

    No row's `tile_m` divides those counts, so the launcher runs its `(4, 1)`
    (9-23 rows) or `(4, 2)` (25-31) band with the row guard on. This is what
    keeps those rows worth their guard: the `(8, 2)` band the launcher falls
    back to above 32 rows is far slower at this many rows, guarded or not.
    """
    # Sized for the widest row count, so all three share one weight ring.
    var ops = Operands[MINIMAX_N, MINIMAX_K](ctx, 25)
    var marker_buf = ctx.enqueue_create_buffer[DType.float32](1)

    comptime for m in [9, 17, 25]:
        print("== guarded m =", m)
        comptime for i in [1, 2, 3]:
            comptime band = _SM100_ROUTER_GATE_BANDS[i]
            comptime label = (
                "m="
                + String(m)
                + " gemv_split_k ("
                + String(band[0])
                + ", "
                + String(band[1])
                + ") guard"
            )
            band_arm[
                m,
                MINIMAX_N,
                MINIMAX_K,
                band[0],
                band[1],
                band[2],
                band[3],
                True,
            ](ctx, ops, marker_buf, label)


def main() raises:
    with DeviceContext() as ctx:
        # The decode row counts this router is sized for: c1's 8, c2's 16,
        # c4's 32, c8's 64, and the row counts between and above them. 2, 3,
        # 6, 9, 20 and 36 reach no unguarded wide row with a `tile_m` above 1:
        # no such `tile_m` divides the first four, and the `(4, 1)` row that
        # divides 20 and 36 stops at 16 rows. They time the guarded rows
        # against the bands they fall back from.
        comptime for m in [1, 2, 3, 6, 8, 9, 16, 20, 32, 36, 64, 128]:
            print("== m =", m)
            run[m](ctx)
        run_guarded_rows(ctx)
