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
"""Manifold-Constrained Hyper-Connections (mHC) gates and stream write-back.

Reference: Xie et al. 2026, "Manifold-Constrained Hyper-Connections", section
2.2 equation 8.

An mHC site replaces the residual add between two transformer sublayers with a
learned mixing of `hc_mult` parallel residual streams. A single projection of
the normalized streams produces `2 * hc_mult + hc_mult**2` values per token,
which this module turns into the three gate tensors the site applies:

- `pre`: stream-collapse weights, folding the parallel streams into the single
  sequence the sublayer consumes.
- `post`: where the sublayer output lands across the streams, in `[0, 2]`.
- `comb`: the `hc_mult x hc_mult` stream mixer, softmax-initialized and then
  Sinkhorn-Knopp projected onto the doubly-stochastic manifold.

After the sublayer runs, `mhc_post` writes its output back into the streams:
`streams'[k] = post[k] * y + sum_j comb[j, k] * streams[j]`.

`hyper_connection_gates` leaves the projection and the final collapse to the
graph. On gfx950 `mhc_pre` fuses the whole pre site instead: projection, RMS
scale, gates, collapse and an optional RMSNorm in one op, which is one launch
at large token counts and two below them, where K is split across blocks.
"""

import max.gpu.primitives.warp as warp
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    lane_id,
    thread_idx,
    warp_id,
)
from std.bit import log2_floor
from std.math import align_up, ceildiv, exp, min, rsqrt
from std.sys import size_of
from std.sys._assembly import inlined_assembly
from std.sys.info import is_amd_gpu
from std.utils.index import StaticTuple

from layout import TensorLayout, TileTensor, row_major, stack_allocation
from max.gpu.compute.mma import mma
from max.gpu.sync import barrier, schedule_barrier
from max.gpu.host import DeviceAttribute, DeviceContext, default_accelerator
from max.gpu.host.info import is_gpu
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from max.runtime.tracing import Trace, TraceLevel

from nn.activations import sigmoid


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(warps_per_block * WARP_SIZE)
    )
)
@__name(
    t"hyper_connection_gates_h{hc_mult}_s{hc_sinkhorn_iters}_w{warps_per_block}"
)
def hyper_connection_gates_kernel[
    PreLayoutType: TensorLayout,
    PostLayoutType: TensorLayout,
    CombLayoutType: TensorLayout,
    ProjLayoutType: TensorLayout,
    BiasLayoutType: TensorLayout,
    ScaleLayoutType: TensorLayout,
    hc_mult: Int,
    hc_sinkhorn_iters: Int,
    warps_per_block: Int,
](
    pre: TileTensor[mut=True, .float32, PreLayoutType, MutAnyOrigin],
    post: TileTensor[mut=True, .float32, PostLayoutType, MutAnyOrigin],
    comb: TileTensor[mut=True, .float32, CombLayoutType, MutAnyOrigin],
    hc_proj: TileTensor[.float32, ProjLayoutType, ImmutAnyOrigin],
    pre_post_comb_b: TileTensor[.float32, BiasLayoutType, ImmutAnyOrigin],
    pre_post_comb_scale: TileTensor[.float32, ScaleLayoutType, ImmutAnyOrigin],
    hc_eps: Float32,
    num_rows: Int32,
):
    """Computes the mHC `pre`, `post` and `comb` gates.

    A warp carries `WARP_SIZE // hc_mult**2` rows at once, so no lane idles:
    two rows per warp at `hc_mult=4` on a 32-lane warp, four on a 64-lane one.
    Lane `l` owns row `l / hc_mult**2` of the warp's group and, within it,
    `comb` element `(t / hc_mult, t % hc_mult)` for `t = l % hc_mult**2`.

    That mapping makes both Sinkhorn reductions lane-group reductions of the
    same warp: `lane_group_sum[num_lanes=hc_mult]` sums across a matrix row
    (torch's `dim=-1`) and `lane_group_sum[num_lanes=hc_mult,
    stride=hc_mult]` sums down a matrix column (torch's `dim=-2`). Every xor
    mask either reduction uses is smaller than `hc_mult**2`, so a group never
    reaches out of the row that owns it and the rows stay independent.

    Parameters:
        PreLayoutType: Layout of the `pre` output.
        PostLayoutType: Layout of the `post` output.
        CombLayoutType: Layout of the `comb` output.
        ProjLayoutType: Layout of the `hc_proj` input.
        BiasLayoutType: Layout of the `pre_post_comb_b` input.
        ScaleLayoutType: Layout of the `pre_post_comb_scale` input.
        hc_mult: Number of parallel residual streams.
        hc_sinkhorn_iters: Sinkhorn-Knopp iterations used to project `comb`.
        warps_per_block: Warps per block.

    Args:
        pre: Stream-collapse weights. Shape: `[num_rows, hc_mult]`.
        post: Sublayer-output placement weights. Shape: `[num_rows, hc_mult]`.
        comb: Row-major stream mixer. Shape: `[num_rows, hc_mult * hc_mult]`.
        hc_proj: Projected streams. Shape:
            `[num_rows, 2 * hc_mult + hc_mult * hc_mult]`.
        pre_post_comb_b: Per-output bias, concatenated in `pre`, `post`, `comb`
            order. Shape: `[2 * hc_mult + hc_mult * hc_mult]`.
        pre_post_comb_scale: Per-output scale, in `pre`, `post`, `comb` order.
            Shape: `[3]`.
        hc_eps: Epsilon guarding the Sinkhorn divisions.
        num_rows: Number of rows to process.
    """
    comptime H = hc_mult
    comptime HH = H * H
    comptime rows_per_warp = WARP_SIZE // HH
    comptime rows_per_block = warps_per_block * rows_per_warp

    # log2_floor in `lane_group_reduce` silently truncates a non-power-of-two
    # group, which would drop matrix entries instead of failing.
    comptime assert H.is_power_of_two(), "hc_mult must be a power of two"
    comptime assert (
        HH <= WARP_SIZE
    ), "hc_mult * hc_mult must fit within a single warp"
    # A power-of-two `hc_mult` divides the warp evenly, so no lane is left over
    # and every lane belongs to exactly one row.
    comptime assert rows_per_warp * HH == WARP_SIZE
    comptime assert hc_sinkhorn_iters >= 1, "hc_sinkhorn_iters must be >= 1"
    comptime assert pre.flat_rank == 2
    comptime assert post.flat_rank == 2
    comptime assert comb.flat_rank == 2
    comptime assert hc_proj.flat_rank == 2
    comptime assert pre_post_comb_b.flat_rank == 1
    comptime assert pre_post_comb_scale.flat_rank == 1
    comptime assert pre.static_shape[1] == H
    comptime assert post.static_shape[1] == H
    comptime assert comb.static_shape[1] == HH

    var lane = Int(lane_id())
    var elem = lane % HH
    var row = (
        block_idx.x * rows_per_block
        + Int(warp_id()) * rows_per_warp
        + lane // HH
    )

    # The scale and the bias are checkpoint weights, so they do not come from
    # the grid this one waits on. Loading them first puts the fetch in flight
    # across the wait instead of behind it. Every index here is within the
    # bias, whatever `elem` is, so none of it needs a bound.
    var s_pre = pre_post_comb_scale.load((0,))
    var s_post = pre_post_comb_scale.load((1,))
    var s_comb = pre_post_comb_scale.load((2,))
    var pre_b = pre_post_comb_b.load((elem,))
    var post_b = pre_post_comb_b.load((H + elem,))
    var comb_b = pre_post_comb_b.load((2 * H + elem,))

    wait_on_dependent_grids()
    launch_dependent_grids()

    # Not warp-uniform: the tail block's last warp straddles `num_rows`. Every
    # lane still runs every shuffle below -- skipping one would leave the
    # warp's reduction undefined -- and a lane group never reaches outside the
    # row that owns it, so an out-of-range row cannot perturb a live one.
    var in_range = row < Int(num_rows)

    if in_range and elem < H:
        var pre_w = hc_proj.load((row, elem))
        pre[row, elem] = _mhc_pre_gate(pre_w, s_pre, pre_b, hc_eps)

        var post_w = hc_proj.load((row, H + elem))
        post[row, elem] = _mhc_post_gate(post_w, s_post, post_b)

    var logit = Float32(0)
    if in_range:
        logit = hc_proj.load((row, 2 * H + elem)) * s_comb + comb_b

    var c = _mhc_comb_gate[H, hc_sinkhorn_iters](logit, hc_eps)

    if in_range:
        comb[row, elem] = c


@inline(.always)
def _mhc_pre_gate(
    pre_w: Float32, s_pre: Float32, pre_b: Float32, hc_eps: Float32
) -> Float32:
    """Returns one `pre` gate from its projected logit."""
    return sigmoid(pre_w * s_pre + pre_b) + hc_eps


@inline(.always)
def _mhc_post_gate(
    post_w: Float32, s_post: Float32, post_b: Float32
) -> Float32:
    """Returns one `post` gate from its projected logit."""
    return 2 * sigmoid(post_w * s_post + post_b)


@inline(.always)
def _mhc_comb_gate[
    hc_mult: Int, hc_sinkhorn_iters: Int
](logit: Float32, hc_eps: Float32) -> Float32:
    """Returns this lane's `comb` entry from its scaled, biased logit.

    Lanes carry one row's `hc_mult**2` entries row-major in consecutive
    lanes, as in `hyper_connection_gates_kernel`, and every lane must call
    this so the lane-group reductions are defined.

    Parameters:
        hc_mult: Number of parallel residual streams.
        hc_sinkhorn_iters: Sinkhorn-Knopp iterations used to project `comb`.

    Args:
        logit: `comb` logit after the scale and bias.
        hc_eps: Epsilon guarding the Sinkhorn divisions.

    Returns:
        The doubly-stochastic mixer entry.
    """
    comptime H = hc_mult
    # Softmax over the matrix row, max-subtracted to match torch.
    var e = exp(logit - warp.lane_group_max[num_lanes=H](logit))
    var c = e / warp.lane_group_sum[num_lanes=H](e) + hc_eps

    # Sinkhorn-Knopp: the softmax already normalized the rows, so the first
    # half-step is the column pass.
    c /= warp.lane_group_sum[num_lanes=H, stride=H](c) + hc_eps

    comptime for _ in range(hc_sinkhorn_iters - 1):
        c /= warp.lane_group_sum[num_lanes=H](c) + hc_eps
        c /= warp.lane_group_sum[num_lanes=H, stride=H](c) + hc_eps
    return c


@inline(.always)
def hyper_connection_gates[
    hc_mult: Int,
    hc_sinkhorn_iters: Int,
    target: StaticString,
    warps_per_block: Int = 4,
](
    pre: TileTensor[mut=True, .float32, ...],
    post: TileTensor[mut=True, .float32, ...],
    comb: TileTensor[mut=True, .float32, ...],
    hc_proj: TileTensor[mut=False, .float32, ...],
    pre_post_comb_b: TileTensor[mut=False, .float32, ...],
    pre_post_comb_scale: TileTensor[mut=False, .float32, ...],
    hc_eps: Float32,
    context: DeviceContext,
) raises:
    """Launches the mHC gate kernel, one warp per row.

    Parameters:
        hc_mult: Number of parallel residual streams.
        hc_sinkhorn_iters: Sinkhorn-Knopp iterations used to project `comb`.
        target: The target device to run the kernel on.
        warps_per_block: Warps per block. Each carries
            `WARP_SIZE // hc_mult**2` rows.

    Args:
        pre: Stream-collapse weights. Shape: `[num_rows, hc_mult]`.
        post: Sublayer-output placement weights. Shape: `[num_rows, hc_mult]`.
        comb: Row-major stream mixer. Shape: `[num_rows, hc_mult * hc_mult]`.
        hc_proj: Projected streams. Shape:
            `[num_rows, 2 * hc_mult + hc_mult * hc_mult]`.
        pre_post_comb_b: Per-output bias, concatenated in `pre`, `post`, `comb`
            order. Shape: `[2 * hc_mult + hc_mult * hc_mult]`.
        pre_post_comb_scale: Per-output scale, in `pre`, `post`, `comb` order.
            Shape: `[3]`.
        hc_eps: Epsilon guarding the Sinkhorn divisions.
        context: The device context.

    Raises:
        If the target is not a GPU or the input widths disagree with `hc_mult`.
    """
    comptime assert is_gpu[
        target
    ](), "hyper_connection_gates is only supported on GPU"

    comptime mix_width = 2 * hc_mult + hc_mult * hc_mult

    var mix_dim = Int(hc_proj.dim(1))
    if mix_dim != mix_width:
        raise Error(
            "expected hc_proj width ",
            mix_width,
            " for hc_mult ",
            hc_mult,
            " but got ",
            mix_dim,
        )
    var bias_dim = Int(pre_post_comb_b.dim(0))
    if bias_dim != mix_width:
        raise Error(
            "expected pre_post_comb_b of size ",
            mix_width,
            " but got ",
            bias_dim,
        )
    var scale_dim = Int(pre_post_comb_scale.dim(0))
    if scale_dim != 3:
        raise Error(
            "expected pre_post_comb_scale of size 3 but got ",
            scale_dim,
        )

    var num_rows = Int(hc_proj.dim(0))
    if num_rows == 0:
        return

    var gpu_ctx = context

    with Trace[TraceLevel.OP, target=target](
        "mo.hyper_connection.gates", task_id=Int(gpu_ctx.device().id)
    ):
        comptime kernel = hyper_connection_gates_kernel[
            pre.LayoutType,
            post.LayoutType,
            comb.LayoutType,
            hc_proj.LayoutType,
            pre_post_comb_b.LayoutType,
            pre_post_comb_scale.LayoutType,
            hc_mult,
            hc_sinkhorn_iters,
            warps_per_block,
        ]

        comptime rows_per_block = warps_per_block * (
            WARP_SIZE // (hc_mult * hc_mult)
        )

        gpu_ctx.enqueue_function[kernel](
            pre,
            post,
            comb,
            hc_proj,
            pre_post_comb_b,
            pre_post_comb_scale,
            hc_eps,
            Int32(num_rows),
            grid_dim=ceildiv(num_rows, rows_per_block),
            block_dim=warps_per_block * WARP_SIZE,
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )


@inline(.always)
def mhc_fn_split[
    width: SIMDLength
](v: SIMD[.float32, width]) -> Tuple[
    SIMD[.bfloat16, width], SIMD[.bfloat16, width], SIMD[.bfloat16, width]
]:
    """Splits fp32 values into three bf16 terms whose sum is exactly `v`.

    Each round-to-nearest step keeps 8 significant bits and leaves a
    remainder at most half an ulp of the term, so the second remainder has at
    most 8 significant bits and the third term is exact (for normal `v`
    within the bfloat16 range; larger finite `v` rounds `hi` to inf).

    Parameters:
        width: SIMD width.

    Args:
        v: The fp32 values.

    Returns:
        The `(hi, lo, lo2)` terms, in decreasing magnitude.
    """
    var hi = v.cast[.bfloat16]()
    var r = v - hi.cast[.float32]()
    var lo = r.cast[.bfloat16]()
    var lo2 = (r - lo.cast[.float32]()).cast[.bfloat16]()
    return (hi, lo, lo2)


@inline(.always)
def _pin[
    dtype: DType, width: SIMDLength
](v: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Returns `v` through an empty asm the compiler cannot move or redo.

    Without it the scheduler sinks loads whose results wait across a barrier
    or a few steps to their use, or reloads them there, so each load is
    followed by a `vmcnt` wait right after it issues instead of overlapping
    the work in between. It can go once removing it leaves no such wait in
    the `.amdgcn` of `mhc_pre_kernel`, `mhc_pre_partials_kernel` and
    `mhc_pre_finish_kernel`.
    """
    return inlined_assembly[
        "", SIMD[dtype, width], constraints="=v,0", has_side_effect=True
    ](v)


comptime _mhc_pre_hc_mult = 4
"""The only `hc_mult` the `mhc_pre` kernels map: one stream per wave."""

comptime _mhc_pre_block_rows = 16
"""Token rows per `mhc_pre` block: the M of the bf16 16x16x32 MFMA."""

comptime _mhc_pre_lane_groups = 4
"""16-lane groups per wave: `WARP_SIZE // _mhc_pre_block_rows` on gfx950.

Each group feeds its own `d` run to the MFMA, and at the gates finishes one
token, so a block's `hc_mult` waves finish its 16 tokens.
"""

comptime _mhc_pre_lane_k = 16
"""Contiguous `d` per lane per projection step: two K=8 MFMA slots."""

comptime _mhc_pre_step_k = _mhc_pre_lane_groups * _mhc_pre_lane_k
"""`d` per projection step: one `_mhc_pre_lane_k` run per lane group."""

comptime _mhc_pre_k_splits_cap = 256
"""Largest K-split: the finish kernel stages and sums at most 256 slices."""

comptime _mhc_pre_target_blocks = 256
"""Blocks the K-split heuristic aims for: MI355X's CU count, one per CU."""

comptime _mhc_pre_threads = 256
"""Threads per `mhc_pre` block: one wave per stream."""

comptime _mhc_pre_vec[dtype: DType] = 16 // size_of[dtype]()
"""Elements of `dtype` per 16-byte collapse access."""


@inline(.always)
def _mhc_pre_project[
    dtype: DType,
    fn_dtype: DType,
    hidden_size: Int,
    terms: Int,
    num_steps: Int,
    masked: Bool,
](
    streams: TileTensor[dtype, ...],
    hc_fn: TileTensor[fn_dtype, ...],
    x_row: Int,
    stream: Int,
    d0: Int,
    fn_rows: Tuple[Int, Int],
    g: Int,
    valid_steps: Int,
    mut acc: Array[SIMD[.float32, 4], 2 * terms],
    mut sq: Array[SIMD[.float32, 4], 2],
):
    """Accumulates one wave's projection over `num_steps` 64-wide steps.

    The steps start at `d0` of stream `stream`, which is flat K index
    `stream * D + d0` of `hc_fn`. Lane `l` feeds token `l % 16` and, for both
    16-wide output tiles, the same 16 consecutive `d` per step, so the A and
    B operands agree on their K slots whatever order the MFMA sums them in.

    With `masked`, only the first `valid_steps` steps count: later ones
    re-read the last valid step and add zero.
    """
    comptime assert streams.flat_rank == 3
    comptime assert hc_fn.flat_rank == 2
    comptime D = hidden_size
    comptime lane_k = _mhc_pre_lane_k
    comptime step_k = _mhc_pre_step_k
    var k_base = stream * D + d0 + g * lane_k

    # Register rings `x_depth` and `fn_depth` steps deep. x streams from HBM;
    # fn is L2-resident and costs 2-4x x's registers per step, so it runs
    # shallower. Steps past the end re-read the last one rather than branch,
    # and an odd step count pads to the unroll with a zeroed step.
    comptime x_depth = min(2, num_steps)
    comptime fn_depth = 1
    comptime unroll = max(x_depth, fn_depth)
    comptime padded = align_up(num_steps, unroll)
    comptime zero_tail = masked or padded != num_steps
    var last = valid_steps - 1 if masked else num_steps - 1

    @inline(.always)
    def load_x(step: Int) {imm} -> SIMD[dtype, lane_k]:
        return streams.load[width=lane_k, alignment=16](
            (x_row, stream, d0 + min(step, last) * step_k + g * lane_k)
        )

    @inline(.always)
    def load_fn(step: Int, fn_row: Int) {imm} -> SIMD[fn_dtype, lane_k]:
        return hc_fn.load[width=lane_k, alignment=16](
            (fn_row, k_base + min(step, last) * step_k)
        )

    @inline(.always)
    def step(
        x_cur: SIMD[dtype, lane_k],
        f_cur: Array[SIMD[fn_dtype, lane_k], 2],
        index: Int,
    ) {mut acc, mut sq, imm last}:
        # The empty asm makes each operand a register the compiler cannot
        # reload at its use.
        var x_v = _pin(x_cur)
        comptime if zero_tail:
            if index > last:
                x_v = 0
        var f_v = Array[SIMD[fn_dtype, lane_k], 2](uninitialized=True)
        comptime for t in range(2):
            f_v[t] = _pin(f_cur[t])

        var x_f = x_v.cast[.float32]()
        comptime for q in range(4):
            var x_q = x_f.slice[4, offset=4 * q]()
            sq[q % 2] = x_q.fma(x_q, sq[q % 2])

        comptime for h in range(2):
            var a = x_v.slice[8, offset=8 * h]()
            comptime for t in range(2):
                var f = f_v[t].slice[8, offset=8 * h]()
                comptime if fn_dtype == .bfloat16:
                    mma(acc[t], a, f, acc[t])
                else:
                    var parts = mhc_fn_split(rebind[SIMD[.float32, 8]](f))
                    comptime for p in range(terms):
                        mma(
                            acc[t * terms + p],
                            a,
                            parts[p],
                            acc[t * terms + p],
                        )

    var x_ring = Array[SIMD[dtype, lane_k], x_depth](uninitialized=True)
    var f_ring = Array[SIMD[fn_dtype, lane_k], 2 * fn_depth](uninitialized=True)
    comptime for i in range(x_depth):
        x_ring[i] = load_x(i)
    comptime for i in range(fn_depth):
        comptime for t in range(2):
            f_ring[2 * i + t] = load_fn(i, fn_rows[t])

    comptime if num_steps <= unroll:
        # The whole working set is already in flight: no refills.
        comptime for i in range(num_steps):
            var f_cur = Array[SIMD[fn_dtype, lane_k], 2](uninitialized=True)
            comptime if i < fn_depth:
                comptime for t in range(2):
                    f_cur[t] = f_ring[2 * i + t]
            else:
                comptime for t in range(2):
                    f_cur[t] = load_fn(i, fn_rows[t])
            step(x_ring[i], f_cur, i)
    else:
        for s0 in range(0, padded, unroll):
            comptime for i in range(unroll):
                comptime xi = i % x_depth
                comptime fi = i % fn_depth
                var x_cur = x_ring[xi]
                var f_cur = Array[SIMD[fn_dtype, lane_k], 2](uninitialized=True)
                comptime for t in range(2):
                    f_cur[t] = f_ring[2 * fi + t]
                x_ring[xi] = load_x(s0 + i + x_depth)
                comptime for t in range(2):
                    f_ring[2 * fi + t] = load_fn(s0 + i + fn_depth, fn_rows[t])
                # Refills go out before this step waits on its operands, so
                # the wait counts them instead of draining; the barrier keeps
                # the scheduler from sinking the refills.
                schedule_barrier()
                step(x_cur, f_cur, s0 + i)


@inline(.always)
def _mhc_pre_stage_mix[
    terms: Int, mix_width: Int
](
    acc: Array[SIMD[.float32, 4], 2 * terms],
    sq: Array[SIMD[.float32, 4], 2],
    mix_s: TileTensor[mut=True, .float32, address_space=.SHARED, ...],
    sq_s: TileTensor[mut=True, .float32, address_space=.SHARED, ...],
    wave: Int,
    lane: Int,
):
    """Writes one wave's projection tiles and per-lane sums of squares."""
    comptime assert mix_s.flat_rank == 3
    comptime assert sq_s.flat_rank == 3
    comptime BM = _mhc_pre_block_rows
    comptime tile1_cols = mix_width - BM
    var m = lane % BM
    var g = lane // BM
    # MFMA output lane `l` holds rows `4 * (l / 16) + i` of column `l % 16`.
    comptime for t in range(2):
        var total = acc[t * terms]
        comptime for p in range(1, terms):
            total += acc[t * terms + p]
        comptime for i in range(4):
            comptime if t == 0:
                mix_s[wave, 4 * g + i, m] = total[i]
            else:
                if m < tile1_cols:
                    mix_s[wave, 4 * g + i, BM + m] = total[i]
    sq_s[wave, g, m] = (sq[0] + sq[1]).reduce_add()


@fieldwise_init
struct _MhcPreGateWeights(TrivialRegisterPassable):
    """One lane's gate scales and biases for its `pre`, `post`, `comb` entry."""

    var s_pre: Float32
    var s_post: Float32
    var s_comb: Float32
    var b_pre: Float32
    var b_post: Float32
    var b_comb: Float32

    @staticmethod
    @inline(.always)
    def load[
        hc_mult: Int
    ](
        base: TileTensor[.float32, ...],
        scale: TileTensor[.float32, ...],
        elem: Int,
    ) -> Self:
        comptime assert base.flat_rank == 1
        comptime assert scale.flat_rank == 1

        @inline(.always)
        def get(t: TileTensor[.float32, ...], i: Int) -> Float32:
            comptime assert t.flat_rank == 1
            return rebind[Float32](t.load((i,)))

        return Self(
            get(scale, 0),
            get(scale, 1),
            get(scale, 2),
            get(base, elem),
            get(base, hc_mult + elem),
            get(base, 2 * hc_mult + elem),
        )


@inline(.always)
def _mhc_pre_gates[
    hc_mult: Int, hc_sinkhorn_iters: Int
](
    post: TileTensor[mut=True, .float32, ...],
    comb: TileTensor[mut=True, .float32, ...],
    pre: TileTensor[mut=True, .float32, ...],
    weights: _MhcPreGateWeights,
    mix_pre: Float32,
    mix_post: Float32,
    mix_comb: Float32,
    rms: Float32,
    row: Int,
    live: Bool,
    elem: Int,
    hc_eps: Float32,
) -> Float32:
    """Writes one token's gates from its projections and returns its `pre`.

    As in `hyper_connection_gates_kernel`, 16 consecutive lanes carry one
    token, lane `elem` taking `comb` entry `elem` and, below `hc_mult`, the
    `pre` and `post` entries. Every lane of the wave must call this; only
    `live` lanes store.
    """
    comptime assert post.flat_rank == 2
    comptime assert comb.flat_rank == 3
    comptime assert pre.flat_rank == 2
    comptime H = hc_mult
    var pre_v = _mhc_pre_gate(
        mix_pre * rms, weights.s_pre, weights.b_pre, hc_eps
    )
    if elem < H and live:
        pre[row, elem] = pre_v
        post[row, elem] = _mhc_post_gate(
            mix_post * rms, weights.s_post, weights.b_post
        )

    var logit = mix_comb * rms * weights.s_comb + weights.b_comb
    var c = _mhc_comb_gate[H, hc_sinkhorn_iters](logit, hc_eps)
    if live:
        comb[row, elem // H, elem % H] = c
    return pre_v


@inline(.always)
def _mhc_pre_load_row[
    dtype: DType, hc_mult: Int, hidden_size: Int, waves_per_row: Int
](
    streams: TileTensor[dtype, ...],
    t: Int,
    lane: Int,
    slot: Int,
) -> Array[
    SIMD[dtype, _mhc_pre_vec[dtype]],
    ceildiv(
        hidden_size // (WARP_SIZE * _mhc_pre_vec[dtype]),
        waves_per_row,
    )
    * hc_mult,
]:
    """Loads the chunks of token `t`'s streams that wave `slot` collapses.

    Wave `slot` of `waves_per_row` takes chunks `slot + waves_per_row * i`,
    16 bytes per lane per stream; a chunk past the row re-reads the last one.
    """
    comptime assert streams.flat_rank == 3
    comptime D = hidden_size
    comptime width = _mhc_pre_vec[dtype]
    comptime chunk = WARP_SIZE * width
    comptime num_chunks = D // chunk
    comptime wave_chunks = ceildiv(num_chunks, waves_per_row)
    var s_v = Array[SIMD[dtype, width], wave_chunks * hc_mult](
        uninitialized=True
    )
    comptime for i in range(wave_chunks):
        var ci = slot + waves_per_row * i
        comptime if num_chunks % waves_per_row != 0:
            ci = min(ci, num_chunks - 1)
        var d = ci * chunk + lane * width
        comptime for j in range(hc_mult):
            s_v[i * hc_mult + j] = streams.load[width=width, alignment=16](
                (t, j, d)
            )
    return s_v^


@inline(.always)
def _mhc_pre_collapse[
    dtype: DType,
    hc_mult: Int,
    hidden_size: Int,
    has_norm: Bool,
    waves_per_row: Int,
    LoadFn: def(Int, Int) -> SIMD[dtype, _mhc_pre_vec[dtype]],
    GammaFn: def(Int) -> SIMD[dtype, _mhc_pre_vec[dtype]],
](
    collapsed: TileTensor[mut=True, dtype, ...],
    t: Int,
    p: SIMD[.float32, hc_mult],
    load: LoadFn,
    gamma_at: GammaFn,
    lane: Int,
    slot: Int,
    norm_eps: Float32,
):
    """Collapses and stores wave `slot`'s chunks of token `t`.

    Wave `slot` of `waves_per_row` takes chunks `slot + waves_per_row * i`,
    16 bytes per lane per stream; `load(i, j)` returns stream `j` of its
    `i`-th chunk and `gamma_at(i)` the norm weight there. With
    `waves_per_row > 1`, every wave of the block must call this, as the norm
    sums the row across them, and a block may call it at most once: no
    barrier follows the last read of the shared row sums.
    """
    comptime assert collapsed.flat_rank == 2
    comptime D = hidden_size
    comptime H = hc_mult
    comptime width = _mhc_pre_vec[dtype]
    comptime chunk = WARP_SIZE * width
    comptime num_chunks = D // chunk
    comptime wave_chunks = ceildiv(num_chunks, waves_per_row)
    comptime exact = num_chunks % waves_per_row == 0

    var out_v = Array[SIMD[dtype, width], wave_chunks](uninitialized=True)
    var row_sq = SIMD[.float32, width](0)
    comptime for i in range(wave_chunks):
        var ci = slot + waves_per_row * i
        var live = exact or ci < num_chunks
        var d = ci * chunk + lane * width
        var y = load(i, 0).cast[.float32]() * p[0]
        comptime for j in range(1, H):
            y = load(i, j).cast[.float32]().fma(p[j], y)
        out_v[i] = y.cast[dtype]()
        comptime if has_norm:
            var y_r = out_v[i].cast[.float32]()
            if live:
                row_sq = y_r.fma(y_r, row_sq)
        else:
            if live:
                collapsed.store[width=width, alignment=16]((t, d), out_v[i])

    comptime if has_norm:
        # The RMSNorm kernel's sequence: normalize the rounded collapse in
        # fp32, round, then multiply by gamma in `dtype`.
        var row_ss = warp.sum(row_sq.reduce_add())
        comptime if waves_per_row > 1:
            var ss_s = stack_allocation[.float32, address_space=.SHARED](
                row_major[waves_per_row]()
            )
            if lane == 0:
                ss_s[slot] = row_ss
            barrier()
            row_ss = rebind[Float32](ss_s[0])
            comptime for w in range(1, waves_per_row):
                row_ss += rebind[Float32](ss_s[w])
        var factor = rsqrt(row_ss / Float32(D) + norm_eps)
        comptime for i in range(wave_chunks):
            var ci = slot + waves_per_row * i
            var d = ci * chunk + lane * width
            if exact or ci < num_chunks:
                var normed = (out_v[i].cast[.float32]() * factor).cast[
                    dtype
                ]() * gamma_at(i)
                collapsed.store[width=width, alignment=16]((t, d), normed)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](_mhc_pre_threads)
)
@__name(
    t"mhc_pre_h{hc_mult}_d{hidden_size}_s{hc_sinkhorn_iters}_{fn_dtype}x{fn_terms}_n{Int(has_norm)}"
)
def mhc_pre_kernel[
    dtype: DType,
    fn_dtype: DType,
    PostLayoutType: TensorLayout,
    CombLayoutType: TensorLayout,
    CollapsedLayoutType: TensorLayout,
    PreLayoutType: TensorLayout,
    StreamsLayoutType: TensorLayout,
    FnLayoutType: TensorLayout,
    BaseLayoutType: TensorLayout,
    ScaleLayoutType: TensorLayout,
    GammaLayoutType: TensorLayout,
    hc_mult: Int,
    hidden_size: Int,
    hc_sinkhorn_iters: Int,
    has_norm: Bool,
    fn_terms: Int,
](
    post: TileTensor[mut=True, .float32, PostLayoutType, MutAnyOrigin],
    comb: TileTensor[mut=True, .float32, CombLayoutType, MutAnyOrigin],
    collapsed: TileTensor[mut=True, dtype, CollapsedLayoutType, MutAnyOrigin],
    pre: TileTensor[mut=True, .float32, PreLayoutType, MutAnyOrigin],
    streams: TileTensor[dtype, StreamsLayoutType, ImmutAnyOrigin],
    hc_fn: TileTensor[fn_dtype, FnLayoutType, ImmutAnyOrigin],
    base: TileTensor[.float32, BaseLayoutType, ImmutAnyOrigin],
    scale: TileTensor[.float32, ScaleLayoutType, ImmutAnyOrigin],
    gamma: TileTensor[dtype, GammaLayoutType, ImmutAnyOrigin],
    rms_eps: Float32,
    hc_eps: Float32,
    norm_eps: Float32,
    num_rows: Int32,
):
    """Runs a whole mHC pre site for 16 tokens: projection, gates, collapse.

    Wave `j` projects stream `j` (a quarter of `K = hc_mult * hidden_size`)
    for the block's 16 tokens on `v_mfma_f32_16x16x32_bf16`, two 16-wide
    tiles covering the 24 mix outputs (see `_mhc_pre_project`). The waves
    then reduce through LDS in a fixed order, and each wave finishes four
    tokens: the RMS post-scale, the gates, then the collapse.

    Parameters:
        dtype: Element type of the streams and the collapsed output.
        fn_dtype: Element type of the projection weights.
        PostLayoutType: Layout of `post`.
        CombLayoutType: Layout of `comb`.
        CollapsedLayoutType: Layout of `collapsed`.
        PreLayoutType: Layout of `pre`.
        StreamsLayoutType: Layout of `streams`.
        FnLayoutType: Layout of `hc_fn`.
        BaseLayoutType: Layout of `base`.
        ScaleLayoutType: Layout of `scale`.
        GammaLayoutType: Layout of `gamma`.
        hc_mult: Number of parallel residual streams.
        hidden_size: Width of one stream.
        hc_sinkhorn_iters: Sinkhorn-Knopp iterations used to project `comb`.
        has_norm: Whether to RMS-normalize the collapsed row by `gamma`.
        fn_terms: Number of `mhc_fn_split` bf16 terms each fp32 weight
            contributes.

    Args:
        post: Sublayer-output placement weights. Shape: `[T, hc_mult]`.
        comb: Stream mixer; `comb[t, j, k]` weighs input stream `j` in output
            stream `k`. Shape: `[T, hc_mult, hc_mult]`.
        collapsed: The collapsed (optionally normalized) sequence. Shape:
            `[T, hidden_size]`.
        pre: Stream-collapse weights. Shape: `[T, hc_mult]`.
        streams: Residual streams. Shape: `[T, hc_mult, hidden_size]`.
        hc_fn: Projection weights, `pre`, `post`, `comb` rows. Shape:
            `[2 * hc_mult + hc_mult**2, hc_mult * hidden_size]`.
        base: Per-output bias. Shape: `[2 * hc_mult + hc_mult**2]`.
        scale: Per-output scale, in `pre`, `post`, `comb` order. Shape: `[3]`.
        gamma: RMSNorm weight. Shape: `[hidden_size]`, read only with
            `has_norm`.
        rms_eps: Epsilon of the RMS scale on the projection.
        hc_eps: Epsilon guarding the gates and the Sinkhorn divisions.
        norm_eps: Epsilon of the collapsed-row RMSNorm.
        num_rows: Number of tokens `T`.
    """
    comptime assert is_amd_gpu["gfx950"](), "mhc_pre requires gfx950"
    comptime assert post.flat_rank == 2
    comptime assert comb.flat_rank == 3
    comptime assert collapsed.flat_rank == 2
    comptime assert pre.flat_rank == 2
    comptime assert streams.flat_rank == 3
    comptime assert hc_fn.flat_rank == 2
    comptime assert base.flat_rank == 1
    comptime assert scale.flat_rank == 1
    comptime assert gamma.flat_rank == 1

    comptime H = hc_mult
    comptime HH = H * H
    comptime mix_width = 2 * H + HH
    comptime BM = _mhc_pre_block_rows
    comptime D = hidden_size
    comptime num_steps = D // _mhc_pre_step_k
    comptime tile1_cols = mix_width - BM
    comptime terms = 1 if fn_dtype == .bfloat16 else fn_terms
    comptime groups = _mhc_pre_lane_groups
    comptime assert H == _mhc_pre_hc_mult, "mhc_pre maps one stream per wave"
    comptime assert mix_width > BM and mix_width <= 2 * BM
    comptime assert groups * BM == WARP_SIZE
    comptime assert _mhc_pre_threads == H * WARP_SIZE
    comptime assert H * groups == BM, "each lane group finishes one token"

    var lane = lane_id()
    var wave = warp_id()
    var m = lane % BM
    var g = lane // BM
    var rows = Int(num_rows)
    var row0 = block_idx.x * BM

    # Tail tokens read the last live token instead of masking the loads; their
    # results are never stored.
    var x_row = min(row0 + m, rows - 1)
    # The second tile's columns past 24 repeat rows 16-23 of `hc_fn`; they are
    # computed and dropped, which is cheaper than masking their loads.
    var fn_rows = (m, BM + m % tile1_cols)

    var acc = Array[SIMD[.float32, 4], 2 * terms](fill=0)
    var sq = Array[SIMD[.float32, 4], 2](fill=0)
    _mhc_pre_project[dtype, fn_dtype, D, terms, num_steps, masked=False](
        streams, hc_fn, x_row, wave, 0, fn_rows, g, num_steps, acc, sq
    )

    var mix_s = stack_allocation[.float32, address_space=.SHARED](
        row_major[H, BM, mix_width]()
    )
    var sq_s = stack_allocation[.float32, address_space=.SHARED](
        row_major[H, groups, BM]()
    )
    var pre_s = stack_allocation[.float32, address_space=.SHARED](
        row_major[BM, H]()
    )
    _mhc_pre_stage_mix[terms, mix_width](acc, sq, mix_s, sq_s, wave, lane)
    barrier()

    # Gates: as in `hyper_connection_gates_kernel`, 16 lanes per token, so a
    # wave finishes tokens `groups * wave + g`.
    var r = groups * wave + g
    var elem = m
    var row = row0 + r

    @inline(.always)
    def mix(c: Int) {imm} -> Float32:
        var v = mix_s[0, r, c]
        comptime for w in range(1, H):
            v += mix_s[w, r, c]
        return rebind[Float32](v)

    var ss = Float32(0)
    comptime for w in range(H):
        comptime for gg in range(groups):
            ss += rebind[Float32](sq_s[w, gg, r])
    var rms = rsqrt(ss / Float32(H * D) + rms_eps)

    var pre_v = _mhc_pre_gates[H, hc_sinkhorn_iters](
        post,
        comb,
        pre,
        _MhcPreGateWeights.load[H](base, scale, elem),
        mix(elem),
        mix(H + elem),
        mix(2 * H + elem),
        rms,
        row,
        row < rows,
        elem,
        hc_eps,
    )
    if elem < H:
        pre_s[r, elem] = pre_v
    barrier()

    # Collapse: each wave re-reads its `groups` tokens, 16 bytes per lane per
    # stream, and keeps the row in registers when the norm needs its RMS.
    for q in range(groups):
        var r_q = groups * wave + q
        var t = row0 + r_q
        if t >= rows:
            break
        var p = SIMD[.float32, H]()
        comptime for j in range(H):
            p[j] = rebind[Float32](pre_s[r_q, j])
        comptime width = _mhc_pre_vec[dtype]

        @inline(.always)
        def load(i: Int, j: Int) {imm} -> SIMD[dtype, width]:
            return streams.load[width=width, alignment=16](
                (t, j, (i * WARP_SIZE + lane) * width)
            )

        @inline(.always)
        def gamma_at(i: Int) {imm} -> SIMD[dtype, width]:
            return gamma.load[width=width, alignment=16](
                ((i * WARP_SIZE + lane) * width,)
            )

        _mhc_pre_collapse[dtype, H, D, has_norm, waves_per_row=1](
            collapsed, t, p, load, gamma_at, lane, 0, norm_eps
        )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](_mhc_pre_threads)
)
@__name(
    t"mhc_pre_partials_h{hc_mult}_d{hidden_size}_{fn_dtype}x{fn_terms}_k{k_splits}"
)
def mhc_pre_partials_kernel[
    dtype: DType,
    fn_dtype: DType,
    StreamsLayoutType: TensorLayout,
    FnLayoutType: TensorLayout,
    PartialLayoutType: TensorLayout,
    hc_mult: Int,
    hidden_size: Int,
    fn_terms: Int,
    k_splits: Int,
](
    partial: TileTensor[mut=True, .float32, PartialLayoutType, MutAnyOrigin],
    streams: TileTensor[dtype, StreamsLayoutType, ImmutAnyOrigin],
    hc_fn: TileTensor[fn_dtype, FnLayoutType, ImmutAnyOrigin],
    num_rows: Int32,
):
    """Projects 16 tokens over one K slice into fp32 partial sums.

    Block `(mb, ks)` owns tokens `[16 * mb, 16 * mb + 16)` and the contiguous
    slice `[ks * K / k_splits, (ks + 1) * K / k_splits)` of the flat
    `hc_mult * hidden_size` axis, which its waves split into 64-wide steps as
    `mhc_pre_kernel`'s waves split K. It stores the 24 projections and the
    sum of squares of each token's slice at `partial[t, ks * 25 + c]`, so a
    token's partials are one contiguous run for `mhc_pre_finish_kernel`.

    Parameters:
        dtype: Element type of the streams.
        fn_dtype: Element type of the projection weights.
        StreamsLayoutType: Layout of `streams`.
        FnLayoutType: Layout of `hc_fn`.
        PartialLayoutType: Layout of `partial`.
        hc_mult: Number of parallel residual streams.
        hidden_size: Width of one stream.
        fn_terms: Number of `mhc_fn_split` bf16 terms each fp32 weight
            contributes.
        k_splits: Number of K slices, one per block along `y`.

    Args:
        partial: Per-slice sums. Shape: `[16 * ceil(T / 16), >= k_splits *
            (2 * hc_mult + hc_mult**2 + 1)]`.
        streams: Residual streams. Shape: `[T, hc_mult, hidden_size]`.
        hc_fn: Projection weights. Shape:
            `[2 * hc_mult + hc_mult**2, hc_mult * hidden_size]`.
        num_rows: Number of tokens `T`.
    """
    comptime assert is_amd_gpu["gfx950"](), "mhc_pre requires gfx950"
    comptime assert partial.flat_rank == 2
    comptime assert streams.flat_rank == 3
    comptime assert hc_fn.flat_rank == 2

    comptime H = hc_mult
    comptime mix_width = 2 * H + H * H
    comptime BM = _mhc_pre_block_rows
    comptime D = hidden_size
    comptime slice_k = H * D // k_splits
    comptime slice_steps = slice_k // _mhc_pre_step_k
    comptime wave_steps = ceildiv(slice_steps, H)
    comptime terms = 1 if fn_dtype == .bfloat16 else fn_terms
    comptime groups = _mhc_pre_lane_groups
    comptime assert H == _mhc_pre_hc_mult, "mhc_pre maps one stream per wave"
    comptime assert groups * BM == WARP_SIZE
    comptime assert _mhc_pre_threads == H * WARP_SIZE
    comptime assert _mhc_pre_k_splits_ok(D, k_splits)

    var lane = lane_id()
    var wave = warp_id()
    var m = lane % BM
    var g = lane // BM
    var rows = Int(num_rows)
    var row0 = block_idx.x * BM
    var ks = block_idx.y

    var x_row = min(row0 + m, rows - 1)
    var fn_rows = (m, BM + m % (mix_width - BM))
    var first = wave * wave_steps
    var k0 = ks * slice_k + first * _mhc_pre_step_k

    var acc = Array[SIMD[.float32, 4], 2 * terms](fill=0)
    var sq = Array[SIMD[.float32, 4], 2](fill=0)
    # Small slices leave the last waves without a step; they add zeros.
    if first < slice_steps:
        _mhc_pre_project[
            dtype,
            fn_dtype,
            D,
            terms,
            wave_steps,
            masked=slice_steps % wave_steps != 0,
        ](
            streams,
            hc_fn,
            x_row,
            k0 // D,
            k0 % D,
            fn_rows,
            g,
            min(wave_steps, slice_steps - first),
            acc,
            sq,
        )

    var mix_s = stack_allocation[.float32, address_space=.SHARED](
        row_major[H, BM, mix_width]()
    )
    var sq_s = stack_allocation[.float32, address_space=.SHARED](
        row_major[H, groups, BM]()
    )
    _mhc_pre_stage_mix[terms, mix_width](acc, sq, mix_s, sq_s, wave, lane)
    barrier()

    # Fixed-order sums over the waves, as `mhc_pre_kernel` reduces them.
    comptime row_vals = mix_width + 1
    comptime for e0 in range(0, BM * row_vals, _mhc_pre_threads):
        var e = e0 + thread_idx.x
        if e < BM * row_vals:
            var r = e // row_vals
            var c = e % row_vals
            var v = Float32(0)
            if c < mix_width:
                v = rebind[Float32](mix_s[0, r, c])
                comptime for w in range(1, H):
                    v += rebind[Float32](mix_s[w, r, c])
            else:
                comptime for w in range(H):
                    comptime for gg in range(groups):
                        v += rebind[Float32](sq_s[w, gg, r])
            partial[row0 + r, ks * row_vals + c] = v


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](_mhc_pre_threads)
)
@__name(
    t"mhc_pre_finish_h{hc_mult}_d{hidden_size}_s{hc_sinkhorn_iters}_n{Int(has_norm)}"
)
def mhc_pre_finish_kernel[
    dtype: DType,
    PostLayoutType: TensorLayout,
    CombLayoutType: TensorLayout,
    CollapsedLayoutType: TensorLayout,
    PreLayoutType: TensorLayout,
    StreamsLayoutType: TensorLayout,
    PartialLayoutType: TensorLayout,
    BaseLayoutType: TensorLayout,
    ScaleLayoutType: TensorLayout,
    GammaLayoutType: TensorLayout,
    hc_mult: Int,
    hidden_size: Int,
    hc_sinkhorn_iters: Int,
    has_norm: Bool,
](
    post: TileTensor[mut=True, .float32, PostLayoutType, MutAnyOrigin],
    comb: TileTensor[mut=True, .float32, CombLayoutType, MutAnyOrigin],
    collapsed: TileTensor[mut=True, dtype, CollapsedLayoutType, MutAnyOrigin],
    pre: TileTensor[mut=True, .float32, PreLayoutType, MutAnyOrigin],
    streams: TileTensor[dtype, StreamsLayoutType, ImmutAnyOrigin],
    partial: TileTensor[.float32, PartialLayoutType, ImmutAnyOrigin],
    base: TileTensor[.float32, BaseLayoutType, ImmutAnyOrigin],
    scale: TileTensor[.float32, ScaleLayoutType, ImmutAnyOrigin],
    gamma: TileTensor[dtype, GammaLayoutType, ImmutAnyOrigin],
    rms_eps: Float32,
    hc_eps: Float32,
    norm_eps: Float32,
    k_splits: Int32,
):
    """Finishes one token from its K-split partials: gates, then collapse.

    One block per token keeps a decode batch of a few tokens on as many CUs.
    The block sums the token's `k_splits` partials in a fixed order (lane
    `l` of a wave adds splits `l, l + 64, ...` in turn, then a warp
    reduction), so the result does not depend on scheduling.

    Parameters:
        dtype: Element type of the streams and the collapsed output.
        PostLayoutType: Layout of `post`.
        CombLayoutType: Layout of `comb`.
        CollapsedLayoutType: Layout of `collapsed`.
        PreLayoutType: Layout of `pre`.
        StreamsLayoutType: Layout of `streams`.
        PartialLayoutType: Layout of `partial`.
        BaseLayoutType: Layout of `base`.
        ScaleLayoutType: Layout of `scale`.
        GammaLayoutType: Layout of `gamma`.
        hc_mult: Number of parallel residual streams.
        hidden_size: Width of one stream.
        hc_sinkhorn_iters: Sinkhorn-Knopp iterations used to project `comb`.
        has_norm: Whether to RMS-normalize the collapsed row by `gamma`.

    Args:
        post: Sublayer-output placement weights. Shape: `[T, hc_mult]`.
        comb: Stream mixer. Shape: `[T, hc_mult, hc_mult]`.
        collapsed: The collapsed (optionally normalized) sequence. Shape:
            `[T, hidden_size]`.
        pre: Stream-collapse weights. Shape: `[T, hc_mult]`.
        streams: Residual streams. Shape: `[T, hc_mult, hidden_size]`.
        partial: `mhc_pre_partials_kernel`'s sums, rows 16-byte aligned.
        base: Per-output bias. Shape: `[2 * hc_mult + hc_mult**2]`.
        scale: Per-output scale, in `pre`, `post`, `comb` order. Shape: `[3]`.
        gamma: RMSNorm weight. Shape: `[hidden_size]`, read only with
            `has_norm`.
        rms_eps: Epsilon of the RMS scale on the projection.
        hc_eps: Epsilon guarding the gates and the Sinkhorn divisions.
        norm_eps: Epsilon of the collapsed-row RMSNorm.
        k_splits: Number of K slices in `partial`, at most 256.
    """
    comptime assert is_amd_gpu["gfx950"](), "mhc_pre requires gfx950"
    comptime assert partial.flat_rank == 2

    comptime H = hc_mult
    comptime mix_width = 2 * H + H * H
    comptime row_vals = mix_width + 1
    comptime threads = _mhc_pre_threads
    comptime max_vecs = ceildiv(_mhc_pre_k_splits_cap * row_vals, 4 * threads)
    comptime split_passes = _mhc_pre_k_splits_cap // WARP_SIZE
    comptime assert H == _mhc_pre_hc_mult, "mhc_pre maps one stream per wave"
    comptime assert threads == H * WARP_SIZE

    var lane = lane_id()
    var wave = warp_id()
    var tid = thread_idx.x
    var t = block_idx.x
    var splits = Int(k_splits)
    var elem = lane % _mhc_pre_block_rows

    # Everything the token needs besides its partials is independent of
    # them, so it all goes out first: each later round trip would add a
    # memory latency to this latency-bound kernel.
    var weights = _MhcPreGateWeights.load[H](base, scale, elem)
    comptime width = _mhc_pre_vec[dtype]
    comptime num_chunks = hidden_size // (WARP_SIZE * width)
    comptime wave_chunks = ceildiv(num_chunks, H)
    var g_v = Array[SIMD[dtype, width], wave_chunks](fill=0)
    comptime if has_norm:
        comptime assert gamma.flat_rank == 1
        comptime for i in range(wave_chunks):
            var ci = min(wave + H * i, num_chunks - 1)
            g_v[i] = gamma.load[width=width, alignment=16](
                ((ci * WARP_SIZE + lane) * width,)
            )
    var s_v = _mhc_pre_load_row[dtype, H, hidden_size, waves_per_row=H](
        streams, t, lane, wave
    )
    # Vectors past the token's run re-read its last one; their LDS copies are
    # never summed.
    var num_vecs = ceildiv(splits * row_vals, 4)
    var p_v = Array[SIMD[.float32, 4], max_vecs](uninitialized=True)
    comptime for i in range(max_vecs):
        p_v[i] = partial.load[width=4, alignment=16](
            (t, 4 * min(tid + threads * i, num_vecs - 1))
        )
    # The streams stay in registers across the barriers below; the empty asm
    # keeps the compiler from sinking their loads to the collapse.
    comptime for i in range(wave_chunks * H):
        s_v[i] = _pin(s_v[i])
    comptime if has_norm:
        comptime for i in range(wave_chunks):
            g_v[i] = _pin(g_v[i])

    var part_s = stack_allocation[.float32, address_space=.SHARED](
        row_major[max_vecs * threads * 4]()
    )
    var mix_row = stack_allocation[.float32, address_space=.SHARED](
        row_major[row_vals]()
    )
    var pre_row = stack_allocation[.float32, address_space=.SHARED](
        row_major[H]()
    )
    comptime for i in range(max_vecs):
        part_s.store[width=4, alignment=16]((4 * (tid + threads * i),), p_v[i])
    barrier()

    var part_rows = part_s.reshape(row_major[_mhc_pre_k_splits_cap, row_vals]())
    comptime for i in range(ceildiv(row_vals, H)):
        var c = wave + H * i
        if c < row_vals:
            var v = Float32(0)
            comptime for j in range(split_passes):
                var s = lane + WARP_SIZE * j
                var x = rebind[Float32](part_rows[s, c])
                v += x if s < splits else 0
            v = warp.sum(v)
            if lane == 0:
                mix_row[c] = v
    barrier()

    if wave == 0:
        var rms = rsqrt(
            rebind[Float32](mix_row[mix_width]) / Float32(H * hidden_size)
            + rms_eps
        )
        var pre_v = _mhc_pre_gates[H, hc_sinkhorn_iters](
            post,
            comb,
            pre,
            weights,
            rebind[Float32](mix_row[elem]),
            rebind[Float32](mix_row[H + elem]),
            rebind[Float32](mix_row[2 * H + elem]),
            rms,
            t,
            lane < _mhc_pre_block_rows,
            elem,
            hc_eps,
        )
        if lane < H:
            pre_row[lane] = pre_v
    barrier()

    var p = SIMD[.float32, H]()
    comptime for j in range(H):
        p[j] = rebind[Float32](pre_row[j])

    @inline(.always)
    def load(i: Int, j: Int) {imm} -> SIMD[dtype, width]:
        return s_v[i * H + j]

    @inline(.always)
    def gamma_at(i: Int) {imm} -> SIMD[dtype, width]:
        return g_v[i]

    _mhc_pre_collapse[dtype, H, hidden_size, has_norm, waves_per_row=H](
        collapsed, t, p, load, gamma_at, lane, wave, norm_eps
    )


def _mhc_pre_k_splits_ok(hidden_size: Int, k_splits: Int) -> Bool:
    """Returns whether `k_splits` slices of K are whole 64-wide steps each
    and no wave's steps cross a stream."""
    comptime H = _mhc_pre_hc_mult
    comptime step_k = _mhc_pre_step_k
    if (
        k_splits > _mhc_pre_k_splits_cap
        or (H * hidden_size // step_k) % k_splits != 0
    ):
        return False
    # A slice either divides a stream or, at one or two splits, divides
    # evenly across the waves.
    var slice_k = H * hidden_size // k_splits
    var slice_steps = slice_k // step_k
    return hidden_size % slice_k == 0 or (
        slice_steps % H == 0
        and hidden_size % (ceildiv(slice_steps, H) * step_k) == 0
    )


def _mhc_pre_max_k_splits(hidden_size: Int) -> Int:
    """Returns the largest valid power-of-two K-split."""
    var ks = 1
    while _mhc_pre_k_splits_ok(hidden_size, 2 * ks):
        ks *= 2
    return ks


def mhc_pre_launch_config(
    num_rows: Int, hidden_size: Int, force_k_splits: Int = 0
) raises -> Tuple[Int, Int, Int]:
    """Returns how `mhc_pre` launches for `T` tokens at `hc_mult = 4`.

    Args:
        num_rows: Number of tokens `T`.
        hidden_size: Width of one stream.
        force_k_splits: K-splits to use; 0 picks.

    Returns:
        `(MB, KS, num_launches)`: `MB = ceil(T / 16)` token blocks, `KS`
        slices of K per block, and 1 launch when `KS == 1` (the fused
        kernel), 2 otherwise (partials, then finish), 0 when `T == 0`.

    Raises:
        If `force_k_splits` is not 0 or a power of two that divides
        `4 * hidden_size / 64`, at most 256.
    """
    var max_ks = _mhc_pre_max_k_splits(hidden_size)
    var mb = ceildiv(num_rows, _mhc_pre_block_rows)
    var ks = 1
    if force_k_splits != 0:
        if (
            force_k_splits < 0
            or not force_k_splits.is_power_of_two()
            or not _mhc_pre_k_splits_ok(hidden_size, force_k_splits)
        ):
            raise Error(
                "mhc_pre supports force_k_splits 0 or a power of two up to ",
                max_ks,
                " but got ",
                force_k_splits,
            )
        ks = force_k_splits
    elif mb > 0:
        # Enough blocks to cover the GPU once.
        while 2 * ks * mb <= _mhc_pre_target_blocks and 2 * ks <= max_ks:
            ks *= 2
        # From 129 to 192 token blocks the fused grid is 0.5-0.75 blocks per
        # CU and leaves CUs idle. A 2-way split overfills them at 1-1.5 per
        # CU and is no faster; a 4-way split, at 2-3 per CU, wins up to 192
        # blocks. Calibrated at D=4096 with a bf16 hc_fn and the norm on.
        comptime fallback_ks = 4
        comptime tuned_max_blocks_per_cu = 3
        if (
            ks == 1
            and fallback_ks * mb
            <= tuned_max_blocks_per_cu * _mhc_pre_target_blocks
            and fallback_ks <= max_ks
        ):
            ks = fallback_ks
    var launches = 0 if mb == 0 else (1 if ks == 1 else 2)
    return (mb, ks, launches)


@inline(.always)
def mhc_pre[
    hc_mult: Int,
    hidden_size: Int,
    hc_sinkhorn_iters: Int,
    has_norm: Bool,
    target: StaticString,
    fn_terms: Int = 3,
](
    post: TileTensor[mut=True, .float32, ...],
    comb: TileTensor[mut=True, .float32, ...],
    collapsed: TileTensor[mut=True, ...],
    pre: TileTensor[mut=True, .float32, ...],
    streams: TileTensor[mut=False, collapsed.dtype, ...],
    hc_fn: TileTensor[mut=False, ...],
    base: TileTensor[mut=False, .float32, ...],
    scale: TileTensor[mut=False, .float32, ...],
    gamma: TileTensor[mut=False, collapsed.dtype, ...],
    rms_eps: Float32,
    hc_eps: Float32,
    norm_eps: Float32,
    context: DeviceContext,
    force_k_splits: Int = 0,
) raises:
    """Launches the fused mHC pre site on blocks of 16 tokens and 256 threads.

    Computes, per token `t` with `x = streams[t]` flattened to `hc_mult *
    hidden_size` values:

    - `mix = (x @ hc_fn.T) * rsqrt(mean(x * x) + rms_eps)`;
    - `pre`, `post` and `comb` from `mix`, as `hyper_connection_gates` does;
    - `collapsed[t] = sum_j pre[t, j] * streams[t, j]`, rounded to the
      streams' dtype, then with `has_norm` RMS-normalized by `gamma`.

    The projection runs on bf16 MFMA with fp32 accumulation. A bf16 `hc_fn` is
    one exact term; an fp32 `hc_fn` is split into `fn_terms` bf16 terms, and at
    3 their products are exact, so the result is at least as accurate as an
    fp32 GEMM. Requires gfx950.

    `mhc_pre_launch_config` picks the K-splits `KS` that bring the grid to
    about 256 blocks, and `KS = 4` from 129 to 192 token blocks, where the
    fused grid would leave CUs idle. At `KS = 1` one kernel does the
    whole site. Above it, `KS` blocks per 16 tokens each project a slice of K
    into a per-call fp32 workspace, and a second launch on the same stream
    sums each token's slices in a fixed order and finishes it, one block per
    token. Both forms are deterministic.

    Parameters:
        hc_mult: Number of parallel residual streams. Only 4.
        hidden_size: Width of one stream: a multiple of 1024 in
            `[1024, 8192]`.
        hc_sinkhorn_iters: Sinkhorn-Knopp iterations used to project `comb`.
        has_norm: Whether to RMS-normalize the collapsed row by `gamma`.
        target: The target device to run the kernel on.
        fn_terms: Test-only: number of bf16 terms per fp32 `hc_fn` weight,
            1 to 3. Callers keep the default 3; fewer drop precision and exist
            so tests can check that every term matters. Ignored for a bf16
            `hc_fn`.

    Args:
        post: Sublayer-output placement weights. Shape: `[T, hc_mult]`.
        comb: Stream mixer; `comb[t, j, k]` weighs input stream `j` in output
            stream `k`. Shape: `[T, hc_mult, hc_mult]`.
        collapsed: The collapsed sequence, normalized with `has_norm`. Shape:
            `[T, hidden_size]`.
        pre: Stream-collapse weights. Shape: `[T, hc_mult]`.
        streams: The bf16 residual streams. Shape: `[T, hc_mult, hidden_size]`.
        hc_fn: The fp32 or bf16 projection weights. Shape:
            `[2 * hc_mult + hc_mult**2, hc_mult * hidden_size]`.
        base: Per-output bias. Shape: `[2 * hc_mult + hc_mult**2]`.
        scale: Per-output scale, in `pre`, `post`, `comb` order. Shape: `[3]`.
        gamma: RMSNorm weight. Shape: `[hidden_size]` with `has_norm`, else
            `[1]`.
        rms_eps: Epsilon of the RMS scale on the projection.
        hc_eps: Epsilon guarding the gates and the Sinkhorn divisions.
        norm_eps: Epsilon of the collapsed-row RMSNorm.
        context: The device context.
        force_k_splits: Splits of K across blocks; 0 picks, 1 forces none.

    Raises:
        If `force_k_splits` is not one `mhc_pre_launch_config` accepts, a
        shape disagrees with `hc_mult`, `hidden_size` or the token count of
        `streams`, or a tensor is not laid out for the kernel's 16-byte
        vector accesses.
    """
    comptime assert is_gpu[target](), "mhc_pre is only supported on GPU"
    # NOTE: keep these in sync with `mhc_pre_supports` in
    # max/python/max/nn/legacy/kernels.py, which routes only shapes they
    # accept.
    comptime assert hc_mult == _mhc_pre_hc_mult, "mhc_pre supports hc_mult == 4"
    comptime assert (
        hidden_size % 1024 == 0 and 1024 <= hidden_size <= 8192
    ), "mhc_pre supports hidden_size a multiple of 1024 in [1024, 8192]"
    comptime assert (
        collapsed.dtype == .bfloat16
    ), "mhc_pre supports bf16 streams"
    comptime assert hc_fn.dtype in (
        DType.float32,
        DType.bfloat16,
    ), "mhc_pre supports fp32 or bf16 hc_fn"
    comptime assert 1 <= fn_terms <= 3, "fn_terms must be 1, 2 or 3"
    comptime assert hc_sinkhorn_iters >= 1, "hc_sinkhorn_iters must be >= 1"

    comptime mix_width = 2 * hc_mult + hc_mult * hc_mult

    var num_rows = Int(streams.dim(0))
    var config = mhc_pre_launch_config(num_rows, hidden_size, force_k_splits)
    var k_splits = config[1]
    var gamma_size = hidden_size if has_norm else 1
    if (
        Int(streams.dim(1)) != hc_mult
        or Int(streams.dim(2)) != hidden_size
        or Int(collapsed.dim(0)) != num_rows
        or Int(collapsed.dim(1)) != hidden_size
        or Int(post.dim(0)) != num_rows
        or Int(post.dim(1)) != hc_mult
        or Int(pre.dim(0)) != num_rows
        or Int(pre.dim(1)) != hc_mult
        or Int(comb.dim(0)) != num_rows
        or Int(comb.dim(1)) != hc_mult
        or Int(comb.dim(2)) != hc_mult
        or Int(hc_fn.dim(0)) != mix_width
        or Int(hc_fn.dim(1)) != hc_mult * hidden_size
        or Int(base.dim(0)) != mix_width
        or Int(scale.dim(0)) != 3
        or Int(gamma.dim(0)) != gamma_size
    ):
        raise Error(
            "mhc_pre expects streams [T, ",
            hc_mult,
            ", ",
            hidden_size,
            "], collapsed [T, ",
            hidden_size,
            "], post/pre [T, ",
            hc_mult,
            "], comb [T, ",
            hc_mult,
            ", ",
            hc_mult,
            "], hc_fn [",
            mix_width,
            ", ",
            hc_mult * hidden_size,
            "], base [",
            mix_width,
            "], scale [3] and gamma [",
            gamma_size,
            "] with T = ",
            num_rows,
        )
    # An empty batch launches nothing, and its buffers need not pass the
    # layout checks below.
    if num_rows == 0:
        return
    if not (
        _is_vec_ok(streams)
        and _is_vec_ok(collapsed)
        and _is_vec_ok(hc_fn)
        and (not has_norm or _is_vec_ok(gamma))
        and Int(post.layout.stride[1]().value()) == 1
        and Int(pre.layout.stride[1]().value()) == 1
        and Int(comb.layout.stride[1]().value()) == hc_mult
        and Int(comb.layout.stride[2]().value()) == 1
        and Int(base.layout.stride[0]().value()) == 1
        and Int(scale.layout.stride[0]().value()) == 1
    ):
        raise Error(
            "mhc_pre needs unit inner strides, 16-byte aligned rows of"
            " streams/hc_fn/collapsed/gamma and contiguous rows of post, pre"
            " and comb"
        )
    var gpu_ctx = context
    comptime fn_terms_used = 1 if hc_fn.dtype == .bfloat16 else fn_terms
    with Trace[TraceLevel.OP, target=target](
        "mo.mhc.pre", task_id=Int(gpu_ctx.device().id)
    ):
        if k_splits == 1:
            comptime kernel = mhc_pre_kernel[
                collapsed.dtype,
                hc_fn.dtype,
                post.LayoutType,
                comb.LayoutType,
                collapsed.LayoutType,
                pre.LayoutType,
                streams.LayoutType,
                hc_fn.LayoutType,
                base.LayoutType,
                scale.LayoutType,
                gamma.LayoutType,
                hc_mult,
                hidden_size,
                hc_sinkhorn_iters,
                has_norm,
                fn_terms_used,
            ]
            gpu_ctx.enqueue_function[kernel](
                post,
                comb,
                collapsed,
                pre,
                streams,
                hc_fn,
                base,
                scale,
                gamma,
                rms_eps,
                hc_eps,
                norm_eps,
                Int32(num_rows),
                grid_dim=config[0],
                block_dim=_mhc_pre_threads,
            )
            return

        # Each token's run of partials is padded to 16 bytes for the finish
        # kernel's vector loads. The launch boundary orders the two kernels,
        # so the workspace needs no clearing and no atomics.
        comptime row_vals = mix_width + 1
        var row_stride = align_up(k_splits * row_vals, 4)
        var padded_rows = config[0] * _mhc_pre_block_rows
        var workspace = gpu_ctx.enqueue_create_buffer[.float32](
            padded_rows * row_stride
        )
        var partial = TileTensor(
            workspace, row_major((padded_rows, row_stride))
        )

        comptime for e in range(1, log2_floor(_mhc_pre_k_splits_cap) + 1):
            comptime ks = 1 << e
            comptime if _mhc_pre_k_splits_ok(hidden_size, ks):
                if k_splits == ks:
                    comptime partials = mhc_pre_partials_kernel[
                        collapsed.dtype,
                        hc_fn.dtype,
                        streams.LayoutType,
                        hc_fn.LayoutType,
                        partial.LayoutType,
                        hc_mult,
                        hidden_size,
                        fn_terms_used,
                        ks,
                    ]
                    gpu_ctx.enqueue_function[partials](
                        partial,
                        streams,
                        hc_fn,
                        Int32(num_rows),
                        grid_dim=(config[0], ks),
                        block_dim=_mhc_pre_threads,
                    )

        var partial_in = partial.as_imm()
        comptime finish = mhc_pre_finish_kernel[
            collapsed.dtype,
            post.LayoutType,
            comb.LayoutType,
            collapsed.LayoutType,
            pre.LayoutType,
            streams.LayoutType,
            partial_in.LayoutType,
            base.LayoutType,
            scale.LayoutType,
            gamma.LayoutType,
            hc_mult,
            hidden_size,
            hc_sinkhorn_iters,
            has_norm,
        ]
        gpu_ctx.enqueue_function[finish](
            post,
            comb,
            collapsed,
            pre,
            streams,
            partial_in,
            base,
            scale,
            gamma,
            rms_eps,
            hc_eps,
            norm_eps,
            Int32(k_splits),
            grid_dim=num_rows,
            block_dim=_mhc_pre_threads,
        )


@always_inline
def mhc_post_tile[
    dtype: DType,
    OutLayoutType: TensorLayout,
    StreamsLayoutType: TensorLayout,
    YLayoutType: TensorLayout,
    //,
    hc_mult: Int,
    width: Int,
    non_temporal: Bool = False,
](
    output: TileTensor[mut=True, dtype, OutLayoutType, MutAnyOrigin],
    streams: TileTensor[dtype, StreamsLayoutType, ImmutAnyOrigin],
    y: TileTensor[dtype, YLayoutType, ImmutAnyOrigin],
    post: SIMD[.float32, hc_mult],
    comb: SIMD[.float32, hc_mult * hc_mult],
    row: Int,
    col: Int,
):
    """Writes one `width`-wide slice of one token's mHC streams back.

    Covers columns `[col, col + width)`. Every load is issued before the
    first FMA, so all `hc_mult + 1` vectors are in flight together. Math is
    fp32, rounded to `dtype` once.

    Parameters:
        dtype: Element type of the streams, `y` and the output.
        OutLayoutType: Layout of `output`.
        StreamsLayoutType: Layout of `streams`.
        YLayoutType: Layout of `y`.
        hc_mult: Number of parallel residual streams.
        width: Elements per vector access.
        non_temporal: Whether to mark the loads and stores non-temporal.

    Args:
        output: Updated streams. Shape: `[num_rows, hc_mult, hidden]`.
        streams: Incoming streams. Shape: `[num_rows, hc_mult, hidden]`.
        y: Sublayer output. Shape: `[num_rows, hidden]`.
        post: `post[row, :]`, the weight of `y` in each output stream.
        comb: `comb[row, :, :]` row-major, so `comb[j * hc_mult + k]` is the
            weight of input stream `j` in output stream `k`.
        row: Token index.
        col: First column of the slice.
    """
    comptime assert output.flat_rank == 3
    comptime assert streams.flat_rank == 3
    comptime assert y.flat_rank == 2

    var y_v = y.load[width=width, non_temporal=non_temporal]((row, col))
    var s_v = Array[SIMD[dtype, width], hc_mult](uninitialized=True)
    comptime for j in range(hc_mult):
        s_v[j] = streams.load[width=width, non_temporal=non_temporal](
            (row, j, col)
        )

    var y_f = y_v.cast[.float32]()
    comptime for k in range(hc_mult):
        var acc = y_f * post[k]
        comptime for j in range(hc_mult):
            acc = s_v[j].cast[.float32]().fma(comb[j * hc_mult + k], acc)
        output.store[width=width, non_temporal=non_temporal](
            (row, k, col), acc.cast[dtype]()
        )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(block_size))
)
@__name(
    t"mhc_post_h{hc_mult}_d{hidden_size}_b{block_size}_nt{Int(non_temporal)}"
)
def mhc_post_kernel[
    dtype: DType,
    OutLayoutType: TensorLayout,
    StreamsLayoutType: TensorLayout,
    YLayoutType: TensorLayout,
    PostLayoutType: TensorLayout,
    CombLayoutType: TensorLayout,
    hc_mult: Int,
    hidden_size: Int,
    block_size: Int,
    non_temporal: Bool,
](
    output: TileTensor[mut=True, dtype, OutLayoutType, MutAnyOrigin],
    streams: TileTensor[dtype, StreamsLayoutType, ImmutAnyOrigin],
    y: TileTensor[dtype, YLayoutType, ImmutAnyOrigin],
    post: TileTensor[.float32, PostLayoutType, ImmutAnyOrigin],
    comb: TileTensor[.float32, CombLayoutType, ImmutAnyOrigin],
):
    """Writes one token's slice of the mHC streams back.

    Block `(row, kb)` owns columns `[kb * chunk, (kb + 1) * chunk)` of token
    `row`, where `chunk = block_size * width`. Each thread loads `y` and all
    `hc_mult` streams for its `width` columns and writes all `hc_mult`
    outputs, so no data crosses threads and no LDS is needed. The `post` and
    `comb` coefficients depend only on `row`, so the compiler keeps them in
    scalar registers.

    Parameters:
        dtype: Element type of the streams, `y` and the output.
        OutLayoutType: Layout of `output`.
        StreamsLayoutType: Layout of `streams`.
        YLayoutType: Layout of `y`.
        PostLayoutType: Layout of `post`.
        CombLayoutType: Layout of `comb`.
        hc_mult: Number of parallel residual streams.
        hidden_size: Width of one stream.
        block_size: Threads per block.
        non_temporal: Whether to mark the loads and stores non-temporal.

    Args:
        output: Updated streams. Shape: `[num_rows, hc_mult, hidden_size]`.
        streams: Incoming streams. Shape: `[num_rows, hc_mult, hidden_size]`.
        y: Sublayer output. Shape: `[num_rows, hidden_size]`.
        post: Weight of `y` in each output stream. Shape:
            `[num_rows, hc_mult]`.
        comb: Stream mixer; `comb[t, j, k]` weighs input stream `j` in output
            stream `k`. Shape: `[num_rows, hc_mult, hc_mult]`.
    """
    comptime width = 16 // size_of[dtype]()
    comptime chunk = block_size * width
    comptime assert post.flat_rank == 2
    comptime assert comb.flat_rank == 3

    var row = block_idx.x
    var col = block_idx.y * chunk + thread_idx.x * width

    wait_on_dependent_grids()

    var post_v = post.load[width=hc_mult, alignment=4]((row, 0))
    var comb_v = comb.load[width=hc_mult * hc_mult, alignment=4]((row, 0, 0))

    # Safe before the stores: a dependent's `wait_on_dependent_grids()` waits
    # for this whole grid to finish and its writes to land.
    launch_dependent_grids()

    mhc_post_tile[
        hc_mult=hc_mult,
        width=width,
        non_temporal=non_temporal,
    ](output, streams, y, post_v, comb_v, row, col)


@inline(.always)
def mhc_post_block_size[dtype: DType, hidden_size: Int]() -> Int:
    """Returns the threads per block `mhc_post` takes for `hidden_size`.

    256 where their 16-byte vectors split the hidden axis into whole blocks,
    else 128.

    Parameters:
        dtype: Element type of the streams.
        hidden_size: Width of one stream.

    Returns:
        The threads per block.
    """
    comptime width = 16 // size_of[dtype]()
    return 256 if hidden_size % (256 * width) == 0 else 128


@inline(.always)
def mhc_post_non_temporal(num_rows: Int, num_cus: Int) -> Bool:
    """Returns whether `mhc_post` marks its accesses non-temporal.

    On MI355X non-temporal accesses win from one token per CU up (17% at 256
    tokens, 12% at 16384); NVIDIA is unmeasured and keeps 8 per SM.

    Args:
        num_rows: Number of tokens.
        num_cus: Compute units (SMs) on the device.

    Returns:
        True to mark the loads and stores non-temporal.
    """
    comptime if default_accelerator().is_amd_gpu():
        return num_rows >= num_cus
    else:
        return num_rows > 8 * num_cus


@inline(.always)
def _is_vec_ok(t: TileTensor) -> Bool:
    """Returns whether `t` has a unit inner stride and 16-byte aligned rows."""
    comptime rank = t.flat_rank
    var ok = (
        Int(t.unsafe_ptr()) % 16 == 0
        and Int(t.layout.stride[rank - 1]().value()) == 1
    )
    comptime for i in range(rank - 1):
        ok = ok and (
            Int(t.layout.stride[i]().value()) * size_of[t.dtype]() % 16 == 0
        )
    return ok


@inline(.always)
def mhc_post[
    hc_mult: Int,
    hidden_size: Int,
    target: StaticString,
    block_size: Int = 256,
](
    output: TileTensor[mut=True, ...],
    streams: TileTensor[mut=False, output.dtype, ...],
    y: TileTensor[mut=False, output.dtype, ...],
    post: TileTensor[mut=False, .float32, ...],
    comb: TileTensor[mut=False, .float32, ...],
    context: DeviceContext,
    force_non_temporal: Optional[Bool] = None,
) raises:
    """Launches the mHC write-back, `(num_rows, k_blocks)` blocks.

    Computes `output[t, k, :] = post[t, k] * y[t, :] + sum_j comb[t, j, k] *
    streams[t, j, :]` in fp32 and rounds once. Each thread covers one 16-byte
    slice, so `k_blocks = hidden_size / (block_size * width)` blocks split
    each token, and memory accesses turn non-temporal from one token per CU
    on AMD (`num_rows >= CUs`) and past `8 * CUs` tokens elsewhere.

    Parameters:
        hc_mult: Number of parallel residual streams.
        hidden_size: Width of one stream.
        target: The target device to run the kernel on.
        block_size: Threads per block.

    Args:
        output: Updated streams. Shape: `[num_rows, hc_mult, hidden_size]`.
        streams: Incoming streams. Shape: `[num_rows, hc_mult, hidden_size]`.
        y: Sublayer output. Shape: `[num_rows, hidden_size]`.
        post: Weight of `y` in each output stream. Shape:
            `[num_rows, hc_mult]`.
        comb: Stream mixer; `comb[t, j, k]` weighs input stream `j` in output
            stream `k`. Shape: `[num_rows, hc_mult, hc_mult]`.
        context: The device context.
        force_non_temporal: Overrides the non-temporal threshold; for
            benchmarks and tests only.

    Raises:
        If a shape disagrees with `hc_mult`, `hidden_size` or the token
        count of `streams`, or a tensor is not laid out for the kernel's
        16-byte vector accesses.
    """
    comptime assert is_gpu[target](), "mhc_post is only supported on GPU"

    comptime width = 16 // size_of[output.dtype]()
    comptime chunk = block_size * width
    comptime assert hidden_size % chunk == 0, (
        "hidden_size must be a multiple of block_size * (16 bytes / element"
        " size)"
    )
    comptime k_blocks = hidden_size // chunk

    var num_rows = Int(streams.dim(0))
    if (
        Int(output.dim(0)) != num_rows
        or Int(output.dim(1)) != hc_mult
        or Int(output.dim(2)) != hidden_size
        or Int(streams.dim(1)) != hc_mult
        or Int(streams.dim(2)) != hidden_size
        or Int(y.dim(0)) != num_rows
        or Int(y.dim(1)) != hidden_size
        or Int(post.dim(0)) != num_rows
        or Int(post.dim(1)) != hc_mult
        or Int(comb.dim(0)) != num_rows
        or Int(comb.dim(1)) != hc_mult
        or Int(comb.dim(2)) != hc_mult
    ):
        raise Error(
            "mhc_post expects streams/output [T, ",
            hc_mult,
            ", ",
            hidden_size,
            "], y [T, ",
            hidden_size,
            "], post [T, ",
            hc_mult,
            "] and comb [T, ",
            hc_mult,
            ", ",
            hc_mult,
            "] with T = ",
            num_rows,
        )
    # An empty tensor's pointer need not be aligned.
    if num_rows == 0:
        return
    # The kernel moves 16-byte vectors along the hidden dimension and reads
    # each token's coefficients as one contiguous vector.
    if not (
        _is_vec_ok(output)
        and _is_vec_ok(streams)
        and _is_vec_ok(y)
        and Int(post.layout.stride[1]().value()) == 1
        and Int(comb.layout.stride[1]().value()) == hc_mult
        and Int(comb.layout.stride[2]().value()) == 1
    ):
        raise Error(
            "mhc_post needs unit inner strides, 16-byte aligned rows of"
            " streams/y/output and contiguous rows of post and comb"
        )

    var gpu_ctx = context
    var num_cus = gpu_ctx.device().get_attribute(
        DeviceAttribute.MULTIPROCESSOR_COUNT
    )
    var non_temporal = mhc_post_non_temporal(num_rows, num_cus)
    if force_non_temporal:
        non_temporal = force_non_temporal.value()

    with Trace[TraceLevel.OP, target=target](
        "mo.mhc.post", task_id=Int(gpu_ctx.device().id)
    ):

        @always_inline
        def launch[non_temporal: Bool]() raises {var}:
            comptime kernel = mhc_post_kernel[
                output.dtype,
                output.LayoutType,
                streams.LayoutType,
                y.LayoutType,
                post.LayoutType,
                comb.LayoutType,
                hc_mult,
                hidden_size,
                block_size,
                non_temporal,
            ]
            gpu_ctx.enqueue_function[kernel](
                output,
                streams,
                y,
                post,
                comb,
                grid_dim=(num_rows, k_blocks),
                block_dim=block_size,
                attributes=pdl_launch_attributes(PDLLevel.ON),
            )

        if non_temporal:
            launch[True]()
        else:
            launch[False]()
