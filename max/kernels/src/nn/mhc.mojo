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
"""Manifold-Constrained Hyper-Connections (mHC) gate computation.

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

The projection itself and the final collapse stay outside: they are a GEMM and
a broadcast-reduce that the graph compiler already handles well.
"""

import max.gpu.primitives.warp as warp
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    lane_id,
    warp_id,
)
from std.math import ceildiv, exp
from std.utils.index import StaticTuple

from layout import Coord, TensorLayout, TileTensor
from max.gpu.host import DeviceContext
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
        Int(block_idx.x) * rows_per_block
        + Int(warp_id()) * rows_per_warp
        + lane // HH
    )

    # The scale and the bias are checkpoint weights, so they do not come from
    # the grid this one waits on. Loading them first puts the fetch in flight
    # across the wait instead of behind it. Every index here is within the
    # bias, whatever `elem` is, so none of it needs a bound.
    var s_pre = pre_post_comb_scale.load[width=1](Coord(0))
    var s_post = pre_post_comb_scale.load[width=1](Coord(1))
    var s_comb = pre_post_comb_scale.load[width=1](Coord(2))
    var pre_b = pre_post_comb_b.load[width=1](Coord(elem))
    var post_b = pre_post_comb_b.load[width=1](Coord(H + elem))
    var comb_b = pre_post_comb_b.load[width=1](Coord(2 * H + elem))

    wait_on_dependent_grids()
    launch_dependent_grids()

    # Not warp-uniform: the tail block's last warp straddles `num_rows`. Every
    # lane still runs every shuffle below -- skipping one would leave the
    # warp's reduction undefined -- and a lane group never reaches outside the
    # row that owns it, so an out-of-range row cannot perturb a live one.
    var in_range = row < Int(num_rows)

    if in_range and elem < H:
        var pre_w = hc_proj.load[width=1]((row, elem))
        pre[row, elem] = sigmoid(pre_w * s_pre + pre_b) + hc_eps

        var post_w = hc_proj.load[width=1]((row, H + elem))
        post[row, elem] = 2 * sigmoid(post_w * s_post + post_b)

    var logit = Float32(0)
    if in_range:
        logit = hc_proj.load[width=1]((row, 2 * H + elem)) * s_comb + comb_b

    # Softmax over the matrix row, max-subtracted to match torch.
    var e = exp(logit - warp.lane_group_max[num_lanes=H](logit))
    var c = e / warp.lane_group_sum[num_lanes=H](e) + hc_eps

    # Sinkhorn-Knopp: the softmax already normalized the rows, so the first
    # half-step is the column pass.
    c /= warp.lane_group_sum[num_lanes=H, stride=H](c) + hc_eps

    comptime for _ in range(hc_sinkhorn_iters - 1):
        c /= warp.lane_group_sum[num_lanes=H](c) + hc_eps
        c /= warp.lane_group_sum[num_lanes=H, stride=H](c) + hc_eps

    if in_range:
        comb[row, elem] = c


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
        "mo.hyper_connection.gates", task_id=Int(gpu_ctx.id())
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
