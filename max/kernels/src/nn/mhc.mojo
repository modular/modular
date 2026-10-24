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

The projection itself and the final collapse stay outside: they are a GEMM and
a broadcast-reduce that the graph compiler already handles well.
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
from std.math import ceildiv, exp
from std.sys import size_of
from std.utils.index import StaticTuple

from layout import TensorLayout, TileTensor
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
        pre[row, elem] = sigmoid(pre_w * s_pre + pre_b) + hc_eps

        var post_w = hc_proj.load((row, H + elem))
        post[row, elem] = 2 * sigmoid(post_w * s_post + post_b)

    var logit = Float32(0)
    if in_range:
        logit = hc_proj.load((row, 2 * H + elem)) * s_comb + comb_b

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
    if num_rows == 0:
        return

    var gpu_ctx = context
    var num_cus = gpu_ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    var non_temporal = mhc_post_non_temporal(num_rows, num_cus)
    if force_non_temporal:
        non_temporal = force_non_temporal.value()

    with Trace[TraceLevel.OP, target=target](
        "mo.mhc.post", task_id=Int(gpu_ctx.id())
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
