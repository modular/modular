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
"""Tiled MFMA blockwise-scaled FP8 grouped (MoE) matmul for AMD gfx950.

Target: AMD MI350/MI355 (gfx950), MFMA `16x16x128` FP8 (E4M3 OCP).

This is the expert-dispatched variant of the dense
`blockwise_scaled_fp8_matmul_amd`. It carries the bulk of a mixed-FP8
MoE model's FLOPs (GLM gate_up / down projections). The grouped
expert/tile dispatch shell (one block per `(expert, n_tile)` striding
over M-tiles, with the expert selected by `block_idx.z` via `a_offsets` /
`expert_ids`) follows the MXFP4 grouped kernel
`block_scaled_grouped_matmul_amd`; the per-tile MMA body is the dense
kernel's E4M3 + fp32-per-128-K-slab-promotion body, reused verbatim via
`blockwise_scaled_fp8_matmul_amd_tile`. The M-tile stride loop keeps
grid.y a packing hint rather than a coverage bound, so grid.y can be
sized from a capture-safe bound instead of the graph-capture
`max_num_tokens_per_expert` -- at decode that constant (16384 in
production) made ~99.6% of the one-wave CTAs exit at the bounds check.

Scale-index and expert-dispatch semantics mirror
`naive_blockwise_scaled_fp8_grouped_matmul` (`linalg/fp8_quantization.mojo`)
at granularity `(m, n, k) = (1, 128, 128)`:

  C[e][m, n] = sum_k A[m_global, k] * B[e, n, k]
             * a_scale[k//128, m_global] * b_scale[e, n//128, k//128]

where `m_global = a_offsets[slot] + m_local` and `e = expert_ids[slot]`.
`a_scales` is token-minor `[K//128, total_tokens]`, so the reused body is
handed the full `a_scales` plus this expert's `a_start_row` column base;
`b_scales` is sliced per expert to `[N//128, K//128]`, matching the dense
body's 2D `b_scales` exactly.
"""

from std.math import ceildiv

from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    grid_dim,
)
from max.gpu.host import DeviceContext

from layout import Coord, Idx, TensorEngine, TensorLayout, TileTensor
from layout.tile_layout import row_major

from std.utils import StaticTuple
from std.utils.numerics import get_accum_type

from .blockwise_scaled_fp8_matmul_amd import (
    blockwise_scaled_fp8_matmul_amd_tile,
)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32((BM // WM) * (BN // WN) * WARP_SIZE)
    )
)
@__name(
    t"blockwise_scaled_fp8_grouped_matmul_amd_BM{BM}_BN{BN}_BK{BK}_WM{WM}_WN{WN}"
)
def blockwise_scaled_fp8_grouped_matmul_amd_kernel[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    accum_type: DType,
    c_layout: TensorLayout,
    a_layout: TensorLayout,
    b_layout: TensorLayout,
    a_scale_layout: TensorLayout,
    b_scale_layout: TensorLayout,
    a_offsets_layout: TensorLayout,
    expert_ids_layout: TensorLayout,
    c_engine: TensorEngine,
    a_engine: TensorEngine,
    b_engine: TensorEngine,
    a_scale_engine: TensorEngine,
    b_scale_engine: TensorEngine,
    a_offsets_engine: TensorEngine,
    expert_ids_engine: TensorEngine,
    *,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    MMA_M: Int,
    MMA_N: Int,
    MMA_K: Int,
    N_SCALE: Int,
    K_SCALE: Int,
](
    c: TileTensor[mut=True, c_type, c_layout, MutAnyOrigin, Engine=c_engine],
    a: TileTensor[a_type, a_layout, ImmutAnyOrigin, Engine=a_engine],
    b: TileTensor[b_type, b_layout, ImmutAnyOrigin, Engine=b_engine],
    a_scales: TileTensor[
        a_scales_type, a_scale_layout, ImmutAnyOrigin, Engine=a_scale_engine
    ],
    b_scales: TileTensor[
        b_scales_type, b_scale_layout, ImmutAnyOrigin, Engine=b_scale_engine
    ],
    a_offsets: TileTensor[
        mut=False,
        .uint32,
        a_offsets_layout,
        ImmutAnyOrigin,
        Engine=a_offsets_engine,
    ],
    expert_ids: TileTensor[
        mut=False,
        .int32,
        expert_ids_layout,
        ImmutAnyOrigin,
        Engine=expert_ids_engine,
    ],
):
    """Computes the `(expert, n_tile)` output-tile column, striding over
    M-tiles, for a grouped matmul.

    `block_idx.z` selects the active-expert slot, `block_idx.x` the N-tile,
    and `block_idx.y` this CTA's first M-tile within that expert's ragged
    row range. The CTA then loops M-tiles `block_idx.y`, `block_idx.y +
    grid_dim.y`, ... so the grid.y CTAs together cover every M-tile exactly
    once however the launcher sized grid.y: a small grid.y packs the same
    per-expert tiles into fewer CTAs (each looping) instead of launching a
    wave of CTAs that exit at the bounds check, and a stale or undersized
    grid.y can no longer drop rows. Zero-row experts and `-1` (skipped)
    expert slots return early; partial M/N tiles are handled by the reused
    body's OOB masking and clamped scale reads. Every loop iteration runs
    the identical one-tile body with the same arguments the
    one-CTA-per-M-tile launch used, so results are bit-identical, and the
    trip count is CTA-uniform so the body's barriers stay convergent.

    Parameters:
        c_type: Output element type.
        a_type: A input element type (`float8_e4m3fn`).
        b_type: B input element type (`float8_e4m3fn`).
        a_scales_type: A scales element type (`float32`).
        b_scales_type: B scales element type (`float32`).
        accum_type: Accumulation element type (`float32`).
        c_layout: Layout of `c`.
        a_layout: Layout of `a`.
        b_layout: Layout of `b`.
        a_scale_layout: Layout of `a_scales`.
        b_scale_layout: Layout of `b_scales`.
        a_offsets_layout: Layout of `a_offsets`.
        expert_ids_layout: Layout of `expert_ids`.
        c_engine: Tensor engine of `c`.
        a_engine: Tensor engine of `a`.
        b_engine: Tensor engine of `b`.
        a_scale_engine: Tensor engine of `a_scales`.
        b_scale_engine: Tensor engine of `b_scales`.
        a_offsets_engine: Tensor engine of `a_offsets`.
        expert_ids_engine: Tensor engine of `expert_ids`.
        BM: Block tile rows (M).
        BN: Block tile cols (N).
        BK: Block tile K (one scale slab; must equal `K_SCALE`).
        WM: Warp tile rows (M).
        WN: Warp tile cols (N).
        MMA_M: MFMA M (16).
        MMA_N: MFMA N (16).
        MMA_K: MFMA K (128 for FP8).
        N_SCALE: N scale granularity (128).
        K_SCALE: K scale granularity (128).

    Args:
        c: Output `[total_tokens, N]`, indexed by per-expert row offsets.
        a: A input `[total_tokens, K]`, K-major.
        b: B expert weights `[num_experts, N, K]`, K-major (`transpose_b`).
        a_scales: A per-block scales `[K // K_SCALE, total_tokens]`.
        b_scales: B per-block scales `[num_experts, N // N_SCALE,
            K // K_SCALE]`. `N` and `K` must each be an exact multiple of
            `N_SCALE` and `K_SCALE` (128) -- asserted by the launcher.
        a_offsets: Row offsets `[num_active_experts + 1]`; slot `z` spans
            rows `a_offsets[z]` to `a_offsets[z + 1]`.
        expert_ids: Expert id (or `-1` to skip) per active slot.
    """
    comptime assert a_offsets.flat_rank == 1, "a_offsets must be rank 1"
    comptime assert expert_ids.flat_rank == 1, "expert_ids must be rank 1"

    comptime N = type_of(c).static_shape[1]
    comptime K = type_of(a).static_shape[1]
    comptime N_SCALES = N // N_SCALE
    comptime K_SCALES = K // K_SCALE

    var slot = Int(block_idx.z)
    var M = Int(a_offsets[slot + 1]) - Int(a_offsets[slot])
    if M == 0 or N == 0:
        return

    var expert_id = Int(expert_ids[slot])
    if expert_id == -1:
        return

    var a_start_row = Int(a_offsets[slot])

    # Per-expert sub-tiles: C/A gathered at this expert's row offset, B and its
    # scales by expert id; `a_scales` stays whole, read at `a_start_row + m`.
    var c_tile = TileTensor(
        c.ptr + a_start_row * N, row_major(Coord(M, Idx[N]))
    )
    var a_tile = TileTensor(
        a.ptr + a_start_row * K, row_major(Coord(M, Idx[K]))
    )
    var b_tile = TileTensor(b.ptr + expert_id * N * K, row_major[N, K]())
    var b_scale_tile = TileTensor(
        b_scales.ptr + expert_id * N_SCALES * K_SCALES,
        row_major[N_SCALES, K_SCALES](),
    )

    # M-tile stride loop: this CTA strides its expert's M-tiles by grid_dim.y,
    # so coverage is independent of grid.y (any grid.y covers every tile).
    var m_tile = Int(block_idx.y)
    var num_m_tiles = ceildiv(M, BM)
    while m_tile < num_m_tiles:
        blockwise_scaled_fp8_matmul_amd_tile[
            c_type,
            a_type,
            b_type,
            a_scales_type,
            b_scales_type,
            accum_type,
            type_of(c_tile).LayoutType,
            type_of(a_tile).LayoutType,
            type_of(b_tile).LayoutType,
            type_of(a_scales).LayoutType,
            type_of(b_scale_tile).LayoutType,
            type_of(c_tile).Engine,
            type_of(a_tile).Engine,
            type_of(b_tile).Engine,
            type_of(a_scales).Engine,
            type_of(b_scale_tile).Engine,
            BM=BM,
            BN=BN,
            BK=BK,
            WM=WM,
            WN=WN,
            MMA_M=MMA_M,
            MMA_N=MMA_N,
            MMA_K=MMA_K,
            N_SCALE=N_SCALE,
            K_SCALE=K_SCALE,
        ](
            c_tile,
            a_tile,
            b_tile,
            a_scales,
            b_scale_tile,
            m_tile,
            Int(block_idx.x),
            a_start_row,
            M,
        )
        m_tile += Int(grid_dim.y)


def blockwise_scaled_fp8_grouped_matmul_amd[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    b_scales_type: DType,
    //,
    *,
    transpose_b: Bool,
    BM: Int = 64,
    BN: Int = 64,
    BK: Int = 128,
    WM: Int = 64,
    WN: Int = 64,
    N_SCALE: Int = 128,
    K_SCALE: Int = 128,
    accum_type: DType = get_accum_type[c_type](),
    static_grid_z: Bool = False,
](
    c: TileTensor[mut=True, c_type, address_space=.GENERIC, ...],
    a: TileTensor[mut=False, a_type, address_space=.GENERIC, ...],
    b: TileTensor[mut=False, b_type, address_space=.GENERIC, ...],
    a_scales: TileTensor[mut=False, a_scales_type, address_space=.GENERIC, ...],
    b_scales: TileTensor[mut=False, b_scales_type, address_space=.GENERIC, ...],
    a_offsets: TileTensor[mut=False, .uint32, address_space=.GENERIC, ...],
    expert_ids: TileTensor[mut=False, .int32, address_space=.GENERIC, ...],
    max_num_tokens_per_expert: Int,
    num_active_experts: Int,
    ctx: DeviceContext,
    decode_grid_m_cap: Int = 0,
) raises:
    """Enqueues the tiled blockwise-scaled FP8 grouped (MoE) matmul for gfx950.

    Parameters:
        c_type: Output element type.
        a_type: A input element type (`float8_e4m3fn`).
        b_type: B input element type (`float8_e4m3fn`).
        a_scales_type: A scales element type (`float32`).
        b_scales_type: B scales element type (`float32`).
        transpose_b: Must be `True` (B is `[num_experts, N, K]`).
        BM: Block tile rows (M). Default 64 (1 warp/CTA at WM=64): smaller
            CTAs pack more of them per CU, so a K-slab's `barrier()` stall
            in one CTA overlaps another CTA's independent MFMA/LDS work
            instead of idling the whole CU. Measured 23-48% faster than
            128 across the GLM dense/grouped production shapes.
        BN: Block tile cols (N). Same reasoning as `BM`.
        BK: Block tile K (one scale slab).
        WM: Warp tile rows (M).
        WN: Warp tile cols (N).
        N_SCALE: N scale granularity (128).
        K_SCALE: K scale granularity (128).
        accum_type: Accumulation element type (`float32`).
        static_grid_z: Use `b`'s comptime expert-slot count for grid.z
            instead of the runtime `num_active_experts` tensor read. Safe
            only when the caller's `a_offsets`/`expert_ids` are always
            sized to that same static count (true for the
            `moe_create_indices` routing path: every expert in
            `[0, num_experts)` gets a slot, none are ever compacted away,
            so `num_active_experts` already equals `b`'s expert dim on
            every call -- this just stops reading that invariant value
            off a device tensor at every launch). Under device graph
            capture, a grid dimension sourced from a device tensor read is
            frozen at capture time; a comptime dimension is not. Default
            `False` preserves the exact current behavior.

    Args:
        c: Output `[total_tokens, N]`.
        a: A input `[total_tokens, K]`, K-major.
        b: B expert weights `[num_experts, N, K]`, K-major (`transpose_b`).
        a_scales: A per-block scales `[K // K_SCALE, total_tokens]`.
        b_scales: B per-block scales `[num_experts, N // N_SCALE,
            K // K_SCALE]`. `N` and `K` must each be an exact multiple of
            `N_SCALE` and `K_SCALE` (128) -- asserted below.
        a_offsets: Row offsets `[num_active_experts + 1]` uint32.
        expert_ids: Expert id (or `-1`) per active slot, int32.
        max_num_tokens_per_expert: Max rows of any active expert this call
            (grid.y bound), read off a device tensor computed from the
            current step's routing. Under device graph capture this value
            is frozen at capture time, so a later replay whose routing
            sends more rows to some expert than the captured grid.y covers
            would silently drop those rows -- the same hazard the naive
            reference kernel has (`fp8_quantization.mojo`). grid.y is
            clamped to this call's own total token count (a capture-safe
            input shape that no single expert's rows can exceed), and the
            kernel's M-tile stride loop makes grid.y a packing hint only:
            a stale or undersized value can no longer drop rows.
        num_active_experts: Number of active expert slots (grid.z) --
            ignored when `static_grid_z` is set.
        ctx: Device context.
        decode_grid_m_cap: Decode-band gate; `0` disables the gate itself
            (the always-on grid.y clamp below still applies). When positive
            and this call's total token count (`a`'s row count) is `<=` the
            cap, grid.y is sized from that total instead of
            `max_num_tokens_per_expert`. Retained for source compatibility:
            with the M-tile stride loop the always-on clamp to the call's
            own row count already gives the decode band the tight
            capture-safe grid without any opt-in, so this gate no longer
            changes the launch for the production capture bound (16384
            >= any decode call's rows). Above the cap (the prefill band,
            which is not capture-replayed in practice) behavior is
            unchanged: a caller wiring this up passes the cap device
            graph capture is bounded to (e.g. the recipe's max decode batch
            size), mirroring `decode_grid_m_cap` on the AMD MXFP4 grouped
            path (`max/python/max/nn/kernels.py`).
    """
    comptime assert transpose_b, "transpose_b must be True"
    comptime assert b.flat_rank == 3, "b must be [num_experts, N, K]"
    comptime assert b_scales.flat_rank == 3, "b_scales must be rank 3"
    comptime assert a_offsets.flat_rank == 1, "a_offsets must be rank 1"
    comptime assert expert_ids.flat_rank == 1, "expert_ids must be rank 1"
    comptime assert (
        a_type == .float8_e4m3fn and b_type == .float8_e4m3fn
    ), "tiled blockwise FP8 grouped matmul supports only float8_e4m3fn inputs"
    comptime assert (
        a_scales_type == .float32 and b_scales_type == .float32
    ), "tiled blockwise FP8 grouped matmul supports only float32 scales"
    comptime assert accum_type == .float32

    comptime MMA_M = 16
    comptime MMA_N = 16
    comptime MMA_K = 128

    comptime N = type_of(c).static_shape[1]
    comptime K = type_of(a).static_shape[1]
    comptime assert N > 0 and K > 0, "N and K must be static"
    comptime assert K % K_SCALE == 0, "K must be a multiple of K_SCALE (128)"
    comptime assert N % N_SCALE == 0, "N must be a multiple of N_SCALE (128)"

    if max_num_tokens_per_expert == 0 or num_active_experts == 0:
        return

    comptime kernel = blockwise_scaled_fp8_grouped_matmul_amd_kernel[
        c_type,
        a_type,
        b_type,
        a_scales_type,
        b_scales_type,
        accum_type,
        type_of(c).LayoutType,
        type_of(a).LayoutType,
        type_of(b).LayoutType,
        type_of(a_scales).LayoutType,
        type_of(b_scales).LayoutType,
        type_of(a_offsets).LayoutType,
        type_of(expert_ids).LayoutType,
        type_of(c).Engine,
        type_of(a).Engine,
        type_of(b).Engine,
        type_of(a_scales).Engine,
        type_of(b_scales).Engine,
        type_of(a_offsets).Engine,
        type_of(expert_ids).Engine,
        BM=BM,
        BN=BN,
        BK=BK,
        WM=WM,
        WN=WN,
        MMA_M=MMA_M,
        MMA_N=MMA_N,
        MMA_K=MMA_K,
        N_SCALE=N_SCALE,
        K_SCALE=K_SCALE,
    ]

    comptime num_threads = (BM // WM) * (BN // WN) * WARP_SIZE

    # grid.y is a packing hint, not a coverage bound (the M-tile stride loop
    # covers every tile for any grid.y); size it from this call's token count.
    var etm = Int(a.dim[0]())
    var m_cap = (
        etm if decode_grid_m_cap > 0
        and etm <= decode_grid_m_cap else min(max_num_tokens_per_expert, etm)
    )
    if m_cap <= 0:
        return

    # grid.z: comptime expert-slot count (a capture-time constant) when
    # static_grid_z, else the runtime num_active_experts read.
    comptime static_experts = type_of(b).static_shape[0] if static_grid_z else 0
    comptime assert (
        not static_grid_z
    ) or static_experts > 0, (
        "static_grid_z needs a comptime-known expert count on `b`"
    )
    var grid_z = static_experts if static_grid_z else num_active_experts

    ctx.enqueue_function[kernel](
        c,
        a,
        b,
        a_scales,
        b_scales,
        a_offsets,
        expert_ids,
        grid_dim=(
            ceildiv(N, BN),
            ceildiv(m_cap, BM),
            grid_z,
        ),
        block_dim=(num_threads, 1, 1),
    )
