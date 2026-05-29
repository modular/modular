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
"""MHA MMA operator: shape constants, SMEM→register loaders, and
MFMA dispatch used by `HKMhaPrefill`.

Specialized for `v_mfma_f32_32x32x16_bf16` (BF16-input, FP32-accum)
on gfx950. K side uses the two-XOR worker-index swizzle and
`ds_read_b128`; V side uses the identity swizzle and the
`ds_read_tr16_b64_warp` transpose-load.
"""

from std.gpu import lane_id
from std.gpu.compute.mma import mma as gpu_mma
from std.math import exp2 as math_exp2
from std.sys import size_of
from std.utils import IndexList

from layout import TensorLayout
from layout.coord import Coord, Idx
from layout.tile_layout import row_major, row_major_nested

from structured_kernels.amd_tile_io import (
    RegTile,
    SMemTile,
    _load_from_lds,
    ds_read_tr16_b64_warp,
)


# `v_mfma_f32_32x32x16_bf16` on gfx950 distributes its 32×32 FP32 output
# across wave64 such that lane `L` owns 16 elements at columns `L & 31`
# and rows `ACC_ROW_OFFSETS_32x32[p] + ((L >> 5) << 2)` for `p` in
# `[0, 16)`: four stripes of four rows, spaced 8 rows apart, with the
# high half-warp shifted by 4. Hardware-determined; consumers needing to
# map per-lane fragment indices to matrix rows should read this table.
comptime ACC_ROW_OFFSETS_32x32 = SIMD[DType.int32, 16](
    0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27
)


struct HKMhaConfig(ImplicitlyCopyable, Movable):
    """Shape configuration for `HKMhaPrefill`.

    Single source of truth for the shape parameters that drive
    register-tile layouts, SMEM sub-block geometry, grid dimensions,
    and IGLP scheduling. Lives in `hk_mha_mma_op` so both `MhaMmaOp`
    and `HKMhaPrefill` can take it as a parameter without a circular
    import.
    """

    var q_block_size: Int
    """Q rows per warp."""

    var kv_block: Int
    """K/V rows per tile (64 at d=128)."""

    var depth: Int
    """Head depth."""

    var num_heads: Int
    """Q num_heads."""

    var num_kv_heads: Int
    """K/V num_heads. `1` (full GQA) or equal to `num_heads` (MHA);
    other ratios need a stride-aware DMA loader (TODO)."""

    var num_warps: Int
    """Warps per block."""

    var rescale_threshold: Float32
    """Lazy-rescale threshold in log2 units of the running max. Above
    this, `o_reg` / `norm_vec` are rescaled by
    `exp2(max_prev - max_new)`; below this the rescale is deferred —
    the residual contribution from `att_block` is bounded by
    `exp2(-rescale_threshold)` of the new max."""

    var output_dtype: DType
    """Element dtype of the output tile `o`. FP32 by default; BF16 for
    production inference where the dispatcher holds a BF16 output buffer.
    The cast from the FP32 accumulator happens per-lane inside
    `_store_o_to_gmem`."""

    def __init__(
        out self,
        *,
        q_block_size: Int,
        kv_block: Int,
        depth: Int,
        num_heads: Int,
        num_kv_heads: Int,
        num_warps: Int = 8,
        rescale_threshold: Float32 = 8.0,
        output_dtype: DType = DType.float32,
    ):
        self.q_block_size = q_block_size
        self.kv_block = kv_block
        self.depth = depth
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_warps = num_warps
        self.rescale_threshold = rescale_threshold
        self.output_dtype = output_dtype


struct MhaMmaOp[T: DType, config: HKMhaConfig]:
    """Namespace-style struct holding the shape constants, register-tile
    layouts, and SMEM→register loaders for `HKMhaPrefill`. All call
    sites go through static methods on this struct.

    Specialized for `v_mfma_f32_32x32x16_bf16`; the MFMA shape, SMEM
    sub-block geometry, and per-lane fragment decomposition are
    hard-coded inside (the rest of the kernel — swizzle, sub-tile
    layouts, ds_read variants — is wired to this exact MFMA).

    Parameters:
        T: Element data type (BF16).
        config: Shape configuration.
    """

    # Config aliases — body code references `Self.Q_BLOCK_SIZE` etc.
    # for readability; the values come from `config`.
    comptime Q_BLOCK_SIZE = Self.config.q_block_size
    comptime KV_BLOCK = Self.config.kv_block
    comptime DEPTH = Self.config.depth

    # MFMA shape — fixed at `v_mfma_f32_32x32x16_bf16`.
    comptime MMA_M = 32
    comptime MMA_N = 32
    comptime MMA_K = 16

    # Shape constants. Sub-block dims are the SMEM tile geometry that
    # matches the SMEM swizzle; FRAG_ELTS / ROWL_* are the per-lane
    # decomposition of the MFMA input fragments.

    comptime K_SUB_ROWS = 32
    """K SMEM sub-block rows (two-XOR swizzle)."""

    comptime K_SUB_COLS = 32
    """K SMEM sub-block cols."""

    comptime V_SUB_ROWS = 8
    """V SMEM sub-block rows (identity swizzle)."""

    comptime V_SUB_COLS = 32
    """V SMEM sub-block cols."""

    comptime ROWL_HALF_LANES = 32
    """Lanes per half-warp in the row_l rt_32x16 decomposition.
    Lanes [0, 32) own one row, [32, 64) own a shifted col block."""

    comptime ROWL_STRIDE = 8
    """Per-lane fragment width in the row_l rt_32x16 decomposition
    (`MMA_K // 2`): half-warp partitioning packs two strips per base tile."""

    comptime FRAG_ELTS = 8
    """BF16 elements per lane per MFMA base tile (`MMA_M * MMA_K / 64`)."""

    # Register-tile layouts. The outer two dims count base tiles; the
    # third dim is the per-lane fragment size.

    comptime Q_LAYOUT = row_major[
        Self.Q_BLOCK_SIZE // Self.MMA_M,
        Self.DEPTH // Self.MMA_K,
        (Self.MMA_M * Self.MMA_K) // 64,
    ]()
    """Q register tile."""

    comptime K_LAYOUT = row_major[
        Self.KV_BLOCK // Self.MMA_M,
        Self.DEPTH // Self.MMA_K,
        (Self.MMA_M * Self.MMA_K) // 64,
    ]()
    """K register tile (whole K, pre-loaded across cluster boundaries)."""

    comptime V_LAYOUT = row_major[
        Self.KV_BLOCK // Self.MMA_K,
        Self.DEPTH // Self.MMA_N,
        (Self.MMA_K * Self.MMA_N) // 64,
    ]()
    """V register tile."""

    comptime ATT_LAYOUT = row_major[
        Self.KV_BLOCK // Self.MMA_M,
        Self.Q_BLOCK_SIZE // Self.MMA_N,
        (Self.MMA_M * Self.MMA_N) // 64,
    ]()
    """Attention block (QK output, col_l rt_32x32 FP32)."""

    # Per-lane decomposition of a col_l rt_32x32 base tile: each lane
    # owns 16 FP32 within one column band — 16 rows by 1 col.
    comptime FRAG_H_COL_L = 16
    """Per-lane row count for one col_l rt_32x32 base tile."""
    comptime FRAG_W_COL_L = 1
    """Per-lane col count for one col_l rt_32x32 base tile."""

    comptime ATT_LAYOUT_NESTED = row_major_nested(
        Coord(
            Coord(
                Idx[Self.KV_BLOCK // Self.MMA_M],
                Idx[Self.FRAG_H_COL_L],
            ),
            Coord(
                Idx[Self.Q_BLOCK_SIZE // Self.MMA_N],
                Idx[Self.FRAG_W_COL_L],
            ),
        )
    )
    """Nested form of `ATT_LAYOUT`: rank 2 (flat_rank 4) — outer dims
    count base tiles, inner sub-dims count per-lane positions. Use via
    `.tile[FRAG_H_COL_L, FRAG_W_COL_L](i, j)` to grab the (16, 1)
    fragment at outer base-tile `(i, j)`."""

    comptime ATT_BF16_SUB_LAYOUT = row_major[
        16 // 16,
        Self.Q_BLOCK_SIZE // Self.MMA_N,
        (16 * Self.MMA_N) // 64,
    ]()
    """One PV-A subtile (16-row strip of att, BF16)."""

    comptime ATT_BF16_FULL_LAYOUT = row_major[
        Self.KV_BLOCK // 16,
        Self.Q_BLOCK_SIZE // Self.MMA_N,
        (16 * Self.MMA_N) // 64,
    ]()
    """Full att BF16 tile pre-cast from FP32 (indexed by subtile_idx
    to feed `mma_PV` strip-by-strip)."""

    comptime O_LAYOUT = row_major[
        Self.DEPTH // Self.MMA_M,
        Self.Q_BLOCK_SIZE // Self.MMA_N,
        (Self.MMA_M * Self.MMA_N) // 64,
    ]()
    """Output accumulator (col_l rt_32x32 FP32)."""

    comptime O_LAYOUT_NESTED = row_major_nested(
        Coord(
            Coord(
                Idx[Self.DEPTH // Self.MMA_M],
                Idx[Self.FRAG_H_COL_L],
            ),
            Coord(
                Idx[Self.Q_BLOCK_SIZE // Self.MMA_N],
                Idx[Self.FRAG_W_COL_L],
            ),
        )
    )
    """CuTe-nested form of `O_LAYOUT`. See `ATT_LAYOUT_NESTED`."""

    comptime O_T_LAYOUT = row_major[
        Self.Q_BLOCK_SIZE // Self.MMA_M,
        Self.DEPTH // Self.MMA_N,
        (Self.MMA_M * Self.MMA_N) // 64,
    ]()
    """Output transpose (row_l view of the same storage as `O_LAYOUT`)."""

    # K SMEM uses a byte-level two-XOR swizzle: the pair
    # `Swizzle(1, 1, 4)` + `Swizzle(1, 0, 6)` realizes
    # `bit5 ^= bit9; bit4 ^= bit10` on the per-lane 16-byte vec index.
    # FP32 SMEM (unused here) would take the identity branch.

    @staticmethod
    @always_inline
    def _swizzle_K_sub(r: Int, c: Int) -> Int:
        """Returns the byte offset of `(r, c)` within a K sub-block,
        with the two-XOR swizzle applied for BF16."""
        var offset = size_of[Self.T]() * (r * Self.K_SUB_COLS + c)
        comptime if size_of[Self.T]() == 2:
            var first = ((offset % 1024) >> 9) << 5  # bit 5 ^= bit 9
            var second = ((offset % 2048) >> 10) << 4  # bit 4 ^= bit 10
            return offset ^ first ^ second
        else:
            comptime assert (
                size_of[Self.T]() == 4
            ), "MhaMmaOp._swizzle_K_sub: BF16/F32 only"
            return offset

    @staticmethod
    @always_inline
    def load_K[
        layout_dst: TensorLayout,
        layout_src: TensorLayout,
        //,
    ](
        mut dst: RegTile[Self.T, layout_dst, MutExternalOrigin],
        src: SMemTile[Self.T, layout_src, MutAnyOrigin],
    ):
        """Loads the whole `(KV_BLOCK, DEPTH)` K tile from SMEM into the
        row_l rt_32x16 register tile, unswizzling on the way.

        Caller must declare K SMEM with shape
        `row_major[KV_BLOCK * (DEPTH / K_SUB_COLS), K_SUB_COLS]` so the
        sub-block id linearizes via `.tile[K_SUB_ROWS, K_SUB_COLS](id, 0)`.
        """
        comptime assert Self.T == DType.bfloat16, "MhaMmaOp.load_K: BF16 only"
        comptime assert (
            layout_dst.static_shape[2] == Self.FRAG_ELTS
        ), "MhaMmaOp.load_K: dst per-base-tile elts must be FRAG_ELTS"

        comptime height = layout_dst.static_shape[0]
        comptime width = layout_dst.static_shape[1]
        comptime width_st = (width * Self.MMA_K) // Self.K_SUB_COLS

        # row_l fragment: lane `lid` owns `(row, col)` in a 32x32 base tile
        # at `row = lid % 32`, `col = 8 * (lid // 32)`.
        var lid = Int(lane_id())
        var row_offset = lid % Self.ROWL_HALF_LANES
        var col_offset = Self.ROWL_STRIDE * (lid // Self.ROWL_HALF_LANES)

        # Precompute lane-local swizzled offsets for the two col parities;
        # each base-tile cycle picks one.
        var swizzled_elts_even = (
            Self._swizzle_K_sub(row_offset, col_offset) // size_of[Self.T]()
        )
        var swizzled_elts_odd = (
            Self._swizzle_K_sub(row_offset, Self.MMA_K + col_offset)
            // size_of[Self.T]()
        )

        var dst_v = dst.vectorize[1, 1, Self.FRAG_ELTS]()
        comptime assert dst_v.flat_rank == 3

        comptime for register_row in range(height):
            comptime for register_col in range(width):
                comptime sub_id = register_row * width_st + register_col // 2
                comptime is_odd_col = (register_col % 2) == 1

                var subtile = src.tile[Self.K_SUB_ROWS, Self.K_SUB_COLS](
                    sub_id, 0
                )
                var swizzled_elts = (
                    swizzled_elts_odd if is_odd_col else swizzled_elts_even
                )
                var smem_at_lane = (subtile.ptr + swizzled_elts).bitcast[
                    Scalar[DType.bfloat16]
                ]()
                var frag = _load_from_lds[width=Self.FRAG_ELTS](smem_at_lane)
                dst_v[register_row, register_col, 0] = rebind[
                    dst_v.ElementType
                ](frag)

    @staticmethod
    @always_inline
    def load_V[
        layout_dst: TensorLayout,
        layout_src: TensorLayout,
        //,
    ](
        mut dst: RegTile[Self.T, layout_dst, MutExternalOrigin],
        src: SMemTile[Self.T, layout_src, MutAnyOrigin],
    ):
        """Loads the whole V tile from SMEM into the col_l rt_16x32 register
        tile via `ds_read_tr16_b64_warp` transpose-loads.

        Each output base tile spans two `V_SUB_ROWS`-tall SMEM sub-blocks
        (top + bot); they are joined into one 8-element MMA fragment.
        Caller must declare V SMEM with shape
        `row_major[KV_BLOCK * (DEPTH / V_SUB_COLS), V_SUB_COLS]`.
        """
        comptime assert Self.T == DType.bfloat16, "MhaMmaOp.load_V: BF16 only"
        comptime assert (
            layout_dst.static_shape[2] == Self.FRAG_ELTS
        ), "MhaMmaOp.load_V: dst per-base-tile elts must be FRAG_ELTS"

        comptime height = layout_dst.static_shape[0]
        comptime width = layout_dst.static_shape[1]
        comptime subtiles_per_row = width
        comptime mma_shape = IndexList[3](Self.MMA_M, Self.MMA_N, Self.MMA_K)

        var dst_v = dst.vectorize[1, 1, Self.FRAG_ELTS]()
        comptime assert dst_v.flat_rank == 3

        comptime for i in range(height):
            comptime for j in range(width):
                comptime top_sid = (2 * i) * subtiles_per_row + j
                comptime bot_sid = (2 * i + 1) * subtiles_per_row + j

                var top_tile = src.tile[Self.V_SUB_ROWS, Self.V_SUB_COLS](
                    top_sid, 0
                )
                var bot_tile = src.tile[Self.V_SUB_ROWS, Self.V_SUB_COLS](
                    bot_sid, 0
                )

                var part_top = ds_read_tr16_b64_warp[mma_shape](top_tile)
                var part_bot = ds_read_tr16_b64_warp[mma_shape](bot_tile)
                var frag = part_top.join(part_bot)

                dst_v[i, j, 0] = rebind[dst_v.ElementType](frag)

    # MMA dispatch — MFMA over RegTile, named by attention semantic.

    @staticmethod
    @always_inline
    def mma_QK[
        T_att: DType,
        layout_att: TensorLayout,
        layout_k: TensorLayout,
        layout_q: TensorLayout,
        //,
    ](
        mut att: RegTile[T_att, layout_att, MutExternalOrigin],
        mut k: RegTile[Self.T, layout_k, MutExternalOrigin],
        mut q: RegTile[Self.T, layout_q, MutExternalOrigin],
    ):
        """QK MFMA: `att += k @ q^T`. K is A (M-outer), Q is B (N-outer).

        For each output base tile `(n, m)`:
        `att[n, m] += sum_k k[n, k] * q[m, k]`.
        """
        comptime ATT_h = layout_att.static_shape[0]
        comptime ATT_w = layout_att.static_shape[1]
        comptime K_count = layout_k.static_shape[1]

        comptime assert (
            layout_k.static_shape[1] == layout_q.static_shape[1]
        ), "mma_QK: K.K (width) != Q.K (width)"
        comptime assert (
            layout_k.static_shape[0] == ATT_h
        ), "mma_QK: K.M (height) != att.M (height)"
        comptime assert (
            layout_q.static_shape[0] == ATT_w
        ), "mma_QK: Q.N (height) != att.N (width)"

        var att_v = att.vectorize[1, 1, layout_att.static_shape[2]]()
        var k_v = k.vectorize[1, 1, layout_k.static_shape[2]]()
        var q_v = q.vectorize[1, 1, layout_q.static_shape[2]]()
        comptime assert att_v.flat_rank == 3
        comptime assert k_v.flat_rank == 3
        comptime assert q_v.flat_rank == 3

        comptime for n in range(ATT_h):
            comptime for m in range(ATT_w):
                var d_simd = att_v[n, m, 0]
                comptime for kk in range(K_count):
                    gpu_mma(d_simd, k_v[n, kk, 0], q_v[m, kk, 0], d_simd)
                att_v[n, m, 0] = d_simd

    @staticmethod
    @always_inline
    def mma_PV[
        T_o: DType,
        layout_o: TensorLayout,
        layout_v: TensorLayout,
        layout_p: TensorLayout,
        //,
    ](
        mut o: RegTile[T_o, layout_o, MutExternalOrigin],
        mut v: RegTile[Self.T, layout_v, MutExternalOrigin],
        mut p: RegTile[Self.T, layout_p, MutExternalOrigin],
    ):
        """PV MFMA: `o += v^T @ p`. V is A (K-outer), P is B (K-outer,
        JIT-cast BF16 from `att_block`).

        For each output base tile `(n, m)`:
        `o[n, m] += sum_k v[k, n] * p[k, m]`.
        """
        comptime O_h = layout_o.static_shape[0]
        comptime O_w = layout_o.static_shape[1]
        comptime K_count = layout_v.static_shape[0]

        comptime assert (
            layout_v.static_shape[0] == layout_p.static_shape[0]
        ), "mma_PV: V.K (height) != P.K (height)"
        comptime assert (
            layout_v.static_shape[1] == O_h
        ), "mma_PV: V.M (width) != o.M (height)"
        comptime assert (
            layout_p.static_shape[1] == O_w
        ), "mma_PV: P.N (width) != o.N (width)"

        var o_v = o.vectorize[1, 1, layout_o.static_shape[2]]()
        var v_v = v.vectorize[1, 1, layout_v.static_shape[2]]()
        var p_v = p.vectorize[1, 1, layout_p.static_shape[2]]()
        comptime assert o_v.flat_rank == 3
        comptime assert v_v.flat_rank == 3
        comptime assert p_v.flat_rank == 3

        comptime for n in range(O_h):
            comptime for m in range(O_w):
                var d_simd = o_v[n, m, 0]
                comptime for kk in range(K_count):
                    gpu_mma(d_simd, v_v[kk, n, 0], p_v[kk, m, 0], d_simd)
                o_v[n, m, 0] = d_simd

    @staticmethod
    @always_inline
    def exp2_inplace_range[
        T_att: DType,
        layout: TensorLayout,
        //,
        start: Int,
        end: Int,
    ](
        mut tile: RegTile[T_att, layout, MutExternalOrigin],
    ) where T_att.is_floating_point():
        """In-place `exp2` over a base-tile-aligned per-lane slice
        `tile[start:end]`. `start` and `end` must be multiples of the
        fragment width so the slice maps to whole base tiles. Used to
        split first / second half of the softmax exp2 across PV MFMAs."""
        comptime _W = layout.static_shape[1]
        comptime _F = layout.static_shape[2]
        comptime _N = layout.static_shape[0] * _W * _F
        comptime assert (
            0 <= start <= end <= _N
        ), "exp2_inplace_range: [start, end) must lie in [0, total)"
        comptime assert (
            start % _F == 0 and end % _F == 0
        ), "exp2_inplace_range: start/end must be multiples of fragment width"
        var v = tile.vectorize[1, 1, _F]()
        comptime assert v.flat_rank == 3
        comptime for b in range(start // _F, end // _F):
            comptime _h, _w = divmod(b, _W)
            v[_h, _w, 0] = math_exp2(v[_h, _w, 0])
