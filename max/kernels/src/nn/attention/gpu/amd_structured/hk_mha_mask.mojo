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
"""Generic mask functor application for the HK MHA `att_block`.

`apply_mask_to_att_block` is the kernel-facing entry. It accepts any
`MHAMask` and the per-tile `TileMaskStatus` returned by
`mask_functor.status(...)`, and rewrites the per-lane FP32 fragment
in place. The comptime dispatcher inside picks one of three paths:

- `NullMask` — comptime-elided (no-op). The status will be
  `NO_MASK` so callers don't even reach this path in practice.
- `CausalMask` — the 16-wide SIMD fast path (one `v_cmp` + one
  `v_cndmask` per stripe), generalized for `start_pos` so the
  causal cap moves with the cache start position.
- Anything else (`SlidingWindowCausalMask`, `ChunkedCausalMask`,
  `MaterializedMask`, fused combinations) — per-element loop over
  the 16 fragment slots, calling `mask_functor.mask(coord, score)`
  with the global `(seq, head, q_idx, k_idx)` coord. Compiler
  inlines and (often) re-vectorizes; the `mask_op.mojo` production
  AMD path uses the same shape.

Per-element row-within-tile mapping comes from the
`v_mfma_f32_32x32x16_bf16` accumulator fragment geometry; see
`MhaMmaOp.ACC_ROW_OFFSETS_32x32`.
"""

from std.sys.intrinsics import _type_is_eq
from std.utils.index import IndexList

from layout import TensorLayout
from nn.attention.mha_mask import CausalMask, MHAMask, NullMask, TileMaskStatus
from structured_kernels.amd_tile_io import RegTile

from .hk_mha_mma_op import ACC_ROW_OFFSETS_32x32


@always_inline
def _apply_causal_mask_fast[
    layout: TensorLayout,
    //,
    Q_BLOCK_SIZE: Int,
    KV_BLOCK_SIZE: Int,
](
    mut dst: RegTile[DType.float32, layout, MutExternalOrigin],
    q_tile_idx: Int32,
    k_tile_idx: Int32,
    start_pos: Int32,
    lane: Int32,
):
    """Causal mask fast path. One `v_cmp` + one `v_cndmask` per stripe.

    Identical structure to the prior `mask_kv_tile` but with
    `start_pos` baked into the Q position so the causal cap moves
    with the cache start position for prefill-with-cache."""
    comptime dst_height = layout.static_shape[0]
    comptime dst_width = layout.static_shape[1]
    comptime base_tile_elts = layout.static_shape[2]

    comptime assert base_tile_elts == 16, (
        "_apply_causal_mask_fast: requires base_tile_elts == 16 (col_l"
        " rt_32x32 fragment)"
    )

    comptime _NEG_INF_VEC = SIMD[DType.float32, 16](Float32(-3.4028235e38))
    comptime _ROW_OFFSETS = ACC_ROW_OFFSETS_32x32

    var col_in_tile = lane & Int32(31)
    var row_extra = (lane >> Int32(5)) << Int32(2)
    var q_pos = q_tile_idx * Int32(Q_BLOCK_SIZE) + col_in_tile + start_pos
    var k_base = k_tile_idx * Int32(KV_BLOCK_SIZE)

    var dst_vec = dst.vectorize[1, 1, 16]()
    comptime assert dst_vec.flat_rank == 3

    comptime for i in range(dst_height):
        # `rel < _ROW_OFFSETS[p]` is the per-element form of `k_pos > q_pos`
        # once the per-element row offset is folded in.
        var rel = q_pos - (k_base + Int32(i * 32) + row_extra)
        var mask = SIMD[DType.int32, 16](rel).lt(_ROW_OFFSETS)
        comptime for j in range(dst_width):
            dst_vec[i, j, 0] = mask.select(_NEG_INF_VEC, dst_vec[i, j, 0])


@always_inline
def _fill_dst_neg_inf[
    layout: TensorLayout,
    //,
](mut dst: RegTile[DType.float32, layout, MutExternalOrigin]):
    """FULL_MASK path: every fragment slot to `-inf`. Subsequent
    `exp2` sends them all to zero. Cheaper than `mask_functor.mask`
    when the entire tile is masked out."""
    comptime dst_height = layout.static_shape[0]
    comptime dst_width = layout.static_shape[1]
    comptime _NEG_INF_VEC = SIMD[DType.float32, 16](Float32(-3.4028235e38))
    var dst_vec = dst.vectorize[1, 1, 16]()
    comptime assert dst_vec.flat_rank == 3
    comptime for i in range(dst_height):
        comptime for j in range(dst_width):
            dst_vec[i, j, 0] = _NEG_INF_VEC


@always_inline
def _apply_mask_generic[
    mask_t: MHAMask,
    layout: TensorLayout,
    //,
    Q_BLOCK_SIZE: Int,
    KV_BLOCK_SIZE: Int,
](
    mask_functor: mask_t,
    mut dst: RegTile[DType.float32, layout, MutExternalOrigin],
    q_tile_idx: Int32,
    k_tile_idx: Int32,
    start_pos: Int32,
    head_idx: UInt32,
    batch_idx: UInt32,
    lane: Int32,
):
    """Generic per-element mask path. Calls `mask_functor.mask(coord,
    score)` 16 times per fragment stripe with width-1 SIMD. Used
    when the mask is not `CausalMask` / `NullMask`; the compiler
    inlines `mask` and vectorizes back where possible."""
    comptime dst_height = layout.static_shape[0]
    comptime dst_width = layout.static_shape[1]

    var col_in_tile = lane & Int32(31)
    var row_extra = (lane >> Int32(5)) << Int32(2)
    var q_pos = q_tile_idx * Int32(Q_BLOCK_SIZE) + col_in_tile + start_pos
    var k_base = k_tile_idx * Int32(KV_BLOCK_SIZE)

    var dst_vec = dst.vectorize[1, 1, 16]()
    comptime assert dst_vec.flat_rank == 3
    comptime for i in range(dst_height):
        comptime for j in range(dst_width):
            var frag = dst_vec[i, j, 0]
            comptime for k_local in range(16):
                var k_pos = (
                    k_base
                    + Int32(i * 32)
                    + row_extra
                    + Int32(ACC_ROW_OFFSETS_32x32[k_local])
                )
                var score = SIMD[DType.float32, 1](frag[k_local])
                var masked = mask_functor.mask(
                    IndexList[4, element_type=DType.uint32](
                        Int(batch_idx),
                        Int(head_idx),
                        Int(q_pos),
                        Int(k_pos),
                    ),
                    score,
                )
                frag[k_local] = masked[0]
            dst_vec[i, j, 0] = frag


@always_inline
def apply_mask_to_att_block[
    mask_t: MHAMask,
    layout: TensorLayout,
    //,
    Q_BLOCK_SIZE: Int,
    KV_BLOCK_SIZE: Int,
](
    mask_functor: mask_t,
    mut dst: RegTile[DType.float32, layout, MutExternalOrigin],
    q_tile_idx: Int32,
    k_tile_idx: Int32,
    start_pos: Int32,
    head_idx: UInt32,
    batch_idx: UInt32,
    lane: Int32,
    mask_status: TileMaskStatus,
):
    """Generic mask functor application for an HK `att_block` tile.

    Dispatches on `mask_status` + `mask_t` at comptime:

    - `NO_MASK` — return (caller already guarded; defensive only).
    - `FULL_MASK` — fill `dst` with `-inf`. The subsequent `exp2`
      zeros every entry, which is the correct softmax result for
      an entirely-masked tile.
    - `PARTIAL`:
      - `CausalMask` → the 16-wide SIMD fast path with `start_pos`
        shift (one `v_cmp` + one `v_cndmask` per stripe).
      - `NullMask` → never reached in practice (status is always
        `NO_MASK`); kept as a comptime no-op for safety.
      - Any other `MHAMask` → per-element `mask_functor.mask(coord,
        score)` loop over the 16 fragment slots.

    Parameters:
        mask_t: The mask functor type (any `MHAMask`).
        layout: `dst` layout, `row_major[height, width, 16]` for col_l
            rt_32x32 (16 = fragment elements per lane).
        Q_BLOCK_SIZE: Q rows per tile (`HKMhaConfig.q_block_size`).
        KV_BLOCK_SIZE: K rows per tile (`HKMhaConfig.kv_block`).

    Args:
        mask_functor: Mask instance (may carry state — sliding window
            radius, materialized buffer pointer, etc.).
        dst: Attention block tile (mutated in place).
        q_tile_idx: Absolute Q tile index (this warp's first row /
            Q_BLOCK_SIZE; usually `block_tile_idx * NUM_WARPS + w_id`).
        k_tile_idx: Absolute K tile index.
        start_pos: KV-cache start position. Shifts the Q absolute
            position by `start_pos` to account for cache reuse.
        head_idx: Q head index (`block_idx.x` in HK).
        batch_idx: Batch index (`block_idx.z` in HK).
        lane: `lane_id()` cast to `Int32`.
        mask_status: Result of `mask_functor.status(...)` for this
            (q_tile, k_tile) pair. Caller is responsible for calling
            `status()` once per tile and passing the result here.
    """
    if mask_status == TileMaskStatus.FULL_MASK:
        _fill_dst_neg_inf(dst)
        return
    if mask_status == TileMaskStatus.NO_MASK:
        return

    # PARTIAL: actually apply the mask.
    comptime if _type_is_eq[mask_t, NullMask]():
        # NullMask says NO_MASK for every tile, so reaching PARTIAL is
        # impossible — comptime-elide.
        return
    elif _type_is_eq[mask_t, CausalMask]():
        _apply_causal_mask_fast[
            Q_BLOCK_SIZE=Q_BLOCK_SIZE, KV_BLOCK_SIZE=KV_BLOCK_SIZE
        ](dst, q_tile_idx, k_tile_idx, start_pos, lane)
    else:
        _apply_mask_generic[
            Q_BLOCK_SIZE=Q_BLOCK_SIZE, KV_BLOCK_SIZE=KV_BLOCK_SIZE
        ](
            mask_functor,
            dst,
            q_tile_idx,
            k_tile_idx,
            start_pos,
            head_idx,
            batch_idx,
            lane,
        )
