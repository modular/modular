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
"""Softmax warp group logic for FA4 (SM100 Flash Attention)."""

from std.math import exp2, recip, align_up
from std.math.constants import log2e
from std.utils.numerics import min_or_neg_inf
from std.sys import size_of, get_defined_int
from std.sys.info import _accelerator_arch
import std.gpu.primitives.warp as warp
from std.gpu.globals import WARPGROUP_SIZE
from std.gpu.memory import AddressSpace, fence_async_view_proxy
from std.gpu.sync import (
    named_barrier,
    cp_async_bulk_commit_group,
    cp_async_bulk_wait_group,
    umma_arrive_leader_cta,
)
from std.gpu.primitives.cluster import block_rank_in_cluster
from std.gpu.compute.arch.tcgen05 import (
    tcgen05_dealloc,
    tcgen05_fence_after,
    tcgen05_fence_before,
    tcgen05_ld,
    tcgen05_release_allocation_lock,
    tcgen05_store_wait,
)
from std.gpu.primitives.warp import _vote_nvidia_helper
from layout.swizzle import make_swizzle
from layout.tma_async import RaggedTMA3DTile
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from nn.attention.gpu.nvidia.sm100.attention import (
    FA4Config,
    EnableForcedOrdering,
    EnableEarlyAdd,
)
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    add_ftz,
    apply_mask,
    combine_pack_o_row,
    elect,
    exp2_emulation,
    fma_ftz,
    llvm_opaque_tid,
    load_cluster_smem,
    maximum,
    MBarType,
    mul_ftz,
    pack_row,
    peel_mask,
    scale_pack_o_row,
    SharedMemPointer,
    SM100TensorAccumulator,
    splitk_partition_idx,
    splitk_window,
    st_shared_v4_b32,
    sub_ftz,
    TMemTile,
)
from nn.attention.gpu.nvidia.common import (
    MHAPosition,
    NullPointer,
    OptionalPointer,
)
from nn.attention.mha_mask import MHAMask, TileMaskStatus, MaskStrategy
from nn.attention.mha_operand import MHAOperand
from nn.attention.gpu.nvidia.mha_tile_scheduler import SeqInfo
from nn.attention.mha_utils import OptionallyStaticInt, _is_decoding
from std.utils.index import Index
from std.utils.static_tuple import StaticTuple
from .smem import SM100AttentionSMem


@always_inline
def fa4_scale_write_output[
    output_type: DType,
    //,
    config: FA4Config,
    output_swizzle_mode: TensorMapSwizzle,
    tma_bpo: Int,
](
    local_row: UInt32,
    local_warp_idx: UInt32,
    warp_group_idx: UInt32,
    inv_row_sum: Float32,
    o_smem_arg: SharedMemPointer[Scalar[output_type]],
    o_tmem_arg: TMemTile[
        DType.float32, config.BM // config.num_q, config.padded_ov_depth
    ],
    ragged_tma_store: RaggedTMA3DTile[
        output_type,
        output_swizzle_mode,
        # `config.BM // config.num_q` is "rows this WG writes": 128 in
        # 2Q (BM=256, two WGs split rows) and 128 in 1Q (BM=128, one WG
        # writes the full set in the T==1 fast path; the multi-tile 1Q
        # path uses fa4_lse_combine_write instead). Same numeric value
        # in both modes.
        BM=config.BM // config.num_q,
        BN=config.ov_depth,
        middle_dim=_,
        group=config.group if config.fuse_gqa else 1,
        # `tma_bpo` (blocks per batched TMA op) is inferred from the store arg:
        # >0 => one batched copy per phase (rank-4 group==1, rank-5 group>1),
        # 0 => per-block (swizzled-output fallback).
        tma_blocks_per_op=tma_bpo,
    ],
    num_output_rows: Int32,
    out_head_idx: UInt32,
    out_row_idx: UInt32,
):
    comptime accum_dtype = DType.float32
    # Rows this WG writes: 128 in both 2Q (BM=256, two WGs split rows in half)
    # and 1Q (BM=128, one WG owns all rows in the T==1 fast path). Matches the
    # TMA store descriptor's BM.
    comptime BM = config.BM // config.num_q
    comptime ov_depth = config.ov_depth

    # O SMEM is row-major (SWIZZLE_NONE): the O accumulator is loaded
    # one-row-per-thread via `tcgen05_ld[datapaths=32]` (warp w, lane l -> row
    # 32*w + l; 4 warps cover 128 rows), exactly the row ownership the S
    # reductions use. So `inv_row_sum` is already the rescale factor for the
    # row this thread writes -- no `warp.shuffle_idx` -- and the per-row 16 B
    # stores stay bank-conflict-free (8 rows * 16 B = 128 B = all 32 banks
    # once). The inner swizzle is the identity for SWIZZLE_NONE, so each
    # k-block is plain row-major [BM, o_sw_K].
    comptime o_swizzle = make_swizzle[output_type, output_swizzle_mode]()
    comptime o_sw_K = output_swizzle_mode.bytes() // size_of[output_type]()
    # The group-of-8 / 16 B store path requires a 2-byte output element
    # (8 elems * 2 B = 16 B = uint32x4). This is the same constraint the
    # previous `output_reg_to_smem_st_matrix` path enforced.
    comptime assert (
        size_of[output_type]() == 2
    ), "fa4_scale_write_output requires a 2-byte output dtype (bf16/f16)"

    # Output column count is aligned to the OUTPUT swizzle granularity
    # (o_sw_K, the SWIZZLE_NONE 16 B box = 8 bf16), NOT the QKV swizzle's
    # `padded_ov_depth` (which aligns to SWIZZLE_128B = 64 elems). ov_depth is
    # already a multiple of o_sw_K for every supported head size, so this is
    # exact and needs no padding (e.g. depth 72 -> 9 boxes, not 128/64=2).
    comptime o_sw_blocks = align_up(ov_depth, o_sw_K) // o_sw_K
    comptime batched = tma_bpo > 0
    comptime if batched:
        comptime assert (
            tma_bpo == (o_sw_blocks + 1) // 2
        ), "batched scale_write expects a half-depth (ceil(blocks/2)) box."

    var e = elect()
    if local_warp_idx == 0:
        if e != 0:
            ragged_tma_store.prefetch_descriptor()

    # Each thread owns output row `local_row` (= tid % 128). Load that row from
    # TMEM for one o_sw_K-wide block, scale+pack (f32x2 compute, wide store; see
    # `scale_pack_o_row`), and write one 16 B row-major store.
    @parameter
    @always_inline
    def write_block[blk: Int]():
        comptime col = blk * o_sw_K
        var o_vals = tcgen05_ld[
            datapaths=32,
            bits=32,
            repeat=o_sw_K,
            dtype=accum_dtype,
            pack=False,
            width=o_sw_K,
        ](o_tmem_arg.tmem_addr + UInt32(col))

        var packed = scale_pack_o_row[output_type, w=o_sw_K](
            o_vals, inv_row_sum
        )

        # Block `blk` is one k-block [BM, o_sw_K]; col % o_sw_K == 0.
        var o_inner = Int(local_row) * o_sw_K
        (o_smem_arg + blk * BM * o_sw_K + o_swizzle(o_inner)).bitcast[
            Scalar[DType.uint32]
        ]().store(packed)

    comptime if batched:
        # Single issuer, 2-phase pipeline: write the first half to smem and kick
        # off its batched TMA, then write the second half (which overlaps the
        # first TMA's copy) and kick off its TMA. The two halves touch disjoint
        # smem blocks, so there is no read/write hazard. The batched store now
        # also covers fused GQA (group > 1): the (middle_dim, rows) selector
        # merge in RaggedTMA3DTile keeps the descriptor within the 5D limit
        # (rank-4 for group==1, rank-5 for group>1); write_block is unchanged
        # (its BM already includes group).
        comptime for blk in range(tma_bpo):
            write_block[blk]()
        named_barrier[Int32(WARPGROUP_SIZE)](Int32(warp_group_idx))
        if local_warp_idx == 0:
            fence_async_view_proxy()
            ragged_tma_store.async_copy_batched[0](
                o_smem_arg,
                ragged_idx=out_row_idx,
                dynamic_dim=UInt32(num_output_rows),
                middle_idx=out_head_idx,
                elect=e,
            )

        comptime for blk in range(tma_bpo, o_sw_blocks):
            write_block[blk]()
        named_barrier[Int32(WARPGROUP_SIZE)](Int32(warp_group_idx))
        if local_warp_idx == 0:
            fence_async_view_proxy()
            # col_start = tma_bpo; the box may overhang the last block
            # for odd o_sw_blocks -> masked off by the TMA.
            ragged_tma_store.async_copy_batched[tma_bpo](
                o_smem_arg,
                ragged_idx=out_row_idx,
                dynamic_dim=UInt32(num_output_rows),
                middle_idx=out_head_idx,
                elect=e,
            )
            cp_async_bulk_commit_group()
    else:
        # tma_bpo == 0: swizzled-output callers (e.g. an MLA variant with a
        # SWIZZLE_128B output store) can't use the blocked-smem batched box, so
        # fall back to one per-block TMA each. (Fused GQA with SWIZZLE_NONE now
        # takes the batched branch above.)
        comptime for blk in range(o_sw_blocks):
            write_block[blk]()
        named_barrier[Int32(WARPGROUP_SIZE)](Int32(warp_group_idx))
        if local_warp_idx == 0:
            fence_async_view_proxy()
            comptime for blk in range(o_sw_blocks):
                ragged_tma_store.async_copy_from_col[blk](
                    o_smem_arg,
                    ragged_idx=out_row_idx,
                    dynamic_dim=UInt32(num_output_rows),
                    middle_idx=out_head_idx,
                    elect=e,
                )
            cp_async_bulk_commit_group()
    cp_async_bulk_wait_group[0]()


@always_inline
def fa4_lse_combine_write[
    output_type: DType,
    //,
    config: FA4Config,
    wg_j_offset: Int,
    iters_per_wg: Int,
    output_swizzle_mode: TensorMapSwizzle,
    tma_bpo: Int,
](
    local_row: UInt32,
    local_warp_idx: UInt32,
    warp_group_idx: UInt32,
    final_scale_local: Float32,
    final_scale_peer: Float32,
    o_smem_arg: SharedMemPointer[Scalar[output_type]],
    own_o_tmem: TMemTile[DType.float32, config.BM, config.padded_ov_depth],
    peer_o_tmem: TMemTile[DType.float32, config.BM, config.padded_ov_depth],
    ragged_tma_store: RaggedTMA3DTile[
        output_type,
        output_swizzle_mode,
        # 1Q only: equals config.BM (= 128); kept as `config.BM //
        # config.num_q` for typewise consistency with the fa4_softmax
        # signature and kernel.mojo construction (which use the same
        # expression to give 128 in both 2Q and 1Q).
        BM=config.BM // config.num_q,
        BN=config.ov_depth,
        middle_dim=_,
        group=config.group if config.fuse_gqa else 1,
        # `tma_bpo` (blocks per batched TMA op) is inferred from the store arg:
        # >0 => this WG issues one batched copy over its half (rank-4 group==1,
        # rank-5 group>1), 0 => per-block.
        tma_blocks_per_op=tma_bpo,
    ],
    num_output_rows: Int32,
    out_head_idx: UInt32,
    out_row_idx: UInt32,
):
    """LSE-combine two TMEM_O fragments and TMA-store a depth-column slice.

    1Q-only sibling of `fa4_scale_write_output`. Each WG handles a disjoint
    range `j in [wg_j_offset, wg_j_offset + iters_per_wg)` of swizzle-block
    columns. For each `j`, the WG loads both its own and the peer's TMEM_O
    fragments, combines them in registers via per-row scales
    (`final_scale_local` for own, `final_scale_peer` for peer), writes the
    combined output to the shared `o_smem_arg` at the `j` slot, then
    TMA-stores that slot to gmem. Both WGs target the same `BM` Q rows but
    disjoint depth columns, so smem and gmem regions never overlap.

    The caller must have already waited on both `pipeline_o0` and
    `pipeline_o1` producer barriers (and issued `tcgen05_fence_after()`)
    before invoking this helper, so the TMEM fragments are visible.
    """
    comptime assert config.num_q == 1

    comptime accum_dtype = DType.float32
    # 1Q: config.BM == 128 == config.BM // config.num_q (matches the TMA
    # store descriptor's BM and the shared o_smem tile extent).
    comptime BM = config.BM // config.num_q

    # O SMEM is row-major (SWIZZLE_NONE): O is loaded one-row-per-thread via
    # `tcgen05_ld[datapaths=32]`, so `final_scale_local`/`final_scale_peer`
    # are already this thread's row scales (no `warp.shuffle_idx`), the combine
    # is a pure per-thread register op, and the per-row 16 B stores stay
    # bank-conflict-free (8 rows * 16 B = 128 B = all 32 banks once). The inner
    # swizzle is the identity for SWIZZLE_NONE, so each k-block is plain
    # row-major [BM, o_sw_K].
    comptime o_swizzle = make_swizzle[output_type, output_swizzle_mode]()
    comptime o_sw_K = output_swizzle_mode.bytes() // size_of[output_type]()
    comptime assert (
        size_of[output_type]() == 2
    ), "fa4_lse_combine_write requires a 2-byte output dtype (bf16/f16)"

    # Each WG handles a disjoint range of o_sw_K-wide column blocks
    # [wg_j_offset, wg_j_offset + iters_per_wg). `iters` matches the caller's
    # `iters_total` and is the output column count aligned to the OUTPUT
    # swizzle granularity (o_sw_K), NOT the QKV swizzle's `padded_ov_depth`
    # (which aligns to SWIZZLE_128B = 64 elems). ov_depth is already a multiple
    # of o_sw_K for every supported head size, so this is exact. Under
    # SWIZZLE_NONE the block size is small (o_sw_K = 8 for bf16), so depth=64
    # yields iters=8 and both WGs participate; the caller's `if iters_per_wg1
    # > 0` guard still skips a WG only when its range is empty.
    comptime iters = align_up(config.ov_depth, o_sw_K) // o_sw_K
    comptime assert iters_per_wg >= 1, (
        "fa4_lse_combine_write requires at least one column block per"
        " call; the caller must skip WG1 when iters_per_wg would be 0."
    )
    comptime assert wg_j_offset + iters_per_wg <= iters

    # Batched: each WG issues ONE TMA over its block range (rank-4 for group==1,
    # rank-5 for group>1 fused GQA, after the RaggedTMA3DTile selector merge).
    # The box is the half-depth `ceil(iters/2)`; WG0 (wg_j_offset=0) fills it
    # exactly, WG1 (wg_j_offset=ceil) overhangs the last block for odd `iters`,
    # which the TMA masks off.
    comptime batched = tma_bpo > 0
    comptime if batched:
        comptime assert (
            tma_bpo == (iters + 1) // 2
        ), "batched combine expects a half-depth (ceil(iters/2)) box."
        comptime assert (
            wg_j_offset == 0 or wg_j_offset == tma_bpo
        ), "batched combine expects wg_j_offset in {0, tma_bpo}."

    var e = elect()
    if local_warp_idx == 0:
        if e != 0:
            ragged_tma_store.prefetch_descriptor()

    # Each thread owns output row `local_row` (= tid % 128). Combine own+peer
    # for this WG's block range and write row-major to SMEM.
    comptime for iter in range(iters_per_wg):
        comptime j = wg_j_offset + iter
        comptime col_start = j * o_sw_K
        var own_arr = tcgen05_ld[
            datapaths=32,
            bits=32,
            repeat=o_sw_K,
            dtype=accum_dtype,
            pack=False,
            width=o_sw_K,
        ](own_o_tmem.tmem_addr + UInt32(col_start))
        var peer_arr = tcgen05_ld[
            datapaths=32,
            bits=32,
            repeat=o_sw_K,
            dtype=accum_dtype,
            pack=False,
            width=o_sw_K,
        ](peer_o_tmem.tmem_addr + UInt32(col_start))

        # combined = own * final_scale_local + peer * final_scale_peer, packed
        # into one 16 B store (f32x2 compute, wide store; see combine_pack_o_row).
        var packed = combine_pack_o_row[output_type](
            own_arr, peer_arr, final_scale_local, final_scale_peer
        )

        # Block `j` is one k-block [BM, o_sw_K]; col % o_sw_K == 0.
        var o_inner = Int(local_row) * o_sw_K
        (o_smem_arg + j * BM * o_sw_K + o_swizzle(o_inner)).bitcast[
            Scalar[DType.uint32]
        ]().store(packed)

    # Sync all WARPGROUP_SIZE threads before the TMA store.
    named_barrier[Int32(WARPGROUP_SIZE)](Int32(warp_group_idx))

    # TMA store: one elected thread issues this WG's column-block stores.
    if local_warp_idx == 0:
        fence_async_view_proxy()
        comptime if batched:
            # One batched copy over this WG's half [wg_j_offset, wg_j_offset+tma_bpo).
            ragged_tma_store.async_copy_batched[wg_j_offset](
                o_smem_arg,
                ragged_idx=out_row_idx,
                dynamic_dim=UInt32(num_output_rows),
                middle_idx=out_head_idx,
                elect=e,
            )
        else:
            # tma_bpo == 0: swizzled-output fallback -> one per-block TMA each.
            comptime for iter in range(iters_per_wg):
                comptime j = wg_j_offset + iter
                ragged_tma_store.async_copy_from_col[j](
                    o_smem_arg,
                    ragged_idx=out_row_idx,
                    dynamic_dim=UInt32(num_output_rows),
                    middle_idx=out_head_idx,
                    elect=e,
                )
        cp_async_bulk_commit_group()

    # Wait for all TMA stores to complete.
    cp_async_bulk_wait_group[0]()


@always_inline
def fa4_splitk_stage_partial[
    band_cols: Int,
    //,
    output_type: DType,
    config: FA4Config,
    own_off: Int,
    own_iters: Int,
    single_source: Bool = False,
    zero_fill: Bool = False,
    do_peers: Bool = True,
    do_own: Bool = True,
    output_swizzle_mode: TensorMapSwizzle = config.swizzle_mode,
](
    local_row: UInt32,
    final_scale_local: Float32,
    final_scale_peer: Float32,
    stage_smem: SharedMemPointer[Scalar[DType.float32]],
    own_o_tmem: TMemTile[DType.float32, config.BM, config.padded_ov_depth],
    peer_o_tmem: TMemTile[DType.float32, config.BM, config.padded_ov_depth],
    mut o_final: InlineArray[Scalar[DType.float32], band_cols],
):
    """Split-K pass 1: LSE-combine the two WG O fragments into this partition's
    NORMALIZED `O_cta` (f32) over the FULL `[0, iters)` depth range, routing each
    block by destination. The OWN band `[own_off, own_off+own_iters)` is written
    into the `o_final` register array UNSCALED (the combine scales it by `w[b]`
    after the publish barrier); every other (PEER-visible) block is stored to
    `stage_smem` in the SWIZZLE_NONE block-major, bank-conflict-free layout
    `fa4_scale_write_output` uses, at f32 width.

    The own-band smem columns are read by no peer (each peer reads only ITS band
    from us, and the bands partition `[0, iters)` disjointly), so keeping them in
    registers elides a pure self round-trip (the old code wrote them to smem and
    the combine read them straight back). The per-block normalized `O_cta` value
    is byte-identical to what the smem path stored; only the destination (regs vs
    smem) and WHEN `w[b]` is applied differ.

    `do_peers`/`do_own` split this into two passes so the caller can publish the
    peer bands (`arrive_cluster`) BEFORE the own band is computed: the peers pass
    (`do_own=False`) writes only the smem-routed blocks; the own pass
    (`do_peers=False`) writes only `o_final`. Computing the own band LAST (after
    the publish) overlaps its TMEM load with the cluster sync latency and keeps
    `o_final` out of the peer-staging live range. Both default True (single pass).

    Each thread owns row `local_row` (= tid%128, via `tcgen05_ld[datapaths=32]`).
    For each 4-element f32 block `fblk` a smem-routed write is one
    `st.shared.v4.b32` (16 B) at `stage_smem[fblk*BM*F + local_row*F + k]`,
    `F = 16/size_of[f32] = 4`; the offset is a pure function of (fblk, local_row,
    k) identical across partitions, so a peer's same-offset DSMEM read lands in
    the matching slot. Each output block (`o_sw_K` bf16 cols) maps to `o_sw_K//F`
    f32 blocks. `final_scale_*` are already this thread's per-row scales (NO
    `warp.shuffle_idx`).

    `single_source` (the T==1 split-K path): only `o0` exists (WG1 returned
    early; `peer_o_tmem`/`o1` is never produced), so skip the peer load and just
    scale `o0` by `final_scale_local` (= `inv_row_sum`). `final_scale_peer`/
    `peer_o_tmem` are unused.

    `zero_fill` (the empty-partition path): this partition has no tiles, so no O
    was produced -- write finite 0.0 at every block (own band -> `o_final`,
    peers -> smem) so the writer's weight-0 DSMEM read of this peer is a finite
    zero (avoids `0*inf = NaN`). No TMEM is read; `final_scale_*`/`own_o_tmem`/
    `peer_o_tmem` are all unused.

    `own_iters == 0` (owns no band, e.g. non-pow2 depth at high P): the own band
    is empty so the complement is the FULL range -- every block is staged to
    smem (peers still need this partition's O_cta for THEIR bands). `o_final` is
    zero-length and untouched.

    `output_swizzle_mode` MUST be the OUTPUT store's swizzle (SWIZZLE_NONE for
    MHA split-K), NOT the QKV `config.swizzle_mode`; the caller passes
    `type_of(ragged_tma_store).swizzle_mode`. `o_sw_K`/`iters` then match the
    caller's `iters_total` (the compile assert below catches a mismatch).
    """
    comptime assert config.num_q == 1
    comptime o_sw_K = output_swizzle_mode.bytes() // size_of[output_type]()
    comptime iters = align_up(config.ov_depth, o_sw_K) // o_sw_K
    comptime assert own_iters == 0 or own_off + own_iters <= iters
    comptime assert band_cols == own_iters * o_sw_K
    comptime BM = config.BM
    # 16 B / 4 B = 4 f32 per `st.shared.v4.b32` (the bank-conflict-free 16 B
    # store width; an 8-f32 store would be two instructions). Each output block
    # (o_sw_K bf16 cols) is `subblocks` f32 blocks.
    comptime F = 16 // size_of[DType.float32]()
    comptime assert o_sw_K % F == 0
    comptime subblocks = o_sw_K // F
    comptime accum_dtype = DType.float32

    var row_F = local_row * UInt32(F)

    # Stage the FULL O_cta over ALL `iters` blocks; route OWN band -> o_final
    # (regs, unscaled), every other block -> stage_smem. The f32x2 combine feeds
    # the destination register directly -- no `o_vals` round-trip.
    comptime for j in range(iters):
        comptime in_own = own_iters > 0 and j >= own_off and j < (
            own_off + own_iters
        )
        # `do_peers`/`do_own` split staging into two passes so the orchestrator
        # can publish (`arrive_cluster`) the peer bands BEFORE the own band is
        # computed: peers pass (`do_own=False`) writes only smem-routed blocks,
        # own pass (`do_peers=False`) writes only `o_final`. Gating the whole
        # block (incl. the `tcgen05_ld`) means each pass touches only its subset;
        # total loads/stores are unchanged. Defaults run both (single pass).
        comptime process = (in_own and do_own) or ((not in_own) and do_peers)
        comptime if process:
            comptime col = j * o_sw_K
            var vblk = InlineArray[SIMD[accum_dtype, F], subblocks](
                uninitialized=True
            )
            comptime if zero_fill:
                # No TMEM read: finite 0.0 (peers' weight-0 DSMEM read must be
                # finite, else 0*inf=NaN; the own band's 0 is scaled by w[b]==0).
                comptime for sb in range(subblocks):
                    vblk[sb] = SIMD[accum_dtype, F](0)
            else:
                var own = tcgen05_ld[
                    datapaths=32,
                    bits=32,
                    repeat=o_sw_K,
                    dtype=accum_dtype,
                    pack=False,
                    width=o_sw_K,
                ](own_o_tmem.tmem_addr + UInt32(col))
                comptime if single_source:
                    # T==1: only o0; normalize by inv_row_sum (final_scale_local).
                    comptime for sb in range(subblocks):
                        var v = SIMD[accum_dtype, F]()
                        comptime for c in range(F // 2):
                            comptime e = sb * F + 2 * c
                            var pair = (
                                SIMD[accum_dtype, 2](own[e], own[e + 1])
                                * final_scale_local
                            )
                            v[2 * c] = pair[0]
                            v[2 * c + 1] = pair[1]
                        vblk[sb] = v
                else:
                    var peer = tcgen05_ld[
                        datapaths=32,
                        bits=32,
                        repeat=o_sw_K,
                        dtype=accum_dtype,
                        pack=False,
                        width=o_sw_K,
                    ](peer_o_tmem.tmem_addr + UInt32(col))
                    # own*final_scale_local + peer*final_scale_peer (f32x2 fma),
                    # identical to combine_pack_o_row modulo f32 not bf16.
                    comptime for sb in range(subblocks):
                        var v = SIMD[accum_dtype, F]()
                        comptime for c in range(F // 2):
                            comptime e = sb * F + 2 * c
                            var own_c = SIMD[accum_dtype, 2](own[e], own[e + 1])
                            var peer_c = SIMD[accum_dtype, 2](
                                peer[e], peer[e + 1]
                            )
                            var comb = peer_c.fma(
                                SIMD[accum_dtype, 2](final_scale_peer),
                                own_c * final_scale_local,
                            )
                            v[2 * c] = comb[0]
                            v[2 * c + 1] = comb[1]
                        vblk[sb] = v

            comptime if in_own:
                # Own band -> o_final registers (UNSCALED; combine applies w[b]).
                comptime local_iter = j - own_off
                comptime for sb in range(subblocks):
                    comptime ocol = local_iter * o_sw_K + sb * F
                    comptime for k in range(F):
                        o_final[ocol + k] = vblk[sb][k]
            else:
                # Peer-visible block -> stage_smem (16 B v4 f32, block-major).
                comptime for sb in range(subblocks):
                    comptime fblk = j * subblocks + sb
                    (stage_smem + UInt32(fblk * BM * F) + row_F).store(vblk[sb])


@always_inline
def fa4_splitk_combine_write[
    output_type: DType,
    band_cols: Int,
    //,
    config: FA4Config,
    P: Int,
    wg_j_offset: Int,
    iters_per_wg: Int,
    use_fma: Bool,
    output_swizzle_mode: TensorMapSwizzle = config.swizzle_mode,
    tma_bpo: Int = 0,
](
    local_row: UInt32,
    local_warp_idx: UInt32,
    warp_group_idx: UInt32,
    own_rank: UInt32,
    own_max: Float32,
    own_sum: Float32,
    scale_log2e: Float32,
    stage_smem: SharedMemPointer[Scalar[DType.float32]],
    maxsum_smem: SharedMemPointer[Scalar[DType.float32]],
    o_smem_arg: SharedMemPointer[Scalar[output_type]],
    publish_mbar: MBarType,
    ragged_tma_store: RaggedTMA3DTile[
        output_type,
        output_swizzle_mode,
        BM=config.BM // config.num_q,
        BN=config.ov_depth,
        middle_dim=_,
        group=config.group if config.fuse_gqa else 1,
        tma_blocks_per_op=tma_bpo,
    ],
    num_output_rows: Int32,
    out_head_idx: UInt32,
    out_row_idx: UInt32,
    mut o_final: InlineArray[Scalar[DType.float32], band_cols],
):
    """Split-K pass 2 (reduce-scatter): EVERY partition's WG0 owns the
    depth-column band `[wg_j_offset, wg_j_offset+iters_per_wg)` (output blocks).
    It reads every partition's `O_cta` FOR ITS BAND + every partition's
    `(max,sum)`, forms the cluster LSE `Gmax`/`Gsum` and per-row weights
    `w_p = sum_p*exp2((max_p-Gmax)*c)/Gsum` (per-thread scalars -- this thread
    owns its row, NO `warp.shuffle_idx`), weight-sums
    `O_final = Σ_p O_cta_p*w_p` (the `sum_p` cancels the `O_cta` normalization),
    casts to bf16, and TMA-stores its band columns to gmem.

    `o_final` arrives PRE-SEEDED by `fa4_splitk_stage_partial` with THIS
    partition's own-rank `b` NORMALIZED `O_cta` band (UNSCALED). The own band is
    read by no peer (each peer reads only ITS band from us), so it is handed off
    in registers rather than round-tripped through `stage_smem`. This pass scales
    that seed in place by `w[b]`, then accumulates the other np-1 ranks via DSMEM
    reads of their `stage_smem` bands.

    Layout is the SWIZZLE_NONE row-per-thread pattern (mirrors
    `fa4_scale_write_output`): the f32 stage is 16 B (`F=4`) block-major, read in
    bulk v4 DSMEM loads; the bf16 write is 16 B (`o_sw_K=8`) block-major.

    NO round-2 cluster barrier: the bf16 pack is retargeted into THIS partition's
    OWN-band f32 stage slice (which no peer reads -- each peer reads only ITS band
    from us), so the write clobbers nothing peers are reading. The peer-read PEER
    bands of `stage_smem` are kept alive until every partition finishes reading by
    the kernel's terminal `cluster_sync()` (kernel.mojo) -- the same trailing-sync
    discipline the DSMEM spike validated. Only round-1 (publish_mbar phase 0, in
    the caller) is needed, to make peers' staged O_cta + (max,sum) visible before
    the DSMEM reads. (An earlier reduce-scatter revision aliased the bf16 write
    over peer-read f32 bytes and so needed a phase-1 barrier; the own-slice
    retarget removes that hazard. Writer-combines-all needed only a WG-local
    barrier because peers never wrote.)
    """
    comptime assert config.num_q == 1
    # The SIMD-2 weight-normalize loop below strides `range(0, P, 2)` and reads
    # `w[p+1]`, so P must be EVEN; the split-K whitelist
    # (FA4Config.supported()) admits only even P (2,4,6,8,10,16). Fail loudly if
    # a future odd P ever slips through instead of reading `w[P]` OOB.
    comptime assert (
        P % 2 == 0
    ), "split-K combine requires an even partition count P"
    comptime o_sw_K = output_swizzle_mode.bytes() // size_of[output_type]()
    comptime iters = align_up(config.ov_depth, o_sw_K) // o_sw_K
    comptime assert iters_per_wg >= 1
    comptime assert wg_j_offset + iters_per_wg <= iters
    comptime BM = config.BM
    # 16 B / 4 B = 4 f32 per v4 load (the staging block width); each output
    # block (o_sw_K bf16 cols) is `subblocks` f32 blocks.
    comptime F = 16 // size_of[DType.float32]()
    comptime assert o_sw_K % F == 0
    comptime subblocks = o_sw_K // F
    comptime assert band_cols == iters_per_wg * o_sw_K
    comptime accum_dtype = DType.float32
    comptime o_swizzle = make_swizzle[output_type, output_swizzle_mode]()

    var e = elect()
    if local_warp_idx == 0:
        if e != 0:
            ragged_tma_store.prefetch_descriptor()

    # Per-thread (max,sum) slot. combine runs WG0-only, so tid == local_row;
    # kept general to match how the caller wrote maxsum (`maxsum[tid*2]`).
    tid = warp_group_idx * UInt32(WARPGROUP_SIZE) + local_row

    # --- Loop A: gather every partition's (max,sum) for THIS thread's row,
    # indexed by GLOBAL cluster rank r in [0, np). r == own_rank uses this
    # partition's own (max,sum) from regs; every other rank is a DSMEM read of
    # that rank's slot. (Reduce-scatter: EVERY partition runs this, so "own" is
    # rank `b`, not rank 0 -- reading absolute ranks 1..np-1 would skip rank 0
    # and double-count `b` for b != 0.) `P` (comptime ceiling) sizes the
    # buffers; loops run to RUNTIME `np`, so slots [np, P) are never read.
    var b: Int = Int(own_rank)
    var pmax = InlineArray[Float32, P](uninitialized=True)
    var psum = InlineArray[Float32, P](uninitialized=True)
    var gmax: Float32 = own_max
    # Own partition's (max,sum) goes to comptime slot 0 (not slot `b`); the
    # other ranks fill comptime slots `p+1`. This keeps every pmax/psum index
    # static so the arrays stay in registers (dynamic indexing would spill to
    # local memory). The slot<->rank mapping is just a permutation; the
    # reductions below are order-independent, and the actual rank `r` is still
    # tracked for the DSMEM read.
    pmax[0] = own_max
    psum[0] = own_sum
    comptime for p in range(P - 1):
        var r = p if p < b else p + 1
        ms = load_cluster_smem[DType.float32, 2](
            maxsum_smem + tid * 2, UInt32(r)
        )
        pmax[p + 1] = ms[0]
        psum[p + 1] = ms[1]
        gmax = max(gmax, pmax[p + 1])

    var w = InlineArray[Float32, P](uninitialized=True)
    var gsum: Float32 = 0
    comptime for p in range(P):
        var d: Float32 = pmax[p] - gmax
        comptime if use_fma:
            d = d * scale_log2e
        w[p] = psum[p] * exp2(d)
        gsum = gsum + w[p]
    inv_gsum = recip(gsum)
    comptime for p in range(0, P, 2):
        wp = SIMD[DType.float32, 2](w[p], w[p + 1]) * inv_gsum
        w[p] = wp[0]
        w[p + 1] = wp[1]

    # --- Pass 2a: O_final[band] = Σ_r O_cta_r[row, band] * w_r over GLOBAL
    # cluster ranks r in [0, np). `w_r` is a per-thread scalar (this thread's
    # row). The seed (rank `b`, this partition's own band) arrives in `o_final`
    # from staging UNSCALED -- scale it in place by `w[0]` (rank `b`'s weight,
    # stored at slot 0; NO stage_smem
    # read-back; those columns are read by no peer). Then accumulate the other
    # np-1 ranks, skipping `b`, via 16 B v4 DSMEM reads of their `stage_smem`.
    # The partition loop is the outermost runtime loop so each rank's unrolled v4
    # reads issue back-to-back (DSMEM-latency MLP). `o_final` spans the FULL band
    # and is written once per block, so the depth-128 column-clobber (#36) is
    # impossible.
    var row_F = local_row * UInt32(F)

    # Seed scale: o_final holds the own-rank `b` normalized O_cta band; * its
    # weight, which lives at comptime slot 0 (own rank was stored there above).
    var wb = w[0]
    comptime for i in range(0, band_cols, 2):
        var pair = SIMD[accum_dtype, 2](o_final[i], o_final[i + 1]) * wb
        o_final[i] = pair[0]
        o_final[i + 1] = pair[1]

    # Accumulate the other np-1 ranks (partition loop outermost), skipping the
    # local partition `b` that already seeded `o_final`.
    for r_base in range(P - 1):
        var r = r_base if r_base < b else r_base + 1
        # Rank `r` was stored at comptime slot `r_base + 1` in Loop A (same
        # `p<->r_base` enumeration), so its weight is `w[r_base + 1]`. Select
        # it via a static-index mux so `w` stays in registers (a dynamic `w[r]`
        # would spill to local memory). The `= 0` init is dead (exactly one
        # branch matches, since r_base in [0, np-1) is within [0, P-1)) but is
        # needed for definite assignment.
        var wr: Float32 = 0
        comptime for p in range(P - 1):
            wr = w[p + 1] if p == r_base else wr
        comptime for iter in range(iters_per_wg):
            comptime j = wg_j_offset + iter
            comptime for sb in range(subblocks):
                comptime fblk = j * subblocks + sb
                comptime ocol = iter * o_sw_K + sb * F
                var vec: SIMD[accum_dtype, F] = load_cluster_smem[
                    DType.float32, F
                ](stage_smem + UInt32(fblk * BM * F) + row_F, UInt32(r))
                comptime for c in range(F // 2):
                    var acc = SIMD[accum_dtype, 2](
                        o_final[ocol + 2 * c], o_final[ocol + 2 * c + 1]
                    )
                    var pair = SIMD[accum_dtype, 2](
                        vec[2 * c], vec[2 * c + 1]
                    ).fma(SIMD[accum_dtype, 2](wr), acc)
                    o_final[ocol + 2 * c] = pair[0]
                    o_final[ocol + 2 * c + 1] = pair[1]

    # --- Pass 2b: pack O_final band -> bf16 row-major -> o_smem (scale 1.0,
    # already normalized), then per-block TMA of the band columns to gmem.
    #
    # NO round-2 cluster barrier here (the old `publish_mbar.wait(1)`): the bf16
    # pack is retargeted into THIS partition's OWN-band f32 stage slice, which no
    # peer ever reads (each peer reads only ITS band from us; the bands partition
    # `[0, iters)` disjointly). So the write clobbers nothing peers are reading
    # and needs no cluster-wide fence -- the kernel's terminal `cluster_sync()`
    # (kernel.mojo) keeps the peer-read PEER bands alive until every partition
    # finishes reading. The own band's f32 data starts at f32-block
    # `wg_j_offset*subblocks`, so shift the bf16 base by
    # `wg_j_offset*(subblocks-1)*BM*o_sw_K` output elements; bf16 block
    # `j == wg_j_offset+iter` then lands at byte `(wg_j_offset*subblocks+iter)*
    # BM*16`, strictly inside the dead own slice (no overlap with peer bands above
    # or below). `async_copy_from_col[j]` reads smem at `base + j*BM*o_sw_K` but
    # its gmem coordinate is `j*swizzle_granularity` (independent of `base`), so
    # the shifted base + global `j` still stores to the correct global gmem
    # column. (SWIZZLE_NONE -> simple block stride; the shift is block-aligned, so
    # the within-block `o_swizzle(o_inner)` row layout is undisturbed.)
    comptime o_own_shift = wg_j_offset * (subblocks - 1) * BM * o_sw_K
    var o_smem_own = o_smem_arg + o_own_shift
    comptime for iter in range(iters_per_wg):
        comptime j = wg_j_offset + iter
        var packed = pack_row[output_type, w=o_sw_K, start=iter * o_sw_K](
            o_final
        )
        var o_inner = Int(local_row) * o_sw_K
        # Explicit v4.b32 (STS.128): a plain `.store(packed)` scalarizes to 4x
        # STS.32 here because `packed` is packed from the long-lived `o_final`
        # accumulator (non-contiguous in-place F2FP pack outputs), which ptxas
        # can't fuse -- the 4 B stores then hit only every 4th bank (4-way
        # conflict). Forcing the wide store keeps it one bank-conflict-free 16 B
        # transaction (see `st_shared_v4_b32`).
        st_shared_v4_b32(
            o_smem_own, j * BM * o_sw_K + o_swizzle(o_inner), packed
        )

    named_barrier[Int32(WARPGROUP_SIZE)](Int32(warp_group_idx))
    if local_warp_idx == 0:
        fence_async_view_proxy()
        comptime for iter in range(iters_per_wg):
            comptime j_global = wg_j_offset + iter
            ragged_tma_store.async_copy_from_col[j_global](
                o_smem_own,
                ragged_idx=out_row_idx,
                dynamic_dim=UInt32(num_output_rows),
                middle_idx=out_head_idx,
                elect=e,
            )
        cp_async_bulk_commit_group()
    cp_async_bulk_wait_group[0]()


@always_inline
def fa4_splitk_reduce_scatter_write[
    output_type: DType,
    //,
    config: FA4Config,
    P: Int,
    use_fma: Bool,
    single_source: Bool = False,
    zero_fill: Bool = False,
    output_swizzle_mode: TensorMapSwizzle = config.swizzle_mode,
    tma_bpo: Int = 0,
](
    local_row: UInt32,
    local_warp_idx: UInt32,
    warp_group_idx: UInt32,
    partition_idx: UInt32,
    own_max: Float32,
    own_sum: Float32,
    scale_log2e: Float32,
    final_scale_local: Float32,
    final_scale_peer: Float32,
    stage_smem: SharedMemPointer[Scalar[DType.float32]],
    maxsum_smem: SharedMemPointer[Scalar[DType.float32]],
    o_smem_arg: SharedMemPointer[Scalar[output_type]],
    publish_mbar: MBarType,
    own_o_tmem: TMemTile[DType.float32, config.BM, config.padded_ov_depth],
    peer_o_tmem: TMemTile[DType.float32, config.BM, config.padded_ov_depth],
    ragged_tma_store: RaggedTMA3DTile[
        output_type,
        output_swizzle_mode,
        BM=config.BM // config.num_q,
        BN=config.ov_depth,
        middle_dim=_,
        group=config.group if config.fuse_gqa else 1,
        tma_blocks_per_op=tma_bpo,
    ],
    num_output_rows: Int32,
    out_head_idx: UInt32,
    out_row_idx: UInt32,
):
    """Split-K pass 2 dispatch (reduce-scatter): pick THIS partition's
    depth-column band from the STATIC partition count `P` and the runtime
    `partition_idx`, then STAGE + publish + combine over it. `P` is
    `config.splitk_partitions` (each split-K kernel is compiled once per static
    `P`), so the band's `wg_j_offset`/`iters_per_wg` are already comptime; only
    `partition_idx` is lifted via a comptime-for over the `P` partitions -- one
    runtime branch taken, the band params comptime per branch.

    Per band (WG0), staging split so the publish lands between peers and own:
      1. `fa4_splitk_stage_partial[do_own=False]` stages only the PEER-visible
         blocks to `stage_smem` (peers read those for THEIR bands).
      2. Phase-0 publish: WG0 `arrive_cluster`s every peer (peers read only our
         peer bands + maxsum, never our own band, so publish now).
      3. `fa4_splitk_stage_partial[do_peers=False]` stages the OWN band into the
         register array `o_final` (UNSCALED) LAST -- its TMEM load overlaps the
         cluster sync, and `o_final` stays out of the peer-staging live range.
      4. ALL entering WGs `wait(0)` (peers' staged `O_cta`/`(max,sum)` visible),
         then `fa4_splitk_combine_write` (WG0) scales the `o_final` seed by
         `w[b]`, DSMEM-accumulates the other ranks' bands, and TMA-stores its
         band -- packing the bf16 into its OWN-band dead f32 slice (no peer reads
         it), so no round-2 cluster barrier is needed (the terminal
         `cluster_sync()` keeps the peer-read bands alive through peers' reads).

    Called by BOTH warp groups (WG1 only participates in the phase-0 wait; WG0
    does the staging/combine). EVERY partition -- empty or not -- joins round-1
    (publish_mbar phase 0), so the publish_mbar count always completes. The
    own-no-band partition (non-pow2 depth at high P) still stages its FULL `O_cta`
    to smem (its complement is the full range) so peers can read it for their
    bands; it writes no bf16 output, so it just falls through to the kernel's
    terminal `cluster_sync()` after round-1.
    """
    comptime assert config.num_q == 1
    comptime o_sw_K = output_swizzle_mode.bytes() // size_of[output_type]()
    comptime iters_total = align_up(config.ov_depth, o_sw_K) // o_sw_K

    # `P` (== `config.splitk_partitions`, the static cluster size) is EVEN, in
    # {2,4,6,8,10,16}; only `partition_idx` (runtime) needs lifting to comptime,
    # via the comptime-for below. The band split `bpp = ceildiv(iters_total, P)`
    # is P-general: a non-pow2 P (6, 10) just yields uneven and/or empty
    # trailing bands, both of which are already handled below.

    @parameter
    @always_inline
    def reduce_scatter_p[P_static: Int]():
        comptime bpp = (iters_total + P_static - 1) // P_static
        comptime for p_static in range(P_static):
            comptime ob = p_static * bpp
            comptime ipw = min(bpp, iters_total - ob) if ob < iters_total else 0
            if partition_idx == UInt32(p_static):
                comptime if ipw > 0:
                    comptime band_cols = ipw * o_sw_K
                    var o_final = InlineArray[Scalar[DType.float32], band_cols](
                        uninitialized=True
                    )
                    if warp_group_idx == UInt32(0):
                        # 1. Stage PEER bands -> smem (own band skipped).
                        fa4_splitk_stage_partial[
                            output_type,
                            config,
                            own_off=ob,
                            own_iters=ipw,
                            single_source=single_source,
                            zero_fill=zero_fill,
                            do_own=False,
                            output_swizzle_mode=output_swizzle_mode,
                        ](
                            local_row,
                            final_scale_local,
                            final_scale_peer,
                            stage_smem,
                            own_o_tmem,
                            peer_o_tmem,
                            o_final,
                        )
                        # 2. Phase-0 publish (per-row, every peer): peers
                        # read only our peer bands + maxsum, never our
                        # own band, so publish before computing own.
                        comptime for pp in range(P_static):
                            publish_mbar[].arrive_cluster(UInt32(pp))
                        # 3. Stage OWN band -> o_final LAST: its TMEM
                        # load overlaps the cluster sync, and o_final
                        # stays out of the peer-staging live range.
                        fa4_splitk_stage_partial[
                            output_type,
                            config,
                            own_off=ob,
                            own_iters=ipw,
                            single_source=single_source,
                            zero_fill=zero_fill,
                            do_peers=False,
                            output_swizzle_mode=output_swizzle_mode,
                        ](
                            local_row,
                            final_scale_local,
                            final_scale_peer,
                            stage_smem,
                            own_o_tmem,
                            peer_o_tmem,
                            o_final,
                        )
                    # 4. Both WGs wait; then combine over the band (WG0).
                    publish_mbar[].wait(UInt32(0))
                    if warp_group_idx == UInt32(0):
                        fa4_splitk_combine_write[
                            config,
                            P=P_static,
                            wg_j_offset=ob,
                            iters_per_wg=ipw,
                            use_fma=use_fma,
                            output_swizzle_mode=output_swizzle_mode,
                            tma_bpo=tma_bpo,
                        ](
                            local_row,
                            local_warp_idx,
                            UInt32(0),
                            partition_idx,
                            own_max,
                            own_sum,
                            scale_log2e,
                            stage_smem,
                            maxsum_smem,
                            o_smem_arg,
                            publish_mbar,
                            ragged_tma_store,
                            num_output_rows,
                            out_head_idx,
                            out_row_idx,
                            o_final,
                        )
                else:
                    # Owns no band (non-pow2 depth at high P): the
                    # complement is the FULL range, so still stage the
                    # whole O_cta to smem (peers read it for THEIR bands)
                    # and publish (round-1). No bf16 write here -> no
                    # round-2 needed; the staged O_cta stays alive for
                    # peers via the kernel's terminal `cluster_sync()`.
                    var o_final_empty = InlineArray[Scalar[DType.float32], 0](
                        uninitialized=True
                    )
                    if warp_group_idx == UInt32(0):
                        fa4_splitk_stage_partial[
                            output_type,
                            config,
                            own_off=0,
                            own_iters=0,
                            single_source=single_source,
                            zero_fill=zero_fill,
                            output_swizzle_mode=output_swizzle_mode,
                        ](
                            local_row,
                            final_scale_local,
                            final_scale_peer,
                            stage_smem,
                            own_o_tmem,
                            peer_o_tmem,
                            o_final_empty,
                        )
                        comptime for pp in range(P_static):
                            publish_mbar[].arrive_cluster(UInt32(pp))
                    publish_mbar[].wait(UInt32(0))

    reduce_scatter_p[P]()


@always_inline
def fa4_softmax[
    QScaleType: OptionalPointer,
    KScaleType: OptionalPointer,
    qkv_dtype: DType,
    rope_dtype_: Optional[DType],
    scale_dtype_: Optional[DType],
    output_type: DType,
    MaskType: MHAMask,
    //,
    KVLUTType: MHAOperand,
    config: FA4Config[
        qkv_dtype, rope_dtype_=rope_dtype_, scale_dtype_=scale_dtype_
    ],
    ValidLengthType: OptionalPointer,
    SinkType: OptionalPointer,
    _is_cache_length_accurate: Bool,
    MaxSeqLenType: OptionallyStaticInt,
    # Statically guarantees `num_output_rows > 0` for every warp group,
    # folding the 2Q output guard away. Only pass True when the calling
    # kernel routes every tile short enough for an empty row half
    # (`seq_len - prompt_offset <= wg_row_offset_seq`) to the 1Q body
    # instead (see `can_switch_to_1q()` use in `kernel.mojo` for MHA and the
    # thin `mla_prefill_kernel_generic` / `_per_token_scale` entrypoints for
    # MLA). A pure-2Q kernel (or one whose mask needs the runtime FULL_MASK
    # slow path, where MLA cannot switch) must leave this False.
    output_nonempty: Bool = False,
    # The generic MLA-prefill single-O path physically drops the 2nd softmax
    # warpgroup (launches 3 WGs / 384 threads). When True, WG0 is the ONLY
    # softmax WG, so the WG0<->WG1 pair rendezvous `named_barrier[2*WG](2)`
    # before the terminal TMEM dealloc must be skipped (no WG1 arrives ->
    # otherwise a 256-thread barrier with only 128 arrivals hangs). Default
    # False keeps the per-token-scale / blockscale siblings (still 2 softmax
    # WGs even for their single-O configs) and every non-single-O path
    # byte-identical.
    single_softmax_wg: Bool = False,
](
    smem: SM100AttentionSMem[config],
    tmem_addr: UInt32,
    score_row: UInt32,
    seq_info: SeqInfo,
    mask: MaskType,
    num_keys: UInt32,
    scale: Float32,
    max_seq_len: UInt32,
    ragged_tma_store: RaggedTMA3DTile[
        output_type,
        _,
        # 2Q: BM=128 (one Q-half per WG). 1Q: BM=128 (both WGs cover the
        # full BM=128 and write disjoint depth-column ranges). Use
        # `config.BM // config.num_q` so the type is consistent across
        # both modes; in 2Q this equals the historical `config.BM // 2`.
        BM=config.BM // config.num_q,
        BN=config.ov_depth,
        middle_dim=_,
        group=config.group if config.fuse_gqa else 1,
        # Inferred from the store the kernel built; forwarded to the writeback
        # helpers, which infer their own `tma_bpo` from this arg.
        tma_blocks_per_op=_,
    ],
    sink_weights: SinkType,
    q_scale: QScaleType = NullPointer[DType.float32, AddressSpace.SHARED](),
    k_scale: KScaleType = NullPointer[DType.float32, AddressSpace.SHARED](),
):
    # Local aliases matching SM100MHA2Q comptime members
    comptime qkv_type = KVLUTType.dtype
    comptime accum_dtype = DType.float32
    comptime BM = config.BM
    comptime BN = config.BN
    comptime HalfBM = BM // 2
    comptime group = config.group
    comptime fuse_gqa = config.fuse_gqa
    comptime BM_mask: Int = config.PairBM_eff()
    comptime padded_ov_depth = config.padded_ov_depth
    comptime page_size = KVLUTType.page_size
    comptime ragged = not ValidLengthType.is_null
    comptime cta_group = config.cta_group()

    var mbars = smem.misc_mbars()
    comptime MiscMBarsType = type_of(mbars)

    # MMA types for TMEM access
    comptime UMMA0Type = SM100TensorAccumulator[
        qkv_type,
        accum_dtype,
        MMA_M=config.MMA_M,
        MMA_N=BN,
        BK=align_up(config.qk_depth, config.MMA_K),
        a_tmem=False,
        swizzle_a=config.swizzle_mode,
        swizzle_b=config.swizzle_mode,
        transpose_b=True,
        num_stages=config.num_qk_stages,
        cta_group=cta_group,
    ]
    comptime UMMA1Type = SM100TensorAccumulator[
        qkv_type,
        accum_dtype,
        MMA_M=config.MMA_M,
        MMA_N=padded_ov_depth,
        BK=BN,
        a_tmem=True,
        swizzle_b=config.swizzle_mode,
        transpose_b=False,
        num_stages=config.num_pv_stages,
        cta_group=cta_group,
    ]
    comptime PositionType = MHAPosition[
        config.BM,
        config.BN,
        config.qk_depth,
        config.padded_qk_depth,
        config.num_q_heads,
        config.group,
        _is_decoding[MaxSeqLenType](),
    ]

    # `tmem_addr` passed in by register (read once post-barrier in the kernel
    # prologue); do NOT re-read `smem.tmem_addr_ptr()` here.
    var o_smem = smem.o_smem[output_type]()
    var o_prod_mbar: MBarType = (
        mbars.mbar_base + MiscMBarsType.O_producer_offset
    )
    var s_tmem: UInt32 = tmem_addr + UInt32(config.TMEM_S0)

    # var tid = UInt32(thread_idx.x)
    var tid = llvm_opaque_tid()
    var row = tid % 128
    var warp_idx: UInt32 = warp.broadcast(tid // 32)
    var warp_group_idx: UInt32 = warp.broadcast(tid // 128)
    # Per-thread BM row within the current Q tile.
    # 2Q (BM = 256): WG0 covers BM rows [0, 128) and WG1 covers
    # [128, 256), so `tid` directly indexes the BM row.
    # 1Q (BM = 128): both WGs share the same BM rows [0, 128); folding
    # WG1's `tid` (128..255) back to [0, 128) via `tid % BM` ensures
    # the per-thread (Q row, head) mapping is identical across WGs.
    # Using bare `tid` in 1Q would shift WG1's score_row by `BM_eff`,
    # which leaks OOB K positions into the softmax for tiles whose
    # `score_row + BM_eff` exceeds `num_keys` (the OOB columns are not
    # masked by SlidingWindow's UPPER|LOWER strategy and TMA-padded
    # K=0 / V=0 then dilutes the output toward 0).
    var thread_tile_row: UInt32 = tid % UInt32(config.BM)

    var cta_q_offset: UInt32 = 0
    comptime if config.pair_cta:
        cta_q_offset = UInt32(
            warp.broadcast(block_rank_in_cluster()) % 2
        ) * UInt32(config.BM_eff())

    # 2-Q path: S1 is at +BN columns
    s_tmem += UInt32(config.BN) * warp_group_idx

    p_tmem = s_tmem
    s_tile = UMMA0Type.CType(s_tmem)
    p_tile = UMMA1Type.AType(p_tmem)

    var pipeline_s = mbars.consumer_s(warp_group_idx)
    pipeline_c = mbars.producer_c(warp_group_idx)
    var order_phase: UInt32 = 1 - warp_group_idx

    var order_s_wait: Optional[MBarType] = None
    var order_s_arrive: Optional[MBarType] = None
    comptime if EnableForcedOrdering:
        order_s_wait = mbars.pipeline_order_wait(warp_group_idx)
        order_s_arrive = mbars.pipeline_order_arrive(warp_group_idx)

    # When fuse_gqa, head_idx is a kv_head_idx
    # the output will match, so `head_idx` is what we use for writing
    # sink and mask want q_head_idx
    var head_idx: UInt32 = seq_info.head_idx
    var q_head_idx: UInt32 = head_idx
    comptime if config.fuse_gqa:
        q_head_idx = UInt32(config.group) * head_idx + row % UInt32(
            config.group
        )

    var scale_log2e: Scalar[accum_dtype] = scale
    var correction_smem = smem.correction_smem() + tid

    comptime if not MaskType.apply_log2e_after_mask:
        scale_log2e *= log2e

    # Fuse scale*log2e multiplication and row_max subtraction into a
    # single FMA in store_exp. Only valid on the default scaling path
    # where apply_log2e_after_mask is off.
    # Disabled when sink weights are used because the sink logit lives
    # in a different domain (scaled by log2e only, not scale*log2e).
    # To disable for NaN debugging, set use_fma = False.
    comptime use_fma = (
        not MaskType.apply_log2e_after_mask
    ) and SinkType.is_null and QScaleType.is_null

    # Fixed P scale for FP8-QKV only. The un-normalized softmax
    # probabilities P = exp2(score - row_max) sit in the e4m3 subnormal floor;
    # lifting them before the fp8 cast that feeds the P@V GEMM reduces PV-GEMM
    # quantization error. Because the softmax uses exp2, the scale is exactly
    # an additive +bias in the exp2 argument (added raw, NOT multiplied by
    # scale_log2e). The final output is normalized by 1/row_sum, and row_sum
    # is accumulated from the SAME scaled P (plus the sink mass, also biased),
    # so the scale factor appears in both numerator and denominator and
    # cancels exactly -- no explicit descale.
    #
    # `p_fp8_bias` and the online-softmax lazy-rescale gate `rescale_threshold`
    # are the same knob (both live in the exp2/log2 domain), linked as
    # `p_fp8_bias = 8 + rescale_threshold`:
    #   fp8 : rescale_threshold = -2,  p_fp8_bias = 6   (a 64x P lift out of
    #         the e4m3 subnormal floor)
    #   bf16: rescale_threshold = -8,  p_fp8_bias = 0   (no bias)
    # The fp8 threshold of -2 (bias 6) was chosen as the perf sweet spot: a
    # prefill sweep of the threshold magnitude gained ~6% and saturated at
    # T=2, and it is accuracy-neutral vs the prior x256 default (bit-identical
    # tail-stress regression test + byte-identical Gemma-4-31B fp8-KV 16k
    # e2e). The bias is applied ONLY inside `comptime if p_fp8_bias != 0:`
    # branches below, so the bf16 codegen is byte-identical (a `+ 0.0` would
    # otherwise survive as a real fadd). Overflow-safe: the lazy-rescale
    # gate (threshold -2) lets a non-rescaled tile's max lag the true max
    # by up to 2 (log2), so max P = exp2(2 + 6) = 256 < 448 (e4m3 max).
    comptime rescale_threshold: Scalar[accum_dtype] = Scalar[accum_dtype](
        -8
    ) if size_of[qkv_type]() >= 2 else Scalar[accum_dtype](-2)
    comptime p_fp8_bias: Scalar[accum_dtype] = 8 + rescale_threshold

    @parameter
    @always_inline
    def mask_row[
        BN: Int, //, mask_strategy: MaskStrategy
    ](mut s: InlineArray[Scalar[accum_dtype], BN], kv_row: UInt32):
        apply_mask[
            mask_strategy=mask_strategy,
            skip_scale=use_fma,
        ](
            s,
            mask,
            scale_log2e,
            prompt_idx=seq_info.prompt_idx,
            q_head_idx=q_head_idx,
            kv_tile_start_row=Int32(kv_row),
            max_seq_len=max_seq_len,
            num_keys=Int32(num_keys),
            score_row=Int32(
                score_row
                + cta_q_offset
                + (
                    thread_tile_row
                    // UInt32(group) if fuse_gqa else thread_tile_row
                )
            ),
        )

    # while waiting, offset output
    #
    # Q-tile geometry:
    # - `per_qo_BM` is the row count of one output tile. 2Q emits two
    #   BM/2-row outputs (one per WG); 1Q emits one full-BM output
    #   combined across both WGs. Both modes have per_qo_BM == 128.
    # - `wg_row_offset` is the gap between WG0's and WG1's row ranges
    #   in BM-direct units (used for q_scale indexing). 2Q: BM/2. 1Q: 0
    #   (both WGs share the same Q rows).
    # - `wg_row_offset_seq` is the same gap in seq-space units
    #   (fuse_gqa-aware, used for num_output_rows / gmem_row).
    comptime per_qo_BM = BM // config.num_q
    comptime per_qo_BM_seq = per_qo_BM // group if fuse_gqa else per_qo_BM
    comptime wg_row_offset: Int = (BM // 2) if config.num_q == 2 else 0
    comptime wg_row_offset_seq: Int = (
        wg_row_offset // group if fuse_gqa else wg_row_offset
    )
    num_output_rows = min(
        Int32(seq_info.seq_len)
        - Int32(seq_info.prompt_offset)
        - Int32(cta_q_offset)
        - Int32(warp_group_idx) * Int32(wg_row_offset_seq),
        Int32(per_qo_BM_seq),
    )

    gmem_row = PositionType.get_q_gmem_row[ragged=ragged](seq_info, max_seq_len)
    var s = InlineArray[Scalar[accum_dtype], config.BN](uninitialized=True)

    # Per-token k_scale buffer offset. The load warp cycles k_scale through
    # num_k_scale_bufs staged buffers (each BN elements wide). The softmax
    # must advance this offset after each K tile to read the correct buffer.
    # 1Q: each softmax WG consumes every other K tile (WG0 even, WG1 odd),
    # so WG1 starts one buffer in and both advance by TWO buffers per
    # processed tile (see the stride-2 advance in load_mask_max_impl).
    var k_scale_off: UInt32 = 0
    comptime if config.num_q == 1 and not KScaleType.is_null:
        k_scale_off = warp_group_idx * UInt32(config.BN)
    comptime k_scale_wrap = config.num_k_scale_bufs() * config.BN
    comptime assert KScaleType.is_null == (k_scale_wrap == 0), String(
        "KScaleType.is_null = ",
        KScaleType.is_null,
        "\nconfig.num_k_scale_bufs() = ",
        config.num_k_scale_bufs(),
        "\nBN = ",
        config.BN,
    )

    comptime max_unroll = 8

    comptime f32x2 = SIMD[DType.float32, 2]

    @parameter
    @always_inline
    def apply_k_scale[
        N: Int, //, offset: Int
    ](mut s0: InlineArray[Float32, N], k_scale_off: UInt32):
        comptime if not QScaleType.is_null:
            comptime for n in range(0, N, 2):
                var k_sc: f32x2 = (
                    k_scale.value()
                    .load[width=2](k_scale_off + UInt32(n + offset))
                    .cast[accum_dtype]()
                )
                sn = mul_ftz(k_sc, f32x2(s0[n], s0[n + 1]))
                s0[n] = sn[0]
                s0[n + 1] = sn[1]

    @parameter
    @always_inline
    def load_mask_max_impl[
        *, mask_strategy: MaskStrategy
    ](kv_row: UInt32) -> StaticTuple[Float32, max_unroll]:
        comptime if EnableForcedOrdering:
            order_s_wait.unsafe_value()[].wait(order_phase)
        # break up into sets of 32
        # minimize wait time by using smallest first
        comptime BM = config.BM // 2
        comptime batch_size = 32
        comptime has_remainder = (config.BN % batch_size) != 0
        comptime first_cols = (
            config.BN % batch_size
        ) if has_remainder else batch_size
        s0 = TMemTile[accum_dtype, BM, first_cols](s_tmem).load_async()
        apply_k_scale[0](s0, k_scale_off)
        s1 = TMemTile[accum_dtype, BM, batch_size](
            s_tmem + UInt32(first_cols)
        ).load_async()
        mask_row[mask_strategy=mask_strategy](s0, kv_row)
        vrow_max = maximum[width=max_unroll](s0)

        comptime for _i in range(first_cols):
            s[_i] = s0[_i]
        comptime cols = config.BN - first_cols + batch_size

        comptime for i in range(cols // (2 * batch_size)):
            comptime offset0 = first_cols + batch_size * (2 * i)
            comptime offset1 = first_cols + batch_size * (2 * i + 1)
            comptime offset2 = first_cols + batch_size * (2 * i + 2)

            comptime if offset1 >= config.BN:
                apply_k_scale[offset0](s1, k_scale_off)
                mask_row[mask_strategy=mask_strategy](
                    s1, kv_row + UInt32(offset0)
                )
                vrow_max = maximum(s1, vrow_max)

                comptime for _i in range(batch_size):
                    s[offset0 + _i] = s1[_i]
            else:
                s2 = TMemTile[accum_dtype, BM, batch_size](
                    s_tmem + UInt32(offset1)
                ).load_async()
                apply_k_scale[offset0](s1, k_scale_off)
                mask_row[mask_strategy=mask_strategy](
                    s1, kv_row + UInt32(offset0)
                )
                vrow_max = maximum(s1, vrow_max)

                comptime for _i in range(batch_size):
                    s[offset0 + _i] = s1[_i]

                comptime if offset2 < config.BN:
                    s1 = TMemTile[accum_dtype, BM, batch_size](
                        s_tmem + UInt32(offset2)
                    ).load_async()
                apply_k_scale[offset1](s2, k_scale_off)
                mask_row[mask_strategy=mask_strategy](
                    s2, kv_row + UInt32(offset1)
                )
                vrow_max = maximum(s2, vrow_max)

                comptime for _i in range(batch_size):
                    s[offset1 + _i] = s2[_i]

        comptime if not KScaleType.is_null:
            comptime if config.num_q == 1:
                # Stride 2 buffers per processed tile (each WG sees every
                # other K tile). `num_k_scale_bufs` may be odd, so use a
                # modular wrap rather than the equality trick below.
                k_scale_off += UInt32(2 * config.BN)
                if k_scale_off >= UInt32(k_scale_wrap):
                    k_scale_off -= UInt32(k_scale_wrap)
            else:
                k_scale_off = (k_scale_off + UInt32(config.BN)) if (
                    k_scale_off != UInt32(k_scale_wrap - config.BN)
                ) else 0
        return vrow_max

    @parameter
    @always_inline
    def init_load_mask_max[
        mask_strategy: MaskStrategy
    ](kv_row: UInt32) -> Float32:
        return maximum(load_mask_max_impl[mask_strategy=mask_strategy](kv_row))

    @parameter
    @always_inline
    def load_mask_max[
        mask_strategy: MaskStrategy
    ](kv_row: UInt32, old_max: Float32) -> Float32:
        pipeline_s.wait()
        tcgen05_fence_after()
        return maximum(
            load_mask_max_impl[mask_strategy=mask_strategy](kv_row), old_max
        )

    @parameter
    @always_inline
    def store_exp(row_max: Float32) -> f32x2:
        comptime exp_simd = 2
        comptime vs_len = config.BN // exp_simd  # 128 // 2 = 64
        comptime assert (vs_len % config.num_pv_stages) == 0
        comptime use_3_then_1_split = UMMA1Type.use_3_then_1_split
        comptime batch_size = 32 if config.num_pv_stages == 1 else vs_len // (
            4 if use_3_then_1_split else config.num_pv_stages
        )
        comptime num_batch_iters, remainder = divmod(vs_len, batch_size)
        comptime assert num_batch_iters > 0
        comptime BatchTileType = TMemTile[
            qkv_type, config.BM // 2, batch_size * exp_simd
        ]
        comptime RemainderTileType = TMemTile[
            qkv_type, config.BM // 2, remainder * exp_simd
        ]
        comptime assert (config.BN % exp_simd) == 0

        @parameter
        @always_inline
        def s_load[i: Int]() -> f32x2:
            return f32x2(s[2 * i], s[2 * i + 1])

        @parameter
        @always_inline
        def s_store[i: Int](v: f32x2):
            s[2 * i] = v[0]
            s[2 * i + 1] = v[1]

        var vrow_max: f32x2
        var vscale: f32x2
        var vneg_max_scaled: f32x2

        comptime if use_fma:
            vscale = f32x2(scale_log2e)
            # expression byte-identical (no `+ 0.0` instruction emitted).
            comptime if p_fp8_bias != 0:
                vneg_max_scaled = fma_ftz(
                    f32x2(-row_max), f32x2(scale_log2e), f32x2(p_fp8_bias)
                )
            else:
                vneg_max_scaled = f32x2(-row_max * scale_log2e)
            vrow_max = f32x2(0)  # unused
        else:
            comptime if p_fp8_bias != 0:
                vrow_max = f32x2(row_max - p_fp8_bias)
            else:
                vrow_max = f32x2(row_max)
            vscale = f32x2(0)  # unused
            vneg_max_scaled = f32x2(0)  # unused

        @parameter
        @always_inline
        def score_to_logit(score: f32x2) -> f32x2:
            comptime if use_fma:
                return fma_ftz(score, vscale, vneg_max_scaled)
            else:
                return sub_ftz(score, vrow_max)

        # --- Experiment parameters ---
        # Schedule the score-to-logit conversion `ratio` iterations ahead
        # of its corresponding exp2 to hide latency.  1 = strict
        # interleave, 4 = ~4-iteration prefetch (current tuned value).
        comptime score_to_logit_ratio: Int = 4
        # Number of exp2s per pass to route through the polynomial
        # emulation path (`exp2_emulation`) rather than hardware
        # `ex2.approx`.  Default 16 on sm_100; disabled on sm_103 where
        # the emulation does not pay off.
        comptime default_emulate_count: Int = 0 if "sm_103" in _accelerator_arch() else 16
        # `default_emulate_count` is calibrated at vs_len=64; the
        # `// 64` normalizes it back to that reference so non-default
        # vs_len scales the count proportionally.  Override at compile
        # time with `-D EXP2_EMULATE_COUNT=N`.
        comptime num_emulated: Int = (
            get_defined_int["EXP2_EMULATE_COUNT", default_emulate_count]()
            * vs_len
        ) // 64  # target emulated exp2s out of vs_len
        comptime emulation_start: Int = batch_size  # emulation window start
        # comptime emulation_start: Int = vs_len // score_to_logit_ratio  # emulation window start
        comptime emulation_end: Int = 0 if num_emulated == 0 else vs_len  # emulation window end
        comptime order_arrive_offset: Int = batch_size - 1  # within last batch
        # Derived: stride to distribute ~num_emulated across [emul_start, emul_end)
        comptime emulation_window: Int = 0 if num_emulated == 0 else emulation_end - emulation_start
        comptime emulation_stride_freq = 1 if num_emulated == 0 else emulation_window // num_emulated
        # num_emulated = emulation_window_freq / emulation_stride_freq
        # +  (emulation_window - emulation_window_freq) / (emulation_stride_freq + 1)
        #
        # num_emulated * emulation_stride_freq * (emulation_stride_freq + 1)
        #   = (emulation_stride_freq + 1)*emulation_window_freq
        #    + emulation_stride_freq * (emulation_window - emulation_window_freq)
        #   = (emulation_stride_freq + 1)*emulation_window_freq
        #    + emulation_stride_freq * emulation_window
        #    - emulation_stride_freq * emulation_window_freq
        #   = emulation_window_freq + emulation_stride_freq * emulation_window
        #
        # Thus:
        comptime emulation_window_freq = num_emulated * emulation_stride_freq * (
            emulation_stride_freq + 1
        ) - emulation_stride_freq * emulation_window
        comptime emulation_window_unfreq_start = emulation_start + emulation_window_freq
        comptime assert vs_len % score_to_logit_ratio == 0
        comptime assert (
            num_emulated >= 0
            and num_emulated <= emulation_window
            and emulation_window >= 0
        )
        comptime assert (
            num_emulated
            == emulation_window_freq // emulation_stride_freq
            + (emulation_window - emulation_window_freq)
            // (emulation_stride_freq + 1)
        )

        @parameter
        @always_inline
        def exp_iter[idx: Int]():
            comptime if idx < vs_len // score_to_logit_ratio:
                comptime for i in range(score_to_logit_ratio):
                    comptime j = score_to_logit_ratio * idx + i
                    s_store[j](score_to_logit(s_load[j]()))

            var x = s_load[idx]()
            comptime if (
                (
                    idx >= emulation_start
                    and (idx < emulation_window_unfreq_start)
                    and ((idx - emulation_start) % emulation_stride_freq == 0)
                )
                or (
                    idx >= emulation_window_unfreq_start
                    and (idx < emulation_end)
                    and (
                        (idx - emulation_start) % (emulation_stride_freq + 1)
                        == 0
                    )
                )
            ):
                x = exp2_emulation(x)
            else:
                x = exp2(x)
            s_store[idx](x)

        # --- Batch 0 ---
        comptime for idx in range(batch_size):
            exp_iter[idx]()

        var acc = s_load[0]()
        comptime if EnableEarlyAdd:
            comptime for i in range(1, batch_size // 2):
                acc = add_ftz(acc, s_load[i]())

        BatchTileType(p_tmem).store_async(s)

        comptime for b in range(1, num_batch_iters):
            comptime offset = batch_size * b

            comptime if use_3_then_1_split:
                comptime if 4 * b == 3 * num_batch_iters:
                    tcgen05_store_wait()
                    tcgen05_fence_before()
                    comptime if config.pair_cta:
                        umma_arrive_leader_cta(pipeline_s.consumer_mbar[0]())
                    else:
                        pipeline_s.release_no_step[0]()
            elif config.num_pv_stages > 1:
                comptime assert config.num_pv_stages == num_batch_iters
                tcgen05_store_wait()
                tcgen05_fence_before()

                comptime assert config.num_pv_stages == num_batch_iters
                comptime if config.pair_cta:
                    umma_arrive_leader_cta(pipeline_s.consumer_mbar[b - 1]())
                else:
                    pipeline_s.release_no_step[b - 1]()

            comptime for idx in range(offset, offset + batch_size):
                exp_iter[idx]()
                comptime if (
                    EnableForcedOrdering
                    and b == max(1, num_batch_iters - 1)
                    and idx == offset + order_arrive_offset
                ):
                    _ = order_s_arrive.unsafe_value()[].arrive()
                    order_phase ^= 1

            comptime el_offset = offset * exp_simd
            comptime tmem_offset = (el_offset * size_of[qkv_type]()) // size_of[
                accum_dtype
            ]()
            BatchTileType(p_tmem + UInt32(tmem_offset)).store_async[
                src_offset=el_offset
            ](s)

        comptime if remainder > 0:
            comptime offset = batch_size * num_batch_iters

            comptime for idx in range(offset, offset + remainder):
                exp_iter[idx]()

            comptime el_offset = offset * exp_simd
            comptime tmem_offset = (el_offset * size_of[qkv_type]()) // size_of[
                accum_dtype
            ]()
            RemainderTileType(p_tmem + UInt32(tmem_offset)).store_async[
                src_offset=el_offset
            ](s)

        tcgen05_store_wait()
        tcgen05_fence_before()
        comptime if config.pair_cta:
            umma_arrive_leader_cta(
                pipeline_s.consumer_mbar[config.num_pv_stages - 1]()
            )
            pipeline_s.step()
        else:
            pipeline_s.release[config.num_pv_stages - 1]()

        pipeline_c.acquire()
        # now we can sum the remaining elements of `acc`
        comptime add_offset = batch_size // 2 if EnableEarlyAdd else 0
        var acc0: f32x2
        var acc1: f32x2
        var acc2: f32x2
        var acc3: f32x2

        comptime if EnableEarlyAdd:
            acc0 = acc
            acc1 = s_load[batch_size // 2]()
            acc2 = s_load[batch_size // 2 + 1]()
            acc3 = add_ftz(
                s_load[batch_size // 2 + 2](),
                s_load[batch_size // 2 + 3](),
            )
        else:
            acc0 = acc
            acc1 = s_load[1]()
            acc2 = s_load[2]()
            acc3 = s_load[3]()

        comptime for i in range(add_offset + 4, vs_len, 4):
            acc0 = add_ftz(acc0, s_load[i]())
            acc1 = add_ftz(acc1, s_load[i + 1]())
            acc2 = add_ftz(acc2, s_load[i + 2]())
            acc3 = add_ftz(acc3, s_load[i + 3]())
        return add_ftz(add_ftz(acc0, acc1), add_ftz(acc2, acc3))

    var kv_row: UInt32 = mask.start_column[BM_mask, BN, page_size](
        seq_info.prompt_idx, score_row
    )
    # 1Q: WG0 takes even-indexed K/V tiles (start = kv_row); WG1 takes
    # odd-indexed (+BN). Both advance by 2*BN per main-loop iter (set
    # below). 2Q: both WGs share the same kv_row stride of BN.
    # single-O (1Q wide-V): WG0 owns EVERY tile (WG1 no-op), so no
    # per-WG start offset and a stride of BN (set below).
    comptime if config.num_q == 1 and not config.single_o:
        kv_row += warp_group_idx * UInt32(config.BN)
    comptime mask_sets = MaskType.nonfull_sets[BM_mask, BN]()
    comptime mask_strategies = MaskType.mask_strategies[BM_mask, BN]()
    comptime num_sets = len(mask_strategies)
    comptime assert len(mask_sets) == num_sets

    var row_max: Float32
    var mask_iters: StaticTuple[UInt32, num_sets] = {}

    # `total_iters_combined` is the combined K-tile count across both
    # WGs in 1Q (= MMA's `mask.total_iters` view). Needed for the peer
    # `o_prod_mbar` wait phase in the 1Q LSE combine below.
    var total_iters_combined: UInt32 = 0

    comptime if mask_sets[0] != TileMaskStatus.UNKNOWN_MASK:
        mask_ends = mask.masked_set_ends[
            BM=BM_mask, BN=BN, page_size=page_size
        ](seq_info.prompt_idx, score_row, num_keys)

        # Split-K (1Q): restrict this CTA to a balanced sub-window
        # [part_cb, part_ce) of the combined tile range [0, T) (T =
        # masked_set_ends[-1]) and offset kv_row to its first tile. Same
        # window as load/mma/correction -- masked_set_ends[-1] ==
        # total_iters for check_mask==False masks (mha_mask.mojo:651-658).
        var part_cb: UInt32 = 0
        var part_ce: UInt32 = mask_ends[num_sets - 1]
        comptime if config.num_q == 1 and config.splitk_partitions > 1:
            var _np: UInt32 = UInt32(config.splitk_partitions)
            var _w = splitk_window(
                mask_ends[num_sets - 1],
                _np,
                splitk_partition_idx(_np),
            )
            part_cb = _w[0]
            part_ce = _w[1]
            kv_row += part_cb * UInt32(config.BN)

        # Per-set tile counts within [part_cb, part_ce): clamp each set end
        # into the window (prev starts at part_cb). Folds to the full
        # per-set counts when split-K is off (part_cb=0, part_ce=T).
        var _prev: UInt32 = part_cb
        comptime for i in range(num_sets):
            _e = max(part_cb, min(mask_ends[i], part_ce))
            mask_iters[i] = _e - _prev
            _prev = _e

        comptime if config.num_q == 1:
            total_iters_combined = part_ce - part_cb
            # single-O: WG0 owns EVERY tile (WG1 no-op), so keep the full
            # combined `mask_iters` (no per-WG split).
            comptime if not config.single_o:
                # Per-WG split with cumulative-parity carry. WG0 owns
                # combined indices with parity 0 (even cumulative position);
                # WG1 owns parity 1. Within set i starting at cumulative
                # combined index `cum`:
                #   parity=0: WG0 takes ceil(iters_combined_i/2), WG1 floor.
                #   parity=1: WG0 takes floor, WG1 ceil.
                # Parity is local to the partition (cumulative starts at 0),
                # consistent with the unconditional WG offset (+warp_group_idx
                # *BN) applied to kv_row above.
                var cumulative: UInt32 = 0
                comptime for i in range(num_sets):
                    iters_combined_i = mask_iters[i]
                    parity = cumulative & UInt32(1)
                    if warp_group_idx == UInt32(0):
                        mask_iters[i] = (
                            iters_combined_i + UInt32(1) - parity
                        ) // UInt32(2)
                    else:
                        mask_iters[i] = (iters_combined_i + parity) // UInt32(2)
                    cumulative += iters_combined_i
    else:
        comptime if config.num_q == 1:
            # Unmasked-only path has no precomputed mask_ends. Derive
            # the combined K-tile count from the [start_column, num_keys)
            # range, matching MMA's `mask.total_iters` view.
            total_iters_combined = mask.total_iters[BM_mask, BN, page_size](
                seq_info.prompt_idx, score_row, num_keys
            )

    comptime assert num_sets >= 1 and num_sets <= 3
    comptime assert num_sets == 1 or mask_sets[0] != TileMaskStatus.UNKNOWN_MASK
    # M2 split-K slices the contiguous masked-set range only. UNKNOWN_MASK
    # (Materialized/And/Or) iterates via a runtime FULL_MASK skip whose
    # tile count is an upper bound -- partitioning it correctly needs the
    # per-partition skip (M5). Block the combo so a future caller cannot
    # silently get redundant full-range partials (which the combine would
    # then sum P times).
    comptime assert (
        config.splitk_partitions == 1
        or mask_sets[0] != TileMaskStatus.UNKNOWN_MASK
    ), (
        "split-K (M2) supports only check_mask==False masks; UNKNOWN_MASK"
        " (Materialized/And/Or) masks are M5"
    )

    # Split-K M4 combine activation, hoisted above the T<=1 / empty guards (it
    # gates the empty-partition neutral-stage path below as well as the T==1 /
    # T>=2 combine paths in the 1Q output phase). Unconditional whenever split-K
    # is active (`splitk_partitions > 1`); folds away when split-K is off.
    comptime splitk_combine_active = config.splitk_partitions > 1

    # 1Q T<=1: WG1 owns the odd-indexed K-tiles. At total_iters_combined == 1
    # (only K_e[0]) AND == 0 (empty partition -- front-load leaves only TRAILING
    # partitions empty, never the rank-0 writer; also covers M6 idle CTAs) WG1
    # has zero work and MMA never commits s1, so pipeline_s.wait() below would
    # hang; peel_mask (num_sets==1) would also underflow mask_iters[0]. Skip
    # everything WG1 would do and drop to the final cross-WG sync that gates
    # TMEM dealloc (the dealloc, `warp_idx == 0`, runs at the kernel terminal).
    #
    # single-O (wide-V fallback): the two per-WG O partials do NOT fit in
    # the 512-col TMEM (`2*BN + 2*padded_ov > 512`), so single-O aliases
    # O1 onto O0 and cannot run the two-WG even/odd LSE-combine (the peer
    # read would overrun TMEM). Instead WG0 processes ALL K-tiles serially
    # into the single O0 accumulator and WG1 is a full no-op for every T
    # (not just T==1) — mirroring the T==1 fast path, generalized. The MMA
    # and correction warps take matching single-O single-WG paths.
    comptime if config.num_q == 1:
        if (
            config.single_o or total_iters_combined <= UInt32(1)
        ) and warp_group_idx == UInt32(1):
            named_barrier[Int32(2 * WARPGROUP_SIZE)](2)
            return

        # Empty partition (total_iters_combined == 0; WG0 only -- WG1 returned
        # just above). No scores/O were produced, so skip pipeline_s.wait +
        # peel + main loop + LSE combine. When the combine is active, stage a
        # NEUTRAL softmax identity (max = -inf, sum = 0, O = 0) and publish it
        # (so peers' weight-0 DSMEM read of this partition is a finite 0). Under
        # REDUCE-SCATTER this partition STILL owns a depth band, so it then runs
        # the same pass-2 helper as the non-empty partitions: it reads its band
        # from all peers (own weight 0; the non-empty peers carry the data) and
        # writes it into its own dead band slice. No round-2 cluster barrier (the
        # kernel's terminal `cluster_sync()` gates peer reads). No sink fold here
        # (partition-0 folds it once).
        comptime if config.splitk_partitions > 1:
            if total_iters_combined == UInt32(0):
                comptime if splitk_combine_active:
                    stage = smem.o_smem[DType.float32]()
                    maxsum = stage + (config.BM * padded_ov_depth)
                    # Neutral (max,sum): -inf never wins the cluster Gmax, and
                    # sum 0 zeroes this partition's combine weight regardless.
                    maxsum[row * 2] = min_or_neg_inf[DType.float32]()
                    maxsum[row * 2 + 1] = Float32(0)
                    # Zero-fill all depth columns (WG0 covers the full range at
                    # 1Q) so peers' weight-0 DSMEM read of this partition is a
                    # finite 0.
                    own_o_empty = TMemTile[
                        accum_dtype, config.BM, padded_ov_depth
                    ](tmem_addr + UInt32(config.TMEM_O0))
                    # Reduce-scatter: stage this (empty) partition's neutral
                    # O_cta (zero_fill), publish, then read its OWN band from all
                    # peers (own weight 0; non-empty peers carry the data) and
                    # write it into its own dead band slice. No round-2 cluster
                    # barrier (terminal `cluster_sync()` gates peer reads).
                    fa4_splitk_reduce_scatter_write[
                        config,
                        P=config.splitk_partitions,
                        use_fma=use_fma,
                        single_source=True,
                        zero_fill=True,
                    ](
                        row,
                        warp_idx & 3,
                        warp_group_idx,
                        splitk_partition_idx(UInt32(config.splitk_partitions)),
                        min_or_neg_inf[DType.float32](),
                        Float32(0),
                        scale_log2e,
                        Float32(0),
                        Float32(0),
                        stage,
                        maxsum,
                        o_smem,
                        smem.misc_mbars().publish_mbar(),
                        own_o_empty,
                        own_o_empty,
                        ragged_tma_store,
                        num_output_rows,
                        head_idx,
                        gmem_row + cta_q_offset,
                    )
                # Release WG1 (parked on this 2*WG barrier above) before the
                # kernel terminal cluster_sync + TMEM dealloc.
                named_barrier[Int32(2 * WARPGROUP_SIZE)](2)
                return

    pipeline_s.wait()
    tcgen05_fence_after()
    # Apply per-token q_scale
    comptime if not QScaleType.is_null:
        scale_log2e *= q_scale.value()[
            warp_group_idx * UInt32(wg_row_offset) + row
        ].cast[accum_dtype]()

    var row_max: Float32 = peel_mask[
        rebind[StaticTuple[MaskStrategy, num_sets]](mask_strategies),
        init_load_mask_max,
    ](mask_iters, kv_row)
    var sink_weight: Scalar[accum_dtype]

    # Split-K: the sink mass must land in the *cluster-global* sum exactly
    # once per Q row, not once per partition CTA. Each partition's `sum_cta`
    # is rescaled into `global_sum_cluster` by `exp2((max_cta - Gmax)*c)` at
    # the terminal combine, so folding the sink into partition 0's `sum_cta`
    # (and clamping its `row_max` so `max_cta` already accounts for the sink)
    # makes the sink contribute `exp2((sink - Gmax)*c)` exactly once. Gate
    # both the clamp and the fold on partition 0; folds to current behavior
    # when split-K is off. (Mirrors the WG0-only intra-CTA sink rule and
    # `mla_decode_combine.mojo`'s once-per-row sink accounting.)
    var fold_sink: Bool = True
    comptime if config.splitk_partitions > 1:
        fold_sink = splitk_partition_idx(
            UInt32(config.splitk_partitions)
        ) == UInt32(0)

    comptime if not SinkType.is_null:
        var sink_weights_ptr = rebind[
            UnsafePointer[Scalar[qkv_type], ImmutAnyOrigin]
        ](sink_weights.value())

        comptime if use_fma:
            sink_weight = sink_weights_ptr[q_head_idx].cast[accum_dtype]()
        else:
            sink_weight = (
                sink_weights_ptr[q_head_idx].cast[accum_dtype]() * log2e
            )
        if fold_sink:
            row_max = max(row_max, sink_weight)
    else:
        sink_weight = 0.0

    var row_sum: f32x2 = store_exp(row_max)

    var o_phase: UInt32 = 0  # initial wait is phase 0

    comptime if not SinkType.is_null:
        # The sink mass must land in `global_sum` exactly once per Q row.
        #
        # 2Q: each WG owns a disjoint set of Q rows, so adding the sink to
        # every WG's `row_sum` already contributes it once per row.
        #
        # 1Q: both WGs cover the SAME Q rows but stride over disjoint halves
        # of the K/V stream (`kv_row_stride = 2*BN`), then LSE-combine their
        # `row_sum`s (`global_sum = row_sum_total*scale_local +
        # peer_sum*scale_peer`, ~L1548). Adding the sink to BOTH WGs would
        # double-count it in `global_sum`, inflating the denominator and
        # shrinking every output (the gpt-oss-20b sink bug). Add it in WG0
        # only. WG0 must be the carrier because the T==1 fast path returns
        # WG1 early (~L1257) before any LSE exchange, so WG0 always survives
        # to fold the sink into the combined denominator.
        # The sink mass must enter row_sum in the SAME scale as the P values
        # stored by store_exp, so it cancels through the final 1/row_sum
        # normalize. fp8 adds the same +p_fp8_bias as store_exp; the
        # `comptime if p_fp8_bias != 0` keeps the bf16 sink expression
        # byte-identical.
        @parameter
        @always_inline
        def sink_mass() -> Float32:
            comptime if use_fma:
                comptime if p_fp8_bias != 0:
                    return exp2(
                        (sink_weight - row_max) * scale_log2e + p_fp8_bias
                    )
                else:
                    return exp2((sink_weight - row_max) * scale_log2e)
            else:
                comptime if p_fp8_bias != 0:
                    return exp2(sink_weight - row_max + p_fp8_bias)
                else:
                    return exp2(sink_weight - row_max)

        comptime if config.num_q == 1:
            # WG0-only (intra-CTA once-per-row) AND, under split-K,
            # partition-0-only (once-per-cluster). `fold_sink` carries the
            # partition-0 gate; folds to `True` when split-K is off.
            if warp_group_idx == UInt32(0) and fold_sink:
                row_sum[0] += sink_mass()
        else:
            row_sum[0] += sink_mass()

    # Lazy-rescale gate for online softmax: only re-scale the accumulator
    # (and adopt the new running max) when `old_max - new_row_max <
    # rescale_threshold` in the log2 domain. Below that, we keep the stale max
    # and skip the rescale; the new exp2(score - old_max) terms stay within
    # 2^|rescale_threshold| of the existing scale, which fp32 accumulation can
    # absorb without meaningful loss. bf16 uses -8 (256x); fp8 uses -2 (a
    # tighter gate that rescales sooner, chosen with `p_fp8_bias` above -- the
    # knob it is linked to via `p_fp8_bias = 8 + rescale_threshold`).

    # 1Q advances kv_row by 2*BN (each WG strides over its half of the
    # K/V stream); 2Q advances by BN (each WG processes every K tile).
    # single-O (1Q wide-V): WG0 processes CONSECUTIVE tiles (WG1 no-op),
    # so it strides by BN like 2Q.
    comptime kv_row_stride: Int = (
        2
        * config.BN if (
            config.num_q == 1 and not config.single_o
        ) else config.BN
    )

    comptime if mask_sets[0] != TileMaskStatus.UNKNOWN_MASK:
        comptime for i in range(num_sets):
            comptime mask_status = mask_sets[i]
            comptime mask_strategy = mask_strategies[i]
            var iters: UInt32

            iters = warp.broadcast(mask_iters[i])
            while iters != 0:
                iters -= 1
                kv_row += UInt32(kv_row_stride)
                # calculate rowmax
                old_max = row_max
                var new_row_max: Float32 = load_mask_max[mask_strategy](
                    kv_row, old_max
                )

                diff = sub_ftz(old_max, new_row_max)

                comptime if use_fma:
                    diff = mul_ftz(diff, scale_log2e)
                var correction: Float32

                comptime if rescale_threshold < 0:
                    # old_max - new_row_max < -8
                    # 8 < new_row_max - old_max
                    if _vote_nvidia_helper(diff < rescale_threshold) != 0:
                        row_max = new_row_max
                        correction = exp2(diff)
                    else:
                        correction = 1
                else:
                    row_max = new_row_max
                    correction = exp2(diff)
                correction_smem[] = correction
                pipeline_c.commit()
                # update s->p
                local_rowsum = store_exp(row_max)
                row_sum = fma_ftz(row_sum, f32x2(correction), local_rowsum)
                o_phase ^= 1
    else:
        while True:
            kv_row += UInt32(kv_row_stride)
            if kv_row >= num_keys:
                break
            cur_mask_status = mask.status(
                seq_info.prompt_idx,
                Index[dtype=DType.int32](Int(score_row), Int(kv_row)),
                Index[dtype=DType.int32](BM_mask, BN),
            )
            if cur_mask_status == TileMaskStatus.FULL_MASK:
                continue
            # calculate rowmax
            old_max = row_max
            var new_row_max: Scalar[accum_dtype]
            if cur_mask_status == TileMaskStatus.PARTIAL_MASK:
                new_row_max = load_mask_max[
                    MaskStrategy.COMPUTED | MaskStrategy.OUT_OF_BOUNDS
                ](kv_row, old_max)
            else:
                new_row_max = load_mask_max[MaskStrategy.OUT_OF_BOUNDS](
                    kv_row, old_max
                )

            diff = sub_ftz(old_max, new_row_max)

            comptime if use_fma:
                diff = mul_ftz(diff, scale_log2e)
            var correction: Float32

            comptime if rescale_threshold < 0:
                # old_max - new_row_max < -8
                # 8 < new_row_max - old_max
                if _vote_nvidia_helper(diff < rescale_threshold) != 0:
                    row_max = new_row_max
                    correction = exp2(diff)
                else:
                    correction = 1
            else:
                row_max = new_row_max
                correction = exp2(diff)
            correction_smem[] = correction
            pipeline_c.commit()
            # update s->p
            local_rowsum = store_exp(row_max)
            row_sum = fma_ftz(row_sum, f32x2(correction), local_rowsum)
            o_phase ^= 1
    # Do the final correction and write.
    comptime assert size_of[output_type]() >= size_of[qkv_type]()

    comptime if config.num_q == 2:
        # 2Q: each WG writes its row half independently.
        inv_row_sum = recip(row_sum.reduce_add())
        # `BM // config.num_q` matches the helper's signature
        # (`config.BM // config.num_q`) at the comptime-expression level;
        # numerically identical to HalfBM = BM // 2 inside this 2Q branch.
        o_tile = TMemTile[accum_dtype, BM // config.num_q, padded_ov_depth](
            tmem_addr
            + UInt32(config.TMEM_O0)
            + warp_group_idx * UInt32(padded_ov_depth)
        )

        # wait on the o_pipeline producer
        @parameter
        @always_inline
        def wait_and_write_output():
            o_prod_mbar[warp_group_idx].wait(o_phase)  # consumer wait
            tcgen05_fence_after()  # example 1
            # TODO: pass in a dedicated barrier that a q-writer can wait on in a persistent kernel?

            fa4_scale_write_output[config](
                row,
                warp_idx & 3,
                warp_group_idx,
                inv_row_sum,
                o_smem + warp_group_idx * UInt32(HalfBM * padded_ov_depth),
                o_tile,
                ragged_tma_store,
                num_output_rows,
                head_idx,
                gmem_row
                + cta_q_offset
                + warp_group_idx
                * UInt32(HalfBM // group if fuse_gqa else HalfBM),
            )

        # `output_nonempty` statically discharges the guard: the entry
        # kernel routed every tile with
        # `seq_len - prompt_offset <= wg_row_offset_seq` to the 1Q body,
        # so both WGs' halves are non-empty whenever this 2Q path runs.
        comptime if output_nonempty:
            debug_assert(
                num_output_rows > 0,
                "1Q switch must take every tile with an empty output half",
            )
            wait_and_write_output()
        else:
            if num_output_rows > 0:
                wait_and_write_output()
    else:
        # 1Q output. T==1 takes a fast path (WG0 has the full output
        # in TMEM_O0 and WG1 has no work / already returned); T>=2
        # combines per-WG partials via LSE exchange.
        #
        # No `num_output_rows > 0` guards on the 1Q write paths: 1Q has no
        # per-WG row offset (`wg_row_offset == 0`) and is single-CTA
        # (`supported()` forbids pair_cta), so `num_output_rows =
        # min(seq_len - prompt_offset, per_qo_BM_seq)`, which is >= 1
        # because every caller dispatches the softmax warps only for tiles
        # with `seq_info.is_valid()` (seq_len > prompt_offset).
        debug_assert(
            num_output_rows > 0,
            "1Q tiles always have output rows (is_valid() holds)",
        )
        # Split-K (num_q==1): each partition computes its OWN K-slice, so its
        # normalized output is a per-partition partial. `splitk_combine_active`
        # (hoisted above the T<=1 guard, since the empty-partition neutral-stage
        # path needs it too) gates the M4 writer-combines-all combine, which is
        # unconditional whenever split-K is active (`splitk_partitions > 1`):
        # every partition stages its partial + (max,sum), and the writer
        # DSMEM-reads all P partitions and writes the FULL combined O. The test
        # compares that combined O against `mha_gpu_naive` over the full key
        # range. `is_writer` (= `splitk_partition_idx == FA4_1Q_SPLITK_WRITER`,
        # default partition 0) selects the writer for the T==1 combine path; it
        # folds to a constant `True` when split-K is off (`splitk_partitions ==
        # 1`), leaving pair-CTA and single-CTA unchanged. `block_rank_in_cluster()`
        # is block-uniform, so gating on `is_writer` does not diverge the
        # `named_barrier` calls below.
        var is_writer: Bool = True
        comptime if config.splitk_partitions > 1:
            comptime SPLITK_WRITER = get_defined_int[
                "FA4_1Q_SPLITK_WRITER", 0
            ]()
            is_writer = splitk_partition_idx(
                UInt32(config.splitk_partitions)
            ) == UInt32(SPLITK_WRITER)
        if config.single_o or total_iters_combined == UInt32(1):
            # T==1 fast path AND the single-O all-T path: skip the
            # LSE-exchange entirely and reuse the 2Q row-scale + stmatrix
            # + TMA helper directly. No peer partial to combine; no per-WG
            # smem/gmem-row offsets. `BM // config.num_q` is the helper's
            # expected row count and numerically equals config.BM (= 128)
            # in 1Q. For single-O, WG0 has accumulated ALL K-tiles' P@V
            # into the single O0 and holds the full `row_sum`, so this
            # writer produces the complete output; WG1 already returned.
            row_sum_total = row_sum.reduce_add()
            inv_row_sum = recip(row_sum_total)
            o_tile = TMemTile[accum_dtype, BM // config.num_q, padded_ov_depth](
                tmem_addr + UInt32(config.TMEM_O0)
            )
            # Only o0 is produced (MMA skipped the o1 commit at T==1).
            o_prod_mbar[0].wait(o_phase)
            tcgen05_fence_after()
            comptime if splitk_combine_active:
                # T==1 split-K combine (reduce-scatter): WG1 returned early, so
                # WG0 alone stages its single normalized `O_cta` (= o0*inv_row_sum)
                # over ALL depth columns + this partition's (row_max, row_sum),
                # publishes via the cluster mbar, then EVERY partition's WG0 reads
                # its OWN depth band from all peers and writes it. (The non-empty
                # partition here carries the data; trailing empty partitions take
                # the empty-partition path above and combine their bands from it.)
                stage = smem.o_smem[DType.float32]()
                maxsum = stage + (config.BM * padded_ov_depth)
                # o0 tile typed with `config.BM` exactly — `o_tile`'s
                # `BM // config.num_q` does not fold to `config.BM` at parse
                # time, so it can't convert to stage_partial's param type.
                own_o_t1 = TMemTile[accum_dtype, config.BM, padded_ov_depth](
                    tmem_addr + UInt32(config.TMEM_O0)
                )
                # Publish this partition's (row_max, row_sum). WG0-only at
                # T==1, so `row` (in-WG tid) is the full softmax tid; partition
                # 0 already folded the sink into row_max/row_sum above.
                maxsum[row * 2] = row_max
                maxsum[row * 2 + 1] = row_sum_total
                # Single-source O (o0*inv_row_sum): staged (own band -> regs,
                # peers -> smem), published, then combined over this partition's
                # own band.
                fa4_splitk_reduce_scatter_write[
                    config,
                    P=config.splitk_partitions,
                    use_fma=use_fma,
                    single_source=True,
                ](
                    row,
                    warp_idx & 3,
                    warp_group_idx,
                    splitk_partition_idx(UInt32(config.splitk_partitions)),
                    row_max,
                    row_sum_total,
                    scale_log2e,
                    inv_row_sum,
                    Float32(0),
                    stage,
                    maxsum,
                    o_smem,
                    smem.misc_mbars().publish_mbar(),
                    own_o_t1,
                    own_o_t1,
                    ragged_tma_store,
                    num_output_rows,
                    head_idx,
                    gmem_row + cta_q_offset,
                )
            else:
                if is_writer:
                    fa4_scale_write_output[config](
                        row,
                        warp_idx & 3,
                        UInt32(0),
                        inv_row_sum,
                        o_smem,
                        o_tile,
                        ragged_tma_store,
                        num_output_rows,
                        head_idx,
                        gmem_row + cta_q_offset,
                    )
            # WG1 already participated in `named_barrier[2*WG](2)` and
            # returned; WG0 must hit it here so the pair-WG sync resolves
            # before TMEM dealloc. Mirrors the unconditional sync below.
            # When the generic single-O path drops WG1 entirely
            # (`single_softmax_wg`), there is no peer WG to rendezvous with, so
            # this 256-thread barrier would hang with only WG0's 128 arrivals.
            # The correction / MMA -> dealloc ordering is carried by the
            # o-producer mbars (unchanged), not this softmax-only barrier.
            comptime if not single_softmax_wg:
                named_barrier[Int32(2 * WARPGROUP_SIZE)](2)
            # Pair-CTA and 1Q split-K defer dealloc to the kernel terminal
            # (after the cluster_sync); only plain single-CTA deallocs inline.
            comptime if not config.pair_cta and config.splitk_partitions == 1:
                if warp_idx == 0:
                    tcgen05_release_allocation_lock[Int32(cta_group)]()
                    tcgen05_dealloc[Int32(cta_group)](
                        tmem_addr, UInt32(config.sm100_tmem_cols)
                    )
            return

        # 1Q: LSE-combine both WGs' TMEM_O fragments into the shared
        # o_smem in depth-column slices, then both WGs TMA-store
        # disjoint column ranges to gmem. Both WGs cover the same Q
        # rows; no per-WG row offset on the write side.

        # 1. WG-local LSE reduce.
        row_sum_total = row_sum.reduce_add()

        # 2. Wait on OWN pipeline_o producer. After this, MMA1 has finished
        # its last V·P, so the last P in own's s_tmem has been consumed and
        # the slot is safe to repurpose for the cross-WG LSE exchange below.
        # MMA1 always runs and the TMEM reuse requires this wait.
        o_prod_mbar[warp_group_idx].wait(o_phase)
        tcgen05_fence_after()

        # 3. LSE exchange through the (now-dead) s_tmem slot. Each WG writes
        # (row_max, row_sum_total) into the first two TMEM columns of its
        # s_tmem slot; the peer reads those two columns from the other WG's
        # slot. Replaces an earlier smem-aliased exchange buffer that had
        # to overlay the K region and could collide with the load warp's K
        # TMA writes.
        # TMEM layout (per Q row r):
        #     col TMEM_S0+0     = WG0 row_max
        #     col TMEM_S0+1     = WG0 row_sum_total
        #     col TMEM_S0+BN+0  = WG1 row_max
        #     col TMEM_S0+BN+1  = WG1 row_sum_total
        var own_lse: InlineArray[Scalar[accum_dtype], 2] = [
            row_max,
            row_sum_total,
        ]
        TMemTile[accum_dtype, BM, 2](s_tmem).store_async(own_lse)
        tcgen05_store_wait()
        tcgen05_fence_before()
        named_barrier[Int32(2 * WARPGROUP_SIZE)](5)
        tcgen05_fence_after()

        # 4. Read peer's slice from the peer WG's s_tmem.
        peer_wg = UInt32(1) - warp_group_idx
        var peer_s_tmem: UInt32 = (tmem_addr + UInt32(config.TMEM_S0)) + UInt32(
            config.BN
        ) * peer_wg
        var peer_lse = TMemTile[accum_dtype, BM, 2](peer_s_tmem).load_async()
        peer_max = peer_lse[0]
        peer_sum = peer_lse[1]

        global_max = max(row_max, peer_max)
        # Match the per-WG online softmax convention: when `use_fma`,
        # `row_max` is tracked in raw (unscaled) score units and the
        # inner-loop diff is multiplied by `scale_log2e` before `exp2`
        # (see `diff = mul_ftz(diff, scale_log2e)` above). The LSE
        # combine must apply the same conversion, otherwise the
        # cross-WG weights are `exp2(raw_diff)` instead of
        # `exp2(raw_diff * scale_log2e)` and the 1Q output drifts ~1
        # ULP whenever the two WGs' raw maxes differ. Without this
        # scaling the bug is masked when K is constant (raw maxes
        # equal across WGs) or V is constant (per-WG O ∝ row_sum so
        # the wrong weights cancel through global_sum normalization).
        # Vectorize the (local, peer) pair: one f32x2 sub / mul / exp2 in
        # place of two scalars each (matching `store_exp`'s f32x2 `exp2`),
        # and fuse the denominator's two products into a single FMA. Note the
        # `*= scale_log2e` here is now `mul_ftz` (ftz), matching the inner
        # loop's `diff = mul_ftz(diff, scale_log2e)`.
        var diffs: f32x2 = f32x2(row_max, peer_max) - f32x2(global_max)
        comptime if use_fma:
            diffs = mul_ftz(diffs, f32x2(scale_log2e))
        var scales: f32x2 = exp2(diffs)  # (scale_local, scale_peer)
        # global_sum = row_sum_total*scale_local + peer_sum*scale_peer, fused.
        global_sum = peer_sum.fma(scales[1], row_sum_total * scales[0])
        var final_scales: f32x2 = scales * recip(global_sum)
        final_scale_local = final_scales[0]
        final_scale_peer = final_scales[1]

        # 5. Wait on PEER pipeline_o producer so peer's TMEM_O is safe to
        # read. Per-pipeline iter counts differ by
        # `total_iters_combined & 1` for odd combined-T, so peer's phase
        # XORs in that bit. (Own's producer was already waited on above
        # before the LSE exchange.)
        peer_phase = o_phase ^ (total_iters_combined & UInt32(1))
        o_prod_mbar[peer_wg].wait(peer_phase)
        tcgen05_fence_after()

        # 5. Build own + peer TMEM tiles at full-BM extent.
        own_o_tile = TMemTile[accum_dtype, BM, padded_ov_depth](
            tmem_addr
            + UInt32(config.TMEM_O0)
            + warp_group_idx * UInt32(padded_ov_depth)
        )
        peer_o_tile = TMemTile[accum_dtype, BM, padded_ov_depth](
            tmem_addr
            + UInt32(config.TMEM_O0)
            + peer_wg * UInt32(padded_ov_depth)
        )

        # 6. Per-WG comptime j-range specialization for the helper.
        # Ceil/floor split: WG0 takes ceil(iters/2) blocks starting
        # at j=0, WG1 takes floor(iters/2) starting at j=ceil(iters/2).
        # Under SWIZZLE_NONE the block is small (o_sw_K = 8 bf16), so
        # iters = ov_depth/8 >= 8 for every supported head size and both
        # WGs always participate (iters_per_wg1 >= 4 > 0). When batched,
        # the store descriptor's box is the half-depth ceil(iters/2): WG0
        # fills it exactly and WG1, for odd iters (e.g. depth=72 -> 9),
        # overhangs the last block, which the TMA masks off. The
        # `iters_per_wg1 > 0` guard below is now always true but kept for
        # safety.
        #
        # The block size must come from the OUTPUT store's swizzle, not
        # `config.swizzle_mode`: fa4_lse_combine_write infers its
        # `output_swizzle_mode` from `ragged_tma_store`, and the two
        # differ for FP8-QKV MLA (64B QKV swizzle, 128B BF16 output
        # store). For MHA the store is built with `config.swizzle_mode`,
        # so this folds to the previous expression.
        comptime swizzle_granularity = (
            type_of(ragged_tma_store).swizzle_mode.bytes()
            // size_of[output_type]()
        )
        # Block count is the output column count aligned to the OUTPUT swizzle
        # granularity (SWIZZLE_NONE -> 8 bf16), NOT the QKV swizzle's
        # `padded_ov_depth` (aligned to 64). ov_depth is already a multiple of
        # the output granularity for every supported head size.
        comptime iters_total = (
            align_up(config.ov_depth, swizzle_granularity)
            // swizzle_granularity
        )
        comptime iters_per_wg0 = (iters_total + 1) // 2
        comptime iters_per_wg1 = iters_total // 2
        # In 1Q both WGs write the same Q rows; no per-WG gmem-row
        # offset (the depth column j drives the gmem position).
        out_row_idx = gmem_row + cta_q_offset
        comptime if splitk_combine_active:
            # Reduce-scatter: every partition stages its FULL normalized O_cta
            # (f32) + per-row (max,sum) into the dead Q+KV span and publishes via
            # the round-1 cluster mbarrier; then (Pass 2) every partition's WG0
            # reads its OWN depth band from all P peers and TMA-stores that band
            # (see fa4_splitk_reduce_scatter_write below). Zero per-CTA smem
            # growth (reuses dead Q+KV) — growing it hangs P>=4.
            comptime assert (
                config.BM * config.padded_ov_depth * 4 + 2 * 256 * 4
                <= type_of(smem).q_bytes + type_of(smem).kv_bytes
            ), "split-K f32 O+(max,sum) staging must fit in the dead Q+KV span"
            stage = smem.o_smem[DType.float32]()
            maxsum = stage + (config.BM * config.padded_ov_depth)

            # --- Publish (max,sum) for both WGs, then stage + reduce-scatter. ---
            tid = warp_group_idx * UInt32(WARPGROUP_SIZE) + row
            maxsum[tid * 2] = global_max
            maxsum[tid * 2 + 1] = global_sum
            # Reduce-scatter (both WGs call; WG0 stages + combines, WG1 only
            # joins the phase-0 wait inside the helper): WG0 stages this CTA's
            # FULL O_cta -- its OWN depth band into the `o_final` registers, every
            # other block to the `stage` smem (peers read those for THEIR bands)
            # -- combining its own o0 with the peer o1 it already waited on
            # (single_source=False). Then every partition's WG0 reads its OWN band
            # from all P partitions and writes the combined bf16 into its OWN-band
            # f32 stage slice (no peer reads it), so NO round-2 cluster barrier is
            # needed -- the kernel's terminal `cluster_sync()` keeps the peer-read
            # PEER bands alive until every partition finishes reading. Band
            # ownership in Pass 2 is across PARTITIONS, not WGs, so WG1 stages
            # nothing (keeping the per-partition combine WG0-only avoids the known
            # WG1-owns-a-band deadlock).
            # MUST-VERIFY-ON-B200: no `fence_async_view_proxy()` between the
            # plain-`st.shared` stage dump and the publish arrive (both now inside
            # the helper) — proven (test_cluster_publish_smoke) unnecessary for
            # the peer `ld.shared::cluster` reads. If P>=4 hangs or reads stale O,
            # the first fix to try is a fence before the arrive in
            # `fa4_splitk_reduce_scatter_write`.
            fa4_splitk_reduce_scatter_write[
                config,
                P=config.splitk_partitions,
                use_fma=use_fma,
            ](
                row,
                warp_idx & 3,
                warp_group_idx,
                splitk_partition_idx(UInt32(config.splitk_partitions)),
                global_max,
                global_sum,
                scale_log2e,
                final_scale_local,
                final_scale_peer,
                stage,
                maxsum,
                o_smem,
                smem.misc_mbars().publish_mbar(),
                own_o_tile,
                peer_o_tile,
                ragged_tma_store,
                num_output_rows,
                head_idx,
                out_row_idx,
            )
        else:
            if warp_group_idx == UInt32(0):
                if is_writer:
                    fa4_lse_combine_write[
                        config,
                        wg_j_offset=0,
                        iters_per_wg=iters_per_wg0,
                    ](
                        row,
                        warp_idx & 3,
                        warp_group_idx,
                        final_scale_local,
                        final_scale_peer,
                        o_smem,
                        own_o_tile,
                        peer_o_tile,
                        ragged_tma_store,
                        num_output_rows,
                        head_idx,
                        out_row_idx,
                    )
            else:
                comptime if iters_per_wg1 > 0:
                    if is_writer:
                        fa4_lse_combine_write[
                            config,
                            wg_j_offset=iters_per_wg0,
                            iters_per_wg=iters_per_wg1,
                        ](
                            row,
                            warp_idx & 3,
                            warp_group_idx,
                            final_scale_local,
                            final_scale_peer,
                            o_smem,
                            own_o_tile,
                            peer_o_tile,
                            ragged_tma_store,
                            num_output_rows,
                            head_idx,
                            out_row_idx,
                        )
    named_barrier[Int32(2 * WARPGROUP_SIZE)](4)
    # Pair-CTA and 1Q split-K defer dealloc to the kernel after cluster_sync so
    # that no CTA exits while a peer's cluster-scoped access is in flight
    # (stmatrix for pair-CTA; the DSMEM `(max,sum)` peer reads for split-K).
    # Only plain single-CTA deallocs inline here.
    comptime if not config.pair_cta and config.splitk_partitions == 1:
        if warp_idx == 0:
            tcgen05_release_allocation_lock[Int32(cta_group)]()
            tcgen05_dealloc[Int32(cta_group)](
                tmem_addr, UInt32(config.sm100_tmem_cols)
            )
