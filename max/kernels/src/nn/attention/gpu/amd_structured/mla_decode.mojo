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
"""Gfx950 MLA (Multi-Latent Attention) decode kernel, built on KVBuffer.

Forked from `mha_decode.mojo` so the MLA-specific pipeline (K==V SMEM
aliasing, `output_depth < depth` rope tail, BN×64 SMEM block split for
depth=576) can evolve independently from the MHA path.

Key invariants (enforced via `mla_decode` callsite in `mla.mojo`):
  * `mla_kv_alias = True`: V reads from K's swizzled SMEM, so no V DMA.
  * `shared_kv = True` (required by `mla_kv_alias`): V SMEM is a bitcast
    view of K SMEM (see `attention.mojo` constructor).
  * `BN = 128` ⇒ `double_buffer_k_only = False`: single-buffer K loop,
    except under `mla_dma_warp`, where K streams through two LDS stages.
  * `output_depth ≤ depth`: per-warp V buffer slices the V-nope portion
    out of K's full-depth SMEM.

Recipe (see `process_tile`):
  * Wait K → optional FULL_MASK skip → load K from SMEM → mma_qk →
    softmax → write P to SMEM → barrier → load V (= aliased K) from
    SMEM → prefetch next K → update output → mma_pv → iter-end drain.

NOTE: `process_tile`, `prefetch_next`, the empty-partition handling,
the softmax block, and the epilogue are largely line-for-line shared
with `mha_decode.mojo`; the two files only diverge in the K==V
aliasing path, the single-buffer-K loop, the `bk_smem<BK` K SMEM
split, the counted K-DMA waits (`_split_k_dma`; `mha_decode.mojo`
keeps full drains), and the FP8 fold's DMA producer warp
(`mla_dma_warp`), which
fills the next K stage while the compute warps run `process_tile` on
the stage its `slot` argument names.  Bug fixes to the shared logic
(softmax numerics, barrier placement, etc.) must be mirrored to both
files until the kernels are re-converged behind a shared helper.
"""

from std.sys import get_defined_bool
from std.math import align_up, ceildiv
from std.math.uutils import umod, ufloordiv
from max.gpu import WARP_SIZE, block_idx, thread_idx
from max.gpu import warp_id as get_warp_id
from max.gpu.sync import s_waitcnt
from std.utils.numerics import get_accum_type, min_or_neg_inf

from nn.attention.mha_mask import TileMaskStatus
from nn.attention.mha_utils import get_start_and_end_for_partitions

from .attention import Attention
from .mha_prefill import barrier
from .mma import TiledMmaOp

from .kv_buffer import KVBuffer, _get_k_swizzle

# Partition alignment for a predicated K operand: one indexed-load strip.
comptime _PREDICATED_SPLIT_ALIGN = 16

# Barriers the DMA producer warp meets per tile: K ready, P written, tile
# done. Under `dma_warp`, `process_tile` must execute exactly this many, or
# the producer and the compute warps fall out of step on a CTA barrier.
comptime _DMA_WARP_BARRIERS_PER_TILE = 3


@inline(.always)
def _k_dma_group_wait[
    tiles_per_warp: Int,
    num_warps: Int,
    key_groups: Int,
    max_depth_group: Int,
]() -> UInt32:
    """Returns the VMEM ops still outstanding once every wave has issued its
    DMA tiles for depth column groups `<= max_depth_group`.

    The loader issues two `buffer_load_dwordx4` per (32 key, 64 depth) tile in
    warp tile order, so a wave's first `2 * (t + 1)` ops cover its tiles
    `0..t`. Every wave executes the same `s_waitcnt`, so the value is set by
    the wave that needs the most of its prefix retired.

    Parameters:
        tiles_per_warp: DMA tiles each wave issues per KV tile.
        num_warps: Waves issuing the DMA.
        key_groups: Key groups per depth column group.
        max_depth_group: The last depth column group that must have landed.

    Returns:
        The `vmcnt` value for the wait.
    """
    var outstanding = 2 * tiles_per_warp
    for w in range(num_warps):
        var issued = 0
        for t in range(tiles_per_warp):
            if (t * num_warps + w) // key_groups <= max_depth_group:
                issued = 2 * (t + 1)
        outstanding = min(outstanding, 2 * tiles_per_warp - issued)
    return UInt32(outstanding)


__extension Attention:
    @inline(.always)
    def mla_decode(
        mut self,
        exp_sum_ptr: MutPointer[
            Scalar[get_accum_type[Self.q_type]()], MutAnyOrigin
        ],
        qk_max_ptr: MutPointer[
            Scalar[get_accum_type[Self.q_type]()], MutAnyOrigin
        ],
        num_partitions: Int,
    ):
        """gfx950 MLA decode on KVBuffer.  See module docstring."""
        comptime assert Self.token_gen, "mla_decode requires token_gen=True"
        comptime assert (
            Self.mla_kv_alias
        ), "mla_decode requires mla_kv_alias=True"
        comptime assert (
            Self.amd_structured_config.shared_kv
        ), "mla_decode requires shared_kv=True (implied by mla_kv_alias)"
        comptime assert (
            Self.BK == 32 or Self.BK == 64 or Self.BK == 128
        ), "BK must be 32, 64, or 128"
        # K-tail zero-pad path: at BK=128 we allow depth%128 != 0 as
        # long as depth is still a multiple of 64.  The kernel's inner-K
        # loop zero-pads the partial 128-element tile so the 16x16x128
        # MFMA sees zeros in the out-of-bounds half.
        comptime assert Self.depth % Self.BK == 0 or (
            Self.BK == 128 and Self.depth % 64 == 0
        ), "depth must be a multiple of BK (or of 64 when BK==128)"
        comptime assert Self.BN % Self.BK == 0, "BN must be a multiple of BK"

        # Use the SMEM physical block width (`_bk_smem`) chosen by the
        # `Attention` struct.  Diverges from `BK` only for the FP8 MLA
        # decode case (BK=128, depth%128 != 0) where it becomes 64 to
        # avoid an 8 KB padded block; otherwise `_bk_smem == BK`.
        comptime _bk_smem = Self._bk_smem
        comptime k_swizzle = _get_k_swizzle[Self.mma_shape[0], _bk_smem]()

        var warp_id = UInt32(get_warp_id[broadcast=True]())
        # Warp index of the DMA producer, one past the compute warps.
        comptime producer_warp_id = UInt32(Self.num_warps_m * Self.num_warps_n)

        # Split-K: compute this partition's key range. A predicated
        # (in-place sparse) K operand's partitions need only strip alignment,
        # so a top-k decode can spread one sequence over many CUs: its
        # loader skips (and zeroes) the keys past the partition end or the
        # entry's valid length, and the mask bounds each tile at `num_keys`.
        comptime split_align = (
            _PREDICATED_SPLIT_ALIGN if Self.k_t.rows_predicated else Self.BN
        )
        var start, end = get_start_and_end_for_partitions[split_align](
            self.num_keys, num_partitions, block_idx.x
        )
        comptime if Self.k_t.rows_predicated:
            if block_idx.x == 0 and thread_idx.x == 0:
                self.k.check_indexed_rows(UInt32(self.batch_idx))

        # Empty partitions (from power-of-two bucketing): reset to
        # rowsum=0/rowmax=-inf so the reduce masks them via `scale > 0`.
        # In sink mode, __init__ primes `rowsum=1/rowmax=sink_weight` —
        # we must override that here or the reduce reads uninitialized
        # partition outputs with a nonzero scale.
        if start >= end:
            _ = self.softmax.rowmax_tensor.fill(
                min_or_neg_inf[Self.accum_type]()
            )
            _ = self.softmax.rowsum_tensor.fill(0)
            comptime if Self.mla_dma_warp:
                # The producer warp owns no stat rows.
                if warp_id == producer_warp_id:
                    return
            self.store_partition_info(num_partitions, exp_sum_ptr, qk_max_ptr)
            return

        # K: full BN × depth SMEM, per-warp DMA cooperation, transpose read.
        # smem_depth is padded to a multiple of `_bk_smem` so every block
        # is exactly `_bk_smem`-wide.  When `_bk_smem == BK` the formula
        # simplifies to `align_up(depth, BK)`; when `_bk_smem < BK`
        # (MLA decode rope tail) it equals `depth`, since the depth%64==0
        # gate ensures depth is a multiple of `_bk_smem`.
        comptime _k_smem_depth = align_up(Self.depth, _bk_smem)
        comptime KBufT = KVBuffer[
            kv_t=Self.k_t,
            mma_shape=Self.mma_shape,
            swizzle=k_swizzle,
            BN=Self.BN,
            WN=Self.WN,
            BK=Self.BK,
            num_threads=Self.num_threads,
            depth=Self.depth,
            kv_num_heads=Self.num_heads // Self.group,
            transpose=True,
            smem_depth=_k_smem_depth,
            bk_smem=_bk_smem,
        ]
        var k_buffer = KBufT(
            self.k,
            self.batch_idx,
            self.kv_head_idx(),
            KBufT.SmemParentType(
                self.k_smem_ptr.as_unsafe_any_origin(),
                KBufT._SmemParentLayout(),
            ),
            end,
            warp_id,
        )

        # Counted DMA waits: start QK on the shallow depth strips while the
        # deeper ones are still landing (`KBufT.dma_col_major` issues them in
        # that order). The three groups are K strips `[0, _strip_hi_0)`,
        # `[_strip_hi_0, _strip_hi_1)` and `[_strip_hi_1, _num_k_strips)`, so
        # each needs only the first `2 * hi - 1` 64-deep column groups of the
        # tile. A mask with interior FULL_MASK tiles skips a tile without
        # draining its DMA (it issues the next one instead), so it keeps the
        # full-drain path.
        comptime _num_k_strips = ceildiv(Self.depth, Self.BK)
        comptime _depth_groups = _k_smem_depth // _bk_smem
        comptime _strip_hi_0 = min(2, _num_k_strips)
        comptime _strip_hi_1 = min(4, _num_k_strips)
        comptime dma_warp = Self.mla_dma_warp
        comptime _split_k_dma = (
            KBufT.dma_col_major
            and _num_k_strips > 2
            and not Self.mask_t.check_mask_during_decoding
            and not dma_warp
        )
        comptime _dma_key_groups = Self.BN // KBufT.warp_tile_rows
        comptime _dma_wait_0 = _k_dma_group_wait[
            KBufT._tiles_per_warp,
            KBufT._num_warps,
            _dma_key_groups,
            min(2 * _strip_hi_0 - 1, _depth_groups - 1),
        ]()
        comptime _dma_wait_1 = _k_dma_group_wait[
            KBufT._tiles_per_warp,
            KBufT._num_warps,
            _dma_key_groups,
            min(2 * _strip_hi_1 - 1, _depth_groups - 1),
        ]()

        # V is aliased onto K's SMEM (mla_kv_alias).  We need only the
        # per-warp V reader; no V DMA buffer.  MHA: output_depth == depth.
        # MLA: output_depth < depth (V-nope only; rope tail consumed in QK).
        comptime depth_per_warp = Self.output_depth // Self.num_warps_n
        # `num_warps_n` must evenly split `output_depth`, and each slice must be
        # `_bk_smem`-aligned (or smaller than one block) so `v_warp_smem_offset`
        # lands on the K writer's block boundaries rather than aliasing wrong K
        # bytes. Holds for num_warps_n in {1,2,4} (depth_per_warp {512,256,128},
        # all 64-aligned); S=1 is num_warps_n=4.
        comptime assert Self.output_depth % Self.num_warps_n == 0
        comptime assert (depth_per_warp % _bk_smem == 0) or (
            depth_per_warp < _bk_smem
        )
        # SMEM block width seen by each warp.  Must be >= _bk_smem so the
        # blocked product layout has at least one repeat.
        comptime smem_depth_per_warp = max(depth_per_warp, _bk_smem)

        # Per-warp V SMEM offset in the blocks-stacked physical layout
        # (row-major (BN, _bk_smem) blocks laid out contiguously): for
        # (row=0, col) → (col // _bk_smem) * BN * _bk_smem + col % _bk_smem.
        # Uses `_bk_smem` (not `BK`) because the K writer lays out SMEM in
        # `_bk_smem`-wide blocks.
        var v_warp_col = umod(Int(warp_id), Self.num_warps_n)
        var v_col_start = v_warp_col * depth_per_warp
        var v_warp_smem_offset = ufloordiv(
            v_col_start, _bk_smem
        ) * Self.BN * _bk_smem + umod(v_col_start, _bk_smem)

        # V reads from K's swizzled SMEM (mla_kv_alias), so V's
        # `ds_read_tr8_b64` addressing applies the same swizzle K uses;
        # `load_v_fp8_strip` consumes that swizzle and XORs the byte
        # offset at vec granularity to land on the writer's permuted slot.
        comptime VBufT = KVBuffer[
            kv_t=Self.v_t,
            mma_shape=Self.mma_shape,
            swizzle=k_swizzle,
            BN=Self.BN,
            WN=Self.BN,
            BK=Self.BK,
            num_threads=Self.num_threads,
            depth=depth_per_warp,
            kv_num_heads=Self.num_heads // Self.group,
            transpose=False,
            cache_depth=Self.cache_depth,
            # A V stage step must equal a K stage step when K is staged twice.
            smem_depth=_k_smem_depth if dma_warp else smem_depth_per_warp,
            bk_smem=_bk_smem,
        ]
        var v_buffer = VBufT(
            self.v,
            self.batch_idx,
            self.kv_head_idx(),
            VBufT.SmemParentType(
                self.v_smem_ptr.unsafe_offset(
                    v_warp_smem_offset
                ).as_unsafe_any_origin(),
                VBufT._SmemParentLayout(),
            ),
            end,
            warp_id,
        )

        # Advance K iterator and mask tracking to partition start.  V is
        # aliased onto K's SMEM so V has no iterator of its own.
        k_buffer.kv_cache_iter.tile_start_row = start
        self.kv_start_row = UInt32(start)

        # Pre-scale Q by (scale * log2e).  Default on for bf16: the deferred
        # `_fma` scaling path can overflow before softmax for large decode
        # scores (for example Gemma 4's depth=512, scale=1.0 global layers).
        # Disabled for fp8 because quantizing scaled Q back to fp8 loses too
        # much precision, so fp8 stays on `_fma`.
        # Sink forces pre-scaling (rowmax in scaled-log2 space).
        comptime prescale_q = (
            get_defined_bool["PRESCALE_Q", True]()
            and not Self.q_type.is_float8()
        ) or Self.sink
        comptime if prescale_q:
            self.scale_q_buffer()

        comptime accum_type = get_accum_type[Self.k_t.dtype]()

        comptime QKMmaOp = TiledMmaOp[
            accum_type,
            Self.q_type,
            Self.mma_shape,
            transpose_b=True,
        ]
        comptime PVMmaOp = TiledMmaOp[
            accum_type,
            Self.q_type,
            Self.mma_shape,
            transpose_b=True,
        ]

        @inline(.always)
        def mma_qk[
            lo: Int = 0, hi: Int = _num_k_strips, zero_p: Bool = True
        ]() {imm}:
            # ceildiv (not floor): when depth % BK != 0 (Kimi MLA: 576 %
            # 128 == 64) we must still process the partial K-tile.  Q's
            # OOB lanes are zero'd in QRegisterBuffer (buffers.mojo) and
            # K's OOB cols come back as zero via buffer_load's
            # resource-bound clamp, so the partial tile MFMA is correct
            # without any explicit masking inside this loop.
            comptime if zero_p:
                self.zero_p_buffer[0]()
            comptime for i in range(lo, hi):
                comptime for k_mma in range(Self.num_k_mmas2):
                    QKMmaOp.mma[swap_a_b=Self.swap_a_b](
                        self.q_buffer.mma_tile[i, k_mma](),
                        k_buffer.mma_subtile[k_mma, i](),
                        self.p_reg_buffer.stage_tile[0](),
                    )

        @inline(.always)
        def mma_pv() {imm}:
            # Each warp's v_buffer holds only depth_per_warp tiles
            # (loaded from the warp's LDS depth offset), so mma_subtile
            # directly gives the warp's depth range.
            comptime for i in range(Self.BN // Self.BK):
                comptime for k_mma in range(v_buffer.num_k_mmas2):
                    PVMmaOp.mma[swap_a_b=Self.swap_a_b](
                        self.p_reg_buffer.mma_tile[i, k_mma, 0](),
                        v_buffer.mma_subtile[k_mma, i](),
                        self.out_reg_buffer.reg_tile,
                    )

        # Masks with interior FULL_MASK tiles (e.g. ChunkedCausalMask) need
        # a per-tile status check: when a tile is fully masked, skip QK,
        # softmax and PV but keep the DMA prefetch pipeline moving.
        # CausalMask during decode never produces FULL_MASK tiles.
        comptime has_interior_full_mask = (
            Self.mask_t.check_mask_during_decoding
        )

        @inline(.always)
        def prefetch_next() {mut k_buffer}:
            """Prefetch next K after current LDS reads have drained.

            MLA: V is aliased onto K's SMEM, so there's no separate V DMA;
            single-buffer K (no `double_buffer_k_only` since BN=128 > 64),
            overwrites slot 0.
            """
            s_waitcnt[lgkmcnt=0]()
            barrier()
            _ = k_buffer.load_from_dram[0]()

        @inline(.always)
        def process_tile[has_next: Bool](slot: Int) {mut self, imm}:
            """Process one KV tile.

            K is already in LDS stage `slot` when called.  V is loaded from
            the same SMEM (mla_kv_alias) mid-tile after K is consumed by QK.
            """
            comptime if not _split_k_dma:
                # Drain the tile's K DMA before the FULL_MASK check, so a
                # skipped tile never leaves it in flight. DMA producer
                # barrier 1 of `_DMA_WARP_BARRIERS_PER_TILE`.
                s_waitcnt[vmcnt=0, lgkmcnt=0]()
                barrier()

            # Skip fully-masked tiles (ChunkedCausalMask interior).
            comptime if has_interior_full_mask:
                var tile_status = self.mask_status(self.kv_start_row)
                if tile_status == TileMaskStatus.FULL_MASK:
                    self.kv_start_row += UInt32(Self.BN)
                    self.mask_advance()
                    comptime if has_next:
                        prefetch_next()
                    return

            comptime if _split_k_dma:
                # Counted DMA waits, one per depth strip group: the strips
                # already in LDS run their QK MMAs while the rest of the
                # tile is still landing.
                comptime for g in range(3):
                    comptime lo = (
                        0 if g
                        == 0 else (_strip_hi_0 if g == 1 else _strip_hi_1)
                    )
                    comptime hi = (
                        _strip_hi_0 if g
                        == 0 else (_strip_hi_1 if g == 1 else _num_k_strips)
                    )
                    comptime wait = (
                        _dma_wait_0 if g
                        == 0 else (_dma_wait_1 if g == 1 else UInt32(0))
                    )
                    s_waitcnt[vmcnt=wait]()
                    barrier()
                    comptime for i in range(lo, hi):
                        k_buffer.load_from_shared[i](0)
                    # Register-side zero pad for the partial K-tile's OOB
                    # tail (a no-op when depth % BK == 0).
                    comptime if hi == _num_k_strips:
                        k_buffer.zero_partial_tile_pad()
                    mma_qk[lo, hi, zero_p=(g == 0)]()
            else:
                k_buffer.load_from_shared(slot)
                # Set the pad columns of the partial K-tile to zero. When
                # depth % BK == 0, there is no pad.
                k_buffer.zero_partial_tile_pad()
                mma_qk()
            # Online softmax on P (cross-warp reduction via SMEM).
            # Prescaled path is numerically safer (Qwen-class models need
            # it — `_fma` accumulates NaN after a few decode steps on real
            # weights).  `_fma` variants fold the score scale into exp2 and
            # save VALU; opt in via `-D PRESCALE_Q=False` if tolerable.
            comptime if prescale_q:
                self.online_softmax_step_0_prescaled[0]()
                self.online_softmax_step_1_prescaled[0]()
            else:
                self.online_softmax_step_0_fma[0]()
                self.online_softmax_step_1_fma[0]()

            # Write P to shared memory so all warps can read it for PV.
            self.p_reg_buffer.copy_to_shared()
            s_waitcnt[lgkmcnt=0]()
            # DMA producer barrier 2 of `_DMA_WARP_BARRIERS_PER_TILE`.
            barrier()

            v_buffer.load_from_shared(slot)

            # Prefetch next tile after V→REG.  Barrier ensures all warps
            # finished V LDS reads before DMA.
            comptime if has_next and not dma_warp:
                prefetch_next()

            self.online_softmax_update_output()
            mma_pv()

            # Iter-end drain + barrier.  The entry barrier at the next
            # iteration is not sufficient on its own: without this
            # explicit drain, mma_pv's pipelined P SMEM ds_reads can overlap
            # with iter N+1's copy_to_shared writes.  Waitcnt-only and
            # barrier-only are both empirically insufficient — both are
            # required.  With the counted K DMA waits the vector counter is
            # left alone here: iter N+1's first strip-group wait closes it.
            comptime if _split_k_dma:
                s_waitcnt[lgkmcnt=0]()
            else:
                s_waitcnt[vmcnt=0, lgkmcnt=0]()
            # DMA producer barrier 3 of `_DMA_WARP_BARRIERS_PER_TILE`.
            barrier()

        var num_tiles = ceildiv(end - start, Self.BN)

        comptime if dma_warp:
            if warp_id == producer_warp_id:
                # Producer: tile t+1's DMA lands in the other stage while the
                # compute warps run tile t. It meets their three barriers per
                # tile: K ready, P written, tile done.
                comptime DmaBufT = KVBuffer[
                    kv_t=Self.k_t,
                    mma_shape=Self.mma_shape,
                    swizzle=k_swizzle,
                    BN=Self.BN,
                    WN=Self.BN,
                    BK=Self.BK,
                    num_threads=WARP_SIZE,
                    depth=Self.depth,
                    kv_num_heads=Self.num_heads // Self.group,
                    transpose=True,
                    smem_depth=_k_smem_depth,
                    bk_smem=_bk_smem,
                ]
                var dma = DmaBufT(
                    self.k,
                    self.batch_idx,
                    self.kv_head_idx(),
                    DmaBufT.SmemParentType(
                        self.k_smem_ptr.as_unsafe_any_origin(),
                        DmaBufT._SmemParentLayout(),
                    ),
                    end,
                    0,
                )
                dma.kv_cache_iter.tile_start_row = start

                dma.load_from_dram[0, desc_in_registers=True]()
                for t in range(0, num_tiles, 2):
                    comptime for slot in range(2):
                        if t + slot < num_tiles:
                            s_waitcnt[vmcnt=0]()
                            barrier()
                            if t + slot + 1 < num_tiles:
                                dma.load_from_dram[
                                    1 - slot, desc_in_registers=True
                                ]()
                            comptime for _ in range(
                                _DMA_WARP_BARRIERS_PER_TILE - 1
                            ):
                                barrier()
                return
        else:
            # Prologue: load first tile's K (slot 0).  V is aliased onto K's
            # SMEM and is consumed mid-tile after K's reads finish.
            _ = k_buffer.load_from_dram[0]()

        # Capping the key bound at the row's valid length makes the tile
        # mask cover the padding; a partition wholly past it is empty,
        # regardless of any prefetched rows.
        comptime if Self.k_t.rows_indexed:
            var valid_end = min(
                end, self.k.indexed_valid_length(UInt32(self.batch_idx))
            )
            if start >= valid_end:
                _ = self.softmax.rowmax_tensor.fill(
                    min_or_neg_inf[Self.accum_type]()
                )
                _ = self.softmax.rowsum_tensor.fill(0)
                self.store_partition_info(
                    num_partitions, exp_sum_ptr, qk_max_ptr
                )
                # Without a split-K reduce nothing else writes this entry,
                # so store the zero accumulator, as the gather path does.
                if num_partitions <= 1:
                    self.store_output()
                # The prologue's LDS DMA may still be in flight; drain it so
                # no write lands after this workgroup's LDS is released.
                s_waitcnt[vmcnt=0]()
                return
            self.num_keys = valid_end
            num_tiles = ceildiv(valid_end - start, Self.BN)

        comptime if dma_warp:
            for t in range(num_tiles):
                process_tile[False](t & 1)
        else:
            # Single-buffer K: simple linear loop.  MLA never enables
            # `double_buffer_k_only` because BN=128 > 64.
            for _ in range(num_tiles - 1):
                process_tile[True](0)

            if num_tiles > 0:
                process_tile[False](0)

        self.out_reg_buffer.apply_softmax_denominator[
            p_scale_log2=type_of(self.p_reg_buffer).p_scale_log2
        ](self.softmax.rowsum_tensor)
        self.store_partition_info(num_partitions, exp_sum_ptr, qk_max_ptr)
        self.store_output()
