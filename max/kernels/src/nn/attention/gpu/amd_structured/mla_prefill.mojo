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
"""MLA (Multi-Latent Attention) prefill kernel for gfx950.

Double-buffered MLA prefill with K_rope support. Uses TileTensor
throughout.

Two-phase QK matmul per tile:
  Phase 1 (nope): Q[:,:depth] @ K^T
  Phase 2 (rope): Q[:,depth:q_depth] @ K_rope^T
"""

from std.collections import OptionalReg
from std.math import ceildiv, clamp, log2
from std.math.uutils import ufloordiv
from std.sys import align_of, simd_width_of
from max.gpu.intrinsics import readfirstlane
from max.gpu import lane_id, warp_id as get_warp_id
from std.memory import bitcast, unsafe_stack_allocation
from layout import TileTensor, row_major
from layout.swizzle import Swizzle
from nn.attention.mha_mask import CausalMask, TileMaskStatus
from nn.attention.mha_operand import MHAOperand
from std.utils.numerics import get_accum_type, min_or_neg_inf

from .attention import Attention
from .iglp import _iglp_opt, AMDIGLPStrategy
from .kv_buffer import KVBuffer
from .mha_prefill import barrier, block_sync_lds_direct_load
from .mma import TiledMmaOp


__extension Attention:
    @inline(.always)
    def store_split_lse(
        self,
        lse_ptr: OptionalReg[MutPointer[Scalar[DType.float32], _]],
        split_idx: Int,
        num_splits: Int,
        total_rows: Int,
        row_base: Int,
        empty: Bool,
    ):
        """Writes each query row's log2-sum-exp for one KV split.

        The row stats are broadcast across the MFMA lane group, so lane
        `l < MMA_M` owns row `l` of its M tile. `empty` marks a split with no
        key tiles; the combine skips it.

        Args:
            lse_ptr: The `[num_splits, total_rows, num_heads]` stack of
                log2-sum-exp planes, one per split.
            split_idx: This CTA's KV split.
            num_splits: Number of KV splits.
            total_rows: Ragged query rows the stack covers.
            row_base: Ragged row of this sequence's first query.
            empty: Whether this split visited no key tiles.
        """
        # Required whenever `split_kv` is set; the launcher always passes it
        # for a split launch. A device-side `__guard`/`abort` on it does not
        # lower (the offload pass cannot prove the binding dominates), so this
        # is the one place the split path unwraps the pointer.
        var lse_tt = TileTensor(
            lse_ptr.value(),
            row_major((num_splits, total_rows, Self.num_heads)),
        )
        var lane = Int(lane_id())
        if lane >= Self.mma_shape[0]:
            return
        comptime for m_mma in range(Self.num_m_mmas):
            var row = (
                Self.q_tile_idx() * Self.BM
                + self.warp_row * Self.WM
                + m_mma * Self.mma_shape[0]
                + lane
            )
            if row < self.seq_len:
                var row_lse = min_or_neg_inf[DType.float32]()
                if not empty:
                    row_lse = self.softmax.rowmax_tensor[m_mma, 0][0].cast[
                        DType.float32
                    ]() + log2(
                        self.softmax.rowsum_tensor[m_mma, 0][0].cast[
                            DType.float32
                        ]()
                    )
                lse_tt[split_idx, row_base + row, Self.q_head_idx()] = row_lse

    @inline(.always)
    def mla_prefill[
        k_rope_t: MHAOperand,
        //,
        split_kv: Bool = False,
        split_min_keys: Int = 1024,
    ](
        mut self,
        k_rope: k_rope_t,
        split_idx: Int = 0,
        num_splits: Int = 1,
        lse_ptr: OptionalReg[MutPointer[Scalar[DType.float32], _]] = None,
        lse_row_base: Int = 0,
        lse_total_rows: Int = 0,
    ):
        """Double-buffered gfx950 MLA prefill with K_rope support.

        Uses gfx950 double-buffered KVBuffer for K, V, and K_rope
        (DRAM->SMEM direct). K_rope has its own SMEM double-buffer.
        Two-phase QK matmul:
        Phase 1 (nope): Q[:,:depth] @ K^T  (KVBuffer path)
        Phase 2 (rope): Q[:,depth:q_depth] @ K_rope^T  (KVBuffer path)

        With `split_kv`, the CTA visits only split `split_idx` of
        `num_splits` contiguous slices of its key tiles, stores the
        normalized fp32 partial output, and writes each row's log2-sum-exp
        to split `split_idx` of the `lse_ptr` stack for the combine.

        Parameters:
            k_rope_t: The K_rope operand type (inferred).
            split_kv: Whether this CTA computes one KV split.
            split_min_keys: Key positions each split keeps. The launch sizes
                `num_splits` from the K buffer's extent, which can exceed the
                keys a sequence really holds, so a CTA drops the splits past
                `ceil(num_tiles * BN / split_min_keys)` and they store nothing.

        Args:
            k_rope: The K_rope operand.
            split_idx: This CTA's KV split.
            num_splits: Number of KV splits.
            lse_ptr: The `[num_splits, total_rows, num_heads]` stack of
                log2-sum-exp planes (split_kv only; required then).
            lse_row_base: Ragged row of this sequence's first query.
            lse_total_rows: Ragged query rows the stack covers.
        """
        comptime cache_num_heads = 1
        comptime cache_depth = 576
        comptime rope_depth = q_depth - Self.depth
        comptime cache_group = Self.num_heads // cache_num_heads

        comptime assert Self.BK == 32 or Self.BK == 64, "BK must be 32 or 64"
        # Validated at depth 128; generalizes to any nope depth that tiles
        # cleanly with BK plus a cleanly-tiling rope (GLM: depth=192, rope=64).
        comptime assert Self.depth <= 256, "MLA nope depth must be <= 256"
        comptime assert (
            Self.depth % Self.BK == 0
        ), "depth must be a multiple of BK"
        comptime assert Self.BN % Self.BK == 0, "BN must be a multiple of BK"
        comptime assert (
            rope_depth % Self.BK == 0
        ), "rope_depth must be a multiple of BK"

        comptime k_swizzle = (
            Swizzle(3, 0, 4) if Self.mma_shape[0]
            == 32 else Optional[Swizzle](None)
        )

        var warp_id = UInt32(
            readfirstlane(bitcast[.int32](UInt32(get_warp_id())))
        )

        # K buffer (nope): depth=128, double-buffered gfx950 style.
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
            uniform_dma_offset=True,
            uniform_row_math=True,
        ]
        var k_buffer = KBufT(
            self.k,
            self.batch_idx,
            self.kv_head_idx(),
            KBufT.SmemParentType(
                self.k_smem_ptr.as_unsafe_any_origin(),
                KBufT._SmemParentLayout(),
            ),
            self.num_keys,
            warp_id,
        )

        # V buffer: depth=128, double-buffered gfx950 style.
        # FP8 V uses Swizzle(3, 0, 4) (same as K) to eliminate the
        # ds_read_tr8_b64 bank conflicts inherent to BK=64 row stride
        # (4 lane-quads per 32-bank LDS row → 4-way conflict). Both the
        # SubTileLoaderLDS writer and the load_v_fp8_strip reader honor
        # the same XOR so the data is read back correctly.
        comptime v_swizzle = (
            Swizzle(3, 0, 4) if Self.v_t.dtype.is_float8()
            and Self.mma_shape[0] == 32 else Optional[Swizzle](None)
        )
        comptime VBufT = KVBuffer[
            kv_t=Self.v_t,
            mma_shape=Self.mma_shape,
            swizzle=v_swizzle,
            BN=Self.BN,
            WN=Self.WN,
            BK=Self.BK,
            num_threads=Self.num_threads,
            depth=Self.output_depth,
            kv_num_heads=Self.num_heads // Self.group,
            transpose=False,
            uniform_dma_offset=True,
            uniform_row_math=True,
        ]
        var v_buffer = VBufT(
            self.v,
            self.batch_idx,
            self.kv_head_idx(),
            VBufT.SmemParentType(
                self.v_smem_ptr.as_unsafe_any_origin(),
                VBufT._SmemParentLayout(),
            ),
            self.num_keys,
            warp_id,
        )

        # K_rope buffer: depth=rope_depth, double-buffered gfx950 style.
        # Separate SMEM allocation since K_rope DMA overlaps with K_nope.
        comptime alignment = align_of[
            SIMD[k_rope_t.dtype, simd_width_of[k_rope_t.dtype]()]
        ]()
        comptime k_rope_smem_elems = 2 * Self.BN * rope_depth
        var k_rope_smem_ptr = unsafe_stack_allocation[
            k_rope_smem_elems,
            k_rope_t.dtype,
            address_space=.SHARED,
            alignment=alignment,
        ]()
        comptime KRopeBufT = KVBuffer[
            kv_t=k_rope_t,
            mma_shape=Self.mma_shape,
            swizzle=k_swizzle,
            BN=Self.BN,
            WN=Self.WN,
            BK=Self.BK,
            num_threads=Self.num_threads,
            depth=rope_depth,
            kv_num_heads=cache_num_heads,
            transpose=True,
            cache_depth=cache_depth,
            head_dim_offset=cache_depth - rope_depth,
            uniform_dma_offset=True,
            uniform_row_math=True,
        ]
        var k_rope_buffer = KRopeBufT(
            k_rope,
            self.batch_idx,
            ufloordiv(Int(self.kv_head_idx()), cache_group),
            KRopeBufT.SmemParentType(
                k_rope_smem_ptr.as_unsafe_any_origin(),
                KRopeBufT._SmemParentLayout(),
            ),
            self.num_keys,
            warp_id,
        )

        comptime accum_type = get_accum_type[Self.k_t.dtype]()
        # The fp8 P pre-scale `2^p_scale_log2` folds into the `exp2` argument
        # (an integer argument shift, i.e. an exact power-of-two scale), leaving
        # the row sum that many times too large for the epilogue to divide back
        # out. `PRegisterBuffer._scale_p` then leaves P unscaled.
        comptime p_scale_log2 = type_of(self.p_reg_buffer).p_scale_log2
        comptime p_scale_in_exp = (Self.p_scale_in_exp and p_scale_log2 != 0)
        comptime p_exp_bias_log2 = p_scale_log2 if p_scale_in_exp else 0

        # Phase 1: Q_nope @ K^T (depth // BK iterations)
        @inline(.always)
        def mma_qk_nope() {imm}:
            comptime MmaOp = TiledMmaOp[
                accum_type,
                Self.q_type,
                Self.mma_shape,
                transpose_b=True,
            ]
            self.zero_p_buffer[0]()

            comptime for i in range(Self.depth // Self.BK):
                comptime for k_mma in range(Self.num_k_mmas2):
                    MmaOp.mma[swap_a_b=Self.swap_a_b](
                        self.q_buffer.mma_tile[i, k_mma](),
                        k_buffer.get_mma_tile[k_mma, i](),
                        self.p_reg_buffer.stage_tile[0](),
                    )

        # Phase 2: Q_rope @ K_rope^T (rope_depth // BK iterations)
        @inline(.always)
        def mma_qk_rope() {imm}:
            comptime MmaOp = TiledMmaOp[
                accum_type,
                Self.q_type,
                Self.mma_shape,
                transpose_b=True,
            ]
            comptime nope_tiles = Self.depth // Self.BK

            comptime for i in range(rope_depth // Self.BK):
                comptime for k_mma in range(k_rope_buffer.num_k_mmas2):
                    MmaOp.mma[swap_a_b=Self.swap_a_b](
                        self.q_buffer.mma_tile[nope_tiles + i, k_mma](),
                        k_rope_buffer.get_mma_tile[k_mma, i](),
                        self.p_reg_buffer.stage_tile[0](),
                    )

        @inline(.always)
        def mma_pv() {imm}:
            comptime PVMmaOp = TiledMmaOp[
                accum_type,
                Self.q_type,
                Self.mma_shape,
                transpose_b=True,
            ]

            comptime for i in range(Self.BN // Self.BK):
                comptime for k_mma in range(v_buffer.num_k_mmas2):
                    PVMmaOp.mma[swap_a_b=Self.swap_a_b](
                        self.p_reg_buffer.mma_tile[i, k_mma, 0](),
                        v_buffer.get_mma_tile[k_mma, i](),
                        self.out_reg_buffer.reg_tile,
                    )

        # Calculate iteration bounds using mask helpers.
        var score_row = UInt32(self.mask_block_row + UInt32(self.start_pos))
        var start_col = self.mask.start_column[Self.BM, Self.BN, 1](
            UInt32(self.batch_idx), score_row
        )
        var num_tiles = Int(
            self.mask.last_masked_set_end[Self.BM, Self.BN, 1](
                UInt32(self.batch_idx), score_row, UInt32(self.num_keys)
            )
        )

        comptime if split_kv:
            comptime assert (
                Self.mask_t == CausalMask
            ), "split-KV MLA prefill supports only the causal mask"
            # The launch's split count comes from the K buffer's extent, so on
            # a short context inside an over-sized buffer it can ask for more
            # splits than the mask's key tiles support. Cap it by the real key
            # count: every split keeps `split_min_keys`, and the splits past
            # the cap find no tiles, store an empty plane and are skipped by
            # the combine.
            var key_bounded_splits = ceildiv(
                num_tiles * Self.BN, split_min_keys
            )
            var splits = clamp(key_bounded_splits, 1, num_splits)
            var tiles_per_split = ceildiv(num_tiles, splits)
            var first_tile = split_idx * tiles_per_split
            num_tiles = clamp(num_tiles - first_tile, 0, tiles_per_split)
            start_col += UInt32(first_tile * Self.BN)
            if num_tiles == 0:
                self.store_split_lse(
                    lse_ptr,
                    split_idx,
                    num_splits,
                    lse_total_rows,
                    lse_row_base,
                    empty=True,
                )
                return

        # Advance KV iterators and mask tracking to start_col.
        k_buffer.kv_cache_iter.set_start_row(Int(start_col))
        v_buffer.kv_cache_iter.set_start_row(Int(start_col))
        k_rope_buffer.kv_cache_iter.set_start_row(
            Int(start_col) + self.cache_start_pos
        )
        # Pinned so the steady loop's per-tile classification stays wave-uniform.
        comptime if Self.uniform_mask_row:
            self.kv_start_row = UInt32(readfirstlane(Int32(start_col)))
        else:
            self.kv_start_row = start_col
        self.mask_warp_col += start_col

        var num_pairs = num_tiles // 2

        # Pre-load first tile K+K_rope+V into LDS slot 0.
        # Issue order: K, K_rope, V — so waiting for vmcnt=V.vm
        # ensures K and K_rope are done while V may still be in-flight.
        _ = k_buffer.load_from_dram[0]()
        _ = k_rope_buffer.load_from_dram[0]()
        _ = v_buffer.load_from_dram[0]()

        comptime has_interior_full_mask = Self.mask_t != CausalMask

        @inline(.always)
        def prefetch_next_tile[
            next_slot: Int
        ](
            mut k_buffer: KBufT,
            mut k_rope_buffer: KRopeBufT,
            mut v_buffer: VBufT,
        ):
            # The pipeline prefetches one tile ahead even from the final tile;
            # re-read the last valid tile rather than read past `num_keys`.
            k_buffer.kv_cache_iter.clamp_prefetch_tile()
            k_rope_buffer.kv_cache_iter.clamp_prefetch_tile()
            v_buffer.kv_cache_iter.clamp_prefetch_tile()
            _ = k_buffer.load_from_dram[next_slot]()
            _ = k_rope_buffer.load_from_dram[next_slot]()
            _ = v_buffer.load_from_dram[next_slot]()

        @inline(.always)
        def process_tile[
            slot: Int, has_next: Bool
        ]() {mut k_buffer, mut k_rope_buffer, mut self, mut v_buffer, imm}:
            comptime next_slot = 1 - slot

            # One barrier per tile: this tile's K, K_rope and V were all issued
            # by the previous tile's prefetch, so the only outstanding DMA here
            # is this slot's and `vmcnt(0)` is the exact count. One full drain
            # plus barrier then covers both the RAW on V and the WAR of the
            # mid-tile prefetch into `next_slot`, which every wave reaches only
            # after its previous tile's LDS reads (consumed by the PV MFMAs)
            # have landed.
            block_sync_lds_direct_load[vmcnt=0]()
            barrier()

            # IGroupLP: co-issue the softmax exp with the next tile's QK MMA to
            # hide the online-softmax latency on this overlap-bound kernel.
            _iglp_opt[AMDIGLPStrategy.MFMA_EXP_INTERLEAVE]()

            # Skip fully masked tiles for non-causal masks.
            comptime if has_interior_full_mask:
                var tile_status = self.mask_status(self.kv_start_row)
                if tile_status == TileMaskStatus.FULL_MASK:
                    self.kv_start_row += UInt32(Self.BN)
                    self.mask_advance()
                    comptime if has_next:
                        prefetch_next_tile[next_slot](
                            k_buffer, k_rope_buffer, v_buffer
                        )
                        barrier()
                    return

            # Load K (nope) from shared -> registers and compute Q_nope @ K^T
            k_buffer.load_from_shared(slot)
            mma_qk_nope()

            # Load K_rope from shared -> registers and compute Q_rope @ K_rope^T
            k_rope_buffer.load_from_shared(slot)
            mma_qk_rope()

            # Prefetch next K+K_rope+V tile before softmax to hide latency.
            comptime if has_next:
                prefetch_next_tile[next_slot](k_buffer, k_rope_buffer, v_buffer)

            # Online softmax: deferred-scale variant that emits packed FMAs
            # (`v_pk_fma_f32 score, scale, neg_scaled_max`) instead of the
            # separate `v_pk_add (score - max)` + `v_pk_mul (* scale)` pair.
            # `arg_bias_log2` is the fp8 P pre-scale, folded into `exp2`.
            self.online_softmax_step_0_pkfma[0, arg_bias_log2=p_exp_bias_log2]()
            self.online_softmax_step_1_pkfma[0, arg_bias_log2=p_exp_bias_log2]()

            v_buffer.load_from_shared(slot)

            self.online_softmax_update_output()

            mma_pv()

        # Main loop: process tiles in pairs (double-buffered).
        for _ in range(num_pairs):
            process_tile[0, True]()
            process_tile[1, True]()

        # Remainder: last odd tile from slot 0.
        if num_tiles % 2 != 0:
            process_tile[0, False]()

        # P carries the `2^p_scale_log2` fp8 pre-scale inside its exp2
        # argument, so the row sum is that many times too large. Divide it back
        # out (an exact power of two) before the split-K LSE and the output
        # denominator read it.
        comptime if p_scale_in_exp:
            comptime for m_mma in range(Self.num_m_mmas):
                self.softmax.rowsum_tensor[
                    m_mma, 0
                ] = self.softmax.rowsum_tensor[m_mma, 0] * Scalar[
                    Self.accum_type
                ](
                    1.0 / Float64(1 << p_scale_log2)
                )

        comptime if split_kv:
            self.store_split_lse(
                lse_ptr,
                split_idx,
                num_splits,
                lse_total_rows,
                lse_row_base,
                empty=False,
            )

        # Apply final softmax denominator and store.
        self.out_reg_buffer.apply_softmax_denominator[
            p_scale_log2=type_of(self.p_reg_buffer).p_scale_log2
        ](self.softmax.rowsum_tensor)
        self.store_output()
