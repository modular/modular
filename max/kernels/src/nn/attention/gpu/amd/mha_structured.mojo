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
"""MHA prefill kernel for gfx950 with structured scheduling.

Supports depth=64, 128, 256. Uses TileTensor throughout for register
and SMEM tile management, with TiledMmaOp for MMA dispatch.
"""

from std.math import ceildiv
from std.sys import simd_width_of, llvm_intrinsic, get_defined_bool
from std.sys.intrinsics import readfirstlane
from std.gpu import WARP_SIZE
from std.gpu import warp_id as get_warp_id
from std.gpu.sync import (
    schedule_barrier,
    s_waitcnt,
)
from layout.swizzle import Swizzle
from .mma import TiledMmaOp
from std.memory import bitcast
from std.utils import IndexList
from std.utils.numerics import get_accum_type

from .mha_gfx950 import Attention
from .kv_buffer import (
    KVBuffer as StructuredKVBuffer,
    KVCacheIterator,
)


# --- Synchronization primitives (same as mha_gfx950.mojo) ---


@always_inline
def set_priority[priority: Int]():
    llvm_intrinsic["llvm.amdgcn.s.setprio", NoneType](Int16(priority))


@always_inline
def barrier[
    *, schedule_barrier_before: Bool = True, schedule_barrier_after: Bool = True
]():
    comptime if schedule_barrier_before:
        schedule_barrier()

    llvm_intrinsic["llvm.amdgcn.s.barrier", NoneType]()

    comptime if schedule_barrier_after:
        schedule_barrier()


@always_inline
def block_sync_lds_direct_load[
    *,
    vmcnt: UInt32 = 0,
]():
    s_waitcnt[vmcnt=vmcnt]()


# =============================================================
# TileTensor-based structured MHA prefill kernel
# =============================================================


__extension Attention:
    def mha_prefill_structured(mut self):
        """TileTensor-based 4-cluster MHA prefill kernel for gfx950."""

        # --- Assertions ---
        comptime assert Self.BK == 32, "BK must be 32"
        comptime assert (
            Self.depth == 64 or Self.depth == 128 or Self.depth == 256
        ), "structured kernel supports depth=64, 128, 256"
        comptime assert (
            Self.attention_config_t.double_buffer
        ), "mha_prefill_structured requires double_buffer=True"

        # Pre-scale Q by (scale * log2e) for depth<=128 so QK matmul
        # produces scaled scores directly.  Disabled for depth>128 to
        # work around LLVM MI Scheduler crash (isReg assertion in
        # RewriteMFMAFormStage).
        comptime prescale_q = get_defined_bool[
            "PRESCALE_Q", True
        ]() and Self.depth <= 128
        comptime num_qk_strips = Self.depth // Self.BK

        # --- Buffer init ---

        var warp_id = UInt32(
            readfirstlane(bitcast[DType.int32](UInt32(get_warp_id())))
        )
        var k_buffer = StructuredKVBuffer[
            mma_shape=Self.mma_shape,
            k_group_size=Self.k_group_size,
            swizzle=Swizzle(3, 0, 4) if Self.mma_shape[0]
            == 32 else Optional[Swizzle](None),
            BN=Self.BN,
            WN=Self.WN,
            BK=Self.BK,
            num_threads=Self.num_threads,
            depth=Self.depth,
            kv_num_heads=Self.num_heads // Self.group,
            transpose=True,
            full_kv=Self.attention_config_t.full_kv,
        ](
            self.k,
            UInt(self.batch_idx),
            UInt(self.kv_head_idx()),
            self.smem_manager.get_k_ptr[type_of(self.k).dtype](),
            UInt(self.num_keys),
            warp_id,
        )

        var v_buffer = StructuredKVBuffer[
            mma_shape=Self.mma_shape,
            k_group_size=Self.k_group_size,
            swizzle=None,
            BN=Self.BN,
            WN=Self.WN,
            BK=Self.BK,
            num_threads=Self.num_threads,
            depth=Self.depth,
            kv_num_heads=Self.num_heads // Self.group,
            transpose=False,
            full_kv=Self.attention_config_t.full_kv,
        ](
            self.v,
            UInt(self.batch_idx),
            UInt(self.kv_head_idx()),
            self.smem_manager.get_v_ptr[type_of(self.v).dtype](),
            UInt(self.num_keys),
            warp_id,
        )

        comptime accum_type = get_accum_type[type_of(self.k).dtype]()

        # QK uses pure TileTensor MMA op.
        comptime QKMmaOp = TiledMmaOp[
            accum_type,
            Self.q_type,
            Self.mma_shape,
            group_size=Self.k_group_size,
            transpose_b=True,
        ]

        # PV MMA op — same TiledMmaOp, PRegisterBuffer.mma_tile handles
        # the f32→bf16 cast+interleave via whole-vector SIMD ops.
        comptime PVMmaOp = TiledMmaOp[
            accum_type,
            Self.q_type,
            Self.mma_shape,
            group_size=Self.k_group_size,
            transpose_b=True,
        ]

        # =============================================================
        # MMA helpers
        # =============================================================

        @always_inline
        @parameter
        def mma_qk_strip[stage: Int, strip: Int]():
            comptime for k_mma in range(Self.num_k_mmas2):
                QKMmaOp.mma[swap_a_b=Self.swap_a_b](
                    self.q_buffer.mma_tile[strip, Int(k_mma)](),
                    k_buffer.get_mma_tile[Int(k_mma), strip](),
                    self.p_reg_buffer.stage_tile[stage](),
                )

        @always_inline
        @parameter
        def mma_pv_strip[stage: Int, strip: Int]():
            comptime for k_mma in range(v_buffer.num_k_mmas2):
                PVMmaOp.mma[swap_a_b=Self.swap_a_b](
                    self.p_reg_buffer.mma_tile[strip, k_mma, stage](),
                    v_buffer.get_mma_tile[k_mma, strip](),
                    self.out_reg_buffer.reg_tile,
                )

        @always_inline
        @parameter
        def mma_qk[stage: Int]():
            self.zero_p_buffer[stage]()
            comptime for i in range(num_qk_strips):
                mma_qk_strip[stage, i]()

        @always_inline
        @parameter
        def mma_pv_incremental[stage: Int](slot: UInt):
            comptime for i in range(Self.BN // Self.BK):
                v_buffer.load_from_shared[i](slot)
                mma_pv_strip[stage, i]()

        # =============================================================
        # Softmax micro-ops
        # =============================================================

        @always_inline
        @parameter
        def softmax_exp_even[stage: Int]():
            var score_tile = self.p_reg_buffer.stage_tile[stage]()
            self.softmax.exp[start=0, stride=2](score_tile)

        @always_inline
        @parameter
        def softmax_exp_odd[stage: Int]():
            var score_tile = self.p_reg_buffer.stage_tile[stage]()
            self.softmax.exp[start=1, stride=2](score_tile)

        @always_inline
        @parameter
        def softmax_qk_sum[stage: Int]():
            var score_tile = self.p_reg_buffer.stage_tile[stage]()
            var warp_scratch = self.warp_scratch_tensor.tile[
                2 * Self.num_warps_n, Self.WM
            ](0, 0)
            self.softmax.calculate_qk_sum(score_tile, warp_scratch)

        @always_inline
        @parameter
        def softmax_sum_correction():
            self.softmax.apply_sum_correction()

        @always_inline
        @parameter
        def softmax_update_sum():
            self.softmax.update_sum_additive()

        @always_inline
        @parameter
        def softmax_apply_mask[stage: Int]():
            comptime if prescale_q:
                self.apply_mask[stage, scale=False]()
            else:
                self.apply_mask[stage]()

        @always_inline
        @parameter
        def softmax_qk_max[stage: Int]():
            var score_tile = self.p_reg_buffer.stage_tile[stage]()
            var warp_scratch = self.warp_scratch_tensor.tile[
                2 * Self.num_warps_n, Self.WM
            ](0, 0)
            self.softmax.calculate_qk_max(score_tile, warp_scratch)

        @always_inline
        @parameter
        def softmax_correction() -> Bool:
            var needs_rescale = False
            comptime num_colwise_tiles = Self.num_m_mmas
            comptime frag_num_rows = (Self.fragment_layout.shape[0].value())
            comptime for col_tile in range(num_colwise_tiles):
                comptime for row in range(frag_num_rows):
                    if rebind[Scalar[Self.accum_type]](
                        self.softmax.score_frag_rowmax[col_tile, row]
                    ) > rebind[Scalar[Self.accum_type]](
                        self.softmax.rowmax_tensor[col_tile, row]
                    ):
                        needs_rescale = True
            if needs_rescale:
                self.softmax.calculate_correction()
            return needs_rescale

        @always_inline
        @parameter
        def softmax_update_output():
            self.softmax.update_output(self.out_reg_buffer.reg_tile)

        @always_inline
        @parameter
        def softmax_update_max():
            self.softmax.update_max()

        # =============================================================
        # process_tile — hand-optimized interleaving
        # =============================================================

        @always_inline
        @parameter
        def process_tile[
            stage: Int, has_next: Bool
        ](pending_scale: Bool) -> Bool:
            comptime prev = 1 - stage
            var needs_rescale = False

            block_sync_lds_direct_load[vmcnt=v_buffer.vm_instrs_per_load]()
            barrier[
                schedule_barrier_before=False,
                schedule_barrier_after=False,
            ]()

            # C0 [COMPUTE]: K LDS + QK MFMAs + finish_softmax(prev)
            k_buffer.load_from_shared(UInt(stage))

            var warp_scratch = self.warp_scratch_tensor.tile[
                2 * Self.num_warps_n, Self.WM
            ](0, 0)
            var prev_tile = self.p_reg_buffer.stage_tile[prev]()

            self.zero_p_buffer[stage]()
            mma_qk_strip[stage, 0]()

            # Interleave finish_softmax(prev) between QK strips.
            # For depth>=128 (>=4 strips), spread across strips 0-3.
            # For depth=64 (2 strips), do all softmax after strip 1.
            comptime if num_qk_strips > 2:
                self.softmax.exp[start=1, stride=2](prev_tile)

            mma_qk_strip[stage, 1]()

            comptime if num_qk_strips > 2:
                self.softmax.calculate_qk_sum(prev_tile, warp_scratch)

            comptime if num_qk_strips > 2:
                mma_qk_strip[stage, 2]()
                if pending_scale:
                    softmax_sum_correction()
                softmax_update_sum()

            comptime if num_qk_strips > 3:
                mma_qk_strip[stage, 3]()

            # Extra strips for depth=256 (strips 4..7).
            comptime for i in range(4, num_qk_strips):
                mma_qk_strip[stage, i]()

            # For depth=64, finish_softmax(prev) after all QK strips.
            comptime if num_qk_strips <= 2:
                self.softmax.exp[start=1, stride=2](prev_tile)
                self.softmax.calculate_qk_sum(prev_tile, warp_scratch)
                if pending_scale:
                    softmax_sum_correction()
                softmax_update_sum()

            # C2 [COMPUTE]: V LDS (incr) + PV MFMAs + start_softmax + exp_even
            set_priority[1]()

            var cur_tile = self.p_reg_buffer.stage_tile[stage]()

            v_buffer.load_from_shared[0](UInt(prev))
            mma_pv_strip[prev, 0]()

            v_buffer.load_from_shared[1](UInt(prev))

            comptime if prescale_q:
                self.apply_mask[stage, scale=False]()
            else:
                self.apply_mask[stage]()
            self.softmax.calculate_qk_max(cur_tile, warp_scratch)

            mma_pv_strip[prev, 1]()

            comptime num_colwise_tiles = Self.num_m_mmas
            comptime frag_num_rows = (Self.fragment_layout.shape[0].value())
            comptime for col_tile in range(num_colwise_tiles):
                comptime for row in range(frag_num_rows):
                    if rebind[Scalar[Self.accum_type]](
                        self.softmax.score_frag_rowmax[col_tile, row]
                    ) > rebind[Scalar[Self.accum_type]](
                        self.softmax.rowmax_tensor[col_tile, row]
                    ):
                        needs_rescale = True

            if needs_rescale:
                self.softmax.calculate_correction()
                self.softmax.update_output(self.out_reg_buffer.reg_tile)

            self.softmax.update_max()

            self.softmax.exp[start=0, stride=2](cur_tile)

            set_priority[0]()

            # DMA next tile.
            comptime if has_next:
                barrier[
                    schedule_barrier_before=False,
                    schedule_barrier_after=False,
                ]()
                _ = k_buffer.load_from_dram[prev]()
                _ = v_buffer.load_from_dram[prev]()

            return needs_rescale

        # =============================================================
        # Iteration bounds
        # =============================================================

        var score_row = UInt32(self.mask_block_row + UInt32(self.start_pos))
        var start_col = self.mask.start_column[Self.BM, Self.BN, 1](score_row)
        var num_tiles = Int(
            self.mask.last_masked_set_end[Self.BM, Self.BN, 1](
                score_row, UInt32(self.num_keys)
            )
        )

        k_buffer.kv_cache_iter.tile_start_row = Int(start_col)
        v_buffer.kv_cache_iter.tile_start_row = Int(start_col)
        self.kv_start_row = start_col
        self.mask_warp_col += start_col

        comptime if prescale_q:
            self.scale_q_buffer()

        # =============================================================
        # PROLOGUE: DMA K[0]+V[0], QK[0], softmax_start + exp_even
        # =============================================================

        _ = k_buffer.load_from_dram[0]()
        _ = v_buffer.load_from_dram[0]()

        block_sync_lds_direct_load[vmcnt=0]()
        barrier[
            schedule_barrier_before=False,
            schedule_barrier_after=False,
        ]()

        k_buffer.load_from_shared(0)
        mma_qk[0]()

        # Prefetch tile 1 during softmax to hide DMA latency.
        if num_tiles > 1:
            _ = k_buffer.load_from_dram[1]()
            _ = v_buffer.load_from_dram[1]()

        # Prologue start_softmax + exp_even.
        softmax_apply_mask[0]()
        softmax_qk_max[0]()
        var needs_rescale = softmax_correction()
        if needs_rescale:
            softmax_update_output()
        softmax_update_max()
        softmax_exp_even[0]()
        var pending_scale = needs_rescale

        # =============================================================
        # STEADY STATE: tiles 1 .. num_tiles-1
        # =============================================================
        if num_tiles > 1:
            var num_steady = num_tiles - 1
            var num_pairs = num_steady // 2

            for _ in range(num_pairs):
                pending_scale = process_tile[1, True](pending_scale)
                pending_scale = process_tile[0, True](pending_scale)

            if num_steady % 2 != 0:
                pending_scale = process_tile[1, False](pending_scale)

            # --- EPILOGUE: finish_softmax on last tile ---
            if (num_tiles - 1) % 2 == 0:
                softmax_exp_odd[0]()
                softmax_qk_sum[0]()
                if pending_scale:
                    softmax_sum_correction()
                softmax_update_sum()
                block_sync_lds_direct_load[vmcnt=0]()
                barrier[
                    schedule_barrier_before=False,
                    schedule_barrier_after=False,
                ]()
                mma_pv_incremental[0](0)
            else:
                softmax_exp_odd[1]()
                softmax_qk_sum[1]()
                if pending_scale:
                    softmax_sum_correction()
                softmax_update_sum()
                block_sync_lds_direct_load[vmcnt=0]()
                barrier[
                    schedule_barrier_before=False,
                    schedule_barrier_after=False,
                ]()
                mma_pv_incremental[1](1)
        else:
            # Single-tile path.
            softmax_exp_odd[0]()
            softmax_qk_sum[0]()
            softmax_update_sum()
            mma_pv_incremental[0](0)

        self.out_reg_buffer.apply_softmax_denominator(
            self.softmax.rowsum_tensor
        )
        self.store_output()
