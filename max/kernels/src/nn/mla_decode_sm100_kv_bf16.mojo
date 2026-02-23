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

from math import ceildiv
from sys import size_of
import gpu.primitives.warp as warp
from gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    barrier,
    block_idx,
    thread_idx,
    warp_id,
)
from gpu.globals import WARPGROUP_SIZE
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu.primitives.grid_controls import launch_dependent_grids
from gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from gpu.memory import AddressSpace, external_memory, fence_async_view_proxy
from gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_fence_before,
    tcgen05_release_allocation_lock,
)
from layout.tma_async import (
    SharedMemBarrier,
)
from memory import bitcast
from nn.mha_fa3_utils import (
    OptionalPointer,
)
from nn.mha_mask import MHAMask
from nn.mha_operand import MHAOperand
from nn.mha_score_mod import ScoreModTrait
from utils.numerics import get_accum_type
from utils.static_tuple import StaticTuple

from nn.mha_sm100_2q import (
    elect,
)
from nn.mha_fa3_utils import KVTMATile

from nn.mla_decode_sm100_utils import (
    MLA_SM100_Decode_Config,
    MLA_SM100_Decode_Common,
    QOTMATile,
    tma_tile_qo,
    MLA_Decode_Pack,
    num_matrix_view_rows_decode,
    OffsetPosition,
    SharedMemPointer,
    MBarType,
    SharedMemTensor,
    KVPipelineGeneric,
    DecodeSM100MiscMBars,
    DecodeSProducer,
    DecodePConsumer,
    DecodeOProducer,
    OutPipeline,
    DecodeOutProducer,
    DecodeKVProducer,
    DecodeKVConsumer,
    DecodeSM100QKTSS,
    DecodeSM100PVSS,
)


# ------------------------------------------------------------------------------
# MLA decoding kernel struct for SM100
# ------------------------------------------------------------------------------
struct MLA_SM100_Decode_KV_BF16[
    q_type: DType,
    KVLUTType: MHAOperand,
    output_type: DType,
    SplitAccumType: OptionalPointer,
    MaskType: MHAMask,
    ScoreModType: ScoreModTrait,
    config: MLA_SM100_Decode_Config,
    use_score_mod: Bool,
    ValidLengthType: OptionalPointer,
    _is_cache_length_accurate: Bool = False,
    ragged: Bool = False,
](TrivialRegisterPassable):
    comptime kv_type = Self.KVLUTType.dtype
    comptime AccumType = get_accum_type[Self.q_type]()
    # 576 / 64 = 9
    comptime NumQKBlocks = Self.config.padded_q_depth // Self.config.BN
    # 512 / 64 = 8
    comptime NumVOBlocks = Self.config.padded_depth // Self.config.BN
    # 64 * 64 = 4096
    comptime BlockElems = Self.config.BM * Self.config.BN
    # 2 bytes for float16
    comptime bytes_per_element = size_of[Self.q_type]()
    # the stage element is the same for both K and V
    comptime KVStageElems = Self.NumQKBlocks * Self.BlockElems
    comptime output_tile_width = (Self.config.BN // 2) * (
        4 // size_of[Self.output_type]()
    )
    comptime UMMAQKTSS = DecodeSM100QKTSS[
        operand_type = Self.q_type,
        accum_type = Self.AccumType,
        config = Self.config,
    ]
    comptime UMMAPVSS = DecodeSM100PVSS[
        operand_type = Self.q_type,
        accum_type = Self.AccumType,
        config = Self.config,
    ]

    comptime Common_MLA_Op = MLA_SM100_Decode_Common[
        Self.q_type,
        Self.KVLUTType,
        Self.output_type,
        Self.SplitAccumType,
        Self.MaskType,
        Self.ScoreModType,
        Self.config,
        Self.use_score_mod,
        Self.ValidLengthType,
        Self._is_cache_length_accurate,
        Self.ragged,
    ]

    # --------------------------------------------------------------------------
    # MLA decoding main kernel function
    # --------------------------------------------------------------------------
    #    3 Warpgroups: Softmax WG (warps 0-3), Correction WG (warps 4-7),
    #                  MMA+Load+Store WG (warps 8-11)
    #    Warp assignments within WG2: warp 8 = Load, warp 9 = MMA QK,
    #                                 warp 10 = MMA PV, warp 11 = Store
    #
    #    KSlot0 (tile 0)                KSlot1 (tile 1)
    #         |                              |
    #         V                              V
    #    UMMA QK → S0 (warp 9)         UMMA QK → S1 (warp 9)
    #         |                              |
    #   arrive mbar_s0                arrive mbar_s1
    #         |                              |
    #         |---- Softmax WG (warps 0-3) ----|
    #         |                                |
    #         V                                V
    #       wait_s0                          wait_s1
    #       S0 → P0                          S1 → P1
    #         |                                |
    #         |---- Correction WG (warps 4-7) ----|
    #         |   (scale O by correction factor    |
    #         |    before new P*V accumulation)    |
    #         V                                    V
    #    UMMA PV → O (warp 10)          UMMA PV → O (warp 10)
    #    (P0 * V → O accumulate)        (P1 * V → O accumulate)
    #         |                                    |
    #       arrive mbar_o                    arrive mbar_o
    #         |                                    |
    #       corr_done_bars signal -----------------|
    #         |                                    |
    #       wait_O_filled                    wait_O_filled
    #         |                                    |
    #       wait_out                           wait_out
    #         |                                    |
    #       Store warp (warp 11)             Store warp
    #

    # --------------------------------------------------------------------------
    # MLA decoding SMEMDescriptors for Q, K, V, P
    # --------------------------------------------------------------------------

    @staticmethod
    @__llvm_arg_metadata(q_tma, `nvvm.grid_constant`)
    @__llvm_arg_metadata(k_tma, `nvvm.grid_constant`)
    @__llvm_arg_metadata(o_tma, `nvvm.grid_constant`)
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.config.num_threads)
        )
    )
    fn kernel(
        q_tma: QOTMATile[
            dtype = Self.q_type,
            BM = Self.config.BM,  # tile_m =64
            BK = Self.config.BK0,  # tile_n =576
            swizzle_mode = Self.config.swizzle_mode,
        ],
        k_tma: KVTMATile[
            dtype = Self.kv_type,
            swizzle_mode = Self.config.kv_tma_swizzle_mode,
            BN = Self.config.BK1,  # tile_m =64
            BK = Self.config.BK0,  # tile_n =576
        ],
        o_tma: QOTMATile[
            dtype = Self.output_type,
            BM = Self.config.out_rows,
            BK = Self.config.BN,
            swizzle_mode = Self.config.swizzle_mode,
        ],
        kv_lut: Self.KVLUTType,
        scale: Float32,
        batch_size: Int,
        q_max_seq_len: Int,
        num_partitions: Int,
        max_cache_valid_length: Int,  # longest KV cache entry,
        mla_decode_pack: MLA_Decode_Pack[
            ValidLengthType = Self.ValidLengthType,
            MaskType = Self.MaskType,
            ScoreModType = Self.ScoreModType,
            SplitAccumType = Self.SplitAccumType,
        ],
        scales_ptr: UnsafePointer[Scalar[DType.float32], origin=MutAnyOrigin],
    ):
        comptime num_reg_softmax = 192
        comptime num_reg_correction = 184
        comptime num_reg_other = 112
        mask = mla_decode_pack.mask
        score_mod = mla_decode_pack.score_mod
        valid_length = mla_decode_pack.valid_length
        var lse_accum_split_ptr = mla_decode_pack.lse_accum_split_ptr
        var offset_position = OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
        ](
            kv_lut,
            valid_length.value(),
            q_max_seq_len,
            num_partitions,
            batch_size,
        )

        # Early exit for split-K: CTAs with no work (num_keys_this_split == 0)
        # must still write -inf LSE, zero o_accum_split, and call
        # launch_dependent_grids() to fulfill the PDL contract with the
        # combine kernel.  Skipping launch_dependent_grids() causes the
        # combine kernel to hang, leading to CUDA_ERROR_ILLEGAL_ADDRESS.
        comptime if Self.config.decoding_warp_split_k:
            if offset_position.num_keys_this_split == 0:
                Self.Common_MLA_Op.pdl_early_exit(
                    offset_position.split_idx,
                    offset_position.batch_idx,
                    offset_position.max_seq_len,
                    offset_position.out_row_offset,
                    batch_size,
                    lse_accum_split_ptr,
                    o_tma,
                )
                return

        # EARLY EXIT: Skip blocks beyond actual sequence length for this batch
        # In ragged mode with split-K, q_max_seq_len can be > 1 (up to 8).
        # block_idx.y ranges from 0 to q_max_seq_len-1, but some sequences
        # may have fewer tokens. CTAs with block_idx.y >= seq_len must still
        # fulfill the PDL contract (write -inf LSE, zero o_accum_split, and
        # call launch_dependent_grids) or the combine kernel will hang.
        comptime if Self.ragged:
            # In ragged mode, block_idx.y is the query token index (0 to q_max_seq_len-1)
            # But this batch might have fewer tokens than q_max_seq_len
            if Int(block_idx.y) >= offset_position.seq_len:
                comptime if Self.config.decoding_warp_split_k:
                    Self.Common_MLA_Op.pdl_early_exit(
                        offset_position.split_idx,
                        offset_position.batch_idx,
                        offset_position.max_seq_len,
                        offset_position.out_row_offset,
                        batch_size,
                        lse_accum_split_ptr,
                        o_tma,
                    )

                return  # This query position doesn't exist for this batch
        q_smem = external_memory[
            Scalar[Self.q_type],
            address_space = AddressSpace.SHARED,
            alignment=128,
            name="mha_dynamic_shared_memory",
        ]()
        var kv_smem = (q_smem + Self.BlockElems * Self.NumQKBlocks).bitcast[
            Scalar[Self.kv_type]
        ]()
        comptime kv_total_stages = Self.config.num_kv_stages
        # to reuse the K for V as well, we break KV as 9 stages of 64x64 to cover 64x576
        comptime kv_smem_total = Self.BlockElems * Self.NumQKBlocks * kv_total_stages

        # we need to use the KSmem for out pointer
        # We move P to the last slot of KV pipeline SO now we have tile of 64x
        # 32 of float or 64x64 of FP16 to save output into
        # tiles in SMEM and smooth the pipeline for the next batch if we use splitk
        var out_smem_start = kv_smem
        # there is potential to have two Tmem for S, because we have two K so we can
        # unblock the MMA while loading S to reg for softmax
        # if it was splitk we need to use the extra P slot. If not we need
        # to clear the KV slot before starting the max because KV slot is used by
        # MMA/load when max is valid.
        var out_smem_total = kv_smem_total

        var out_smem = out_smem_start.bitcast[Scalar[Self.output_type]]()

        var max_smem = (out_smem + out_smem_total).bitcast[
            Scalar[Self.AccumType]
        ]()

        var li_smem = (
            max_smem + WARPGROUP_SIZE
        )  # 128 x1 for SMEM correction for Softmax
        #  Now we have to define MBARS for the kernel
        var mbar_base: MBarType = (li_smem + WARPGROUP_SIZE).bitcast[
            SharedMemBarrier
        ]()

        var mbar_q: MBarType = mbar_base  # q uses 0
        var mbar_kv_base: MBarType = mbar_base + 1  # barrier total[1]

        var kv_pipeline = KVPipelineGeneric[
            num_kv_stages = Self.config.num_kv_stages,  # 2
            num_qk_stages=1,
            num_producer=1,
            num_consumer=2,
        ](mbar_kv_base)

        # Move mbar_base to the first free barrier *after* KV:
        mbar_base = mbar_kv_base + kv_pipeline.num_mbars()  # kv uses 1..4
        # Move mbar_base to the first free barrier *after* k done:
        var s_bars = DecodeSM100MiscMBars[
            num_stages=2, num_producer=1, num_consumer=WARPGROUP_SIZE
        ](
            mbar_base
        )  # S uses 5..8
        mbar_base = s_bars.end()  # barrier total[9]
        var p_bars = DecodeSM100MiscMBars[
            num_stages=2, num_producer=WARPGROUP_SIZE, num_consumer=1
        ](
            mbar_base
        )  # P uses 9 .. 12
        mbar_base = p_bars.end()  # barrier total [13]
        var o_bars = DecodeSM100MiscMBars[
            num_stages=2, num_producer=1, num_consumer=WARPGROUP_SIZE
        ](
            mbar_base
        )  # O uses 13..16
        mbar_base = o_bars.end()  # barrier total [17]
        # C pipeline, Softmax -> Correction
        var c_bars = DecodeSM100MiscMBars[
            num_stages=1,
            num_producer=WARPGROUP_SIZE,
            num_consumer=WARPGROUP_SIZE,
        ](
            mbar_base
        )  # C uses 17..18
        mbar_base = c_bars.end()  # barrier total [19]
        # Correction done barrier: Correction -> Softmax direction
        # Signals when Correction exits its while loop (all corrections done)
        # 2-stage pipeline to overlap correction with next softmax iteration
        var corr_done_bars = DecodeSM100MiscMBars[
            num_stages=2,
            num_producer=WARPGROUP_SIZE,
            num_consumer=WARPGROUP_SIZE,
        ](
            mbar_base
        )  # corr_done uses 19..22
        mbar_base = corr_done_bars.end()  # barrier total [23]
        # We need (num_out_stages * 2) more barriers for the out pipeline.
        # num_out_stages = (Depth/BN) / blocks_per_stage = 8/2 = 4, so 4*2 = 8.
        comptime OutPipeType = DecodeOutProducer[Self.output_type, Self.config]
        var out_pipeline = OutPipeline[
            num_out_stages = OutPipeType.num_out_stages,
            num_producer=WARPGROUP_SIZE,
            num_consumer=1,
        ](
            mbar_base
        )  # Write uses 23 + (num_out_stages)*2
        mbar_base += (
            out_pipeline.num_mbars()
        )  # barrier total [23 + (num_out_stages)*2]
        var warp_idx = UInt32(warp.broadcast(warp_id()))
        var ptr_tmem_addr = (mbar_base).bitcast[UInt32]()
        is_leader = elect() != 0
        if warp_idx == 8:
            if is_leader:
                mbar_q[].init(1)
                # only one thread will load the Q
                kv_pipeline.init()
                s_bars.init()
                p_bars.init()
                o_bars.init()
                c_bars.init()
                out_pipeline.init()
                corr_done_bars.init()
                q_tma.prefetch_descriptor()
                k_tma.prefetch_descriptor()
                o_tma.prefetch_descriptor()
        elif warp_idx == 9:
            tcgen05_alloc[Self.config.cta_group](
                ptr_tmem_addr, Self.config.sm100_tmem_cols
            )
        barrier()

        if warp_idx < 4:  # softmax warpgroup
            warpgroup_reg_alloc[num_reg_softmax]()
            Self.Common_MLA_Op.Softmax(
                ptr_tmem_addr[0],
                s_bars,
                p_bars,
                kv_smem.bitcast[Scalar[Self.q_type]](),
                max_smem,
                li_smem,
                out_smem,
                c_bars,
                corr_done_bars,
                out_pipeline,
                offset_position,
                scale,
                mask,
                score_mod,
                prompt_idx=UInt32(offset_position.batch_idx),
                max_seq_len=UInt32(q_max_seq_len),
                lse_accum_split_ptr=lse_accum_split_ptr,
                batch_size=batch_size,
            )
        elif warp_idx >= 4 and warp_idx < 8:  # correction warpgroup
            warpgroup_reg_alloc[num_reg_correction]()
            Self.Common_MLA_Op.Correction(
                ptr_tmem_addr[0],
                o_bars,
                c_bars,
                corr_done_bars,
                offset_position,
            )
        else:
            warpgroup_reg_dealloc[num_reg_other]()
            if warp_idx == 8:
                Self.load(
                    q_tma,
                    k_tma,
                    kv_lut,
                    q_smem,
                    kv_smem,
                    mbar_q,
                    kv_pipeline,
                    offset_position,
                )
            elif warp_idx == 9:
                Self.mmaQK(
                    ptr_tmem_addr[0],
                    q_smem,
                    (kv_smem).bitcast[Scalar[Self.q_type]](),
                    mbar_q,
                    s_bars,
                    kv_pipeline,
                    offset_position,
                )
            elif warp_idx == 10:
                Self.mmaPV(
                    ptr_tmem_addr[0],
                    (kv_smem).bitcast[Scalar[Self.q_type]](),
                    p_bars,
                    o_bars,
                    kv_pipeline,
                    offset_position,
                )
            elif warp_idx == 11:
                Self.Common_MLA_Op.store(
                    out_pipeline, out_smem, o_tma, offset_position
                )
        barrier()

        # PDL: Signal that this CTA is done so dependent grids (combine kernel) can start.
        # This must be called by all threads in the CTA after all work is complete.
        comptime if Self.config.decoding_warp_split_k:
            launch_dependent_grids()

        if warp_idx == 9:
            tcgen05_release_allocation_lock[Self.config.cta_group]()
            tcgen05_dealloc[Self.config.cta_group](
                ptr_tmem_addr[0], Self.config.sm100_tmem_cols
            )

    # --------------------------------------------------------------------------
    # MLA decoding load_q and load_kv function
    # --------------------------------------------------------------------------
    @staticmethod
    @always_inline
    fn load(
        q_tma: QOTMATile[
            dtype = Self.q_type,
            BM = Self.config.BM,  # tile_m =64
            BK = Self.config.BK0,  # tile_n =576
            swizzle_mode = Self.config.swizzle_mode,
        ],
        k_tma: KVTMATile[
            dtype = Self.kv_type,
            swizzle_mode = Self.config.kv_tma_swizzle_mode,
            BN = Self.config.BK1,  # tile_m =64
            BK = Self.config.BK0,  # tile_n =576
        ],
        kv_lut: Self.KVLUTType,
        q_smem: SharedMemPointer[Scalar[Self.q_type]],
        kv_smem: SharedMemPointer[Scalar[Self.kv_type]],
        mbar_q: MBarType,
        kv_pipeline: KVPipelineGeneric[
            num_kv_stages = Self.config.num_kv_stages,  # 2
            num_qk_stages=1,
            num_producer=1,
            num_consumer=2,
        ],
        offset_position: OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
        ],
    ):
        # Early exit if this split has no work (prevents producer/consumer deadlock)
        if offset_position.num_keys_this_split == 0:
            return

        num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN
        )

        var kv_prod = DecodeKVProducer[Self.kv_type, Self.config](
            kv_pipeline, kv_smem
        )
        var elect_mask = elect()
        var is_leader = elect_mask != 0
        var row: UInt = UInt(offset_position.q_row_offset)
        # Start KV from kv_start_row for split-K support
        var kv_row: UInt32 = UInt32(offset_position.kv_start_row)
        # Clamp kv_row to prevent OOB lookup_table access on the last tile.
        var num_keys_u32 = UInt32(offset_position.num_keys)
        kv_row = min(kv_row, max(num_keys_u32, UInt32(1)) - 1)
        var kv_gmem_row: UInt32 = kv_lut.row_idx(
            UInt32(offset_position.batch_idx), kv_row
        )
        if is_leader:
            # this is the total bytes expected to be transferred to the mbar for Q and K0
            mbar_q[].expect_bytes(
                Int32(
                    Self.config.BM
                    * Self.config.q_depth
                    * size_of[Self.q_type]()
                )
            )
            Self.Common_MLA_Op.load_q(q_tma, q_smem, mbar_q, UInt(0), row)

        var k0_bar: MBarType = kv_prod.producer_mbar[qk_stage=0]()

        if is_leader:
            k0_bar[].expect_bytes(
                Int32(
                    Self.config.BN
                    * Self.config.q_depth
                    * size_of[Self.kv_type]()
                )
            )
            var stage_ptr = kv_prod.stage_base_ptr[qk_stage=0]()
            Self.Common_MLA_Op.load_kv(
                k_tma, stage_ptr, k0_bar, UInt(0), UInt(kv_gmem_row)
            )

        kv_prod.commit_step()

        kv_row += UInt32(Self.config.BN)

        var tile_idx: Int = 1
        while tile_idx < num_k_tiles:
            kv_prod.acquire[qk_stage=0]()
            var stage_ptr = kv_prod.stage_base_ptr[qk_stage=0]()
            var k_mbar = kv_prod.producer_mbar[qk_stage=0]()
            kv_row = min(kv_row, max(num_keys_u32, UInt32(1)) - 1)
            var kv_gmem_row: UInt32 = kv_lut.row_idx(
                UInt32(offset_position.batch_idx), kv_row
            )

            if is_leader:
                k_mbar[].expect_bytes(
                    Int32(
                        Self.config.BN
                        * Self.config.q_depth
                        * size_of[Self.kv_type]()
                    )
                )
                Self.Common_MLA_Op.load_kv(
                    k_tma, stage_ptr, k_mbar, UInt(0), UInt(kv_gmem_row)
                )

            kv_row += UInt32(Self.config.BN)
            kv_prod.commit_step()
            tile_idx += 1

    # --------------------------------------------------------------------------
    # MLA decoding MMA for Q, K, V, P blocks
    # --------------------------------------------------------------------------

    # -------------------------------------------------
    # PIPELINE LOOP:
    #   loop over tiles 1..num_k_tiles-1
    #   each iteration does:
    #     - PV(tile_idx-1) with prev_stage_idx  (then release its KV stage)
    #     - QK(tile_idx) with the next KV stage
    # -------------------------------------------------
    # QK process the Numkey vertically, meaning the C Scale for the first
    # block of all tiles is going to be zero the PV multiply the P horizontally
    # to V meaning only the C scale for prev tile for is going to be Zero for all
    # 9 block and after that it is going to be 1
    #                Q                                              KV0/1
    #   ___ ___ ___ ___ ___ ___ ___ ___ ___       ___ ___ ___ ___ ___ ___ ___ ___ ___
    #  |___|___|___|___|___|___|___|___|___|  T0 |___|___|___|___|___|___|___|___|___|
    #                                         T1 |___|___|___|___|___|___|___|___|___|
    #                                         T2 |___|___|___|___|___|___|___|___|___|
    #                                         T3 |___|___|___|___|___|___|___|___|___|
    #     S0     S1     S0    S1
    #   ______ ______ ______ ______
    #  |__T0__|__T1__|__T2__|__T3__|
    #
    #     P0     P0     P0    P0
    #   ______ ______ ______ ______
    #  |__T0__|__T1__|__T2__|__T3__|

    # We move it to It might be possible to create two P slot and put it at the
    # last slot of KV pipeline, Need to verify if that gives better performance.
    # QK process the Numkey vertically, meaning the C Scale for the first block
    # of all tiles is going to be zero the PV multiply the P horizontally to V
    # meaning only the C scale for prev tile for is going to be Zero for all 9 block
    # and after that it is going to be 1
    #                Q                                              KV0/1
    #   ___ ___ ___ ___ ___ ___ ___ ___ ___       ___ ___ ___ ___ ___ ___ ___ ___ _______
    #  |___|___|___|___|___|___|___|___|___|  T0 |___|___|___|___|___|___|___|___|__P0/1_|
    #                                         T1 |___|___|___|___|___|___|___|___|__P0/1_|
    #                                         T2 |___|___|___|___|___|___|___|___|__P0/1_|
    #                                         T3 |___|___|___|___|___|___|___|___|__P0/1_|
    #     S0    S1    S0    S1
    #   ______ ______ ______ ______
    #  |__T0__|__T1__|__T2__|__T3__|
    #
    #   P0     P1    P0    P1
    #  ______ ______ ______ ______
    # |__T0__|__T1__|__T2__|__T3__|

    @staticmethod
    @always_inline
    fn mmaQK(
        tmem_addr: UInt32,
        q_smem: SharedMemPointer[Scalar[Self.q_type]],
        kv_smem: SharedMemPointer[Scalar[Self.q_type]],
        mbar_q: MBarType,
        s_bars: DecodeSM100MiscMBars[
            num_stages=2, num_producer=1, num_consumer=WARPGROUP_SIZE
        ],
        kv_pipeline: KVPipelineGeneric[
            num_kv_stages = Self.config.num_kv_stages,  # 2
            num_qk_stages=1,
            num_producer=1,
            num_consumer=2,
        ],
        offset_position: OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
        ],
    ):
        var s0_tmem = tmem_addr + UInt32(Self.config.TMEM_S0)
        var elect_mask = elect()

        num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN
        )

        # Early exit if there are no K tiles
        if num_k_tiles == 0:
            return

        var kv_cons = DecodeKVConsumer[Self.q_type, Self.config](
            kv_pipeline, kv_smem
        )
        # ---  S producer wrapper (2-stage pipeline) ---
        var s_prod = DecodeSProducer(s_bars.producer())
        comptime s_stride = UInt32(Self.config.TMEM_S1 - Self.config.TMEM_S0)

        var q_descriptor = Self.UMMAQKTSS.descriptor_q_block(q_smem)
        var k_descriptor = Self.UMMAQKTSS.descriptor_k_block(kv_smem)
        comptime stage_stride_in_bytes = Self.KVStageElems * Self.bytes_per_element

        mbar_q[].wait(0)
        var tile_idx: Int = 0

        while tile_idx < num_k_tiles:
            # wait until the corresponding consumer has freed a slot
            s_prod.acquire()

            var slot_idx: UInt32 = s_prod.slot_index()
            var s_tmem_slot = s0_tmem + slot_idx * s_stride

            kv_cons.wait[qk_stage=0]()
            k_slot_index = kv_cons.stage_index[qk_stage=0]()

            Self.UMMAQKTSS.mma[stage_idx=0](
                a=q_descriptor,
                b=k_descriptor + k_slot_index * UInt32(stage_stride_in_bytes),
                c=s_tmem_slot,
                c_scale=UInt32(0),
                elect=elect_mask,
            )
            tcgen05_fence_before()
            s_prod.commit_mma(elect_mask)
            kv_cons.release[qk_stage=0](elect_mask)
            tile_idx += 1

    @staticmethod
    @always_inline
    fn mmaPV(
        tmem_addr: UInt32,
        kv_smem: SharedMemPointer[Scalar[Self.q_type]],
        p_bars: DecodeSM100MiscMBars[
            num_stages=2, num_producer=WARPGROUP_SIZE, num_consumer=1
        ],
        o_bars: DecodeSM100MiscMBars[
            num_stages=2, num_producer=1, num_consumer=WARPGROUP_SIZE
        ],
        kv_pipeline: KVPipelineGeneric[
            num_kv_stages = Self.config.num_kv_stages,  # 2
            num_qk_stages=1,
            num_producer=1,
            num_consumer=2,
        ],
        offset_position: OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
        ],
    ):
        var o_tmem = tmem_addr + UInt32(Self.config.TMEM_O)
        var elect_mask = elect()
        num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN
        )

        # Early exit if there are no K tiles
        if num_k_tiles == 0:
            return

        # ---  S producer wrapper (2-stage pipeline) ---
        comptime s_stride = UInt32(Self.config.TMEM_S1 - Self.config.TMEM_S0)
        var kv_cons = DecodeKVConsumer[Self.q_type, Self.config](
            kv_pipeline, kv_smem
        )
        var p_cons = DecodePConsumer(p_bars.consumer())
        var o_prod = DecodeOProducer(o_bars.producer())
        var p_smem_base = kv_smem + Self.NumVOBlocks * Self.BlockElems
        var p_descriptor = Self.UMMAPVSS.descriptor_p_block(p_smem_base)
        var v_descriptor = Self.UMMAPVSS.descriptor_v_block(kv_smem)
        comptime block_step = Self.config.MMA_PV_N // Self.config.BN
        comptime stage_stride_in_bytes = Self.KVStageElems * Self.bytes_per_element
        comptime block_stride_in_bytes = Self.BlockElems * Self.bytes_per_element

        var tile_idx: Int = 0
        var c_scale: UInt32 = 0
        while tile_idx < num_k_tiles:
            kv_cons.wait[qk_stage=0]()
            var p_slot_index = p_cons.wait()
            var v_slot_index = kv_cons.stage_index[qk_stage=0]()

            # PV does not have the k-rope so we don't need to do the last block
            comptime for block in range(0, Self.NumVOBlocks, block_step):
                o_prod.acquire()
                Self.UMMAPVSS.mma[stage_idx=0](
                    a=p_descriptor
                    + p_slot_index * UInt32(stage_stride_in_bytes),
                    b=v_descriptor
                    + v_slot_index * UInt32(stage_stride_in_bytes)
                    + UInt32(block * block_stride_in_bytes),
                    c=o_tmem + UInt32(block) * UInt32(Self.config.BN // 2),
                    c_scale=c_scale,
                    elect=elect_mask,
                )
                o_prod.commit_mma(elect_mask)
            p_cons.release_mma(elect_mask)

            kv_cons.release[qk_stage=0](elect_mask)
            tcgen05_fence_before()

            if tile_idx == 0:
                c_scale = 1
            tile_idx += 1
