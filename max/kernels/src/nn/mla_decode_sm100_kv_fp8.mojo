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
    thread_idx,
    block_idx,
    warp_id,
)
from gpu.globals import WARPGROUP_SIZE
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu.primitives.grid_controls import launch_dependent_grids
from gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from gpu.memory import AddressSpace, external_memory, fence_async_view_proxy
from gpu.sync import (
    named_barrier,
)
from gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_fence_before,
    tcgen05_release_allocation_lock,
)
from layout.layout import (
    Layout,
)
from logger import Logger

from layout.swizzle import make_swizzle
from layout.tma_async import (
    PipelineState,
    SharedMemBarrier,
)
from memory import bitcast
from nn.mha_fa3_utils import (
    OptionalPointer,
)
from nn.mha_mask import MHAMask
from nn.mha_operand import MHAOperand
from nn.mha_score_mod import ScoreModTrait
from utils.numerics import get_accum_type, min_or_neg_inf
from utils.static_tuple import StaticTuple

from nn.mha_sm100_2q import (
    elect,
    TMemTile,
    LocalTensor,
    elect_mma_arrive,
)
from layout._layout import row_major
from layout._tile_tensor import stack_allocation as tt_stack_allocation
from nn.mha_fa3_utils import KVTMATile

comptime logger = Logger()
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
    KVLoad2CvtProducer,
    KVLoad2CvtConsumer,
    KVCvt2MmaProducer,
    KVCvt2MmaConsumer,
    DecodeSM100MiscMBars,
    DecodeSProducer,
    DecodeSConsumer,
    DecodePConsumer,
    DecodeOProducer,
    OutPipeline,
    DecodeOutProducer,
    DecodeSM100QKTSS,
    DecodeSM100PVSS,
    ld_shared_v4_u32,
    cvt_fp8x8_from_2xu32_to_bf16x8_packed_u32x4,
    st_shared_v4_b32_at_bf16_elem_off,
)


# ------------------------------------------------------------------------------
# MLA decoding kernel struct for SM100
# ------------------------------------------------------------------------------
struct MLA_SM100_Decode_KV_FP8[
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
    # O: 128 x 256
    comptime O_M = Self.config.BM * 2  # 128
    comptime O_N = Self.config.padded_depth // 2  # 256

    # S: 128 x 32
    comptime S_M = Self.config.BM * 2  # 128
    comptime S_N = Self.config.BN // 2  # 32
    comptime OTMemTile = TMemTile[Self.AccumType, Self.O_M, Self.O_N]
    comptime STMemTile = TMemTile[Self.AccumType, Self.S_M, Self.S_N]
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
    #    KSlot0 (tile 0)        KSlot1 (tile 1)
    #          |                    |
    #          V                    V
    #    UMMA WS → S0         UMMA WS → S1
    #          |                    |
    #   arrive mbar_s0        arrive mbar_s1
    #          |                    |
    #          |---- Softmax Warpgroup ----|
    #          |                           |
    #          V                           V
    #       wait_s0                      wait_s1
    #       S0 → P0                      S1 → P1
    #           |                          |
    #        UMMA WS → O                   |
    #           |                          |
    #           |                          |
    #       arrive mbar_0                  |
    #           |                          |
    #           |                          |
    #           V---- Coorection WG  ------|
    #           |                          |
    #       UMMA WP → O                UMMA WP → 1
    #    arrive mbar_0              arrive mbar_1
    #           |                          |
    #    Coorection WG1 → O           Coorection WG1 → 1
    #         |                            |
    #       wait_O_filled                wait_O_filled
    #        C_WG                         C_WG
    #          |                            |
    #        wair_out                     wair_out
    #          |                            |
    #       Write WG                     Write WG
    #         |                            |
    #        W_WG                         W_WG
    #
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
    ):
        # Softmax now includes the epilogue, so it needs more registers
        # Correction does less work now (no epilogue), so it needs fewer
        comptime num_reg_softmax = 184
        comptime num_reg_correction = 72
        comptime num_reg_keep_mma_load_store = 72
        comptime num_reg_keep_fp8tofp16 = 184
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
        @parameter
        if Self.config.decoding_warp_split_k:
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

        # early exit: Skip blocks beyond actual sequence length for this batch
        # In ragged mode with split-K, q_max_seq_len can be > 1 (up to 8).
        # block_idx.y ranges from 0 to q_max_seq_len-1, but some sequences
        # may have fewer tokens. CTAs with block_idx.y >= seq_len must still
        # fulfill the PDL contract (write -inf LSE, zero o_accum_split, and
        # call launch_dependent_grids) or the combine kernel will hang.
        @parameter
        if Self.ragged:
            # In ragged mode, block_idx.y is the query token index (0 to q_max_seq_len-1)
            # But this batch might have fewer tokens than q_max_seq_len
            if Int(block_idx.y) >= offset_position.seq_len:

                @parameter
                if Self.config.decoding_warp_split_k:
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
        # var kv_smem = q_smem + Self.BlockElems * Self.NumQKBlocks
        var kv_smem_bf16 = q_smem + Self.BlockElems * Self.NumQKBlocks
        var kv_smem_fp8_upper0 = (
            kv_smem_bf16.bitcast[Scalar[Self.KVLUTType.dtype]]()
            + Self.BlockElems * Self.NumQKBlocks
        )
        # This moves the memory to the second half of the KV SMEM give a space for
        # safe FP8/FP16 conversion
        # var kv_smem_fp8 = (q_smem + Self.BlockElems * Self.NumQKBlocks).bitcast[
        #     Scalar[Self.KVLUTType.dtype]
        # ]() + Self.BlockElems * Self.NumQKBlocks

        comptime kv_total_stages = Self.config.num_kv_stages
        # to reuse the K for V as well, we break KV as 9 stages of 64x64 to cover 64x576
        comptime kv_smem_total = Self.BlockElems * Self.NumQKBlocks * kv_total_stages

        # we need to use the KSmem for out pointer
        # We move P to the last slot of KV pipeline SO now we have tile of 64x
        # 32 of flot or 64x64 of FP16 to save output into
        # tiles in SMEM and smooth the pipeline for the next batch if we use splitk
        var out_smem_start = kv_smem_bf16.bitcast[Scalar[Self.output_type]]()
        # there is potential to have two Tmem for S, because we have two K so we can
        # unblock the MMA while loading S to reg for softrmax
        # if it was splitk we need to use the extra P slot. If not we need
        # to clear the KV slot before statting the max because KV slot is used by
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

        var kv_cvt2mma_pipe = KVPipelineGeneric[
            num_kv_stages = Self.config.num_kv_stages,  # 2
            num_qk_stages=1,
            num_producer=WARPGROUP_SIZE,  # 128
            num_consumer=2,
        ](mbar_kv_base)

        # Move mbar_base to the first free barrier *after* KV:
        mbar_base = mbar_kv_base + kv_cvt2mma_pipe.num_mbars()  # kv uses 1..4
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
        # This is used for the pipeline between Load and convert fp8 to bf16
        var kv_load2cvt_pipe = KVPipelineGeneric[
            num_kv_stages = Self.config.num_kv_stages,  # 2
            num_qk_stages=1,
            num_producer=1,
            num_consumer = WARPGROUP_SIZE + 2,  # 128 + 2 mma
        ](
            mbar_base
        )  # kv_load2cvt_pipe uses 23..26
        mbar_base += kv_load2cvt_pipe.num_mbars()  # barrier total [27]
        # we need to add ((Depth/BN) -1) x2 = 8x2 =16 +27 =43
        # more barriers. for the splitk the two added is enough for now.
        comptime OutPipeType = DecodeOutProducer[Self.output_type, Self.config]
        var out_pipeline = OutPipeline[
            num_out_stages = OutPipeType.num_out_stages,
            num_producer=WARPGROUP_SIZE,
            num_consumer=1,
        ](
            mbar_base
        )  # Write uses 27 + (num_out_stages)*2
        mbar_base += out_pipeline.num_mbars()

        # barrier total [27 + (num_out_stages)*2]
        var ptr_tmem_addr = (mbar_base).bitcast[UInt32]()

        var warp_idx = UInt32(warp.broadcast(warp_id()))
        is_leader = elect() != 0
        if warp_idx == 8:
            if is_leader:
                mbar_q[].init(1)
                # only one thread will load the Q
                kv_cvt2mma_pipe.init()
                s_bars.init()
                p_bars.init()
                kv_load2cvt_pipe.init()
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
                kv_smem_bf16,
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
            warpgroup_reg_dealloc[num_reg_correction]()
            Self.Common_MLA_Op.Correction(
                ptr_tmem_addr[0],
                o_bars,
                c_bars,
                corr_done_bars,
                offset_position,
            )
        elif warp_idx >= 8 and warp_idx < 12:
            warpgroup_reg_dealloc[num_reg_keep_mma_load_store]()
            if warp_idx == 8:
                Self.load(
                    q_tma,
                    k_tma,
                    kv_lut,
                    q_smem,
                    kv_smem_fp8_upper0,
                    mbar_q,
                    kv_load2cvt_pipe,
                    offset_position,
                )
            elif warp_idx == 9:
                Self.mmaQK(
                    ptr_tmem_addr[0],
                    q_smem,
                    kv_smem_bf16,
                    mbar_q,
                    s_bars,
                    kv_cvt2mma_pipe,
                    kv_load2cvt_pipe,
                    offset_position,
                )
            elif warp_idx == 10:
                Self.mmaPV(
                    ptr_tmem_addr[0],
                    kv_smem_bf16,
                    p_bars,
                    o_bars,
                    kv_cvt2mma_pipe,
                    kv_load2cvt_pipe,
                    offset_position,
                )
            elif warp_idx == 11:
                Self.Common_MLA_Op.store(
                    out_pipeline, out_smem, o_tma, offset_position
                )
        else:
            warpgroup_reg_alloc[num_reg_keep_fp8tofp16]()
            # Use num_keys_this_split for loop bounds (each split processes its portion)
            var num_k_tiles = ceildiv(
                offset_position.num_keys_this_split, Self.config.BN
            )
            Self.convertFP8ToBF16(
                kv_smem_fp8_upper0,
                kv_smem_bf16,
                kv_load2cvt_pipe,
                kv_cvt2mma_pipe,
                num_k_tiles,
            )
        barrier()

        # PDL: Signal that this CTA is done so dependent grids (combine kernel) can start.
        # This must be called by all threads in the CTA after all work is complete.
        @parameter
        if Self.config.decoding_warp_split_k:
            launch_dependent_grids()

        if warp_idx == 9:
            tcgen05_release_allocation_lock[Self.config.cta_group]()
            tcgen05_dealloc[Self.config.cta_group](
                ptr_tmem_addr[0], Self.config.sm100_tmem_cols
            )

    @staticmethod
    @always_inline
    fn load(
        q_tma: QOTMATile[
            dtype = Self.q_type,
            BM = Self.config.BM,  # tile_m =64
            BK = Self.config.BK0,  # tile_n =576
            swizzle_mode = Self.config.swizzle_mode,
        ],
        k_tma_fp8: KVTMATile[
            dtype = Self.kv_type,
            swizzle_mode = Self.config.kv_tma_swizzle_mode,
            BN = Self.config.BK1,  # tile_m =64
            BK = Self.config.BK0,  # tile_n =576
        ],
        kv_lut: Self.KVLUTType,
        q_smem: SharedMemPointer[Scalar[Self.q_type]],
        kv_smem_fp8: SharedMemPointer[Scalar[Self.kv_type]],
        mbar_q: MBarType,
        kv_load2cvt_pipe: KVPipelineGeneric[
            num_kv_stages = Self.config.num_kv_stages,  # 2
            num_qk_stages=1,
            num_producer=1,
            num_consumer = WARPGROUP_SIZE + 2,  # 128 + 2 mma
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
        num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN
        )

        # Early exit if this split has no work (prevents producer/consumer deadlock)
        if num_k_tiles == 0:
            return

        var kv_load_prod = KVLoad2CvtProducer[Self.kv_type, Self.config](
            kv_load2cvt_pipe,
            kv_smem_fp8,
        )
        var elect_mask = elect()
        var is_leader = elect_mask != 0
        var row: UInt = UInt(offset_position.q_row_offset)
        var pipe_qk = PipelineState[num_stages=1]()
        # Start KV from kv_start_row for split-K support
        var kv_row: UInt32 = UInt32(offset_position.kv_start_row)
        var kv_gmem_row: UInt32 = kv_lut.row_idx(
            UInt32(offset_position.batch_idx), kv_row
        )
        if is_leader:
            mbar_q[].expect_bytes(
                Int32(
                    Self.config.BM
                    * Self.config.q_depth
                    * size_of[Self.q_type]()
                )
            )
            Self.Common_MLA_Op.load_q(q_tma, q_smem, mbar_q, UInt(0), row)

        var k0_bar: MBarType = kv_load_prod.producer_mbar[qk_stage=0]()

        if is_leader:
            k0_bar[].expect_bytes(
                Int32(
                    Self.config.BN
                    * Self.config.q_depth
                    * size_of[Self.kv_type]()
                )
            )
            var stage_ptr = kv_load_prod.stage_base_ptr[qk_stage=0]()
            Self.Common_MLA_Op.load_kv(
                k_tma_fp8, stage_ptr, k0_bar, UInt(0), UInt(kv_gmem_row)
            )

        kv_load_prod.commit_step()

        kv_row += UInt32(Self.config.BN)

        var tile_idx: Int = 1
        while tile_idx < num_k_tiles:
            kv_load_prod.acquire[qk_stage=0]()

            var stage_ptr = kv_load_prod.stage_base_ptr[qk_stage=0]()
            var k_mbar = kv_load_prod.producer_mbar[qk_stage=0]()

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
                    k_tma_fp8, stage_ptr, k_mbar, UInt(0), UInt(kv_gmem_row)
                )

            kv_row += UInt32(Self.config.BN)
            kv_load_prod.commit_step()

            tile_idx += 1

    @staticmethod
    @always_inline
    fn convertFP8ToBF16(
        kv_smem_fp8: SharedMemPointer[Scalar[Self.kv_type]],
        kv_smem_bf16: SharedMemPointer[Scalar[Self.q_type]],
        kv_load2cvt_pipe: KVPipelineGeneric[
            num_kv_stages = Self.config.num_kv_stages,  # 2
            num_qk_stages=1,
            num_producer=1,
            num_consumer = WARPGROUP_SIZE + 2,  # 128 + 2 mma
        ],
        kv_cvt2mma_pipe: KVPipelineGeneric[
            num_kv_stages = Self.config.num_kv_stages,  # 2
            num_qk_stages=1,
            num_producer=WARPGROUP_SIZE,  # 128
            num_consumer=2,
        ],
        num_k_tiles: Int,
    ):
        comptime sw_fp8 = make_swizzle[
            Self.kv_type, Self.config.kv_tma_swizzle_mode
        ]()
        comptime sw_bf16 = make_swizzle[Self.q_type, Self.config.swizzle_mode]()

        comptime BN: Int = Self.config.BN
        comptime BK: Int = Self.config.q_depth
        comptime NumBlocks: Int = BK // BN
        comptime BlockElems: Int = Self.config.BM * BN

        var kv_load_cons_cvt = KVLoad2CvtConsumer[Self.kv_type, Self.config](
            kv_load2cvt_pipe,
            kv_smem_fp8,
        )
        var kv_cvt_prod = KVCvt2MmaProducer[Self.q_type, Self.config](
            kv_cvt2mma_pipe, kv_smem_bf16
        )
        var lane: Int = Int(thread_idx.x) & 0x7F
        var row: Int = lane & 0x3F
        # XOR the half selection with row bits to spread
        # conflicting rows across different column halves.
        # Original pattern: rows 0,8,16,24 all access banks 0-3 (4-way conflict)
        # With this fix: rows 0,16 access col0, rows 8,24 access col32
        var half: Int = (lane >> 6) ^ ((row >> 3) & 1)
        var col0: Int = half * 32

        var direct0: Int = row * BN + col0
        var direct1: Int = row * BN + col0 + 16

        var phys_fp8_0: Int = sw_fp8(direct0)
        var phys_fp8_1: Int = sw_fp8(direct1)

        var phys_bf16_0a: Int = sw_bf16(direct0)
        var phys_bf16_0b: Int = sw_bf16(direct0 + 8)
        var phys_bf16_1a: Int = sw_bf16(direct1)
        var phys_bf16_1b: Int = sw_bf16(direct1 + 8)

        var tile_idx: Int = 0
        while tile_idx < num_k_tiles:
            kv_load_cons_cvt.wait()
            kv_cvt_prod.acquire()

            var src_u8 = kv_load_cons_cvt.stage_base_ptr().bitcast[
                Scalar[DType.uint8]
            ]()
            var dst = kv_cvt_prod.stage_base_ptr()

            # First: Load all FP8 data and convert to BF16 in registers
            # This approach loads ALL blocks first, then uses ONE barrier,
            # then stores ALL blocks. This will significantly reduce the number of barriers.
            # and improve the performance (18 barriers vs 1 barrier here).
            var p0a_all = tt_stack_allocation[
                dtype = DType.uint32, address_space = AddressSpace.LOCAL
            ](row_major[4, NumBlocks]())
            var p0b_all = tt_stack_allocation[
                dtype = DType.uint32, address_space = AddressSpace.LOCAL
            ](row_major[4, NumBlocks]())
            var p1a_all = tt_stack_allocation[
                dtype = DType.uint32, address_space = AddressSpace.LOCAL
            ](row_major[4, NumBlocks]())
            var p1b_all = tt_stack_allocation[
                dtype = DType.uint32, address_space = AddressSpace.LOCAL
            ](row_major[4, NumBlocks]())

            @parameter
            for b in range(NumBlocks):
                var src_block_u8 = src_u8 + b * BlockElems

                var q0 = ld_shared_v4_u32(src_block_u8, phys_fp8_0)
                var q1 = ld_shared_v4_u32(src_block_u8, phys_fp8_1)

                var p0a = cvt_fp8x8_from_2xu32_to_bf16x8_packed_u32x4[
                    fp8_dtype = Self.kv_type,
                    out_dtype = Self.q_type,
                ](q0[0], q0[1])
                p0a_all.ptr.store(b * 4, p0a)

                var p0b = cvt_fp8x8_from_2xu32_to_bf16x8_packed_u32x4[
                    fp8_dtype = Self.kv_type,
                    out_dtype = Self.q_type,
                ](q0[2], q0[3])
                p0b_all.ptr.store(b * 4, p0b)

                var p1a = cvt_fp8x8_from_2xu32_to_bf16x8_packed_u32x4[
                    fp8_dtype = Self.kv_type,
                    out_dtype = Self.q_type,
                ](q1[0], q1[1])
                p1a_all.ptr.store(b * 4, p1a)

                var p1b = cvt_fp8x8_from_2xu32_to_bf16x8_packed_u32x4[
                    fp8_dtype = Self.kv_type,
                    out_dtype = Self.q_type,
                ](q1[2], q1[3])
                p1b_all.ptr.store(b * 4, p1b)

            # Single barrier. All 128 threads finish ALL reads before ANY writes
            named_barrier[Int32(WARPGROUP_SIZE)](3)

            # Second: Store all BF16 data from registers
            @parameter
            for b in range(NumBlocks):
                var dst_block = dst + b * BlockElems

                st_shared_v4_b32_at_bf16_elem_off[out_dtype = Self.q_type](
                    dst_block,
                    phys_bf16_0a,
                    p0a_all.ptr.load[width=4](b * 4),
                )
                st_shared_v4_b32_at_bf16_elem_off[out_dtype = Self.q_type](
                    dst_block,
                    phys_bf16_0b,
                    p0b_all.ptr.load[width=4](b * 4),
                )
                st_shared_v4_b32_at_bf16_elem_off[out_dtype = Self.q_type](
                    dst_block,
                    phys_bf16_1a,
                    p1a_all.ptr.load[width=4](b * 4),
                )
                st_shared_v4_b32_at_bf16_elem_off[out_dtype = Self.q_type](
                    dst_block,
                    phys_bf16_1b,
                    p1b_all.ptr.load[width=4](b * 4),
                )

            fence_async_view_proxy()
            kv_cvt_prod.commit_all()
            kv_load_cons_cvt.release_all()
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
    # QK process the Numkey veritcally, meaning the C Scale for the first
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
    # QK process the Numkey veritcally, meaning the C Scale for the first block
    # of all tiles is going to be zero the PV multiply the P horisontally to V
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
        kv_cvt2mma_pipe: KVPipelineGeneric[
            num_kv_stages = Self.config.num_kv_stages,  # 2
            num_qk_stages=1,
            num_producer=WARPGROUP_SIZE,  # 128
            num_consumer=2,
        ],
        kv_load2cvt_pipe: KVPipelineGeneric[
            num_kv_stages = Self.config.num_kv_stages,  # 2
            num_qk_stages=1,
            num_producer=1,
            num_consumer = WARPGROUP_SIZE + 2,  # 128 + 2 mma
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
        var o_tmem = tmem_addr + UInt32(Self.config.TMEM_O)
        var elect_mask = elect()
        # c_scale = 0 for the very first MMA (overwrite),
        #           1 afterwards (accumulate)
        # Number of K-tiles we have in this row
        # Use num_keys_this_split for loop bounds (each split processes its portion)
        num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN
        )

        # Early exit if there are no K tiles
        if num_k_tiles == 0:
            return

        var kv_cons = KVCvt2MmaConsumer[Self.q_type, Self.config](
            kv_cvt2mma_pipe, kv_smem
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
            elect_mma_arrive(
                kv_load2cvt_pipe.consumer_mbar[0](k_slot_index), elect_mask
            )
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
        kv_cvt2mma_pipe: KVPipelineGeneric[
            num_kv_stages = Self.config.num_kv_stages,  # 2
            num_qk_stages=1,
            num_producer=WARPGROUP_SIZE,  # 128
            num_consumer=2,
        ],
        kv_load2cvt_pipe: KVPipelineGeneric[
            num_kv_stages = Self.config.num_kv_stages,  # 2
            num_qk_stages=1,
            num_producer=1,
            num_consumer = WARPGROUP_SIZE + 2,  # 128 + 2 mma
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
        var kv_cons = KVCvt2MmaConsumer[Self.q_type, Self.config](
            kv_cvt2mma_pipe, kv_smem
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
            # that block is used for P
            @parameter
            for block in range(0, Self.NumVOBlocks, block_step):
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
            elect_mma_arrive(
                kv_load2cvt_pipe.consumer_mbar[0](v_slot_index), elect_mask
            )
            tcgen05_fence_before()
            if tile_idx == 0:
                c_scale = 1
            tile_idx += 1
