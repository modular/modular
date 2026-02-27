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

"""Native FP8 MLA decode kernel for SM100 (B200).

All-FP8 kernel: Q, K, V, and P are all FP8 e4m3 in SMEM.
Uses native FP8 WGMMA (tcgen05.mma.kind::f8f6f4) for both QK and PV.
Same 3-WG structure as the BF16 kernel (softmax WG, correction WG, MMA+Load+Store WG).

Q arrives as FP8 from TMA directly (like FlashInfer), no BF16 conversion.
KV arrives as FP8 from TMA directly (half the bytes of BF16).
P (softmax output) is written as FP8 e4m3 to a separate SMEM region.
The FP8 tensorwise dequant scale is folded into the softmax QK scale.

SMEM Layout (native FP8):
  Q FP8:      64 x 576 x 1 = 36864 bytes   (SWIZZLE_64B)
  KV stages:  N x 64 x 576 x 1 bytes  (SWIZZLE_64B, N=num_kv_stages, typically 4)
  P stages:   N x 64 x 64 x 1 bytes    (SWIZZLE_64B, separate from KV)
  max/li:     128 x 4 x 2 = 1024 bytes
  barriers:   (6N+11) fixed + output barriers
"""

from math import ceildiv, exp2, recip, log2
from math.constants import log2e
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
    tcgen05_fence_after,
    tcgen05_fence_before,
    tcgen05_ld,
    tcgen05_load_wait,
    tcgen05_release_allocation_lock,
    tcgen05_st,
)
from gpu.sync import named_barrier
from gpu.primitives.warp import _vote_nvidia_helper
from layout.tma_async import (
    SharedMemBarrier,
)
from layout._layout import row_major
from layout.layout import Layout
from layout.int_tuple import UNKNOWN_VALUE
from layout.swizzle import make_ldmatrix_swizzle
from memory import bitcast
from nn.mha_fa3_utils import (
    OptionalPointer,
    KVTMATile,
)
from nn.mha_mask import MHAMask
from nn.mha_operand import MHAOperand
from utils.numerics import get_accum_type, min_or_neg_inf
from utils.static_tuple import StaticTuple

from nn.sm100_attention_utils import (
    elect,
    LocalTensor,
    elect_mma_arrive,
    sub_ftz,
)

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
    DecodeSProducerN,
    DecodePConsumerN,
    DecodeOProducer,
    OutPipeline,
    DecodeOutProducer,
    DecodeKVProducer,
    DecodeKVConsumer,
    DecodeSM100QKTSS_FP8,
    DecodeSM100PVSS_FP8,
    clamped_index_coordinate,
)


# ------------------------------------------------------------------------------
# Native FP8 MLA decoding kernel struct for SM100
# All of Q, K, V, P are FP8 in SMEM.
# ------------------------------------------------------------------------------
struct MLA_SM100_Decode_QKV_FP8[
    q_type: DType,
    KVLUTType: MHAOperand,
    output_type: DType,
    SplitAccumType: OptionalPointer,
    MaskType: MHAMask,
    config: MLA_SM100_Decode_Config,
    ValidLengthType: OptionalPointer,
    _is_cache_length_accurate: Bool = False,
    ragged: Bool = False,
](TrivialRegisterPassable):
    comptime kv_type = Self.KVLUTType.dtype  # float8_e4m3fn
    comptime fp8_type = DType.float8_e4m3fn
    comptime AccumType = get_accum_type[Self.q_type]()
    # 576 / 64 = 9
    comptime NumQKBlocks = Self.config.padded_q_depth // Self.config.BN
    # 512 / 64 = 8
    comptime NumVOBlocks = Self.config.padded_depth // Self.config.BN
    # 64 * 64 = 4096
    comptime BlockElems = Self.config.BM * Self.config.BN
    # FP8: 1 byte per element for all SMEM operands (Q, K, V, P)
    comptime fp8_bytes_per_element = size_of[Self.fp8_type]()
    # BF16: 2 bytes per element for output/accumulator
    comptime bf16_bytes_per_element = size_of[Self.q_type]()
    # KV stage element count (FP8): 9 blocks * 4096 elems = 36864
    comptime KVStageElems = Self.NumQKBlocks * Self.BlockElems
    # P stage element count (FP8): 1 block * 4096 elems = 4096
    comptime PStageElems = Self.BlockElems
    comptime output_tile_width = (Self.config.BN // 2) * (
        4 // size_of[Self.output_type]()
    )

    # QK MMA: FP8 (KIND_F8F6F4) — both Q and K are FP8 in SMEM
    comptime UMMAQKTSS = DecodeSM100QKTSS_FP8[
        operand_type = Self.fp8_type,
        accum_type = Self.AccumType,
        config = Self.config,
    ]
    # PV MMA: FP8 (KIND_F8F6F4) — both P and V are FP8 in SMEM
    comptime UMMAPVSS = DecodeSM100PVSS_FP8[
        operand_type = Self.fp8_type,
        accum_type = Self.AccumType,
        config = Self.config,
    ]

    comptime Common_MLA_Op = MLA_SM100_Decode_Common[
        Self.q_type,
        Self.KVLUTType,
        Self.output_type,
        Self.SplitAccumType,
        Self.MaskType,
        Self.config,
        Self.ValidLengthType,
        Self._is_cache_length_accurate,
        Self.ragged,
    ]

    # Number of pipeline stages for KV, S, and P (dynamically computed, typically 4)
    comptime num_stages = Self.config.num_kv_stages

    # --------------------------------------------------------------------------
    # Main kernel function — 3 Warpgroups, 384 threads
    # --------------------------------------------------------------------------
    #    Softmax WG (warps 0-3), Correction WG (warps 4-7),
    #    MMA+Load+Store WG (warps 8-11)
    #    Warp assignments within WG2: warp 8 = Load, warp 9 = MMA QK,
    #                                 warp 10 = MMA PV, warp 11 = Store
    #
    # SMEM layout (all FP8, N=num_kv_stages, typically 4):
    #   Q FP8:     64 x 576 x 1 = 36864 bytes (SWIZZLE_64B)
    #   KV stages: N x 64 x 576 x 1 bytes (SWIZZLE_64B)
    #   P stages:  N x 64 x 64 x 1 bytes (SWIZZLE_64B, separate region)
    #   max/li:    128 x 4 x 2 = 1024 bytes
    #   barriers:  (6N+11) fixed + output barriers
    #
    # The FP8 dequant scale is folded into the softmax scale:
    #   qk_scale = (1.0 / sqrt(d_qk)) * kv_scale
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
        # Q TMA is FP8 with SWIZZLE_64B (same as KV)
        q_tma: QOTMATile[
            dtype = Self.kv_type,
            BM = Self.config.BM,  # tile_m =64
            BK = Self.config.BK0,  # tile_n =576
            swizzle_mode = Self.config.kv_tma_swizzle_mode,  # SWIZZLE_64B
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
        mla_decode_pack: MLA_Decode_Pack[
            ValidLengthType = Self.ValidLengthType,
            MaskType = Self.MaskType,
            SplitAccumType = Self.SplitAccumType,
        ],
        scales_ptr: UnsafePointer[Scalar[DType.float32], origin=MutAnyOrigin],
    ):
        # Register allocation: same as BF16 kernel (3 WGs)
        comptime num_reg_softmax = 192
        comptime num_reg_correction = 184
        comptime num_reg_other = 112
        mask = mla_decode_pack.mask
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

        # Early exit for split-K: CTAs with no work
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

        # Early exit for ragged: skip blocks beyond actual sequence length
        comptime if Self.ragged:
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

                return

        # ---- SMEM layout (all FP8, N stages) ----
        # Q FP8 region: 64 x 576 x 1 bytes = 36864 bytes
        q_smem = external_memory[
            Scalar[Self.fp8_type],
            address_space = AddressSpace.SHARED,
            alignment=128,
            name="mha_dynamic_shared_memory",
        ]()

        # KV FP8 SMEM: N stages, starts right after Q FP8
        # Each stage = 64 x 576 x 1 = 36864 bytes
        var kv_smem = q_smem + Self.BlockElems * Self.NumQKBlocks

        # P FP8 SMEM: separate region after KV stages
        # Each stage = 64 x 64 x 1 = 4096 bytes
        comptime kv_total_stages = Self.config.num_kv_stages
        var p_smem = kv_smem + Self.KVStageElems * kv_total_stages

        # P stages: num_stages * 4096 elems
        comptime p_total_stages = Self.num_stages

        # Output SMEM reuses KV SMEM (same as BF16 kernel)
        var out_smem_start = kv_smem
        var out_smem = out_smem_start.bitcast[Scalar[Self.output_type]]()

        # max_smem/li_smem: placed after P stages
        # max_smem is double-buffered (2 x 128 elements) to avoid a race
        # condition in softmax; li_smem is a single 128-element buffer.
        var max_smem = (p_smem + Self.PStageElems * p_total_stages).bitcast[
            Scalar[Self.AccumType]
        ]()
        var li_smem = max_smem + 2 * WARPGROUP_SIZE

        # ---- Barrier layout (6N+11 fixed for N-stage pipelines) ----
        # bar_q(1) + kv(2N) + s(2N) + p(2N) + o(4) + c(2) + corr_done(4)
        var mbar_base: MBarType = (li_smem + WARPGROUP_SIZE).bitcast[
            SharedMemBarrier
        ]()

        var mbar_q: MBarType = mbar_base
        var mbar_kv_base: MBarType = mbar_base + 1

        var kv_pipeline = KVPipelineGeneric[
            num_kv_stages = Self.config.num_kv_stages,
            num_qk_stages=1,
            num_producer=1,
            num_consumer=2,
        ](mbar_kv_base)

        mbar_base = mbar_kv_base + kv_pipeline.num_mbars()
        # S pipeline: N-stage (matching KV stages)
        var s_bars = DecodeSM100MiscMBars[
            num_stages = Self.num_stages,
            num_producer=1,
            num_consumer=WARPGROUP_SIZE,
        ](mbar_base)
        mbar_base = s_bars.end()
        # P pipeline: N-stage (matching KV stages)
        var p_bars = DecodeSM100MiscMBars[
            num_stages = Self.num_stages,
            num_producer=WARPGROUP_SIZE,
            num_consumer=1,
        ](mbar_base)
        mbar_base = p_bars.end()
        var o_bars = DecodeSM100MiscMBars[
            num_stages=2, num_producer=1, num_consumer=WARPGROUP_SIZE
        ](mbar_base)
        mbar_base = o_bars.end()
        var c_bars = DecodeSM100MiscMBars[
            num_stages=1,
            num_producer=WARPGROUP_SIZE,
            num_consumer=WARPGROUP_SIZE,
        ](mbar_base)
        mbar_base = c_bars.end()
        var corr_done_bars = DecodeSM100MiscMBars[
            num_stages=2,
            num_producer=WARPGROUP_SIZE,
            num_consumer=WARPGROUP_SIZE,
        ](mbar_base)
        mbar_base = corr_done_bars.end()
        comptime OutPipeType = DecodeOutProducer[Self.output_type, Self.config]
        var out_pipeline = OutPipeline[
            num_out_stages = OutPipeType.num_out_stages,
            num_producer=WARPGROUP_SIZE,
            num_consumer=1,
        ](mbar_base)
        mbar_base += out_pipeline.num_mbars()

        var warp_idx = UInt32(warp.broadcast(warp_id()))
        var ptr_tmem_addr = (mbar_base).bitcast[UInt32]()
        is_leader = elect() != 0

        if warp_idx == 8:
            if is_leader:
                mbar_q[].init(1)
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
            Self.Common_MLA_Op.Softmax[
                native_fp8=True,
                num_sp_stages = Self.num_stages,
            ](
                ptr_tmem_addr[0],
                s_bars,
                p_bars,
                p_smem.bitcast[Scalar[Self.Common_MLA_Op.q_type]](),
                max_smem,
                li_smem,
                out_smem,
                c_bars,
                corr_done_bars,
                out_pipeline,
                offset_position,
                scale,
                mask,
                prompt_idx=UInt32(offset_position.batch_idx),
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
                    kv_smem,
                    mbar_q,
                    s_bars,
                    kv_pipeline,
                    offset_position,
                )
            elif warp_idx == 10:
                Self.mmaPV(
                    ptr_tmem_addr[0],
                    kv_smem,
                    p_smem,
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

        # PDL: Signal that this CTA is done
        comptime if Self.config.decoding_warp_split_k:
            launch_dependent_grids()

        if warp_idx == 9:
            tcgen05_release_allocation_lock[Self.config.cta_group]()
            tcgen05_dealloc[Self.config.cta_group](
                ptr_tmem_addr[0], Self.config.sm100_tmem_cols
            )

    # --------------------------------------------------------------------------
    # Load: TMA Q (FP8) directly, TMA KV (FP8) directly
    # --------------------------------------------------------------------------
    @staticmethod
    @always_inline
    fn load(
        q_tma: QOTMATile[
            dtype = Self.kv_type,
            BM = Self.config.BM,
            BK = Self.config.BK0,
            swizzle_mode = Self.config.kv_tma_swizzle_mode,  # SWIZZLE_64B
        ],
        k_tma: KVTMATile[
            dtype = Self.kv_type,
            swizzle_mode = Self.config.kv_tma_swizzle_mode,
            BN = Self.config.BK1,
            BK = Self.config.BK0,
        ],
        kv_lut: Self.KVLUTType,
        q_smem: SharedMemPointer[Scalar[Self.fp8_type]],
        kv_smem: SharedMemPointer[Scalar[Self.fp8_type]],
        mbar_q: MBarType,
        kv_pipeline: KVPipelineGeneric[
            num_kv_stages = Self.config.num_kv_stages,
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
        if offset_position.num_keys_this_split == 0:
            return

        num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN
        )

        var kv_prod = DecodeKVProducer[Self.kv_type, Self.config](
            kv_pipeline, kv_smem.bitcast[Scalar[Self.kv_type]]()
        )
        var elect_mask = elect()
        var is_leader = elect_mask != 0
        var row: UInt = UInt(offset_position.q_row_offset)
        var kv_row: UInt32 = UInt32(offset_position.kv_start_row)
        var num_keys_u32 = UInt32(offset_position.num_keys)
        kv_row = min(kv_row, max(num_keys_u32, UInt32(1)) - 1)
        var kv_gmem_row: UInt32 = kv_lut.row_idx(
            UInt32(offset_position.batch_idx), kv_row
        )

        # Load Q via TMA as FP8 (no conversion needed)
        if is_leader:
            mbar_q[].expect_bytes(
                Int32(
                    Self.config.BM
                    * Self.config.q_depth
                    * Self.fp8_bytes_per_element
                )
            )
            # Q TMA: load FP8 Q directly into q_smem
            var q_block_smem = q_smem
            var q_smem_tensor = SharedMemTensor[
                Self.kv_type, type_of(q_tma).layout
            ](q_block_smem.bitcast[Scalar[Self.kv_type]]())
            q_tma.async_copy(q_smem_tensor, mbar_q[], (Int(UInt(0)), Int(row)))

        # Load first KV tile (FP8)
        var k0_bar: MBarType = kv_prod.producer_mbar[qk_stage=0]()
        if is_leader:
            k0_bar[].expect_bytes(
                Int32(
                    Self.config.BN
                    * Self.config.q_depth
                    * Self.fp8_bytes_per_element
                )
            )
            var stage_ptr = kv_prod.stage_base_ptr[qk_stage=0]()
            Self.Common_MLA_Op.load_kv(
                k_tma, stage_ptr, k0_bar, UInt(0), UInt(kv_gmem_row)
            )
        kv_prod.commit_step()
        kv_row += UInt32(Self.config.BN)

        # Wait for Q TMA to complete. Q arrives as FP8, ready for MMA.
        mbar_q[].wait(0)

        # Load remaining KV tiles
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
                        * Self.fp8_bytes_per_element
                    )
                )
                Self.Common_MLA_Op.load_kv(
                    k_tma, stage_ptr, k_mbar, UInt(0), UInt(kv_gmem_row)
                )

            kv_row += UInt32(Self.config.BN)
            kv_prod.commit_step()
            tile_idx += 1

    # --------------------------------------------------------------------------
    # MMA QK: Q(FP8) x K(FP8) -> S(TMEM)
    # --------------------------------------------------------------------------
    @staticmethod
    @always_inline
    fn mmaQK(
        tmem_addr: UInt32,
        q_smem: SharedMemPointer[Scalar[Self.fp8_type]],
        kv_smem: SharedMemPointer[Scalar[Self.fp8_type]],
        mbar_q: MBarType,
        s_bars: DecodeSM100MiscMBars[
            num_stages = Self.num_stages,
            num_producer=1,
            num_consumer=WARPGROUP_SIZE,
        ],
        kv_pipeline: KVPipelineGeneric[
            num_kv_stages = Self.config.num_kv_stages,
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

        if num_k_tiles == 0:
            return

        var kv_cons = DecodeKVConsumer[Self.fp8_type, Self.config](
            kv_pipeline, kv_smem
        )
        # N-stage S producer
        var s_prod = DecodeSProducerN[Self.num_stages](s_bars.producer())
        comptime s_stride = UInt32(Self.config.TMEM_S1 - Self.config.TMEM_S0)

        # FP8 descriptors for Q and K
        var q_descriptor = Self.UMMAQKTSS.descriptor_q_block(q_smem)
        var k_descriptor = Self.UMMAQKTSS.descriptor_k_block(kv_smem)
        # Stage stride in bytes: FP8 elements (1 byte each)
        comptime stage_stride_in_bytes = Self.KVStageElems * Self.fp8_bytes_per_element

        # Q FP8 is ready (load warp waited on mbar_q and Q arrived as FP8).
        # The barrier() at the end of init ensures visibility.

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
            tile_idx += 1

    # --------------------------------------------------------------------------
    # MMA PV: P(FP8) x V(FP8) -> O(TMEM)
    # --------------------------------------------------------------------------
    @staticmethod
    @always_inline
    fn mmaPV(
        tmem_addr: UInt32,
        kv_smem: SharedMemPointer[Scalar[Self.fp8_type]],
        p_smem: SharedMemPointer[Scalar[Self.fp8_type]],
        p_bars: DecodeSM100MiscMBars[
            num_stages = Self.num_stages,
            num_producer=WARPGROUP_SIZE,
            num_consumer=1,
        ],
        o_bars: DecodeSM100MiscMBars[
            num_stages=2, num_producer=1, num_consumer=WARPGROUP_SIZE
        ],
        kv_pipeline: KVPipelineGeneric[
            num_kv_stages = Self.config.num_kv_stages,
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

        if num_k_tiles == 0:
            return

        comptime s_stride = UInt32(Self.config.TMEM_S1 - Self.config.TMEM_S0)
        var kv_cons = DecodeKVConsumer[Self.fp8_type, Self.config](
            kv_pipeline, kv_smem
        )
        # N-stage P consumer
        var p_cons = DecodePConsumerN[Self.num_stages](p_bars.consumer())
        var o_prod = DecodeOProducer(o_bars.producer())

        # P descriptor: points to the separate P SMEM region
        var p_descriptor = Self.UMMAPVSS.descriptor_p_block(p_smem)
        # V descriptor: points to the KV SMEM (V data is first 512 cols of KV)
        var v_descriptor = Self.UMMAPVSS.descriptor_v_block(kv_smem)
        comptime block_step = Self.config.MMA_PV_N // Self.config.BN
        # FP8: 1 byte per element
        comptime kv_stage_stride_in_bytes = Self.KVStageElems * Self.fp8_bytes_per_element
        comptime p_stage_stride_in_bytes = Self.PStageElems * Self.fp8_bytes_per_element
        comptime block_stride_in_bytes = Self.BlockElems * Self.fp8_bytes_per_element

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
                    + p_slot_index * UInt32(p_stage_stride_in_bytes),
                    b=v_descriptor
                    + v_slot_index * UInt32(kv_stage_stride_in_bytes)
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
