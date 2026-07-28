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

"""Native FP8 sparse MLA decode kernel for SM100 (B200).

Combines the sparse gather4 TMA machinery from mla_decode_sparse_kv_fp8.mojo
with the native FP8 3-WG pipeline from mla_decode_qkv_fp8.mojo.

Eliminates the FP8→BF16 convert warpgroup (WG3) present in
MLA_SM100_Decode_Sparse_KV_FP8 by performing QK and PV MMAs natively in FP8
(tcgen05.mma.kind::f8f6f4). Q arrives as FP8 via SWIZZLE_64B TMA directly.
KV arrives as FP8 via gather4 INT64 TMA directly into kv_smem (no SMEM
conversion stage). P is FP8 in a separate SMEM region.

Going from 4 warpgroups back to 3:
  WG0 (warps  0-3):  Softmax  (native_fp8=True)
  WG1 (warps  4-7):  Correction
  WG2 (warps 8-11):  Load (warp 8) / MMA QK (warp 9) / MMA PV (warp 10) /
                     idx_producer + Store (warp 11)

SMEM layout (FP8, N = num_kv_stages, typically 4):
  Q FP8:     64 x 576 x 1 = 36864 bytes  (SWIZZLE_64B)
  KV stages: N x 64 x 576 x 1 bytes      (SWIZZLE_64B)
  P stages:  N x 64 x 64 x 1 bytes       (SWIZZLE_64B, separate region)
  max/li:    128 x 4 x 3 = 1536 bytes
  barriers:  (6N+11) fixed + output + idx_bars (4) barriers
  ptr_tmem_addr: 4 bytes
  idx_smem:  2 x 64 x 4 = 512 bytes
"""

from std.collections import OptionalReg
from std.math import ceildiv, clamp
from std.math.constants import log2e
from std.sys import size_of
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    barrier,
    thread_idx,
    block_idx,
    warp_id,
    lane_id,
)
from std.gpu.globals import WARPGROUP_SIZE
from std.gpu.primitives.grid_controls import launch_dependent_grids
from std.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from std.gpu.memory import AddressSpace, CacheEviction, external_memory
from std.gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_fence_before,
    tcgen05_release_allocation_lock,
)
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    _gather4_box_width,
)
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from layout import ComptimeInt, RowMajorLayout, TileTensor
from layout.tile_layout import row_major as tt_row_major
from nn.attention.gpu.nvidia.common import OptionalPointer
from nn.attention.mha_mask import MHAMask
from nn.attention.mha_operand import MHAOperand
from std.utils.index import IndexList
from std.utils.numerics import get_accum_type, min_or_neg_inf
from std.utils.static_tuple import StaticTuple

from nn.attention.gpu.nvidia.sm100.attention_utils import (
    elect,
    expect_bytes_pred,
    SharedMemPointer,
    MBarType,
    ProducerPipeline,
    ConsumerPipeline,
)

from nn.attention.gpu.nvidia.sm100.mla_decode_utils import (
    MLA_SM100_Decode_Config,
    MLA_SM100_Decode_Common,
    QOTMATile,
    MLA_Decode_Pack,
    OffsetPosition,
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
)


struct MLA_SM100_Decode_Sparse_QKV_FP8[
    q_type: DType,
    KVLUTType: MHAOperand,
    output_type: DType,
    SplitAccumType: OptionalPointer,
    MaskType: MHAMask,
    config: MLA_SM100_Decode_Config,
    ValidLengthType: OptionalPointer,
    _is_cache_length_accurate: Bool = False,
    ragged: Bool = False,
    has_attn_sink: Bool = False,
    has_extra_kv: Bool = False,
    has_variable_topk: Bool = False,
    fold_shared_index: Bool = False,
    q_len_fold: Int = 1,
](TrivialRegisterPassable):
    """Sparse MLA decode with native FP8 WGMMA for SM100 (B200).

    Sibling of MLA_SM100_Decode_Sparse_KV_FP8 but without the FP8→BF16
    convert warpgroup. Q arrives as FP8 (SWIZZLE_64B), KV arrives as FP8 via
    gather4 INT64 TMA directly into kv_smem. Both QK and PV MMMAs run
    natively in FP8 (tcgen05.mma.kind::f8f6f4). P lives in a separate SMEM
    region (same as MLA_SM100_Decode_QKV_FP8).
    """

    comptime kv_type = Self.KVLUTType.dtype
    comptime fp8_type = DType.float8_e4m3fn
    comptime AccumType = get_accum_type[Self.q_type]()
    comptime NumQKBlocks = Self.config.padded_q_depth // Self.config.BN_QK
    comptime NumVOBlocks = Self.config.padded_depth // Self.config.BN_QK
    comptime BlockElems = Self.config.BM * Self.config.BN_QK
    comptime fp8_bytes_per_element = size_of[Self.fp8_type]()
    comptime KVStageElems = Self.NumQKBlocks * Self.BlockElems
    comptime PStageElems = Self.BlockElems
    comptime output_tile_width = (Self.config.BN_QK // 2) * (
        4 // size_of[Self.output_type]()
    )

    comptime kv_gather4_tile_width = Self.config.padded_q_depth
    comptime kv_gather4_box_w = _gather4_box_width[
        Self.fp8_type,
        Self.kv_gather4_tile_width,
        TensorMapSwizzle.SWIZZLE_64B,
    ]()

    comptime UMMAQKTSS = DecodeSM100QKTSS_FP8[
        operand_type=Self.fp8_type,
        accum_type=Self.AccumType,
        config=Self.config,
    ]
    comptime UMMAPVSS = DecodeSM100PVSS_FP8[
        operand_type=Self.fp8_type,
        accum_type=Self.AccumType,
        config=Self.config,
    ]

    comptime num_stages = Self.config.num_kv_stages

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

    @staticmethod
    @__llvm_arg_metadata(q_tma, `nvvm.grid_constant`)
    @__llvm_arg_metadata(k_tma, `nvvm.grid_constant`)
    @__llvm_arg_metadata(o_tma, `nvvm.grid_constant`)
    @__llvm_arg_metadata(extra_k_tma, `nvvm.grid_constant`)
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.config.num_threads)
        )
    )
    @__llvm_metadata(`nvvm.minctasm`=SIMDLength(1))
    def kernel(
        q_tma: QOTMATile[
            dtype=Self.fp8_type,
            BM=Self.config.BM,
            BK=Self.config.BK_QK,
            swizzle_mode=Self.config.kv_tma_swizzle_mode,
        ],
        k_tma: TMATensorTile[
            Self.fp8_type,
            2,
            tile_shape=IndexList[2](Self.config.BK_PV, Self.kv_gather4_box_w),
            desc_shape=IndexList[2](1, Self.kv_gather4_box_w),
        ],
        o_tma: QOTMATile[
            dtype=Self.output_type,
            BM=Self.config.out_rows,
            BK=Self.config.BN_PV // 4,
            swizzle_mode=Self.config.swizzle_mode,
        ],
        kv_lut: Self.KVLUTType,
        scale: Float32,
        mla_decode_pack: MLA_Decode_Pack[
            ValidLengthType=Self.ValidLengthType,
            MaskType=Self.MaskType,
            SplitAccumType=Self.SplitAccumType,
        ],
        d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
        indices_stride: Int,
        topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
        scales_ptr: UnsafePointer[Scalar[DType.float32], origin=MutAnyOrigin],
        attn_sink_ptr: OptionalReg[
            UnsafePointer[Scalar[DType.float32], origin=MutAnyOrigin]
        ],
        extra_k_tma: TMATensorTile[
            Self.fp8_type,
            2,
            tile_shape=IndexList[2](Self.config.BK_PV, Self.kv_gather4_box_w),
            desc_shape=IndexList[2](1, Self.kv_gather4_box_w),
        ],
        extra_kv_lut: Self.KVLUTType,
        extra_d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
        extra_topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
        extra_indices_stride: Int,
        extra_scales_ptr: OptionalReg[
            UnsafePointer[Scalar[DType.float32], origin=MutAnyOrigin]
        ],
        scalar_args: TileTensor[
            DType.int64, RowMajorLayout[ComptimeInt[3]], MutAnyOrigin
        ],
    ):
        comptime _mask_type_name: String = Self.MaskType.get_type_name()
        comptime assert (
            _mask_type_name == "NullMask" or _mask_type_name == "CausalMask"
        ), (
            "MLA_SM100_Decode_Sparse_QKV_FP8 supports NullMask and CausalMask"
            " only. Sliding-window is not yet wired for sparse."
        )
        comptime assert (
            not Self.fold_shared_index
            or Self.config.num_q_heads * Self.q_len_fold <= Self.config.BM
        ), "fold_shared_index requires num_q_heads * q_len_fold <= BM."
        comptime assert not (
            Self.fold_shared_index and Self.has_extra_kv
        ), "fold_shared_index does not support extra always-attend KV cache."

        comptime num_reg_softmax = 192
        comptime num_reg_correction = 184
        comptime num_reg_other = 112

        var batch_size = Int(scalar_args.ptr[0])
        var q_max_seq_len = Int(scalar_args.ptr[1])
        var num_partitions = mla_decode_pack.num_partitions
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
            sparse=True,
            has_extra_kv=Self.has_extra_kv,
            has_variable_topk=Self.has_variable_topk,
        ](
            kv_lut,
            rebind[
                UnsafePointer[
                    Scalar[Self.ValidLengthType.dtype],
                    ImmutAnyOrigin,
                    address_space=AddressSpace.GENERIC,
                ]
            ](valid_length.value()),
            q_max_seq_len,
            num_partitions,
            batch_size,
            sparse_indices_stride=indices_stride,
            sparse_topk_lengths=topk_lengths,
            sparse_extra_indices_stride=extra_indices_stride,
            sparse_extra_topk_lengths=extra_topk_lengths,
        )

        var topk: Int
        comptime if Self.has_variable_topk:
            topk = Int(
                topk_lengths.unsafe_value()[Int(offset_position.batch_idx)]
            )
        else:
            topk = indices_stride
        var extra_topk: Int = 0
        comptime if Self.has_extra_kv:
            comptime if Self.has_variable_topk:
                extra_topk = Int(
                    extra_topk_lengths.unsafe_value()[
                        Int(offset_position.batch_idx)
                    ]
                )
            else:
                extra_topk = extra_indices_stride
        topk = offset_position.num_keys - extra_topk

        var num_orig_blocks = ceildiv(topk, Self.config.BN_QK)

        @parameter
        @always_inline
        def _pdl_early_exit_all_q():
            comptime if Self.fold_shared_index:
                comptime for q_local in range(Self.q_len_fold):
                    Self.Common_MLA_Op.pdl_early_exit[fold_q=True](
                        offset_position.split_idx,
                        offset_position.batch_idx,
                        offset_position.max_seq_len,
                        offset_position.out_row_offset_at(q_local),
                        batch_size,
                        lse_accum_split_ptr,
                        o_tma,
                        seq_idx_fold=UInt32(q_local),
                    )
            else:
                Self.Common_MLA_Op.pdl_early_exit(
                    offset_position.split_idx,
                    offset_position.batch_idx,
                    offset_position.max_seq_len,
                    offset_position.out_row_offset,
                    batch_size,
                    lse_accum_split_ptr,
                    o_tma,
                )

        comptime if Self.config.decoding_warp_split_k:
            if offset_position.num_keys_this_split == 0:
                _pdl_early_exit_all_q()
                return

        comptime if Self.ragged:
            if block_idx.y >= offset_position.seq_len:
                comptime if Self.config.decoding_warp_split_k:
                    _pdl_early_exit_all_q()
                return

        q_smem = external_memory[
            Scalar[Self.fp8_type],
            address_space=AddressSpace.SHARED,
            alignment=128,
            name="mha_dynamic_shared_memory",
        ]()

        var kv_smem = q_smem + Self.BlockElems * Self.NumQKBlocks

        comptime kv_total_stages = Self.config.num_kv_stages
        var p_smem = kv_smem + Self.KVStageElems * kv_total_stages

        # Output SMEM reuses KV SMEM (same as BF16 and dense native FP8 kernels)
        var out_smem = kv_smem.bitcast[Scalar[Self.output_type]]()

        var max_smem = (p_smem + Self.PStageElems * Self.num_stages).bitcast[
            Scalar[Self.AccumType]
        ]()
        var li_smem = max_smem + 2 * WARPGROUP_SIZE

        var mbar_base: MBarType = (
            (li_smem + WARPGROUP_SIZE)
            .bitcast[SharedMemBarrier]()
            .as_unsafe_any_origin()
        )

        var mbar_q: MBarType = mbar_base
        var mbar_kv_base: MBarType = mbar_base + 1
        var kv_pipeline = KVPipelineGeneric[
            num_kv_stages=Self.config.num_kv_stages,
            num_qk_stages=1,
            num_producer=1,
            num_consumer=2,
        ](mbar_kv_base)
        mbar_base = mbar_kv_base + kv_pipeline.num_mbars()

        var s_bars = DecodeSM100MiscMBars[
            num_stages=Self.num_stages,
            num_producer=1,
            num_consumer=WARPGROUP_SIZE,
        ](mbar_base)
        mbar_base = s_bars.end()

        var p_bars = DecodeSM100MiscMBars[
            num_stages=Self.num_stages,
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
            num_out_stages=OutPipeType.num_out_stages,
            num_producer=WARPGROUP_SIZE,
            num_consumer=1,
        ](mbar_base)
        mbar_base += out_pipeline.num_mbars()

        # idx_bars: warp 11 (producer, 32 threads) → warp 8 (consumer, 32 threads)
        var idx_bars = DecodeSM100MiscMBars[
            num_stages=2, num_producer=32, num_consumer=32
        ](mbar_base)
        mbar_base = idx_bars.end()

        var ptr_tmem_addr = (mbar_base).bitcast[UInt32]()
        var idx_smem_base = (ptr_tmem_addr + 1).bitcast[Int32]()
        comptime idx_smem_stride = Self.config.BN_QK

        var warp_idx = UInt32(warp_id[broadcast=True]())
        is_leader = elect() != 0

        if warp_idx == 8:
            if is_leader:
                mbar_q[].init(1)
                kv_pipeline.init()
                s_bars.init()
                p_bars.init()
                o_bars.init()
                c_bars.init()
                corr_done_bars.init()
                out_pipeline.init()
                idx_bars.init()
                q_tma.prefetch_descriptor()
                k_tma.prefetch_descriptor()
                o_tma.prefetch_descriptor()
                comptime if Self.has_extra_kv:
                    extra_k_tma.prefetch_descriptor()
        elif warp_idx == 9:
            tcgen05_alloc[Self.config.cta_group](
                ptr_tmem_addr, Self.config.sm100_tmem_cols
            )
        barrier()

        if warp_idx < 4:
            warpgroup_reg_alloc[num_reg_softmax]()

            var attn_sink_log2 = Scalar[DType.float32](
                min_or_neg_inf[DType.float32]()
            )
            comptime if Self.has_attn_sink:
                var lane_idx = Int(lane_id())
                var row = lane_idx & 0x3F
                var head_idx_local = Int(block_idx.x) * Self.config.BM + row
                if head_idx_local < Self.config.num_q_heads:
                    attn_sink_log2 = attn_sink_ptr.unsafe_value()[
                        head_idx_local
                    ] * Scalar[DType.float32](log2e)

            Self.Common_MLA_Op.Softmax[
                native_fp8=True,
                num_sp_stages=Self.num_stages,
                has_attn_sink=Self.has_attn_sink,
                fold_q=Self.fold_shared_index,
                q_len_fold=Self.q_len_fold,
            ](
                ptr_tmem_addr[0],
                s_bars,
                p_bars,
                p_smem.bitcast[
                    Scalar[Self.Common_MLA_Op.q_type]
                ]().as_unsafe_any_origin(),
                max_smem.as_unsafe_any_origin(),
                li_smem.as_unsafe_any_origin(),
                out_smem.as_unsafe_any_origin(),
                c_bars,
                corr_done_bars,
                out_pipeline,
                offset_position,
                scale,
                mask,
                prompt_idx=UInt32(offset_position.batch_idx),
                lse_accum_split_ptr=lse_accum_split_ptr,
                batch_size=batch_size,
                attn_sink_log2=attn_sink_log2,
            )
        elif warp_idx >= 4 and warp_idx < 8:
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
                    q_smem.as_unsafe_any_origin(),
                    kv_smem.as_unsafe_any_origin(),
                    mbar_q,
                    kv_pipeline,
                    offset_position,
                    idx_bars,
                    idx_smem_base,
                    num_orig_blocks,
                    topk,
                    extra_k_tma,
                    extra_topk,
                )
            elif warp_idx == 9:
                Self.mmaQK(
                    ptr_tmem_addr[0],
                    q_smem.as_unsafe_any_origin(),
                    kv_smem.as_unsafe_any_origin(),
                    mbar_q,
                    s_bars,
                    kv_pipeline,
                    offset_position,
                )
            elif warp_idx == 10:
                Self.mmaPV(
                    ptr_tmem_addr[0],
                    kv_smem.as_unsafe_any_origin(),
                    p_smem.as_unsafe_any_origin(),
                    p_bars,
                    o_bars,
                    kv_pipeline,
                    offset_position,
                )
            elif warp_idx == 11:
                var batch_d_indices_w11 = d_indices.unsafe_value() + (
                    offset_position.q_token_idx * indices_stride
                )
                var batch_extra_d_indices_w11 = extra_d_indices
                comptime if Self.has_extra_kv:
                    batch_extra_d_indices_w11 = (
                        extra_d_indices.unsafe_value()
                        + (offset_position.q_token_idx * extra_indices_stride)
                    )
                Self.idx_producer(
                    idx_bars,
                    idx_smem_base,
                    kv_lut,
                    batch_d_indices_w11,
                    topk,
                    offset_position,
                    num_orig_blocks,
                    extra_kv_lut,
                    batch_extra_d_indices_w11,
                    extra_topk,
                )
                Self.Common_MLA_Op.store[
                    fold_q=Self.fold_shared_index,
                    q_len_fold=Self.q_len_fold,
                ](
                    out_pipeline,
                    out_smem.as_unsafe_any_origin(),
                    o_tma,
                    offset_position,
                )
        barrier()

        comptime if Self.config.decoding_warp_split_k:
            launch_dependent_grids()

        if warp_idx == 9:
            tcgen05_release_allocation_lock[Self.config.cta_group]()
            tcgen05_dealloc[Self.config.cta_group](
                ptr_tmem_addr[0], Self.config.sm100_tmem_cols
            )

    @staticmethod
    @always_inline
    def _transform_indices_to_smem(
        d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
        idx_smem: SharedMemPointer[Int32],
        indices_base: Int,
        kv_lut: Self.KVLUTType,
        topk: UInt32,
    ):
        var lane = thread_idx.x & 31
        var max_idx = max(topk, UInt32(1)) - 1

        comptime for row_pass in range(2):
            var row_in_tile = lane + row_pass * 32
            var idx_pos = UInt32(indices_base + row_in_tile)
            var clamped_pos = min(idx_pos, max_idx)
            var raw_index = d_indices.unsafe_value()[Int(clamped_pos)]
            var tma_row = kv_lut.get_tma_row(raw_index)
            if raw_index == -1:
                tma_row = -1
            idx_smem[row_in_tile] = tma_row

    @staticmethod
    @always_inline
    def idx_producer(
        idx_bars: DecodeSM100MiscMBars[
            num_stages=2,
            num_producer=32,
            num_consumer=32,
        ],
        idx_smem_base: SharedMemPointer[Int32],
        kv_lut: Self.KVLUTType,
        d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
        topk: Int,
        offset_position: OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
            sparse=True,
            has_extra_kv=Self.has_extra_kv,
            has_variable_topk=Self.has_variable_topk,
        ],
        num_orig_blocks: Int,
        extra_kv_lut: Self.KVLUTType,
        extra_d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
        extra_topk: Int,
    ):
        var num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN_QK
        )
        if num_k_tiles == 0:
            return

        var idx_prod = idx_bars.producer()
        var orig_topk_u32 = UInt32(topk)

        var num_orig_tiles = num_k_tiles
        comptime if Self.has_extra_kv:
            var orig_tokens_in_split = clamp(
                topk - offset_position.kv_start_row,
                0,
                offset_position.num_keys_this_split,
            )
            num_orig_tiles = ceildiv(orig_tokens_in_split, Self.config.BN_QK)

        var first_tile_from_orig = num_orig_tiles > 0
        var orig_indices_base = Int(offset_position.kv_start_row)
        var kv_stage_idx = UInt32(0)

        if first_tile_from_orig:
            var idx_smem = idx_smem_base + kv_stage_idx * UInt32(
                Self.config.BN_QK
            )
            Self._transform_indices_to_smem(
                d_indices,
                idx_smem,
                orig_indices_base,
                kv_lut,
                orig_topk_u32,
            )
            idx_prod.commit()
            orig_indices_base += Self.config.BN_QK
            kv_stage_idx ^= 1

        var remaining_orig = num_orig_tiles - 1 if first_tile_from_orig else 0
        var t: Int = 0
        while t < remaining_orig:
            idx_prod.acquire()
            var idx_smem = idx_smem_base + kv_stage_idx * UInt32(
                Self.config.BN_QK
            )
            Self._transform_indices_to_smem(
                d_indices,
                idx_smem,
                orig_indices_base,
                kv_lut,
                orig_topk_u32,
            )
            idx_prod.commit()
            orig_indices_base += Self.config.BN_QK
            kv_stage_idx ^= 1
            t += 1

        comptime if Self.has_extra_kv:
            var extra_topk_u32 = UInt32(extra_topk)
            var extra_indices_base = max(
                0, Int(offset_position.kv_start_row) - topk
            )
            var num_extra_tiles = num_k_tiles - num_orig_tiles

            if not first_tile_from_orig:
                var idx_smem = idx_smem_base + kv_stage_idx * UInt32(
                    Self.config.BN_QK
                )
                Self._transform_indices_to_smem(
                    extra_d_indices,
                    idx_smem,
                    extra_indices_base,
                    extra_kv_lut,
                    extra_topk_u32,
                )
                idx_prod.commit()
                extra_indices_base += Self.config.BN_QK
                kv_stage_idx ^= 1
                num_extra_tiles -= 1

            var te: Int = 0
            while te < num_extra_tiles:
                idx_prod.acquire()
                var idx_smem = idx_smem_base + kv_stage_idx * UInt32(
                    Self.config.BN_QK
                )
                Self._transform_indices_to_smem(
                    extra_d_indices,
                    idx_smem,
                    extra_indices_base,
                    extra_kv_lut,
                    extra_topk_u32,
                )
                idx_prod.commit()
                extra_indices_base += Self.config.BN_QK
                kv_stage_idx ^= 1
                te += 1

    @staticmethod
    @always_inline
    def load(
        q_tma: QOTMATile[
            dtype=Self.fp8_type,
            BM=Self.config.BM,
            BK=Self.config.BK_QK,
            swizzle_mode=Self.config.kv_tma_swizzle_mode,
        ],
        k_tma: TMATensorTile[
            Self.fp8_type,
            2,
            tile_shape=IndexList[2](Self.config.BK_PV, Self.kv_gather4_box_w),
            desc_shape=IndexList[2](1, Self.kv_gather4_box_w),
        ],
        kv_lut: Self.KVLUTType,
        q_smem: SharedMemPointer[Scalar[Self.fp8_type]],
        kv_smem: SharedMemPointer[Scalar[Self.fp8_type]],
        mbar_q: MBarType,
        kv_pipeline: KVPipelineGeneric[
            num_kv_stages=Self.config.num_kv_stages,
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
            sparse=True,
            has_extra_kv=Self.has_extra_kv,
            has_variable_topk=Self.has_variable_topk,
        ],
        idx_bars: DecodeSM100MiscMBars[
            num_stages=2,
            num_producer=32,
            num_consumer=32,
        ],
        idx_smem_base: SharedMemPointer[Int32],
        num_orig_blocks: Int,
        topk: Int,
        extra_k_tma: TMATensorTile[
            Self.fp8_type,
            2,
            tile_shape=IndexList[2](Self.config.BK_PV, Self.kv_gather4_box_w),
            desc_shape=IndexList[2](1, Self.kv_gather4_box_w),
        ],
        extra_topk: Int,
    ):
        var num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN_QK
        )
        if num_k_tiles == 0:
            return

        var kv_prod = DecodeKVProducer[Self.fp8_type, Self.config](
            kv_pipeline, kv_smem
        )
        var elect_mask = elect()
        var is_leader = elect_mask != 0
        var row: Int = offset_position.q_row_offset

        var num_orig_tiles = num_k_tiles
        comptime if Self.has_extra_kv:
            var orig_tokens_in_split = clamp(
                topk - offset_position.kv_start_row,
                0,
                offset_position.num_keys_this_split,
            )
            num_orig_tiles = ceildiv(orig_tokens_in_split, Self.config.BN_QK)

        expect_bytes_pred(
            mbar_q,
            Int32(
                Self.config.BM
                * Self.config.q_depth
                * Self.fp8_bytes_per_element
            ),
            elect_mask,
        )
        if is_leader:
            comptime q_elems = type_of(q_tma).tile_shape[0] * type_of(
                q_tma
            ).tile_shape[1]
            comptime q_tt_layout = tt_row_major[q_elems]()
            var q_smem_tensor = TileTensor[
                Self.fp8_type,
                type_of(q_tt_layout),
                MutAnyOrigin,
                address_space=AddressSpace.SHARED,
            ](q_smem.bitcast[Scalar[Self.fp8_type]](), q_tt_layout)
            q_tma.async_copy(q_smem_tensor, mbar_q[], (0, row))

        comptime kv_bytes = Self.config.BN_QK * Self.kv_gather4_tile_width * size_of[
            Self.fp8_type
        ]()

        var first_tile_from_orig = num_orig_tiles > 0
        var idx_cons = idx_bars.consumer()

        if first_tile_from_orig:
            Self._load_one_tile(
                kv_prod,
                is_leader,
                k_tma,
                idx_cons,
                idx_smem_base,
                kv_bytes,
            )
            # Wait here so Q load overlaps with the first KV transfer.
            mbar_q[].wait(0)

        var remaining_orig = num_orig_tiles - 1 if first_tile_from_orig else 0
        Self._load_tile_range(
            kv_prod,
            is_leader,
            k_tma,
            idx_cons,
            idx_smem_base,
            remaining_orig,
        )

        comptime if Self.has_extra_kv:
            var num_extra_tiles = num_k_tiles - num_orig_tiles

            if not first_tile_from_orig:
                Self._load_one_tile(
                    kv_prod,
                    is_leader,
                    extra_k_tma,
                    idx_cons,
                    idx_smem_base,
                    kv_bytes,
                )
                # First tile comes from extra_kv since orig was empty.
                mbar_q[].wait(0)
                num_extra_tiles -= 1

            Self._load_tile_range(
                kv_prod,
                is_leader,
                extra_k_tma,
                idx_cons,
                idx_smem_base,
                num_extra_tiles,
            )

    @staticmethod
    @always_inline
    def _load_one_tile(
        mut kv_prod: DecodeKVProducer[Self.fp8_type, Self.config],
        is_leader: Bool,
        cur_k_tma: TMATensorTile[
            Self.fp8_type,
            2,
            tile_shape=IndexList[2](Self.config.BK_PV, Self.kv_gather4_box_w),
            desc_shape=IndexList[2](1, Self.kv_gather4_box_w),
        ],
        mut idx_cons: ConsumerPipeline[2],
        idx_smem_base: SharedMemPointer[Int32],
        kv_bytes: Int,
    ):
        var kv_stage_ptr = kv_prod.stage_base_ptr[qk_stage=0]()
        var k_mbar = kv_prod.producer_mbar[qk_stage=0]()

        idx_cons.wait()
        var idx_smem = idx_smem_base + idx_cons.state.index() * UInt32(
            Self.config.BN_QK
        )

        expect_bytes_pred(k_mbar, Int32(kv_bytes), Int32(is_leader))
        if is_leader:
            cur_k_tma.async_copy_gather4_tile[
                tile_width=Self.config.padded_q_depth,
                eviction_policy=CacheEviction.EVICT_LAST,
            ](
                kv_stage_ptr,
                k_mbar[],
                idx_smem,
                start_idx=0,
            )

        idx_cons.release()
        kv_prod.commit_step()

    @staticmethod
    @always_inline
    def _load_tile_range(
        mut kv_prod: DecodeKVProducer[Self.fp8_type, Self.config],
        is_leader: Bool,
        cur_k_tma: TMATensorTile[
            Self.fp8_type,
            2,
            tile_shape=IndexList[2](Self.config.BK_PV, Self.kv_gather4_box_w),
            desc_shape=IndexList[2](1, Self.kv_gather4_box_w),
        ],
        mut idx_cons: ConsumerPipeline[2],
        idx_smem_base: SharedMemPointer[Int32],
        num_tiles: Int,
    ):
        comptime kv_bytes = Self.config.BN_QK * Self.kv_gather4_tile_width * size_of[
            Self.fp8_type
        ]()

        var t: Int = 0
        while t < num_tiles:
            kv_prod.acquire[qk_stage=0]()

            var kv_stage_ptr = kv_prod.stage_base_ptr[qk_stage=0]()
            var k_mbar = kv_prod.producer_mbar[qk_stage=0]()

            idx_cons.wait()
            var idx_smem = idx_smem_base + idx_cons.state.index() * UInt32(
                Self.config.BN_QK
            )

            expect_bytes_pred(k_mbar, Int32(kv_bytes), Int32(is_leader))
            if is_leader:
                cur_k_tma.async_copy_gather4_tile[
                    tile_width=Self.config.padded_q_depth,
                    eviction_policy=CacheEviction.EVICT_LAST,
                ](
                    kv_stage_ptr,
                    k_mbar[],
                    idx_smem,
                    start_idx=0,
                )

            idx_cons.release()
            kv_prod.commit_step()
            t += 1

    @staticmethod
    @always_inline
    def mmaQK(
        tmem_addr: UInt32,
        q_smem: SharedMemPointer[Scalar[Self.fp8_type]],
        kv_smem: SharedMemPointer[Scalar[Self.fp8_type]],
        mbar_q: MBarType,
        s_bars: DecodeSM100MiscMBars[
            num_stages=Self.num_stages,
            num_producer=1,
            num_consumer=WARPGROUP_SIZE,
        ],
        kv_pipeline: KVPipelineGeneric[
            num_kv_stages=Self.config.num_kv_stages,
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
            sparse=True,
            has_extra_kv=Self.has_extra_kv,
            has_variable_topk=Self.has_variable_topk,
        ],
    ):
        var s0_tmem = tmem_addr + UInt32(Self.config.TMEM_S0)
        var elect_mask = elect()

        num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN_QK
        )
        if num_k_tiles == 0:
            return

        var kv_cons = DecodeKVConsumer[Self.fp8_type, Self.config](
            kv_pipeline, kv_smem
        )
        var s_prod = DecodeSProducerN[Self.num_stages](s_bars.producer())
        comptime s_stride = UInt32(Self.config.TMEM_S1 - Self.config.TMEM_S0)

        var q_descriptor = Self.UMMAQKTSS.descriptor_q_block(q_smem)
        var k_descriptor = Self.UMMAQKTSS.descriptor_k_block(kv_smem)
        comptime stage_stride_in_bytes = (
            Self.KVStageElems * Self.fp8_bytes_per_element
        )

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
            tile_idx += 1

    @staticmethod
    @always_inline
    def mmaPV(
        tmem_addr: UInt32,
        kv_smem: SharedMemPointer[Scalar[Self.fp8_type]],
        p_smem: SharedMemPointer[Scalar[Self.fp8_type]],
        p_bars: DecodeSM100MiscMBars[
            num_stages=Self.num_stages,
            num_producer=WARPGROUP_SIZE,
            num_consumer=1,
        ],
        o_bars: DecodeSM100MiscMBars[
            num_stages=2, num_producer=1, num_consumer=WARPGROUP_SIZE
        ],
        kv_pipeline: KVPipelineGeneric[
            num_kv_stages=Self.config.num_kv_stages,
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
            sparse=True,
            has_extra_kv=Self.has_extra_kv,
            has_variable_topk=Self.has_variable_topk,
        ],
    ):
        var o_tmem = tmem_addr + UInt32(Self.config.TMEM_O)
        var elect_mask = elect()
        num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN_QK
        )
        if num_k_tiles == 0:
            return

        comptime s_stride = UInt32(Self.config.TMEM_S1 - Self.config.TMEM_S0)
        var kv_cons = DecodeKVConsumer[Self.fp8_type, Self.config](
            kv_pipeline, kv_smem
        )
        var p_cons = DecodePConsumerN[Self.num_stages](p_bars.consumer())
        var o_prod = DecodeOProducer(o_bars.producer())

        var p_descriptor = Self.UMMAPVSS.descriptor_p_block(p_smem)
        var v_descriptor = Self.UMMAPVSS.descriptor_v_block(kv_smem)
        comptime block_step = Self.config.MMA_PV_N // Self.config.BN_QK
        comptime kv_stage_stride_in_bytes = (
            Self.KVStageElems * Self.fp8_bytes_per_element
        )
        comptime p_stage_stride_in_bytes = (
            Self.PStageElems * Self.fp8_bytes_per_element
        )
        comptime block_stride_in_bytes = (
            Self.BlockElems * Self.fp8_bytes_per_element
        )

        var tile_idx: Int = 0
        var c_scale: UInt32 = 0
        while tile_idx < num_k_tiles:
            kv_cons.wait[qk_stage=0]()
            var p_slot_index = p_cons.wait()
            var v_slot_index = kv_cons.stage_index[qk_stage=0]()

            comptime for block in range(0, Self.NumVOBlocks, block_step):
                o_prod.acquire()
                Self.UMMAPVSS.mma[stage_idx=0](
                    a=p_descriptor
                    + p_slot_index * UInt32(p_stage_stride_in_bytes),
                    b=v_descriptor
                    + v_slot_index * UInt32(kv_stage_stride_in_bytes)
                    + UInt32(block * block_stride_in_bytes),
                    c=o_tmem + UInt32(block) * UInt32(Self.config.BN_QK // 2),
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
