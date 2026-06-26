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
"""SM100 (B200) sparse MLA decode kernel with BF16 KV cache.

K is loaded by a single BF16 + SWIZZLE_128B gather4 TMA covering the full
576-element row (tile_width=576, box_w=64). `OffsetPosition[sparse=True]`
overrides `num_keys` with the sparse topk; double-buffered `idx_bars` and
`idx_smem` pipe per-tile row indices from warp 11 (producer) to warp 8
(consumer) one tile ahead.

Supports NullMask and CausalMask. Sliding-window attention is FP8-only.
"""

from std.collections import OptionalReg
from std.math import ceildiv, clamp
from std.math.constants import log2e
from std.sys import size_of
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    barrier,
    block_idx,
    lane_id,
    thread_idx,
    warp_id,
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
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    _gather4_box_width,
)
from layout import ComptimeInt, CoordLike, RowMajorLayout, TileTensor
from nn.attention.gpu.nvidia.common import (
    OptionalPointer,
)
from nn.attention.mha_mask import MHAMask
from nn.attention.mha_operand import MHAOperand
from std.utils.index import IndexList
from std.utils.numerics import get_accum_type, min_or_neg_inf
from std.utils.static_tuple import StaticTuple

from nn.attention.gpu.nvidia.sm100.attention_utils import (
    ConsumerPipeline,
    elect,
    expect_bytes_pred,
    SharedMemPointer,
    MBarType,
)

from nn.attention.gpu.nvidia.sm100.mla_decode_utils import (
    MLA_SM100_Decode_Config,
    MLA_SM100_Decode_Common,
    QOTMATile,
    MLA_Decode_Pack,
    OffsetPosition,
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
# SM100 sparse MLA decode kernel — BF16 KV variant.
#
# 3-warpgroup layout: warps 0-3 Softmax, 4-7 Correction, 8-11 Load/MMA-QK/
# MMA-PV/Store.  `num_kv_stages=2`, single `KVPipelineGeneric`, P aliased
# onto the rope slot of KV SMEM.  K TMA is a single BF16 + SWIZZLE_128B
# gather4 descriptor `TMATensorTile[bfloat16, 2, ..., tile_shape=(64, 64)]`;
# warp 11 transforms sparse indices via `kv_lut.get_tma_row()` one tile
# ahead of warp 8.
# ------------------------------------------------------------------------------
struct MLA_SM100_Decode_Sparse_KV_BF16[
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
](TrivialRegisterPassable):
    comptime kv_type = Self.KVLUTType.dtype
    comptime AccumType = get_accum_type[Self.q_type]()
    # 576 / 64 = 9
    comptime NumQKBlocks = Self.config.padded_q_depth // Self.config.BN_QK
    # 512 / 64 = 8
    comptime NumVOBlocks = Self.config.padded_depth // Self.config.BN_QK
    # 64 * 64 = 4096
    comptime BlockElems = Self.config.BM * Self.config.BN_QK
    # 2 bytes for bfloat16
    comptime bytes_per_element = size_of[Self.q_type]()
    # the stage element is the same for both K and V
    comptime KVStageElems = Self.NumQKBlocks * Self.BlockElems
    comptime output_tile_width = (Self.config.BN_QK // 2) * (
        4 // size_of[Self.output_type]()
    )

    # Single BF16 + SWIZZLE_128B gather4 descriptor covers the full
    # 576-element K row in 9 col-groups x 16 4-row chunks = 144 gather4
    # PTX calls per tile.  Box width is 64 BF16 elems = 128 bytes (one
    # swizzle group), so the gather4 SMEM layout is directly consumable
    # by the UMMA K-major descriptor.
    comptime kv_gather4_tile_width = Self.config.padded_q_depth
    comptime kv_gather4_box_w = _gather4_box_width[
        DType.bfloat16,
        Self.kv_gather4_tile_width,
        TensorMapSwizzle.SWIZZLE_128B,
    ]()

    comptime UMMAQKTSS = DecodeSM100QKTSS[
        operand_type=Self.q_type,
        accum_type=Self.AccumType,
        config=Self.config,
    ]
    comptime UMMAPVSS = DecodeSM100PVSS[
        operand_type=Self.q_type,
        accum_type=Self.AccumType,
        config=Self.config,
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

    # --------------------------------------------------------------------------
    # Sparse BF16 KV decode kernel entry.
    # --------------------------------------------------------------------------
    #    3 Warpgroups: Softmax WG (warps 0-3), Correction WG (warps 4-7),
    #                  MMA+Load+Store WG (warps 8-11).
    #    Warp assignments within WG2: warp 8 = Load (consumes idx pipe and
    #    issues gather4 TMA), warp 9 = MMA QK, warp 10 = MMA PV,
    #    warp 11 = idx producer during the main loop and output Store in
    #    the epilogue.

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
    @__llvm_metadata(`nvvm.minctasm`=SIMDSize(1))
    @__name(
        t"sm100_mla_decode_sparse_kv_bf16_{Self.q_type}_{Self.kv_type}_{Self.output_type}_nqh{Self.config.num_q_heads}_nkvh{Self.config.num_kv_heads}",
    )
    def kernel(
        q_tma: QOTMATile[
            dtype=Self.q_type,
            BM=Self.config.BM,  # 64
            BK=Self.config.BK_QK,  # 576
            swizzle_mode=Self.config.swizzle_mode,
        ],
        # Single BF16 gather4 TMA: SWIZZLE_128B, BN_QK rows, box_w=64 BF16
        # elems.  The full K row (576 BF16 = 1152 bytes) is loaded across
        # 9 col-groups by `async_copy_gather4_tile` internally.
        k_tma: TMATensorTile[
            DType.bfloat16,
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
        attn_sink_ptr: OptionalReg[
            UnsafePointer[Scalar[DType.float32], origin=MutAnyOrigin]
        ],
        # Extra KV TMA: BF16, SWIZZLE_128B, tile_width=padded_q_depth=576,
        # box_w=_gather4_box_width[bfloat16, 576, SWIZZLE_128B]()=64.
        # Same descriptor shape as the main K_TMA.
        extra_k_tma: TMATensorTile[
            DType.bfloat16,
            2,
            tile_shape=IndexList[2](Self.config.BK_PV, Self.kv_gather4_box_w),
            desc_shape=IndexList[2](1, Self.kv_gather4_box_w),
        ],
        extra_kv_lut: Self.KVLUTType,
        extra_d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
        extra_topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]],
        extra_indices_stride: Int,
        scalar_args: TileTensor[
            DType.int64,
            RowMajorLayout[ComptimeInt[3]],
            MutAnyOrigin,
        ],
    ):
        # The upstream dispatcher monomorphizes the kernel struct for both
        # `decoding_warp_split_k=True` and `False`; the split-K branch is
        # selected at runtime via `num_partitions > 1`.
        comptime assert Self.KVLUTType.dtype == DType.bfloat16
        comptime assert size_of[Self.KVLUTType.dtype]() == 2
        comptime assert Self.config.supported()
        comptime assert Self.config.scale_block_size == 0
        comptime assert not Self.config.decode_layout_g
        comptime assert Self.config.cta_group == 1
        comptime assert Self.config.num_threads == WARPGROUP_SIZE * 3

        # Mask: NullMask or CausalMask (sliding-window is FP8-only).
        comptime _mask_type_name: String = Self.MaskType.get_type_name()
        comptime assert (
            _mask_type_name == "NullMask" or _mask_type_name == "CausalMask"
        ), (
            "MLA_SM100_Decode_Sparse_KV_BF16 only supports NullMask and"
            " CausalMask. Sliding window is FP8-only."
        )
        comptime num_reg_softmax = 192
        comptime num_reg_correction = 184
        comptime num_reg_other = 112
        var batch_size = Int(scalar_args.raw_load(0))
        var q_max_seq_len = Int(scalar_args.raw_load(1))
        var num_partitions = Int(scalar_args.raw_load(2))
        mask = mla_decode_pack.mask
        valid_length = mla_decode_pack.valid_length
        var lse_accum_split_ptr = mla_decode_pack.lse_accum_split_ptr
        # OffsetPosition[sparse=True] overrides num_keys with topk
        # (clamped to actual_tokens by `OffsetPosition.__init__`).
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

        # Re-derive topk and extra_topk for block-level control flow:
        # these decide which blocks belong to the original vs extra cache
        # inside the kernel loop.
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
        # `num_keys` from OffsetPosition is topk+extra_topk; back-derive
        # the clamped topk.
        topk = offset_position.num_keys - extra_topk

        var num_orig_blocks = ceildiv(topk, Self.config.BN_QK)

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

        # Skip query positions beyond this batch's actual seq_len.  In
        # ragged mode with split-K, q_max_seq_len can be > 1 (up to 8),
        # so block_idx.y can exceed a specific batch's seq_len.  Those
        # CTAs must still fulfill the PDL contract (write -inf LSE, zero
        # o_accum_split, and call launch_dependent_grids) or the combine
        # kernel will hang.
        comptime if Self.ragged:
            if block_idx.y >= offset_position.seq_len:
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

        q_smem = external_memory[
            Scalar[Self.q_type],
            address_space=AddressSpace.SHARED,
            alignment=128,
            name="mha_dynamic_shared_memory",
        ]()
        var kv_smem = (q_smem + Self.BlockElems * Self.NumQKBlocks).bitcast[
            Scalar[Self.kv_type]
        ]()
        comptime kv_total_stages = Self.config.num_kv_stages
        comptime kv_smem_total = Self.BlockElems * Self.NumQKBlocks * kv_total_stages

        var out_smem_start = kv_smem
        var out_smem_total = kv_smem_total

        var out_smem = out_smem_start.bitcast[Scalar[Self.output_type]]()

        var max_smem = (out_smem + out_smem_total).bitcast[
            Scalar[Self.AccumType]
        ]()

        var li_smem = max_smem + 2 * WARPGROUP_SIZE

        var mbar_base: MBarType = (
            (li_smem + WARPGROUP_SIZE)
            .bitcast[SharedMemBarrier]()
            .as_unsafe_any_origin()
        )

        var mbar_q: MBarType = mbar_base  # q uses 0
        var mbar_kv_base: MBarType = mbar_base + 1  # barrier total[1]

        var kv_pipeline = KVPipelineGeneric[
            num_kv_stages=Self.config.num_kv_stages,  # 2
            num_qk_stages=1,
            num_producer=1,
            num_consumer=2,
        ](mbar_kv_base)

        mbar_base = mbar_kv_base + kv_pipeline.num_mbars()  # kv uses 1..4
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
        var c_bars = DecodeSM100MiscMBars[
            num_stages=1,
            num_producer=WARPGROUP_SIZE,
            num_consumer=WARPGROUP_SIZE,
        ](
            mbar_base
        )  # C uses 17..18
        mbar_base = c_bars.end()  # barrier total [19]
        var corr_done_bars = DecodeSM100MiscMBars[
            num_stages=2,
            num_producer=WARPGROUP_SIZE,
            num_consumer=WARPGROUP_SIZE,
        ](
            mbar_base
        )  # corr_done uses 19..22
        mbar_base = corr_done_bars.end()  # barrier total [23]
        comptime OutPipeType = DecodeOutProducer[Self.output_type, Self.config]
        var out_pipeline = OutPipeline[
            num_out_stages=OutPipeType.num_out_stages,
            num_producer=WARPGROUP_SIZE,
            num_consumer=1,
        ](mbar_base)
        mbar_base += out_pipeline.num_mbars()

        # Sparse index pipeline: warp 11 (producer, 32 threads) transforms
        # d_indices -> TMA rows; warp 8 (consumer) waits before issuing TMA.
        # 2-stage double-buffer so warp 11 runs one tile ahead of warp 8.
        var idx_bars = DecodeSM100MiscMBars[
            num_stages=2, num_producer=32, num_consumer=32
        ](mbar_base)
        mbar_base = idx_bars.end()  # +4 barriers

        var ptr_tmem_addr = (mbar_base).bitcast[UInt32]()

        # Double-buffered idx SMEM: 2 * BN_QK Int32 row indices.
        var idx_smem_base = (ptr_tmem_addr + 1).bitcast[Int32]()

        var warp_idx = UInt32(warp_id[broadcast=True]())
        is_leader = elect() != 0
        if warp_idx == 8:
            if is_leader:
                mbar_q[].init(1)
                kv_pipeline.init()
                s_bars.init()
                p_bars.init()
                idx_bars.init()
                o_bars.init()
                c_bars.init()
                out_pipeline.init()
                corr_done_bars.init()
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

        if warp_idx < 4:  # softmax warpgroup
            warpgroup_reg_alloc[num_reg_softmax]()

            # Per-head attn_sink_log2 (one head per thread in the warpgroup).
            # When attn_sink_ptr is null, attn_sink_log2 stays at -inf and
            # exp2(-inf - mi) = 0, leaving the denominator unchanged.
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

            Self.Common_MLA_Op.Softmax[has_attn_sink=Self.has_attn_sink,](
                ptr_tmem_addr[0],
                s_bars,
                p_bars,
                kv_smem.bitcast[Scalar[Self.q_type]]().as_unsafe_any_origin(),
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
                    (kv_smem)
                    .bitcast[Scalar[Self.q_type]]()
                    .as_unsafe_any_origin(),
                    mbar_q,
                    s_bars,
                    kv_pipeline,
                    offset_position,
                )
            elif warp_idx == 10:
                Self.mmaPV(
                    ptr_tmem_addr[0],
                    (kv_smem)
                    .bitcast[Scalar[Self.q_type]]()
                    .as_unsafe_any_origin(),
                    p_bars,
                    o_bars,
                    kv_pipeline,
                    offset_position,
                )
            elif warp_idx == 11:
                # Warp 11 runs the idx producer during the main loop, then
                # transitions into the output store epilogue.
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
                Self.Common_MLA_Op.store(
                    out_pipeline,
                    out_smem.as_unsafe_any_origin(),
                    o_tma,
                    offset_position,
                )
        barrier()

        # PDL: signal that this CTA is done so the combine kernel can
        # start.  Must be called by all threads in the CTA after all work
        # is complete.
        comptime if Self.config.decoding_warp_split_k:
            launch_dependent_grids()

        if warp_idx == 9:
            tcgen05_release_allocation_lock[Self.config.cta_group]()
            tcgen05_dealloc[Self.config.cta_group](
                ptr_tmem_addr[0], Self.config.sm100_tmem_cols
            )

    # --------------------------------------------------------------------------
    # Sparse index transform: physical_block * page_size + offset -> TMA row.
    # All 32 threads of warp 11 cooperatively transform BN_QK=64 entries
    # (each thread handles lane and lane+32).
    # --------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------
    # Index producer (warp 11): transforms one tile's row indices ahead of
    # the gather4 TMA.  With `has_extra_kv`, extra_topk tokens from the
    # separate cache are appended after the original topk tokens in the
    # same loop, matching warp 8's load order.
    # --------------------------------------------------------------------------
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

        # Original vs extra tile counts (mirrors warp 8's load logic).
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

        # First tile: no acquire — idx_free starts ready.
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

        # Remaining original tiles: acquire before produce.
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

        # Extra KV tiles.
        comptime if Self.has_extra_kv:
            var extra_topk_u32 = UInt32(extra_topk)
            var extra_indices_base = max(
                0, Int(offset_position.kv_start_row) - topk
            )
            var num_extra_tiles = num_k_tiles - num_orig_tiles

            if not first_tile_from_orig:
                # First tile drawn from the extra cache — no acquire.
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

    # --------------------------------------------------------------------------
    # Single-tile load helpers shared by the original-cache and extra-cache
    # loops.
    # --------------------------------------------------------------------------
    @staticmethod
    @always_inline
    def _load_one_tile(
        mut kv_prod: DecodeKVProducer[Self.kv_type, Self.config],
        is_leader: Bool,
        elect_mask: Int32,
        cur_k_tma: TMATensorTile[
            DType.bfloat16,
            2,
            tile_shape=IndexList[2](Self.config.BK_PV, Self.kv_gather4_box_w),
            desc_shape=IndexList[2](1, Self.kv_gather4_box_w),
        ],
        mut idx_cons: ConsumerPipeline[2],
        idx_smem_base: SharedMemPointer[Int32],
        kv_bytes: Int,
    ):
        """First KV tile of a cache: no kv pipeline acquire (starts ready)."""
        var kv_stage_ptr = kv_prod.stage_base_ptr[qk_stage=0]()
        var k_mbar = kv_prod.producer_mbar[qk_stage=0]()

        idx_cons.wait()
        var idx_smem = idx_smem_base + idx_cons.state.index() * UInt32(
            Self.config.BN_QK
        )

        expect_bytes_pred(k_mbar, Int32(kv_bytes), elect_mask)
        if is_leader:
            cur_k_tma.async_copy_gather4_tile[
                tile_width=Self.config.padded_q_depth,
                eviction_policy=CacheEviction.EVICT_LAST,
            ](
                kv_stage_ptr.bitcast[Scalar[DType.bfloat16]](),
                k_mbar[],
                idx_smem,
                start_idx=0,
            )
        idx_cons.release()
        kv_prod.commit_step()

    @staticmethod
    @always_inline
    def _load_tile_range(
        mut kv_prod: DecodeKVProducer[Self.kv_type, Self.config],
        is_leader: Bool,
        elect_mask: Int32,
        cur_k_tma: TMATensorTile[
            DType.bfloat16,
            2,
            tile_shape=IndexList[2](Self.config.BK_PV, Self.kv_gather4_box_w),
            desc_shape=IndexList[2](1, Self.kv_gather4_box_w),
        ],
        mut idx_cons: ConsumerPipeline[2],
        idx_smem_base: SharedMemPointer[Int32],
        num_tiles: Int,
    ):
        """Load remaining KV tiles from a cache using a single gather4 TMA."""
        comptime kv_bytes = Self.config.BN_QK * Self.config.q_depth * size_of[
            DType.bfloat16
        ]()

        var t: Int = 0
        while t < num_tiles:
            kv_prod.acquire[qk_stage=0]()

            var stage_ptr = kv_prod.stage_base_ptr[qk_stage=0]()
            var k_mbar = kv_prod.producer_mbar[qk_stage=0]()

            idx_cons.wait()
            var idx_smem = idx_smem_base + idx_cons.state.index() * UInt32(
                Self.config.BN_QK
            )

            expect_bytes_pred(k_mbar, Int32(kv_bytes), elect_mask)
            if is_leader:
                cur_k_tma.async_copy_gather4_tile[
                    tile_width=Self.config.padded_q_depth,
                    eviction_policy=CacheEviction.EVICT_LAST,
                ](
                    stage_ptr.bitcast[Scalar[DType.bfloat16]](),
                    k_mbar[],
                    idx_smem,
                    start_idx=0,
                )
            idx_cons.release()
            kv_prod.commit_step()
            t += 1

    # --------------------------------------------------------------------------
    # Load warp (warp 8): issues one gather4 TMA per K tile.  With
    # `has_extra_kv`, the extra cache is loaded contiguously after the
    # original cache, matching warp 11's idx producer.
    # --------------------------------------------------------------------------
    @staticmethod
    @always_inline
    def load(
        q_tma: QOTMATile[
            dtype=Self.q_type,
            BM=Self.config.BM,
            BK=Self.config.BK_QK,
            swizzle_mode=Self.config.swizzle_mode,
        ],
        k_tma: TMATensorTile[
            DType.bfloat16,
            2,
            tile_shape=IndexList[2](Self.config.BK_PV, Self.kv_gather4_box_w),
            desc_shape=IndexList[2](1, Self.kv_gather4_box_w),
        ],
        kv_lut: Self.KVLUTType,
        q_smem: SharedMemPointer[Scalar[Self.q_type]],
        kv_smem: SharedMemPointer[Scalar[Self.kv_type]],
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
            num_stages=2, num_producer=32, num_consumer=32
        ],
        idx_smem_base: SharedMemPointer[Int32],
        num_orig_blocks: Int,
        topk: Int,
        extra_k_tma: TMATensorTile[
            DType.bfloat16,
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

        var kv_prod = DecodeKVProducer[Self.kv_type, Self.config](
            kv_pipeline, kv_smem
        )
        var elect_mask = elect()
        var is_leader = elect_mask != 0
        var row: Int = offset_position.q_row_offset

        # Number of original KV tiles to load in this split.
        var num_orig_tiles = num_k_tiles
        comptime if Self.has_extra_kv:
            var orig_tokens_in_split = clamp(
                topk - offset_position.kv_start_row,
                0,
                offset_position.num_keys_this_split,
            )
            num_orig_tiles = ceildiv(orig_tokens_in_split, Self.config.BN_QK)

        # Q expect-bytes on mbar_q.
        expect_bytes_pred(
            mbar_q,
            Int32(
                Self.config.BM * Self.config.q_depth * size_of[Self.q_type]()
            ),
            elect_mask,
        )
        if is_leader:
            Self.Common_MLA_Op.load_q(q_tma, q_smem, mbar_q, 0, row)

        # Single gather4 transfers BN_QK rows x q_depth BF16 elems per
        # tile.  Byte count must match what `async_copy_gather4_tile`
        # actually writes: BN_QK * box_w * num_col_groups *
        # sizeof(bfloat16) = BN_QK * q_depth * 2 (box_w=64, num_col_groups
        # = 576/64 = 9) = 64 * 576 * 2 = 73728 bytes.
        comptime kv_bytes = Self.config.BN_QK * Self.config.q_depth * size_of[
            DType.bfloat16
        ]()

        var first_tile_from_orig = num_orig_tiles > 0
        var idx_cons = idx_bars.consumer()

        # First tile: no kv pipeline acquire — starts ready.
        if first_tile_from_orig:
            Self._load_one_tile(
                kv_prod,
                is_leader,
                elect_mask,
                k_tma,
                idx_cons,
                idx_smem_base,
                kv_bytes,
            )

        # Remaining original tiles: acquire before each.
        var remaining_orig = num_orig_tiles - 1 if first_tile_from_orig else 0
        Self._load_tile_range(
            kv_prod,
            is_leader,
            elect_mask,
            k_tma,
            idx_cons,
            idx_smem_base,
            remaining_orig,
        )

        # Extra KV tiles.
        comptime if Self.has_extra_kv:
            var num_extra_tiles = num_k_tiles - num_orig_tiles

            if not first_tile_from_orig:
                # First tile drawn from extra cache — no kv pipeline acquire.
                Self._load_one_tile(
                    kv_prod,
                    is_leader,
                    elect_mask,
                    extra_k_tma,
                    idx_cons,
                    idx_smem_base,
                    kv_bytes,
                )
                num_extra_tiles -= 1

            Self._load_tile_range(
                kv_prod,
                is_leader,
                elect_mask,
                extra_k_tma,
                idx_cons,
                idx_smem_base,
                num_extra_tiles,
            )

    # --------------------------------------------------------------------------
    # MMA QK warp (warp 9).  UMMAQKTSS reads BF16 + SWIZZLE_128B and the
    # gather4 TMA lands BF16 in SWIZZLE_128B SMEM, so the descriptor layout
    # matches the dense BF16 reader.
    # --------------------------------------------------------------------------
    @staticmethod
    @always_inline
    def mmaQK(
        tmem_addr: UInt32,
        q_smem: SharedMemPointer[Scalar[Self.q_type]],
        kv_smem: SharedMemPointer[Scalar[Self.q_type]],
        mbar_q: MBarType,
        s_bars: DecodeSM100MiscMBars[
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
        var s0_tmem = tmem_addr + UInt32(Self.config.TMEM_S0)
        var elect_mask = elect()

        num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN_QK
        )
        if num_k_tiles == 0:
            return

        var kv_cons = DecodeKVConsumer[Self.q_type, Self.config](
            kv_pipeline, kv_smem
        )
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
            tile_idx += 1

    # --------------------------------------------------------------------------
    # MMA PV warp (warp 10).  P SMEM aliases the rope slot of KV SMEM.
    # --------------------------------------------------------------------------
    @staticmethod
    @always_inline
    def mmaPV(
        tmem_addr: UInt32,
        kv_smem: SharedMemPointer[Scalar[Self.q_type]],
        p_bars: DecodeSM100MiscMBars[
            num_stages=2, num_producer=WARPGROUP_SIZE, num_consumer=1
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
        var kv_cons = DecodeKVConsumer[Self.q_type, Self.config](
            kv_pipeline, kv_smem
        )
        var p_cons = DecodePConsumer(p_bars.consumer())
        var o_prod = DecodeOProducer(o_bars.producer())
        # P aliases the upper NumVOBlocks slot of KV SMEM.  PV stops at
        # NumVOBlocks (= 8); the 9th K block is the rope slot that P
        # occupies after softmax.
        var p_smem_base = kv_smem + Self.NumVOBlocks * Self.BlockElems
        var p_descriptor = Self.UMMAPVSS.descriptor_p_block(p_smem_base)
        var v_descriptor = Self.UMMAPVSS.descriptor_v_block(kv_smem)
        comptime block_step = Self.config.MMA_PV_N // Self.config.BN_QK
        comptime stage_stride_in_bytes = Self.KVStageElems * Self.bytes_per_element
        comptime block_stride_in_bytes = Self.BlockElems * Self.bytes_per_element

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
                    + p_slot_index * UInt32(stage_stride_in_bytes),
                    b=v_descriptor
                    + v_slot_index * UInt32(stage_stride_in_bytes)
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
