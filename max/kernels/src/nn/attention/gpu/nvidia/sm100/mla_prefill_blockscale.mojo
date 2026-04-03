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

from std.sys import size_of
from std.math import ceildiv, align_up
from nn.attention.mha_operand import MHAOperand
from nn.attention.mha_mask import MHAMask, TileMaskStatus
from nn.attention.gpu.nvidia.mha_tile_scheduler import (
    SeqInfo,
    TransientScheduler,
)
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    StagedPipeline,
    KProducerPipeline,
    VProducerPipeline,
    KConsumerPipeline,
    VConsumerPipeline,
    SharedMemPointer,
    elect,
)
from nn.attention.gpu.mha import q_num_matrix_view_rows
from nn.attention.gpu.nvidia.sm90.attention import (
    get_seq_info,
    KVTMATile,
    kv_coord,
    NonNullPointer,
    NullPointer,
    Pack,
    q_coord,
    q_tma,
    QTMATile,
)
from layout.tma_async import (
    SharedMemBarrier,
    RaggedTMA3DTile,
)
from layout import TileTensor
from layout.tile_layout import row_major as tt_row_major
from layout.swizzle import make_swizzle
from linalg.arch.sm100.mma import smem_descriptor

from std.gpu.globals import WARP_SIZE
from std.gpu.memory import AddressSpace
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from std.gpu.host import DeviceContext, FuncAttribute
from std.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    barrier,
    thread_idx_uint as thread_idx,
    warp_id_uint as warp_id,
)
from nn.attention.mha_utils import (
    MHAConfig,
    NoPartition,
    OptionallyStaticInt,
)
from std.gpu.compute.arch.tcgen05 import *
from std.utils.static_tuple import StaticTuple
from kv_cache.types import padded_depth

from nn.attention.gpu.nvidia.sm100.mla_prefill_utils import (
    MLAConfig,
    SM100MLA,
    MLAPositionSummary,
    MLAKVLayouts,
    split_smem,
    TMAtoCvtPipeline,
    CvtToMMAPipeline,
    cvt_block_fp8_to_bf16_with_scale,
)
from nn.attention.gpu.nvidia.sm100.softmax_warp import fa4_softmax
from nn.attention.gpu.nvidia.sm100.correction_warp import fa4_correction
from nn.attention.gpu.nvidia.sm100.smem import SM100AttentionSMem


@fieldwise_init
struct WarpRole(Equatable, TrivialRegisterPassable):
    var _role: Int32
    comptime Softmax0 = Self(0)
    comptime Softmax1 = Self(1)
    comptime Correction = Self(2)
    comptime MMA = Self(3)
    comptime Load = Self(4)
    comptime CVTToBF16 = Self(5)
    comptime Empty = Self(6)

    @always_inline
    def __eq__(self, other: Int) -> Bool:
        return self == Self(Int32(other))


def warp_idx_to_role(warp_idx: UInt32) -> WarpRole:
    var wg_idx = warp_idx // 4
    if wg_idx == 0:
        return WarpRole.Softmax0
    elif wg_idx == 1:
        return WarpRole.Softmax1
    elif wg_idx == 2:
        return WarpRole.Correction
    elif warp_idx == 12:
        return WarpRole.MMA
    elif warp_idx == 13:
        return WarpRole.Load
    elif warp_idx >= 14 and warp_idx < 16:
        return WarpRole.CVTToBF16
    else:
        return WarpRole.Empty


__extension SM100MLA:
    @staticmethod
    @__llvm_arg_metadata(q_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(k_nope_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(k_rope_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(v_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(ragged_tma_store, `nvvm.grid_constant`)
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.config.num_threads)
        )
    )
    @__llvm_metadata(`nvvm.minctasm`=Int(1))
    def mla_prefill_kernel_blockscale[
        blockwise_scale: Int = 0,
    ](
        q_tma_op: QTMATile[
            Self.KVLUTType.dtype,
            Self.config.qkv_swizzle_mode,
            BM=Self.config.BM // 2,
            depth=Self.config.BK0,
            group=Self.config.group,
            decoding=False,
        ],
        k_nope_tma_op: KVTMATile[
            Self.KVLUTType.dtype,
            Self.config.qkv_swizzle_mode,
            BN=Self.config.BN,
            BK=Self.nope_depth,
        ],
        k_rope_tma_op: KVTMATile[
            Self.KRopeType.dtype,
            Self.config.rope_gmem_swizzle_mode,
            BN=Self.config.BN,
            BK=Self.rope_depth,
        ],
        v_tma_op: KVTMATile[
            Self.KVLUTType.dtype,
            Self.config.qkv_swizzle_mode,
            BN=Self.config.BN,
            BK=Self.nope_depth,
        ],
        ragged_tma_store: RaggedTMA3DTile[
            Self.output_dtype,
            Self.config.output_swizzle_mode,
            BM=Self.config.fa4_config.BM // 2,
            BN=Self.config.fa4_config.ov_depth,
        ],
        kv_lut: Self.KVLUTType,
        k_rope_lut: Self.KRopeType,
        scale: Float32,
        batch_size: UInt32,
        pack: Pack[
            Self.MaskType,
            Self.SchedulerType,
            Self.ValidLengthType,
            Self.SinkType,
            Self.KVRowOffsetsType,
            Self.MaxSeqLenType,
            Self.PartitionType,
        ],
    ):
        comptime assert Self.KRopeType.dtype == config.rope_gmem_dtype
        comptime assert Self.KVLUTType.dtype == config.rope_mma_dtype
        comptime assert Self.MMA_M == 64 or Self.MMA_M == 128
        comptime assert Self.config.supported(), (
            "\nqk_depth = "
            + String(Self.config.qk_depth)
            + "\nBN = "
            + String(Self.config.BN)
            + "\nnum_kv_stages = "
            + String(Self.config.num_kv_stages)
            + "\ntmem_used = "
            + String(Self.config.tmem_used)
            + "\nsmem_used = "
            + String(Self.config.smem_used)
            + "\n% smem_used = "
            + String(
                Float64(Self.config.smem_used)
                / Float64(Self.config.sm100_smem_carveout)
            )
            + "\nsmem_total = "
            + String(Self.config.sm100_smem_carveout)
        )
        comptime assert (
            not Self.SchedulerType.may_advance
        ), "Persistent kernels not yet supported with FA4"

        mask = pack.mask
        scheduler = pack.scheduler
        valid_length = pack.valid_length
        max_seq_len = pack.max_seq_len
        partition = pack.partition

        comptime num_qo = Self.config.num_qo()
        # TODO: We may want to support num_qo>2 for depth=64?
        comptime assert (
            num_qo == 1 or num_qo == 2
        ), "Currently only support num_qo == 1 or 2"
        # Blockscale uses its own extra barriers for the FP8→BF16 rope
        # conversion pipeline, so skip FA4MiscMBars rope barriers to stay
        # within the 32-barrier hardware limit.
        comptime SmemType = SM100AttentionSMem[Self.config.fa4_config]
        var attn_smem = SmemType()
        var misc_mbars = attn_smem.misc_mbars()
        var q_smem = rebind[SharedMemPointer[Scalar[KVLUTType.dtype]]](
            attn_smem.q_smem()
        )
        var k_smem = rebind[SharedMemPointer[Scalar[KVLUTType.dtype]]](
            attn_smem.k_smem_base()
        )
        var v_smem = rebind[SharedMemPointer[Scalar[KVLUTType.dtype]]](
            attn_smem.v_smem_base()
        )
        var rope_smem = rebind[SharedMemPointer[Scalar[KVLUTType.dtype]]](
            attn_smem.rope_smem_base()
        )
        var ptr_tmem_addr = attn_smem.tmem_addr_ptr()

        # Extra blockscale barriers are placed after the SM100AttentionSMem layout.
        # Align to 8 bytes for SharedMemBarrier.
        comptime extra_barrier_offset = align_up(
            SmemType.smem_size(), size_of[SharedMemBarrier]()
        )
        var extra_base = (attn_smem.base + extra_barrier_offset).bitcast[
            SharedMemBarrier
        ]()
        comptime num_rope_bufs = Self.config.fa4_config.num_rope_buffers()
        var tma_to_cvt_producer_mbars = extra_base
        var tma_to_cvt_consumer_mbars = extra_base + num_rope_bufs
        var cvt_to_mma_producer_mbars = extra_base + 2 * num_rope_bufs
        var cvt_to_mma_consumer_mbars = extra_base + 2 * num_rope_bufs + 2

        var tma_to_cvt_pipeline = TMAtoCvtPipeline[
            num_rope_bufs,
            num_producer=1,
            num_consumer=64,
        ](tma_to_cvt_producer_mbars, tma_to_cvt_consumer_mbars)

        var cvt_to_mma_pipeline = CvtToMMAPipeline[
            2,
            num_producer=64,
            num_consumer=1,
        ](cvt_to_mma_producer_mbars, cvt_to_mma_consumer_mbars)

        # https://github.com/NVIDIA/cutlass/blob/main/examples/77_blackwell_fmha/kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp
        comptime num_reg_softmax = 184
        comptime num_reg_correction = 96
        comptime num_reg_other = 48
        comptime num_reg_empty = 24

        comptime assert not Self.PartitionType.do_partition, (
            "Neither partitioning nor decoding are supported by the 2-q"
            " implementation."
        )

        var warp_idx = UInt32(warp_id[broadcast=True]())
        if warp_idx == 0:
            # Initialize all barriers (S/C/order/Q1Sync/KV/O) in one call
            misc_mbars.init(lane_idx=Int32(thread_idx.x))
            tma_to_cvt_pipeline.init()
            cvt_to_mma_pipeline.init()
        elif warp_idx == 1:
            tcgen05_alloc[Self.cta_group](
                ptr_tmem_addr, Self.config.sm100_tmem_cols
            )
        elif warp_idx == 2:
            e = elect()
            if e != 0:
                q_tma_op.prefetch_descriptor()
            if e != 0:
                k_nope_tma_op.prefetch_descriptor()
            if e != 0:
                k_rope_tma_op.prefetch_descriptor()
            if e != 0:
                v_tma_op.prefetch_descriptor()

        barrier()

        var role = warp_idx_to_role(warp_idx)

        # warp group partitioning
        # Two QO:
        if role == WarpRole.Softmax0 or role == WarpRole.Softmax1:
            # softmax $warp_group_idx
            warpgroup_reg_alloc[num_reg_softmax]()
            var seq_info: SeqInfo = get_seq_info[
                Self.BM,
                Self.num_q_heads,
                Self.MaskType.get_type_name() == "CausalMask",
            ](batch_size, max_seq_len, valid_length, partition)

            if not seq_info.is_valid():
                return

            var pos: MLAPositionSummary = MLAPositionSummary.create[
                _ndbuffer_mha_operand=Self._ndbuffer_mha_operand,
            ](k_rope_lut, seq_info)

            fa4_softmax[
                Self.KVLUTType,
                Self.config.fa4_config,
                Self.ValidLengthType,
                NullPointer[Self.output_dtype],
                False,
                Self.MaxSeqLenType,
            ](
                attn_smem,
                pos.score_row,
                seq_info,
                mask,
                pos.num_keys,
                scale.cast[Self.accum_dtype](),
                max_seq_len.as_uint32(),
                ragged_tma_store,
                NullPointer[Self.output_dtype](),
            )

        elif role == WarpRole.Correction:
            # correction
            warpgroup_reg_dealloc[num_reg_correction]()

            var seq_info: SeqInfo = get_seq_info[
                Self.BM,
                Self.num_q_heads,
                Self.MaskType.get_type_name() == "CausalMask",
            ](batch_size, max_seq_len, valid_length, partition)
            if not seq_info.is_valid():
                return
            var pos: MLAPositionSummary = MLAPositionSummary.create[
                _ndbuffer_mha_operand=Self._ndbuffer_mha_operand,
            ](k_rope_lut, seq_info)
            fa4_correction[
                Self.config.fa4_config,
                Self.page_size,
            ](
                attn_smem,
                pos.score_row,
                pos.num_keys,
                mask,
            )
        elif role == WarpRole.Load:
            warpgroup_reg_dealloc[num_reg_other]()
            var seq_info: SeqInfo = get_seq_info[
                Self.BM,
                Self.num_q_heads,
                Self.MaskType.get_type_name() == "CausalMask",
            ](batch_size, max_seq_len, valid_length, partition)

            if not seq_info.is_valid():
                return
            var pos: MLAPositionSummary = MLAPositionSummary.create[
                _ndbuffer_mha_operand=Self._ndbuffer_mha_operand,
            ](k_rope_lut, seq_info)

            Self.load(
                tma_to_cvt_pipeline,
                pos.score_row,
                pos.num_keys,
                seq_info,
                max_seq_len,
                mask,
                q_tma_op,
                k_nope_tma_op,
                k_rope_tma_op,
                v_tma_op,
                kv_lut,
                k_rope_lut,
                q_smem,
                k_smem,
                v_smem,
                rope_smem,
            )

        elif role == WarpRole.MMA:
            warpgroup_reg_dealloc[num_reg_other]()
            var seq_info: SeqInfo = get_seq_info[
                Self.BM,
                Self.num_q_heads,
                Self.MaskType.get_type_name() == "CausalMask",
            ](batch_size, max_seq_len, valid_length, partition)

            if not seq_info.is_valid():
                tcgen05_release_allocation_lock[Self.cta_group]()
                tcgen05_dealloc[Self.cta_group](
                    ptr_tmem_addr[0], Self.config.sm100_tmem_cols
                )
                return
            var pos: MLAPositionSummary = MLAPositionSummary.create[
                _ndbuffer_mha_operand=Self._ndbuffer_mha_operand,
            ](k_rope_lut, seq_info)
            Self.mma(
                ptr_tmem_addr[0],
                cvt_to_mma_pipeline,
                pos.score_row,
                pos.num_keys,
                mask,
                q_smem,
                k_smem,
                v_smem,
                rope_smem,
            )
        elif role == WarpRole.CVTToBF16:
            warpgroup_reg_dealloc[num_reg_other]()

            var seq_info: SeqInfo = get_seq_info[
                Self.BM,
                Self.num_q_heads,
                Self.MaskType.get_type_name() == "CausalMask",
            ](batch_size, max_seq_len, valid_length, partition)

            if not seq_info.is_valid():
                return

            var pos: MLAPositionSummary = MLAPositionSummary.create[
                _ndbuffer_mha_operand=Self._ndbuffer_mha_operand,
            ](k_rope_lut, seq_info)

            var iter_count: UInt32 = (
                mask.last_masked_set_end[Self.BM, Self.BN, Self.page_size](
                    pos.score_row, pos.num_keys
                )
                - 1
            )

            var local_thread_idx = UInt32(thread_idx.x - 14 * UInt(WARP_SIZE))

            Self.convert_fp8_to_bf16(
                iter_count,
                tma_to_cvt_pipeline,
                cvt_to_mma_pipeline,
                k_rope_lut,
                k_smem,
                rope_smem,
                seq_info,
                pos.num_keys,
                local_thread_idx,
            )

        elif role == WarpRole.Empty:
            warpgroup_reg_dealloc[num_reg_empty]()

    @staticmethod
    @always_inline
    def load(
        mut tma_to_cvt_pipeline: TMAtoCvtPipeline,
        score_row: UInt32,
        num_keys: UInt32,
        seq_info: SeqInfo,
        max_seq_len: Self.MaxSeqLenType,
        mask: Self.MaskType,
        q_tma_op: QTMATile[
            Self.KVLUTType.dtype,
            Self.config.qkv_swizzle_mode,
            BM=Self.config.BM // 2,
            depth=Self.config.BK0,  # padded depth -> 192
            group=Self.config.group,
            decoding=False,
        ],
        k_nope_tma_op: KVTMATile[
            Self.KVLUTType.dtype,
            Self.config.qkv_swizzle_mode,
            BN=Self.config.BN,
            BK=Self.nope_depth,
        ],
        k_rope_tma_op: KVTMATile[
            Self.KRopeType.dtype,
            Self.config.rope_gmem_swizzle_mode,
            BN=Self.config.BN,
            BK=Self.rope_depth,
        ],
        v_tma_op: KVTMATile[
            Self.KVLUTType.dtype,
            Self.config.qkv_swizzle_mode,
            BN=Self.config.BN,
            BK=Self.nope_depth,
        ],
        kv_lut: Self.KVLUTType,
        k_rope_lut: Self.KRopeType,
        q_smem: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
        k_smem_base: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
        v_smem_base: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
        rope_smem_base: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
    ):
        comptime KVPipeType = MLAKVLayouts[
            Self.KVLUTType.dtype,
            Self.KRopeType.dtype,
            DType.invalid,
            Self.config,
        ]

        # If two-qo, we produce qkv in a pattern of
        # q0 & k0, q1, v0, k1, v1, k2, v2...
        # Blockscale uses its own TMA-to-CVT/CVT-to-MMA barriers for rope
        # synchronization.
        comptime BlockscaleSmemType = SM100AttentionSMem[Self.config.fa4_config]
        var mbars = BlockscaleSmemType().misc_mbars()

        comptime SMemTensorLT[elems: Int] = TileTensor[
            Self.KVLUTType.dtype,
            type_of(tt_row_major[elems]()),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ]
        comptime q_elems = type_of(q_tma_op).tile_shape[0] * type_of(
            q_tma_op
        ).tile_shape[1] * type_of(q_tma_op).tile_shape[2]
        comptime QType = SMemTensorLT[q_elems]

        var k_rope_head_idx: UInt32 = seq_info.head_idx // UInt32(Self.group)
        var kv_head_idx: UInt32 = seq_info.head_idx

        comptime q_elements = (Self.config.BM // 2) * Self.config.BK0
        comptime q_bytes = size_of[Self.qkv_dtype]() * q_elements
        comptime k_rope_bytes = size_of[
            Self.KRopeType.dtype
        ]() * Self.BN * Self.rope_depth

        var q_gmem_row: UInt32 = Self.PositionType.get_q_gmem_row[ragged=True](
            seq_info, max_seq_len
        )
        var q_head_idx: UInt32 = seq_info.head_idx
        e = elect()

        var kv_row: UInt32 = mask.start_column[
            Self.BM, Self.BN, Self.page_size
        ](score_row)
        var kv_gmem_row: UInt32 = kv_lut.row_idx(seq_info.prompt_idx, kv_row)
        var k_rope_gmem_row: UInt32 = k_rope_lut.row_idx(
            seq_info.prompt_idx, kv_row
        )
        var iter_count: UInt32 = (
            mask.last_masked_set_end[Self.BM, Self.BN, Self.page_size](
                score_row, num_keys
            )
            - 1
        )

        comptime check_mask = mask.nonfull_sets[Self.BM, Self.BN]()[
            0
        ] == TileMaskStatus.UNKNOWN_MASK

        comptime if Self.config.fa4_config.use_fused_kv:
            # ---- Fused KV mode ----
            # K_nope and V alternate in the same circular buffer (padded_v_depth
            # wide). K_rope (FP8) goes into the separate rope smem buffer via
            # tma_to_cvt_pipeline.
            comptime KNopeType = SMemTensorLT[KVPipeType.k_nope_tma_layout]
            comptime VType = SMemTensorLT[KVPipeType.k_nope_tma_layout]
            comptime KRopeSMemType = TileTensor[
                Self.KRopeType.dtype,
                type_of(tt_row_major[KVPipeType.k_rope_tma_layout]()),
                MutAnyOrigin,
                address_space=AddressSpace.SHARED,
            ]
            comptime kv_stage_elems = (
                Self.config.fa4_config.padded_ov_depth * Self.config.BN
            )
            comptime rope_stage_elems = (
                Self.config.rope_depth * Self.config.BN
            )
            comptime k_nope_bytes = (
                Self.nope_depth * Self.config.BN * size_of[Self.qkv_dtype]()
            )
            comptime k_rope_bytes = KVPipeType.k_rope_bytes
            comptime v_bytes = k_nope_bytes
            comptime qk_fused_bytes = k_nope_bytes + q_bytes

            comptime KVPipeProdType = StagedPipeline[
                Self.config.num_kv_stages, 1
            ]
            var kv_pipeline: KVPipeProdType = {mbars.get_k_mbars()}
            kv_pipeline.state._phase = 1  # producer starts at phase 1

            # ---- Peeled: K0 + Q0 on same KV barrier ----
            var k0_mbar = kv_pipeline.producer_mbar()
            if e != 0:
                k0_mbar[].expect_bytes(Int32(qk_fused_bytes))
            # Copy Q0
            if e != 0:
                q_tma_op.async_copy(
                    QType(q_smem, tt_row_major[q_elems]()),
                    k0_mbar[],
                    q_coord[
                        depth=Self.qk_depth,
                        decoding=False,
                    ](q_gmem_row, q_head_idx),
                )
            # Copy K_nope0 into fused buffer
            if e != 0:
                k_nope_tma_op.async_copy(
                    KNopeType(
                        k_smem_base
                        + kv_pipeline.state.index() * UInt32(kv_stage_elems),
                        tt_row_major[KVPipeType.k_nope_tma_layout](),
                    ),
                    k0_mbar[],
                    kv_coord[depth=Self.nope_depth,](kv_gmem_row, kv_head_idx),
                )
            # Copy K_rope0 (FP8) into rope buffer via tma_to_cvt_pipeline
            if e != 0:
                var k_rope_coord = kv_coord[depth=Self.rope_depth,](
                    k_rope_gmem_row, k_rope_head_idx
                )
                k_rope_coord[0] = UInt32(Self.cache_depth - Self.rope_depth)
                tma_to_cvt_pipeline.producer_mbar()[].expect_bytes(
                    Int32(k_rope_bytes)
                )
                # Advance by BF16 elements on the BF16 pointer first,
                # then bitcast to FP8 for the TMA copy.
                k_rope_tma_op.async_copy(
                    KRopeSMemType(
                        (
                            rope_smem_base
                            + tma_to_cvt_pipeline.state.index()
                            * UInt32(rope_stage_elems)
                        ).bitcast[Scalar[Self.KRopeType.dtype]](),
                        tt_row_major[KVPipeType.k_rope_tma_layout](),
                    ),
                    tma_to_cvt_pipeline.producer_mbar()[],
                    k_rope_coord,
                )
            tma_to_cvt_pipeline.step()
            kv_pipeline.state.step()  # step -> stage 1

            # ---- Q1 (separate barrier) ----
            if e != 0:
                var q1_mbar = mbars.q1_wait_mbar()
                q1_mbar[0].expect_bytes(Int32(q_bytes))
                q_tma_op.async_copy(
                    QType(q_smem + q_elements, tt_row_major[q_elems]()),
                    q1_mbar[0],
                    q_coord[
                        depth=Self.qk_depth,
                        decoding=False,
                    ](q_gmem_row + UInt32(Self.config.BM // 2), q_head_idx),
                )

            # ---- V0 ----
            kv_pipeline.producer_acquire()
            var v0_mbar = kv_pipeline.producer_mbar()
            if e != 0:
                v0_mbar[].expect_bytes(Int32(v_bytes))
            if e != 0:
                v_tma_op.async_copy(
                    VType(
                        k_smem_base
                        + kv_pipeline.state.index() * UInt32(kv_stage_elems),
                        tt_row_major[KVPipeType.k_nope_tma_layout](),
                    ),
                    v0_mbar[],
                    kv_coord[depth=Self.nope_depth,](kv_gmem_row, kv_head_idx),
                )
            kv_pipeline.state.step()

            # ---- KV producer loop ----
            while iter_count != 0:
                iter_count -= 1
                kv_row += UInt32(Self.config.BN)

                comptime if check_mask:
                    if (
                        Self.mask_status(mask, score_row, kv_row)
                        == TileMaskStatus.FULL_MASK
                    ):
                        continue
                kv_gmem_row = kv_lut.row_idx(seq_info.prompt_idx, kv_row)
                k_rope_gmem_row = k_rope_lut.row_idx(
                    seq_info.prompt_idx, kv_row
                )

                # Produce K_nope_n
                kv_pipeline.producer_acquire()
                var kn_mbar = kv_pipeline.producer_mbar()
                if e != 0:
                    kn_mbar[].expect_bytes(Int32(k_nope_bytes))
                if e != 0:
                    k_nope_tma_op.async_copy(
                        KNopeType(
                            k_smem_base
                            + kv_pipeline.state.index()
                            * UInt32(kv_stage_elems),
                            tt_row_major[KVPipeType.k_nope_tma_layout](),
                        ),
                        kn_mbar[],
                        kv_coord[depth=Self.nope_depth,](
                            kv_gmem_row, kv_head_idx
                        ),
                    )
                # K_rope_n (FP8) into rope buffer
                if e != 0:
                    var k_rope_coord = kv_coord[depth=Self.rope_depth,](
                        k_rope_gmem_row, k_rope_head_idx
                    )
                    k_rope_coord[0] = UInt32(Self.cache_depth - Self.rope_depth)
                    tma_to_cvt_pipeline.producer_mbar()[].expect_bytes(
                        Int32(k_rope_bytes)
                    )
                    # Advance by BF16 elements first, then bitcast to FP8.
                    k_rope_tma_op.async_copy(
                        KRopeSMemType(
                            (
                                rope_smem_base
                                + tma_to_cvt_pipeline.state.index()
                                * UInt32(rope_stage_elems)
                            ).bitcast[Scalar[Self.KRopeType.dtype]](),
                            tt_row_major[KVPipeType.k_rope_tma_layout](),
                        ),
                        tma_to_cvt_pipeline.producer_mbar()[],
                        k_rope_coord,
                    )
                tma_to_cvt_pipeline.step()
                kv_pipeline.state.step()

                # Produce Vn
                kv_pipeline.producer_acquire()
                var vn_mbar = kv_pipeline.producer_mbar()
                if e != 0:
                    vn_mbar[].expect_bytes(Int32(v_bytes))
                if e != 0:
                    v_tma_op.async_copy(
                        VType(
                            k_smem_base
                            + kv_pipeline.state.index()
                            * UInt32(kv_stage_elems),
                            tt_row_major[KVPipeType.k_nope_tma_layout](),
                        ),
                        vn_mbar[],
                        kv_coord[depth=Self.nope_depth,](
                            kv_gmem_row, kv_head_idx
                        ),
                    )
                kv_pipeline.state.step()

        else:
            # ---- Split KV mode (original) ----

            comptime KPipeType = KProducerPipeline[
                Self.KVLUTType.dtype, Self.config.fa4_config
            ]
            comptime VPipeType = VProducerPipeline[
                Self.KVLUTType.dtype, Self.config.fa4_config
            ]
            var pipeline_k: KPipeType = {mbars.get_k_mbars(), k_smem_base}
            var pipeline_v: VPipeType = {mbars.get_v_mbars(), v_smem_base}

            # Get K0 destination (no wait needed for first iteration)
            k0_dest = pipeline_k.get_k[qk_stage=0]()

            # copy q0
            if e != 0:
                # Q0 + K0: signal K barrier with combined q + k_nope bytes
                k0_dest.mbar[].expect_bytes(
                    Int32(KVPipeType.k_nope_bytes + q_bytes)
                )
                q_tma_op.async_copy(
                    QType(q_smem, tt_row_major[q_elems]()),
                    k0_dest.mbar[],
                    q_coord[
                        depth=Self.qk_depth,
                        decoding=False,
                    ](q_gmem_row, q_head_idx),
                )
            # copy k0
            k_nope_smem, k_rope_smem = split_smem[
                KVPipeType.k_nope_tma_layout,
                KVPipeType.k_rope_tma_layout,
                Self.KVLUTType.dtype,
                Self.KRopeType.dtype,
            ](k0_dest.smem)
            if e != 0:
                # K0
                k_nope_tma_op.async_copy(
                    k_nope_smem,
                    k0_dest.mbar[],
                    kv_coord[depth=Self.nope_depth,](kv_gmem_row, kv_head_idx),
                )
                # K0 rope
                var k_rope_coord = kv_coord[depth=Self.rope_depth,](
                    k_rope_gmem_row, k_rope_head_idx
                )
                k_rope_coord[0] = UInt32(
                    Self.cache_depth - Self.rope_depth
                )  # only load last 64 head_dims

                tma_to_cvt_pipeline.producer_mbar()[].expect_bytes(
                    Int32(KVPipeType.k_rope_bytes)
                )

                k_rope_tma_op.async_copy(
                    k_rope_smem,
                    tma_to_cvt_pipeline.producer_mbar()[],
                    k_rope_coord,
                )

            tma_to_cvt_pipeline.step()
            pipeline_k.commit_step()

            if e != 0:
                var q1_mbar = mbars.q1_wait_mbar()
                q1_mbar[0].expect_bytes(Int32(q_bytes))
                # Q1
                q_tma_op.async_copy(
                    QType(q_smem + q_elements, tt_row_major[q_elems]()),
                    q1_mbar[0],
                    q_coord[
                        depth=Self.qk_depth,
                        decoding=False,
                    ](q_gmem_row + UInt32(Self.config.BM // 2), q_head_idx),
                )
            # copy v0
            mbarv0 = pipeline_v.get_v(e)
            if e != 0:
                v_tma_op.async_copy(
                    mbarv0.smem,
                    mbarv0.mbar[],
                    kv_coord[depth=Self.nope_depth,](kv_gmem_row, kv_head_idx),
                )
            pipeline_v.commit_step()

            # kv producer loop
            while iter_count != 0:
                iter_count -= 1
                kv_row += UInt32(Self.config.BN)

                comptime if check_mask:
                    if (
                        Self.mask_status(mask, score_row, kv_row)
                        == TileMaskStatus.FULL_MASK
                    ):
                        continue
                kv_gmem_row = kv_lut.row_idx(seq_info.prompt_idx, kv_row)
                k_rope_gmem_row = k_rope_lut.row_idx(
                    seq_info.prompt_idx, kv_row
                )
                # produce k
                pipeline_k.acquire_k[qk_stage=0]()
                kn_dest = pipeline_k.get_k[qk_stage=0]()
                if e != 0:
                    kn_dest.mbar[].expect_bytes(Int32(KVPipeType.k_nope_bytes))
                    k_nope_smem_n, k_rope_smem_n = split_smem[
                        KVPipeType.k_nope_tma_layout,
                        KVPipeType.k_rope_tma_layout,
                        Self.KVLUTType.dtype,
                        Self.KRopeType.dtype,
                    ](kn_dest.smem)

                    k_nope_tma_op.async_copy(
                        k_nope_smem_n,
                        kn_dest.mbar[],
                        kv_coord[depth=Self.nope_depth,](
                            kv_gmem_row, kv_head_idx
                        ),
                    )
                    # K rope
                    tma_to_cvt_pipeline.producer_mbar()[].expect_bytes(
                        Int32(KVPipeType.k_rope_bytes)
                    )

                    var k_rope_coord = kv_coord[depth=Self.rope_depth,](
                        k_rope_gmem_row, k_rope_head_idx
                    )
                    k_rope_coord[0] = UInt32(
                        Self.cache_depth - Self.rope_depth
                    )  # only load last 64 head_dims
                    k_rope_tma_op.async_copy(
                        k_rope_smem_n,
                        tma_to_cvt_pipeline.producer_mbar()[],
                        k_rope_coord,
                    )

                pipeline_k.commit_step()
                tma_to_cvt_pipeline.step()
                pipeline_v.acquire_v()
                mbarvn = pipeline_v.get_v(e)
                if e != 0:
                    v_tma_op.async_copy(
                        mbarvn.smem,
                        mbarvn.mbar[],
                        kv_coord[depth=Self.nope_depth,](
                            kv_gmem_row, kv_head_idx
                        ),
                    )
                pipeline_v.commit_step()

    @staticmethod
    @always_inline
    def convert_fp8_to_bf16(
        mut iter_count: UInt32,
        mut tma_to_cvt_pipeline: TMAtoCvtPipeline,
        mut cvt_to_mma_pipeline: CvtToMMAPipeline,
        k_rope: Self.KRopeType,
        k_smem_base: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
        rope_smem_base: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
        seq_info: SeqInfo,
        num_keys: UInt32,
        local_thread_idx: UInt32,
    ):
        var local_warp_idx, local_lane_idx = divmod(
            local_thread_idx, UInt32(WARP_SIZE)
        )

        var kv_start_tok: UInt32 = 0

        comptime swizzle_fp8 = make_swizzle[
            Self.KRopeType.dtype, TensorMapSwizzle.SWIZZLE_64B
        ]()
        comptime swizzle_bf16 = make_swizzle[
            Self.KVLUTType.dtype, TensorMapSwizzle.SWIZZLE_128B
        ]()

        # In split mode, k_rope sits after k_nope within each K stage.
        # In fused mode, k_rope is in a separate rope smem buffer.
        comptime k_stage_stride = Self.config.padded_qk_depth * Self.config.BN
        comptime k_rope_offset = Self.config.BN * Self.nope_depth
        comptime rope_stage_elems = Self.config.rope_depth * Self.config.BN

        while True:
            tma_to_cvt_pipeline.consumer_wait()

            # In fused mode, k_rope is in a separate rope smem buffer.
            # In split mode, k_rope sits after k_nope within each K stage.
            var k_rope_smem_ptr: SharedMemPointer[Scalar[Self.KVLUTType.dtype]]
            comptime if Self.config.fa4_config.use_fused_kv:
                k_rope_smem_ptr = (
                    rope_smem_base
                    + tma_to_cvt_pipeline.state.index()
                    * UInt32(rope_stage_elems)
                )
            else:
                k_rope_smem_ptr = (
                    k_smem_base
                    + tma_to_cvt_pipeline.state.index() * UInt32(k_stage_stride)
                    + k_rope_offset
                )

            comptime tile_rows = Self.config.BN // 2
            comptime tile_layout = tt_row_major[tile_rows, Self.rope_depth]()
            comptime SmemFP8 = TileTensor[
                Self.KRopeType.dtype,
                type_of(tile_layout),
                MutAnyOrigin,
                address_space=AddressSpace.SHARED,
            ]
            comptime SmemBF16 = TileTensor[
                Self.KVLUTType.dtype,
                type_of(tile_layout),
                MutAnyOrigin,
                address_space=AddressSpace.SHARED,
            ]
            var tile_offset = Int(local_warp_idx) * tile_rows * Self.rope_depth
            var fp8_base = k_rope_smem_ptr.bitcast[
                Scalar[Self.KRopeType.dtype]
            ]()
            var bf16_base = k_rope_smem_ptr.bitcast[
                Scalar[Self.KVLUTType.dtype]
            ]()
            k_rope_tile_fp8 = SmemFP8(fp8_base + tile_offset, tile_layout)
            k_rope_tile_bf16 = SmemBF16(bf16_base + tile_offset, tile_layout)

            cvt_block_fp8_to_bf16_with_scale[
                swizzle_fp8=swizzle_fp8,
                swizzle_bf16=swizzle_bf16,
            ](
                k_rope_tile_fp8,
                k_rope_tile_bf16,
                k_rope,
                seq_info,
                kv_start_tok + UInt32(Self.BN // 2) * local_warp_idx,
                num_keys,
                local_lane_idx,
            )

            tma_to_cvt_pipeline.step()
            cvt_to_mma_pipeline.producer_commit()

            if iter_count == 0:
                break
            iter_count -= 1
            kv_start_tok += UInt32(Self.BN)

    @staticmethod
    @always_inline
    def mma(
        tmem_addr: UInt32,
        mut cvt_to_mma_pipeline: CvtToMMAPipeline,
        score_row: UInt32,
        num_keys: UInt32,
        mask: Self.MaskType,
        q_smem: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
        k_smem_base: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
        v_smem_base: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
        rope_smem_base: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
    ):
        # Construct SmemType to get MiscMBars (blockscale uses its own CVT
        # barriers for rope synchronization).
        comptime BlockscaleSmemType = SM100AttentionSMem[Self.config.fa4_config]
        var mbars = BlockscaleSmemType().misc_mbars()

        s0_tmem = tmem_addr + UInt32(Self.config.TMEM_S0)
        s1_tmem = tmem_addr + UInt32(Self.config.TMEM_S1)
        o0_tmem = tmem_addr + UInt32(Self.config.TMEM_O0)
        o1_tmem = tmem_addr + UInt32(Self.config.TMEM_O1)

        # S pipelines with sub-stages (1 producer, num_pv_stages consumers)
        var pipeline_s0 = mbars.producer_s0()
        var pipeline_s1 = mbars.producer_s1()
        # Keep consumer pointers for acquire operations (shared phase tracking)
        consumer_s0 = pipeline_s0.consumer_mbar_base
        consumer_s1 = pipeline_s1.consumer_mbar_base

        # O pipelines (producer side only; consumer wait is merged into S barriers)
        var pipeline_o0 = mbars.producer_o0()
        var pipeline_o1 = mbars.producer_o1()

        comptime q0_size = (Self.config.BM // 2) * Self.config.padded_qk_depth
        comptime q0_bytes = UInt32(q0_size * size_of[Self.KVLUTType.dtype]())
        q0 = Self.descriptor_q(q_smem)
        q1 = q0 + q0_bytes

        comptime if Self.config.fa4_config.use_fused_kv:
            # ---- Fused KV mode ----
            # K_nope and V alternate in the same buffer (padded_v_depth wide).
            # K_rope (BF16 after CVT conversion) is in a separate rope buffer.
            # Q@K' = Q_nope@K_nope (c_scale=0) + Q_rope@K_rope (c_scale=1).

            comptime kv_stage_bytes = (
                Self.config.fa4_config.padded_ov_depth
                * Self.config.BN
                * size_of[Self.KVLUTType.dtype]()
            )
            comptime rope_stage_bytes = (
                Self.config.rope_depth
                * Self.config.BN
                * size_of[Self.KVLUTType.dtype]()
            )

            # K_nope descriptor: k_major for Q@K_nope'
            kv_desc_k = smem_descriptor[
                BMN=Self.config.BN,
                BK=Self.config.fa4_config.padded_ov_depth,
                swizzle_mode=Self.config.qkv_swizzle_mode,
                is_k_major=True,
            ](k_smem_base)
            # V descriptor: mn_major for P@V
            kv_desc_v = smem_descriptor[
                BMN=Self.config.fa4_config.padded_ov_depth,
                BK=Self.config.BN,
                swizzle_mode=Self.config.qkv_swizzle_mode,
                is_k_major=False,
            ](k_smem_base)
            # K_rope descriptor: k_major for Q@K_rope' (BF16 after CVT)
            rope_desc = smem_descriptor[
                BMN=Self.config.BN,
                BK=Self.config.rope_depth,
                swizzle_mode=Self.config.qkv_swizzle_mode,
                is_k_major=True,
            ](rope_smem_base)

            # Q_rope offset: Q_nope occupies first MMA_M * padded_v_depth
            # elements in Q's column-major tiled layout.
            comptime q_rope_off = UInt32(Self.q_rope_byte_offset)
            q0_rope = q0 + q_rope_off
            q1_rope = q1 + q_rope_off

            comptime KVPipeType = StagedPipeline[
                Self.config.fa4_config.num_kv_stages, 1
            ]
            var kv_pipeline: KVPipeType = {mbars.get_k_mbars()}

            var iter_count: UInt32 = (
                mask.total_iters[Self.BM, Self.BN, Self.page_size](
                    score_row, num_keys
                )
                - 1
            )

            e = elect()

            # Rope stage counter: tracks which cvt_to_mma stage contains the
            # current K_rope BF16 data. Cycles 0,1,...,num_rope_buffers-1.
            var rope_stage: UInt32 = 0
            comptime num_rope_stages = UInt32(
                Self.config.fa4_config.num_rope_buffers()
            )

            # ---- Peeled iteration ----
            # Stage 0 = K0 (K_nope0 + K_rope0 via CVT)
            kv_pipeline.consumer_wait()
            cvt_to_mma_pipeline.consumer_wait()
            k0 = kv_desc_k + UInt32(kv_stage_bytes) * kv_pipeline.state.index()
            Self.UMMA0Type.mma[stage_idx=0](q0, k0, s0_tmem, elect=e, c_scale=0)
            # K_rope0: from CVT pipeline at rope_stage=0
            r0 = rope_desc + UInt32(rope_stage_bytes) * rope_stage
            Self.UMMA0RopeType.mma[stage_idx=0](
                q0_rope, r0, s0_tmem, elect=e, c_scale=1
            )
            pipeline_s0.commit_mma(e)

            # Q1 @ K0 (wait for Q1 first)
            var q1_mbar = mbars.q1_wait_mbar()
            q1_mbar[0].wait()
            Self.UMMA0Type.mma[stage_idx=0](q1, k0, s1_tmem, elect=e, c_scale=0)
            Self.UMMA0RopeType.mma[stage_idx=0](
                q1_rope, r0, s1_tmem, elect=e, c_scale=1
            )
            kv_pipeline.consumer_release(e)  # release K0, step → stage 1
            cvt_to_mma_pipeline.consumer_release(e)
            rope_stage = (rope_stage + 1) % num_rope_stages
            pipeline_s1.commit_mma(e)

            # Stage 1 = V0
            kv_pipeline.consumer_wait()
            var v_prev_idx: UInt32 = kv_pipeline.state.index()
            v0 = kv_desc_v + UInt32(kv_stage_bytes) * v_prev_idx
            comptime for pv_stage in range(Self.config.num_pv_stages):
                _ = consumer_s0[pv_stage].wait(0)
                Self.UMMA1Type.mma[stage_idx=pv_stage](
                    s0_tmem, v0, o0_tmem, elect=e, c_scale=0
                )
            pipeline_o0.commit_mma(e)
            var phase: UInt32 = 0

            var c_scale: UInt32 = 0

            # ---- Main loop ----
            while iter_count != 0:
                iter_count -= 1

                # Advance past held V to get to next K
                kv_pipeline.state.step()

                # Kn (K_nope_n via KV pipeline + K_rope_n via CVT pipeline)
                kv_pipeline.consumer_wait()
                cvt_to_mma_pipeline.consumer_wait()
                kn = (
                    kv_desc_k
                    + UInt32(kv_stage_bytes) * kv_pipeline.state.index()
                )
                Self.UMMA0Type.mma[stage_idx=0](
                    q0, kn, s0_tmem, elect=e, c_scale=0
                )
                rn = rope_desc + UInt32(rope_stage_bytes) * rope_stage
                Self.UMMA0RopeType.mma[stage_idx=0](
                    q0_rope, rn, s0_tmem, elect=e, c_scale=1
                )
                pipeline_s0.commit_mma(e)

                # P1 @ V_{n-1}
                v_prev = kv_desc_v + UInt32(kv_stage_bytes) * v_prev_idx
                comptime for pv_stage in range(Self.config.num_pv_stages):
                    _ = consumer_s1[pv_stage].wait(phase)
                    Self.UMMA1Type.mma[stage_idx=pv_stage](
                        s1_tmem, v_prev, o1_tmem, elect=e, c_scale=c_scale
                    )
                pipeline_o1.commit_mma(e)
                c_scale = 1
                kv_pipeline.consumer_release_at(
                    v_prev_idx, e
                )  # release V_{n-1}

                # Q1 @ Kn
                Self.UMMA0Type.mma[stage_idx=0](
                    q1, kn, s1_tmem, elect=e, c_scale=0
                )
                Self.UMMA0RopeType.mma[stage_idx=0](
                    q1_rope, rn, s1_tmem, elect=e, c_scale=1
                )
                kv_pipeline.consumer_release(e)  # release Kn, step
                cvt_to_mma_pipeline.consumer_release(e)
                rope_stage = (rope_stage + 1) % num_rope_stages
                pipeline_s1.commit_mma(e)
                phase ^= 1

                # Vn
                kv_pipeline.consumer_wait()
                v_prev_idx = kv_pipeline.state.index()
                vn = kv_desc_v + UInt32(kv_stage_bytes) * v_prev_idx
                comptime for pv_stage in range(Self.config.num_pv_stages):
                    _ = consumer_s0[pv_stage].wait(phase)
                    Self.UMMA1Type.mma[stage_idx=pv_stage](
                        s0_tmem, vn, o0_tmem, elect=e, c_scale=1
                    )
                pipeline_o0.commit_mma(e)

            # ---- Epilogue ----
            v_prev = kv_desc_v + UInt32(kv_stage_bytes) * v_prev_idx
            comptime for pv_stage in range(Self.config.num_pv_stages):
                _ = consumer_s1[pv_stage].wait(phase)
                Self.UMMA1Type.mma[stage_idx=pv_stage](
                    s1_tmem, v_prev, o1_tmem, elect=e, c_scale=c_scale
                )
            pipeline_o1.commit_mma(e)
            kv_pipeline.consumer_release_at(v_prev_idx, e)  # release V_last

        else:
            # ---- Split KV mode (original) ----

            # Separate K and V consumer pipelines
            comptime KConType = KConsumerPipeline[
                Self.KVLUTType.dtype, Self.config.fa4_config
            ]
            comptime VConType = VConsumerPipeline[
                Self.KVLUTType.dtype, Self.config.fa4_config
            ]
            var pipeline_k: KConType = {mbars.get_k_mbars(), k_smem_base}
            var pipeline_v: VConType = {mbars.get_v_mbars(), v_smem_base}

            # We peel the first iteration, as we want to wait on q1
            var iter_count: UInt32 = (
                mask.total_iters[Self.BM, Self.BN, Self.page_size](
                    score_row, num_keys
                )
                - 1
            )

            # Q_0 @ K_0'
            # wait for CVT for fp8 case, else wait for producer
            pipeline_k.wait_k()
            cvt_to_mma_pipeline.consumer_wait()

            k0 = pipeline_k.get_k()
            e = elect()

            Self.UMMA0Type.mma(q0, k0, s0_tmem, elect=e, c_scale=0)
            pipeline_s0.commit_mma(e)

            # Q_1 @ K_0'
            mbars.q1_wait_mbar()[0].wait()  # wait on Q1
            Self.UMMA0Type.mma(q1, k0, s1_tmem, elect=e, c_scale=0)
            pipeline_s1.commit_mma(e)

            pipeline_k.release_k(e)  # release K0
            cvt_to_mma_pipeline.step()

            # Wait V0
            pipeline_v.wait_v()
            vlatest = pipeline_v.get_v()
            comptime for pv_stage in range(Self.config.num_pv_stages):
                _ = consumer_s0[pv_stage].wait(0)
                Self.UMMA1Type.mma[stage_idx=pv_stage](
                    s0_tmem, vlatest, o0_tmem, elect=e, c_scale=0
                )
            pipeline_o0.commit_mma(e)
            var phase: UInt32 = 0

            var c_scale: UInt32 = 0

            while iter_count != 0:
                iter_count -= 1

                # Q_0 @ K_n'
                kn = pipeline_k.get_k()
                pipeline_k.wait_k()
                cvt_to_mma_pipeline.consumer_wait()

                Self.UMMA0Type.mma(q0, kn, s0_tmem, elect=e, c_scale=0)
                pipeline_s0.commit_mma(e)

                # O_1 + P_1 @ V_{n-1}
                comptime for pv_stage in range(Self.config.num_pv_stages):
                    _ = consumer_s1[pv_stage].wait(phase)
                    Self.UMMA1Type.mma[stage_idx=pv_stage](
                        s1_tmem, vlatest, o1_tmem, elect=e, c_scale=c_scale
                    )
                pipeline_o1.commit_mma(e)
                c_scale = 1
                pipeline_v.release_v(e)

                # Q_1 @ K_n'
                Self.UMMA0Type.mma(q1, kn, s1_tmem, elect=e, c_scale=0)
                pipeline_k.release_k(e)
                pipeline_s1.commit_mma(e)
                phase ^= 1

                cvt_to_mma_pipeline.step()

                # O_0 + P_0 @ V_n
                vlatest = pipeline_v.get_v()
                pipeline_v.wait_v()
                comptime for pv_stage in range(Self.config.num_pv_stages):
                    _ = consumer_s0[pv_stage].wait(phase)
                    Self.UMMA1Type.mma[stage_idx=pv_stage](
                        s0_tmem, vlatest, o0_tmem, elect=e, c_scale=1
                    )
                pipeline_o0.commit_mma(e)

            comptime for pv_stage in range(Self.config.num_pv_stages):
                _ = consumer_s1[pv_stage].wait(phase)
                Self.UMMA1Type.mma[stage_idx=pv_stage](
                    s1_tmem, vlatest, o1_tmem, elect=e, c_scale=c_scale
                )
            pipeline_o1.commit_mma(e)


@always_inline
def mla_sm100_prefill_blockscale[
    output_dtype: DType,
    q_type: DType,
    KVType: MHAOperand,
    KRopeType: MHAOperand,
    MaskType: MHAMask,
    MaxPromptLenType: OptionallyStaticInt,
    //,
    config: MHAConfig,
    group: Int,
    q_depth: Int,
    cache_depth: Int,
    _ndbuffer_mha_operand: Bool,
    blockwise_scale: Int = 0,
](
    output: TileTensor[output_dtype, address_space=AddressSpace.GENERIC, ...],
    q: TileTensor[q_type, address_space=AddressSpace.GENERIC, ...],
    k: KVType,
    v: KVType,
    k_rope: KRopeType,
    mask_functor: MaskType,
    valid_length: TileTensor[
        DType.uint32, address_space=AddressSpace.GENERIC, ...
    ],
    max_prompt_len: MaxPromptLenType,
    scale: Float32,
    batch_size: Int,
    ctx: DeviceContext,
) raises:
    comptime fa4_config = MLAConfig[
        q_type, rope_gmem_dtype=KRopeType.dtype, rope_mma_dtype=q_type
    ](
        num_q_heads=Int(config.num_heads),
        group=group,
        depth=q_depth,
        page_size=KVType.page_size,
    )

    var num_rows_q = q_num_matrix_view_rows(q)

    comptime RaggedStoreType = RaggedTMA3DTile[
        output_dtype,
        fa4_config.output_swizzle_mode,
        BM=fa4_config.fa4_config.BM // 2,
        BN=fa4_config.fa4_config.ov_depth,
    ]

    var ragged_tma_store = RaggedStoreType.create(
        ctx, output.ptr, rows=num_rows_q, middle_dim=fa4_config.num_q_heads
    )

    q_tma_op = q_tma[
        fa4_config.qkv_swizzle_mode,
        BM=fa4_config.BM // 2,
        depth=fa4_config.qk_depth,
        q_num_heads=fa4_config.num_q_heads,
        group=fa4_config.group,
        decoding=False,
    ](
        ctx,
        q.ptr,
        num_rows_q,
    )

    # [batch_size * num_keys, num_heads, kv_depth]
    k_nope_tma_op = k.create_tma_tile[
        fa4_config.qkv_swizzle_mode,
        BN=fa4_config.BN,
        depth=fa4_config.nope_depth,
    ](ctx)

    # [batch_size, num_keys, cache_num_heads, cache_depth]
    k_rope_tma_op = k_rope.create_tma_tile[
        fa4_config.rope_gmem_swizzle_mode,
        BN=fa4_config.BN,
        depth=cache_depth,
        BK=fa4_config.rope_depth,
    ](ctx)

    # [batch_size * num_keys, num_heads, kv_depth]
    v_tma_op = v.create_tma_tile[
        fa4_config.qkv_swizzle_mode,
        BN=fa4_config.BN,
        depth=fa4_config.nope_depth,
    ](ctx)

    _mla_prefill_sm100_valid_length_dispatch[
        fa4_config=fa4_config,
        cache_depth=cache_depth,
        _ndbuffer_mha_operand=_ndbuffer_mha_operand,
        blockwise_scale=blockwise_scale,
    ](
        ragged_tma_store,
        q_tma_op,
        k_nope_tma_op,
        k_rope_tma_op,
        v_tma_op,
        k,
        k_rope,
        mask_functor,
        valid_length,
        max_prompt_len,
        scale,
        batch_size,
        ctx,
    )


@always_inline
def _mla_prefill_sm100_valid_length_dispatch[
    KVType: MHAOperand,
    output_dtype: DType,
    q_type: DType,
    MaskType: MHAMask,
    KRopeType: MHAOperand,
    MaxPromptLenType: OptionallyStaticInt,
    //,
    fa4_config: MLAConfig,
    cache_depth: Int,
    _ndbuffer_mha_operand: Bool,
    blockwise_scale: Int = 0,
](
    ragged_tma_store: RaggedTMA3DTile[
        output_dtype,
        fa4_config.output_swizzle_mode,
        BM=fa4_config.fa4_config.BM // 2,
        BN=fa4_config.fa4_config.ov_depth,
    ],
    q_tma_op: QTMATile[
        q_type,
        fa4_config.qkv_swizzle_mode,
        BM=fa4_config.BM // 2,
        depth=fa4_config.qk_depth,
        group=fa4_config.group,
        decoding=False,
    ],
    k_nope_tma_op: KVTMATile[
        KVType.dtype,
        fa4_config.qkv_swizzle_mode,
        BN=fa4_config.BN,
        BK=padded_depth[
            KVType.dtype, fa4_config.qkv_swizzle_mode, fa4_config.nope_depth
        ](),
    ],
    k_rope_tma_op: KVTMATile[
        KRopeType.dtype,
        fa4_config.rope_gmem_swizzle_mode,
        BN=fa4_config.BN,
        BK=fa4_config.rope_depth,
    ],
    v_tma_op: KVTMATile[
        KVType.dtype,
        fa4_config.qkv_swizzle_mode,
        BN=fa4_config.BN,
        BK=padded_depth[
            KVType.dtype, fa4_config.qkv_swizzle_mode, fa4_config.nope_depth
        ](),
    ],
    kv_lut: KVType,
    k_rope_lut: KRopeType,
    mask_functor: MaskType,
    valid_length: TileTensor[
        DType.uint32, address_space=AddressSpace.GENERIC, ...
    ],
    max_prompt_len: MaxPromptLenType,
    scale: Float32,
    batch_size: Int,
    ctx: DeviceContext,
) raises:
    comptime SchedulerType = TransientScheduler[
        UInt32(fa4_config.BM),
        UInt32(fa4_config.num_q_heads),
        flip_prompt_idx=MaskType.get_type_name() == "CausalMask",
    ]
    comptime ValidLengthType = NonNullPointer[DType.uint32]
    comptime SinkType = NullPointer[output_dtype]
    comptime KVRowOffsetsType = NullPointer[DType.uint32]
    comptime PartitionType = NoPartition[DType.float32]
    var valid_len: ValidLengthType = {
        rebind[UnsafePointer[UInt32, ImmutAnyOrigin]](valid_length.ptr)
    }

    comptime SM100MLAType = SM100MLA[
        KVType,
        KRopeType,
        output_dtype,
        MaskType,
        SchedulerType,
        fa4_config,
        ValidLengthType,
        SinkType,
        KVRowOffsetsType,
        MaxPromptLenType,
        PartitionType,
        _ndbuffer_mha_operand,
    ]

    comptime kernel = SM100MLAType.mla_prefill_kernel_blockscale[
        blockwise_scale
    ]

    comptime PackType = Pack[
        MaskType,
        SchedulerType,
        ValidLengthType,
        SinkType,
        KVRowOffsetsType,
        MaxPromptLenType,
        PartitionType,
    ]

    var pack: PackType = {
        mask_functor,
        SchedulerType(),
        valid_len,
        SinkType(),
        KVRowOffsetsType(),
        max_prompt_len,
        PartitionType(),
    }

    var max_num_prompt_tiles: UInt32 = ceildiv(
        max_prompt_len.as_uint32(), UInt32(fa4_config.BM)
    )
    var num_blocks: UInt32 = (
        max_num_prompt_tiles * PartitionType().num_partitions()
    )

    comptime num_threads = fa4_config.num_threads
    comptime SmemType = SM100AttentionSMem[fa4_config.fa4_config]
    comptime assert fa4_config.supported(), fa4_config.fa4_config.description()
    # Extra barriers for blockscale: tma_to_cvt (2*num_rope_bufs) + cvt_to_mma (4)
    comptime extra_barrier_count = 2 * fa4_config.fa4_config.num_rope_buffers() + 4
    comptime smem_use = align_up(
        SmemType.smem_size(), size_of[SharedMemBarrier]()
    ) + extra_barrier_count * size_of[SharedMemBarrier]()
    comptime assert SmemType.smem_size() == fa4_config.smem_used, String(
        "SmemType.smem_size() = ",
        SmemType.smem_size(),
        "\nfa4_config.smem_used = ",
        fa4_config.smem_used,
        "\n",
        fa4_config.fa4_config.description(),
    )
    comptime assert smem_use <= fa4_config.sm100_smem_carveout

    ctx.enqueue_function[kernel, kernel](
        q_tma_op,
        k_nope_tma_op,
        k_rope_tma_op,
        v_tma_op,
        ragged_tma_store,
        kv_lut,
        k_rope_lut,
        scale,
        UInt32(batch_size),
        pack,
        grid_dim=SchedulerType.grid_dim(UInt32(batch_size), num_blocks),
        block_dim=(num_threads, 1, 1),
        shared_mem_bytes=smem_use,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_use)
        ),
    )
