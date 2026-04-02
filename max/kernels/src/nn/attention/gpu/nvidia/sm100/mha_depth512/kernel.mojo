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
"""Kernel entry point for depth=512 pair-CTA SM100 (Blackwell) MHA prefill.

Two neighboring SMs cooperate via pair-CTA MMA (cta_group=2,
cluster_shape=(2,1,1)). Each CTA processes BM=64 Q rows; the pair-CTA
MMA instruction operates on MMA_M=128 combined rows.

Warp assignment (384 threads = 12 warps, 3 warp groups of 128):
    Warps 0-3:   Softmax (warp group 0)
    Warps 4-7:   Correction (warp group 1)
    Warp 8:      MMA (leader CTA issues pair-CTA MMA; peer early-returns)
    Warp 9:      Load (both CTAs issue TMA multicast; leader calls expect_bytes)
    Warps 10-11: Spare (no-op)
"""

from std.math import align_up, ceildiv, min
from std.sys import size_of
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    thread_idx_uint as thread_idx,
    warp_id,
)
from std.gpu.globals import WARPGROUP_SIZE, WARP_SIZE
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from std.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from std.gpu.memory import AddressSpace, external_memory, fence_mbarrier_init
from std.gpu.primitives.cluster import block_rank_in_cluster, cluster_sync
from linalg.matmul.gpu.sm100_structured.structured_kernels.tmem import (
    TmemAllocation,
)
from layout.tma_async import (
    SharedMemBarrier,
    RaggedTMA3DTile,
)
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    SharedMemPointer,
    MBarType,
    elect,
)
from nn.attention.gpu.nvidia.sm90.attention import (
    get_seq_info,
    KVTMATile,
    MHAPosition,
    NullPointer,
    OptionalPointer,
    Pack,
    PositionSummary,
    QTMATile,
)
from nn.attention.mha_mask import MHAMask, TileMaskStatus
from nn.attention.mha_operand import MHAOperand
from nn.attention.gpu.nvidia.mha_tile_scheduler import (
    MHATileScheduler,
    SeqInfo,
)
from nn.attention.mha_utils import (
    MHAPartitionScheme,
    OptionallyStaticInt,
    _is_decoding,
)
from std.utils.index import Index
from std.utils.static_tuple import StaticTuple
from .barriers import Depth512MBars
from .config import Depth512SM100Config
from .correction_warp import depth512_correction
from .load_warp import depth512_load
from .mma_warp import depth512_mma
from .smem import Depth512AttentionSMem
from .softmax_warp import depth512_softmax


struct SM100MHADepth512[
    KVLUTType: MHAOperand,
    output_type: DType,
    MaskType: MHAMask,
    SchedulerType: MHATileScheduler,
    config: Depth512SM100Config[KVLUTType.dtype],
    ValidLengthType: OptionalPointer,
    KVRowOffsetsType: OptionalPointer,
    _is_cache_length_accurate: Bool,
    MaxSeqLenType: OptionallyStaticInt,
    PartitionType: MHAPartitionScheme,
](TrivialRegisterPassable):
    comptime qkv_type = Self.KVLUTType.dtype
    comptime accum_type = DType.float32

    comptime cta_group = 2
    comptime BM = Self.config.BM  # 64 per CTA
    comptime PairBM = Self.BM * 2  # 128 across pair
    comptime BN = Self.config.BN
    comptime num_q_heads = Self.config.num_q_heads
    comptime group = Self.config.group
    comptime ragged = not Self.ValidLengthType.is_null
    comptime page_size = Self.KVLUTType.page_size

    comptime TmemAllocType = TmemAllocation[Self.cta_group]
    comptime SmemType = Depth512AttentionSMem[Self.config]

    comptime PositionType = MHAPosition[
        Self.PairBM,
        Self.config.BN,
        Self.config.qk_depth,
        Self.config.qk_depth,  # padded_qk_depth = qk_depth for depth512
        Self.config.num_q_heads,
        Self.config.group,
        _is_decoding[Self.MaxSeqLenType](),
    ]

    @staticmethod
    @__llvm_arg_metadata(q_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(k_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(v_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(ragged_tma_store, `nvvm.grid_constant`)
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.config.num_threads)
        )
    )
    @__llvm_metadata(`nvvm.cluster_dim`=StaticTuple[Int32, 3](2, 1, 1))
    @__llvm_metadata(`nvvm.minctasm`=Int(1))
    def kernel(
        q_tma_op: QTMATile[
            Self.KVLUTType.dtype,
            Self.config.swizzle_mode,
            BM=Self.config.BM,
            depth=Self.config.qk_depth,
            group=Self.config.group,
            decoding=False,
            num_qk_stages=Self.config.num_qk_stages,
        ],
        k_tma_op: KVTMATile[
            Self.KVLUTType.dtype,
            Self.config.swizzle_mode,
            BN=Self.config.BN // 2,
            BK=Self.config.BK0,
        ],
        v_tma_op: KVTMATile[
            Self.KVLUTType.dtype,
            Self.config.swizzle_mode,
            BN=Self.config.BK1,
            BK=Self.config.ov_depth // 4,
        ],
        ragged_tma_store: RaggedTMA3DTile[
            Self.output_type,
            Self.config.swizzle_mode,
            BM=Self.config.BM,
            BN=Self.config.ov_depth,
        ],
        kv_lut: Self.KVLUTType,
        scale: Float32,
        batch_size: UInt32,
        num_keys_arg: UInt32,
        pack: Pack[
            Self.MaskType,
            Self.SchedulerType,
            Self.ValidLengthType,
            NullPointer[DType.float32],  # SinkType (unused for depth512)
            Self.KVRowOffsetsType,
            Self.MaxSeqLenType,
            Self.PartitionType,
        ],
    ):
        comptime assert _is_decoding[Self.MaxSeqLenType]() == False
        comptime assert Self.config.supported(), Self.config.description()
        comptime assert (
            not Self.PartitionType.do_partition
        ), "Partitioning not supported with depth512 pair-CTA"

        mask = pack.mask
        valid_length = pack.valid_length
        kv_input_row_offsets = pack.kv_input_row_offsets
        max_seq_len = pack.max_seq_len
        partition = pack.partition

        var smem = Self.SmemType()

        comptime num_reg_softmax = 256
        comptime num_reg_correction = 184
        comptime num_reg_other = 64

        # ---- Initialization (per-CTA, then cluster sync) ----------------

        var warp_idx = UInt32(warp_id[broadcast=True]())
        if warp_idx == 0:
            # Initialize all barriers.
            Depth512MBars[Self.config.num_kv_stages](smem.mbar_base()).init(
                lane_idx=Int32(thread_idx.x)
            )
        elif warp_idx == 1:
            # TMEM allocation (pair-CTA cooperative).
            _ = Self.TmemAllocType.allocate(
                Self.TmemAllocType.SmemAddrStorage(smem.tmem_addr_ptr())
            )
        elif warp_idx == 2:
            e = elect()
            if e != 0:
                q_tma_op.prefetch_descriptor()
            if e != 0:
                k_tma_op.prefetch_descriptor()
            if e != 0:
                v_tma_op.prefetch_descriptor()

        fence_mbarrier_init()
        cluster_sync()

        # ---- Warp dispatch -----------------------------------------------

        var cta_rank = UInt32(block_rank_in_cluster() % 2)

        if warp_idx < 4:
            # Softmax warp group (warps 0-3, 128 threads).
            warpgroup_reg_alloc[num_reg_softmax]()
            var seq_info: SeqInfo = get_seq_info[
                Self.PairBM,
                Self.num_q_heads,
                Self.MaskType.get_type_name() == "CausalMask",
                pair_cta=True,
            ](batch_size, max_seq_len, valid_length, partition)

            if not seq_info.is_valid():
                return

            var pos: PositionSummary = PositionSummary.create[
                ragged=Self.ragged,
                _is_cache_length_accurate=Self._is_cache_length_accurate,
            ](
                kv_lut,
                seq_info,
                num_keys_arg,
                kv_input_row_offsets,
                max_seq_len,
            )

            # Compute per-CTA output write parameters.
            gmem_row = Self.PositionType.get_q_gmem_row[ragged=Self.ragged](
                seq_info, max_seq_len.as_uint32()
            )
            var out_row_idx = gmem_row + cta_rank * UInt32(Self.BM)
            var out_head_idx = seq_info.head_idx
            var num_output_rows = min(
                Int32(seq_info.seq_len)
                - Int32(seq_info.prompt_offset)
                - Int32(cta_rank) * Int32(Self.BM),
                Int32(Self.BM),
            )

            depth512_softmax[
                Self.MaskType,
                Self.qkv_type,
                Self.output_type,
                Self.config,
                Self.page_size,
            ](
                smem,
                pos.score_row,
                pos.num_keys,
                mask,
                scale.cast[Self.accum_type](),
                ragged_tma_store,
                num_output_rows,
                out_head_idx,
                out_row_idx,
            )

        elif warp_idx < 8:
            # Correction warp group (warps 4-7, 128 threads).
            warpgroup_reg_alloc[num_reg_correction]()

            var seq_info: SeqInfo = get_seq_info[
                Self.PairBM,
                Self.num_q_heads,
                Self.MaskType.get_type_name() == "CausalMask",
                pair_cta=True,
            ](batch_size, max_seq_len, valid_length, partition)
            if not seq_info.is_valid():
                return
            var pos: PositionSummary = PositionSummary.create[
                ragged=Self.ragged,
                _is_cache_length_accurate=Self._is_cache_length_accurate,
            ](
                kv_lut,
                seq_info,
                num_keys_arg,
                kv_input_row_offsets,
                max_seq_len,
            )
            depth512_correction[
                Self.MaskType,
                Self.qkv_type,
                Self.config,
                Self.page_size,
            ](
                smem,
                pos.score_row,
                pos.num_keys,
                mask,
            )

        elif warp_idx == 8:
            # MMA warp (single warp; leader issues pair-CTA MMA, peer
            # early-returns inside depth512_mma).
            warpgroup_reg_dealloc[num_reg_other]()

            var seq_info: SeqInfo = get_seq_info[
                Self.PairBM,
                Self.num_q_heads,
                Self.MaskType.get_type_name() == "CausalMask",
                pair_cta=True,
            ](batch_size, max_seq_len, valid_length, partition)

            if not seq_info.is_valid():
                var tmem = Self.TmemAllocType.from_shared(
                    Self.TmemAllocType.SmemAddrStorage(smem.tmem_addr_ptr())
                )
                tmem.release_lock()
                tmem.deallocate()
                return
            var pos: PositionSummary = PositionSummary.create[
                ragged=Self.ragged,
                _is_cache_length_accurate=Self._is_cache_length_accurate,
            ](
                kv_lut,
                seq_info,
                num_keys_arg,
                kv_input_row_offsets,
                max_seq_len,
            )
            depth512_mma[
                Self.MaskType,
                Self.qkv_type,
                Self.config,
                Self.page_size,
            ](
                smem,
                pos.score_row,
                pos.num_keys,
                mask,
            )

        elif warp_idx == 9:
            # Load warp (single warp; both CTAs issue TMA multicast).
            warpgroup_reg_dealloc[num_reg_other]()

            var seq_info: SeqInfo = get_seq_info[
                Self.PairBM,
                Self.num_q_heads,
                Self.MaskType.get_type_name() == "CausalMask",
                pair_cta=True,
            ](batch_size, max_seq_len, valid_length, partition)

            if not seq_info.is_valid():
                return
            var pos: PositionSummary = PositionSummary.create[
                ragged=Self.ragged,
                _is_cache_length_accurate=Self._is_cache_length_accurate,
            ](
                kv_lut,
                seq_info,
                num_keys_arg,
                kv_input_row_offsets,
                max_seq_len,
            )
            depth512_load[
                Self.KVLUTType,
                Self.MaskType,
                Self.qkv_type,
                Self.config,
                Self.ValidLengthType,
                Self._is_cache_length_accurate,
                Self.MaxSeqLenType,
            ](
                smem,
                pos.score_row,
                pos.num_keys,
                seq_info,
                max_seq_len,
                mask,
                q_tma_op,
                k_tma_op,
                v_tma_op,
                kv_lut,
            )

        else:
            # Spare warps 10-11 (no-op).
            warpgroup_reg_dealloc[24]()

    @staticmethod
    @always_inline
    def mask_status(
        mask: Self.MaskType, score_row: UInt32, kv_row: UInt32
    ) -> TileMaskStatus:
        return mask.status(
            Index[dtype=DType.int32](
                Int(score_row),
                Int(kv_row),
            ),
            Index[dtype=DType.int32](Self.PairBM, Self.BN),
        )
