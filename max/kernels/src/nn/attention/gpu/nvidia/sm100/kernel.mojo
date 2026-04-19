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

from std.math import align_up
from std.sys import simd_width_of, size_of
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    barrier,
    thread_idx,
    warp_id,
)
from std.gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from std.gpu.compute.arch.mma_nvidia_sm100 import MMASmemDescriptorPair
from std.gpu.primitives.warp import broadcast
from std.gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_release_allocation_lock,
)
from std.gpu.memory import fence_mbarrier_init
from std.gpu.primitives.cluster import block_rank_in_cluster, cluster_sync
from layout.tma_async import RaggedTMA3DTile
from nn.attention.gpu.nvidia.sm100.attention import (
    FA4Config,
    EnableForcedOrdering,
)
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    SharedMemPointer,
    SM100TensorAccumulatorSS,
    SM100TensorAccumulatorTS,
    elect,
    FA4MiscMBars,
)
from nn.attention.gpu.nvidia.sm90.attention import (
    get_seq_info,
    KVTMATile,
    MHAPosition,
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
from linalg.arch.sm100.mma import smem_descriptor
from .smem import SM100AttentionSMem
from .softmax_warp import fa4_softmax
from .correction_warp import fa4_correction
from .load_warp import fa4_load
from .mma_warp import fa4_mma


struct SM100MHA2Q[
    KVLUTType: MHAOperand,
    output_type: DType,
    MaskType: MHAMask,
    SchedulerType: MHATileScheduler,
    config: FA4Config[KVLUTType.dtype],
    ValidLengthType: OptionalPointer,
    SinkType: OptionalPointer,
    KVRowOffsetsType: OptionalPointer,
    _is_cache_length_accurate: Bool,
    MaxSeqLenType: OptionallyStaticInt,
    PartitionType: MHAPartitionScheme,
](TrivialRegisterPassable):
    comptime qkv_type = Self.KVLUTType.dtype
    comptime accum_type = DType.float32
    comptime simd_size: Int = simd_width_of[Self.qkv_type]()

    comptime pair_cta: Bool = Self.config.pair_cta
    comptime cta_group: Int = 2 if Self.pair_cta else 1
    comptime BM = Self.config.BM
    comptime BN = Self.config.BN
    comptime depth = Self.config.qk_depth
    comptime padded_depth = Self.config.padded_qk_depth
    comptime num_q_heads = Self.config.num_q_heads
    comptime group = Self.config.group
    comptime fuse_gqa = Self.config.fuse_gqa
    # BM_eff: sequence positions per full tile (BM // group when fusing)
    comptime BM_eff: Int = Self.config.BM_eff()
    # BM_mask: the BM value passed to mask functions.
    # For pair-CTA, use PairBM so both CTAs make identical skip decisions.
    comptime BM_mask: Int = Self.config.PairBM_eff()
    comptime ragged = not Self.ValidLengthType.is_null
    comptime page_size = Self.KVLUTType.page_size

    comptime num_m_mmas = 2
    comptime MMA_M = Self.config.MMA_M  # 128 single-CTA, 256 pair-CTA
    comptime qo_elements = Self.padded_depth * Self.HalfBM
    comptime qkv_dt_size = size_of[Self.qkv_type]()
    comptime HalfBM = Self.BM // 2

    comptime num_qk_stages = Self.config.num_qk_stages
    comptime num_pv_stages = Self.config.num_pv_stages

    # Unified misc barriers type managing all barriers including K/V/O pipelines
    comptime MiscMBarsType = FA4MiscMBars[
        num_qk_stages=Self.num_qk_stages,
        num_pv_stages=Self.num_pv_stages,
        num_kv_stages=Self.config.num_kv_stages,
        use_order_barriers=EnableForcedOrdering,
        use_fused_kv=Self.config.use_fused_kv,
        pair_cta=Self.pair_cta,
    ]

    # First MMA is Q@K' (can be staged by num_qk_stages)
    # (BM x depth) @ (BN x depth)' -> (BM x BN)
    comptime UMMA0Type = SM100TensorAccumulatorSS[
        Self.qkv_type,
        Self.accum_type,
        MMA_M=Self.MMA_M,  # 128 single-CTA, 256 pair-CTA
        MMA_N=Self.BN,
        BK=align_up(Self.depth, Self.config.MMA_K),  # BK in memory depth
        swizzle_a=Self.config.swizzle_mode,
        swizzle_b=Self.config.swizzle_mode,
        transpose_b=True,
        num_stages=Self.num_qk_stages,
        cta_group=Self.cta_group,
    ]
    # Second MMA is P@V (V not staged, but P writing can be staged)
    # (BM x BN) @ (BN x depth) -> (BM x depth)
    comptime UMMA1Type = SM100TensorAccumulatorTS[
        Self.qkv_type,
        Self.accum_type,
        MMA_M=Self.MMA_M,
        MMA_N=Self.config.padded_ov_depth,
        BK=Self.BN,
        swizzle_b=Self.config.swizzle_mode,
        transpose_b=False,
        num_stages=Self.num_pv_stages,
        cta_group=Self.cta_group,
    ]

    comptime swizzle_granularity = Self.config.swizzle_mode.bytes() // Self.qkv_dt_size
    comptime k_elements: UInt32 = UInt32(
        Self.swizzle_granularity * Self.config.BN
    )
    comptime qo_bytes: UInt32 = UInt32(Self.qkv_dt_size * Self.qo_elements)
    comptime k_bytes: UInt32 = UInt32(Self.qkv_dt_size) * Self.k_elements
    comptime MMA_K = 16
    comptime v_bytes_per_mma: UInt32 = UInt32(
        Self.qkv_dt_size * Self.MMA_K * Self.config.padded_ov_depth
    )

    comptime PositionType = MHAPosition[
        Self.config.BM,
        Self.config.BN,
        Self.config.qk_depth,
        Self.config.padded_qk_depth,
        Self.config.num_q_heads,
        Self.config.group,
        _is_decoding[Self.MaxSeqLenType](),
    ]

    comptime SmemType = SM100AttentionSMem[Self.config]

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
    @__llvm_metadata(`nvvm.minctasm`=Int(1))
    @__llvm_metadata(
        `nvvm.cluster_dim`=StaticTuple[Int32, 3](Int32(Self.cta_group), 1, 1)
    )
    @__name(
        t"sm100_mha_2q_depth{Self.config.qk_depth}_{Self.qkv_type}_{Self.output_type}_nqh{Self.config.num_q_heads}_nkvh{Self.config.num_kv_heads}",
        mangle=True,
    )
    def kernel(
        q_tma_op: QTMATile[
            Self.KVLUTType.dtype,
            Self.config.swizzle_mode,
            BM=Self.config.BM // 2,
            depth=Self.config.qk_depth,
            group=Self.config.group,
            decoding=False,
            fuse_gqa=Self.fuse_gqa,
            num_qk_stages=Self.config.num_qk_stages,
        ],
        k_tma_op: KVTMATile[
            Self.KVLUTType.dtype,
            Self.config.swizzle_mode,
            BN=Self.config.k_rows_per_cta(),
            BK=Self.config.BK0,
        ],
        v_tma_op: KVTMATile[
            Self.KVLUTType.dtype,
            Self.config.swizzle_mode,
            BN=Self.config.BN,
            BK=Self.config.v_cols_per_cta(),
        ],
        ragged_tma_store: RaggedTMA3DTile[
            Self.output_type,
            Self.config.swizzle_mode,
            BM=Self.config.BM // 2,
            BN=Self.config.ov_depth,
            group=Self.config.group if Self.fuse_gqa else 1,
        ],
        kv_lut: Self.KVLUTType,
        scale: Float32,
        batch_size: UInt32,
        num_keys_arg: UInt32,
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
        comptime assert (
            Self.MMA_M == 64 or Self.MMA_M == 128 or Self.MMA_M == 256
        )
        comptime assert _is_decoding[Self.MaxSeqLenType]() == False
        comptime assert Self.config.supported(), (
            "depth = "
            + String(Self.config.qk_depth)
            + "\nBN = "
            + String(Self.config.BN)
            + "\nnum_kv_stages = "
            + String(Self.config.num_kv_stages)
            + "\ntmem_used = "
            + String(Self.config.tmem_used)
            + "\nsmem_used = "
            + String(Self.config.smem_used)
        )
        comptime assert (
            not Self.SchedulerType.may_advance
        ), "Persistent kernels not yet supported with FA4"

        mask = pack.mask
        scheduler = pack.scheduler
        valid_length = pack.valid_length
        sink_weights = pack.sink_weights
        kv_input_row_offsets = pack.kv_input_row_offsets
        max_seq_len = pack.max_seq_len
        partition = pack.partition

        comptime num_qo = Self.config.num_qo()
        # TODO: We may want to support num_qo>2 for depth=64?
        comptime assert (
            num_qo == 1 or num_qo == 2
        ), "Currently only support num_qo == 1 or 2"
        var smem = Self.SmemType()
        var misc_mbars = smem.misc_mbars()

        # https://github.com/NVIDIA/cutlass/blob/main/examples/77_blackwell_fmha/kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp
        comptime num_reg_softmax = 192
        comptime num_reg_correction = 88
        comptime num_reg_other = 40

        comptime assert not Self.PartitionType.do_partition, (
            "Neither partitioning nor decoding are supported by the 2-q"
            " implementation."
        )

        var warp_idx = UInt32(warp_id[broadcast=True]())
        if warp_idx == 0:
            # Initialize all barriers (S/C/order/Q1Sync/K/V/O) in one call
            misc_mbars.init(lane_idx=Int32(thread_idx.x))
        elif warp_idx == 1:
            tcgen05_alloc[Int32(Self.cta_group)](
                smem.tmem_addr_ptr(),
                UInt32(512),
            )
        elif warp_idx == 2:
            e = elect()
            if e != 0:
                q_tma_op.prefetch_descriptor()
            if e != 0:
                k_tma_op.prefetch_descriptor()
            if e != 0:
                v_tma_op.prefetch_descriptor()

        # Pair-CTA: cluster_sync ensures both CTAs see each other's barriers.
        # Single-CTA: plain barrier suffices.
        comptime if Self.pair_cta:
            fence_mbarrier_init()
            cluster_sync()
        else:
            barrier()

        # warp group partitioning
        # Two QO:
        #
        # Pair-CTA: early returns are replaced with conditional work so that
        # ALL threads always reach the cluster_sync at the bottom.  Without
        # this, invalid tiles cause some warps to return early while warps
        # 14-15 (and any valid warps) block at cluster_sync forever.
        if warp_idx < 8:
            # softmax $warp_group_idx
            warpgroup_reg_alloc[num_reg_softmax]()
            var seq_info: SeqInfo = get_seq_info[
                Self.BM_mask,
                Self.num_q_heads
                // Self.group if Self.fuse_gqa else Self.num_q_heads,
                Self.MaskType.get_type_name() == "CausalMask",
                pair_cta=Self.pair_cta,
            ](batch_size, max_seq_len, valid_length, partition)

            comptime if not Self.pair_cta:
                if not seq_info.is_valid():
                    return

            if seq_info.is_valid():
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

                fa4_softmax[
                    Self.KVLUTType,
                    Self.config,
                    Self.ValidLengthType,
                    Self.SinkType,
                    Self._is_cache_length_accurate,
                    Self.MaxSeqLenType,
                ](
                    smem,
                    pos.score_row,
                    seq_info,
                    mask,
                    pos.num_keys,
                    scale.cast[Self.accum_type](),
                    max_seq_len.as_uint32(),
                    ragged_tma_store,
                    sink_weights,
                )

        elif warp_idx < 12:
            # correction
            warpgroup_reg_dealloc[num_reg_correction]()

            var seq_info: SeqInfo = get_seq_info[
                Self.BM_mask,
                Self.num_q_heads
                // Self.group if Self.fuse_gqa else Self.num_q_heads,
                Self.MaskType.get_type_name() == "CausalMask",
                pair_cta=Self.pair_cta,
            ](batch_size, max_seq_len, valid_length, partition)

            comptime if not Self.pair_cta:
                if not seq_info.is_valid():
                    return

            if seq_info.is_valid():
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
                fa4_correction[
                    Self.config,
                    Self.page_size,
                ](
                    smem,
                    pos.score_row,
                    pos.num_keys,
                    mask,
                )
        else:
            if warp_idx == 13:  # produce
                warpgroup_reg_dealloc[num_reg_other]()
                var seq_info: SeqInfo = get_seq_info[
                    Self.BM_mask,
                    Self.num_q_heads
                    // Self.group if Self.fuse_gqa else Self.num_q_heads,
                    Self.MaskType.get_type_name() == "CausalMask",
                    pair_cta=Self.pair_cta,
                ](batch_size, max_seq_len, valid_length, partition)

                comptime if not Self.pair_cta:
                    if not seq_info.is_valid():
                        return

                if seq_info.is_valid():
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
                    fa4_load[
                        Self.KVLUTType,
                        Self.MaskType,
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

            elif warp_idx == 12:  # Q @ K', P @ V
                warpgroup_reg_dealloc[num_reg_other]()
                var seq_info: SeqInfo = get_seq_info[
                    Self.BM_mask,
                    Self.num_q_heads
                    // Self.group if Self.fuse_gqa else Self.num_q_heads,
                    Self.MaskType.get_type_name() == "CausalMask",
                    pair_cta=Self.pair_cta,
                ](batch_size, max_seq_len, valid_length, partition)

                comptime if not Self.pair_cta:
                    if not seq_info.is_valid():
                        var tmem_addr = smem.tmem_addr_ptr()[]
                        tcgen05_release_allocation_lock[Int32(Self.cta_group)]()
                        tcgen05_dealloc[Int32(Self.cta_group)](
                            tmem_addr, UInt32(512)
                        )
                        return
                var execute: Bool = seq_info.is_valid()
                comptime if Self.pair_cta:
                    # ---- Pair-CTA: leader-only guard ----
                    execute &= broadcast(block_rank_in_cluster()) % 2 == 0
                if execute:
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
                    fa4_mma[Self.config, page_size=Self.page_size](
                        smem,
                        pos.score_row,
                        pos.num_keys,
                        mask,
                    )
            else:
                warpgroup_reg_dealloc[24]()

        # Pair-CTA: cluster_sync before dealloc so that stmatrix
        # (which uses shared::cluster on SM100) in the peer CTA has
        # finished before either CTA exits and breaks the cluster.
        # All early returns above were converted to fall-through for
        # pair_cta so that every thread reaches this sync point.
        comptime if Self.pair_cta:
            cluster_sync()
            if warp_idx == 0:
                var tmem_addr = smem.tmem_addr_ptr()[]
                tcgen05_release_allocation_lock[Int32(Self.cta_group)]()
                tcgen05_dealloc[Int32(Self.cta_group)](tmem_addr, UInt32(512))

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
            Index[dtype=DType.int32](Self.BM_mask, Self.BN),
        )

    @staticmethod
    @always_inline
    def descriptor_q(
        q_smem: SharedMemPointer[Scalar[Self.qkv_type]],
    ) -> MMASmemDescriptorPair:
        return smem_descriptor[
            BMN=Self.config.BM // 2,
            BK=Self.config.BK0,
            swizzle_mode=Self.config.swizzle_mode,
            is_k_major=True,
        ](q_smem)
