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

from math import align_up
from sys import simd_width_of, size_of
import gpu.primitives.warp as warp
from gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    barrier,
    thread_idx,
    warp_id,
)
from gpu.globals import WARPGROUP_SIZE
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from gpu.memory import AddressSpace, external_memory
from gpu.compute.arch.mma_nvidia_sm100 import MMASmemDescriptorPair
from linalg.matmul.gpu.sm100_structured.structured_kernels.tmem import (
    TmemAllocation,
)
from layout.tma_async import (
    SharedMemBarrier,
    RaggedTMA3DTile,
)
from nn.fa4_config import FA4Config, EnableForcedOrdering
from nn.sm100_attention_utils import (
    SharedMemPointer,
    MBarType,
    SM100TensorAccumulatorSS,
    SM100TensorAccumulatorTS,
    elect,
    FA4MiscMBars,
)
from nn.mha_fa3_utils import (
    get_seq_info,
    KVTMATile,
    MHAPosition,
    OptionalPointer,
    Pack,
    PositionSummary,
    QTMATile,
)
from nn.mha_mask import MHAMask, TileMaskStatus
from nn.mha_operand import MHAOperand
from nn.mha_tile_scheduler import (
    MHATileScheduler,
    SeqInfo,
)
from nn.mha_utils import (
    MHAConfig,
    MHAPartitionScheme,
    OptionallyStaticInt,
    _is_decoding,
)
from utils.index import Index
from utils.static_tuple import StaticTuple
from linalg.arch.sm100.mma import smem_descriptor
from .softmax_warp import fa4_softmax
from .correction_warp import fa4_correction
from .load_warp import fa4_load
from .mma_warp import fa4_mma


struct SM100MHA2Q[
    KVLUTType: MHAOperand,
    output_type: DType,
    MaskType: MHAMask,
    SchedulerType: MHATileScheduler,
    config: FA4Config,
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

    comptime cta_group = 1  # TODO: support 2
    comptime BM = Self.config.BM
    comptime BN = Self.config.BN
    comptime depth = Self.config.depth
    comptime padded_depth = Self.config.padded_depth
    comptime num_q_heads = Self.config.num_q_heads
    comptime group = Self.config.group
    comptime ragged = not Self.ValidLengthType.is_null
    comptime page_size = Self.KVLUTType.page_size

    comptime num_m_mmas = 2
    comptime MMA_M = Self.config.BM // Self.num_m_mmas
    comptime qo_elements = Self.padded_depth * Self.MMA_M
    comptime qkv_dt_size = size_of[Self.qkv_type]()
    comptime HalfBM = Self.BM // 2

    comptime num_qk_stages = Self.config.num_qk_stages
    comptime num_pv_stages = Self.config.num_pv_stages

    # Unified misc barriers type managing all barriers including K/V/O pipelines
    comptime MiscMBarsType = FA4MiscMBars[
        num_qk_stages = Self.num_qk_stages,
        num_pv_stages = Self.num_pv_stages,
        num_kv_stages = Self.config.num_kv_stages,
        separate_kv=True,
        use_order_barriers=EnableForcedOrdering,
    ]

    # TMEM allocation type for this kernel's cta_group configuration
    comptime TmemAllocType = TmemAllocation[Self.cta_group]

    # First MMA is Q@K' (can be staged by num_qk_stages)
    # (BM x depth) @ (BN x depth)' -> (BM x BN)
    comptime UMMA0Type = SM100TensorAccumulatorSS[
        Self.qkv_type,
        Self.accum_type,
        MMA_M = Self.MMA_M,  # generally 128
        MMA_N = Self.BN,
        BK = align_up(Self.depth, Self.config.MMA_K),  # BK in memory depth
        swizzle_a = Self.config.swizzle_mode,
        swizzle_b = Self.config.swizzle_mode,
        transpose_b=True,
        num_stages = Self.num_qk_stages,
    ]
    # Second MMA is P@V (V not staged, but P writing can be staged)
    # (BM x BN) @ (BN x depth) -> (BM x depth)
    comptime UMMA1Type = SM100TensorAccumulatorTS[
        Self.qkv_type,
        Self.accum_type,
        MMA_M = Self.MMA_M,
        MMA_N = Self.config.padded_depth,
        BK = Self.BN,
        swizzle_b = Self.config.swizzle_mode,
        transpose_b=False,
        num_stages = Self.num_pv_stages,
    ]

    comptime swizzle_granularity = Self.config.swizzle_mode.bytes() // Self.qkv_dt_size
    comptime k_elements: UInt32 = UInt32(
        Self.swizzle_granularity * Self.config.BN
    )
    comptime qo_bytes: UInt32 = UInt32(Self.qkv_dt_size * Self.qo_elements)
    comptime k_bytes: UInt32 = UInt32(Self.qkv_dt_size) * Self.k_elements
    comptime MMA_K = 16
    comptime v_bytes_per_mma: UInt32 = UInt32(
        Self.qkv_dt_size * Self.MMA_K * Self.config.padded_depth
    )

    comptime PositionType = MHAPosition[
        Self.config.BM,
        Self.config.BN,
        Self.config.depth,
        Self.config.padded_depth,
        Self.config.num_q_heads,
        Self.config.group,
        _is_decoding[Self.MaxSeqLenType](),
    ]

    comptime q_offset: Int32 = 0
    comptime kv_offset: Int32 = Self.q_offset + Int32(
        Self.config.BM * Self.config.padded_depth
    )
    comptime correction_offset: Int32 = (
        Self.kv_offset
        + Int32(
            2
            * Self.config.num_kv_stages
            * Self.config.padded_depth
            * Self.config.BN
        )
    ) * Int32(size_of[Self.qkv_type]()) // Int32(size_of[DType.float32]())
    comptime mbar_offset = (
        Self.correction_offset + Int32(Self.config.BM)
    ) * Int32(size_of[DType.float32]()) // Int32(size_of[SharedMemBarrier]())

    @staticmethod
    @always_inline
    fn get_tmem_ptr(
        misc_mbars: Self.MiscMBarsType,
    ) -> SharedMemPointer[UInt32]:
        # tmem_ptr comes after all barriers (now unified in MiscMBarsType)
        return (misc_mbars.mbar_base + Self.MiscMBarsType.num_mbars()).bitcast[
            UInt32
        ]()

    @staticmethod
    @always_inline
    fn get_tmem_addr_storage(
        misc_mbars: Self.MiscMBarsType,
    ) -> Self.TmemAllocType.SmemAddrStorage:
        """Get TMEM address storage as typed SMemArray for TmemAllocation."""
        return Self.TmemAllocType.SmemAddrStorage(Self.get_tmem_ptr(misc_mbars))

    @staticmethod
    @always_inline
    fn get_q_smem(
        misc_mbars: Self.MiscMBarsType,
    ) -> SharedMemPointer[Scalar[Self.qkv_type]]:
        return (misc_mbars.mbar_base - Self.mbar_offset).bitcast[
            Scalar[Self.qkv_type]
        ]() + Self.q_offset

    @staticmethod
    @always_inline
    fn get_kv_smem(
        misc_mbars: Self.MiscMBarsType,
    ) -> SharedMemPointer[Scalar[Self.qkv_type]]:
        return (misc_mbars.mbar_base - Self.mbar_offset).bitcast[
            Scalar[Self.qkv_type]
        ]() + Self.kv_offset

    @staticmethod
    @always_inline
    fn get_correction_smem(
        misc_mbars: Self.MiscMBarsType,
    ) -> SharedMemPointer[Float32]:
        return (misc_mbars.mbar_base - Self.mbar_offset).bitcast[
            Float32
        ]() + Self.correction_offset

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
    fn kernel(
        q_tma_op: QTMATile[
            Self.KVLUTType.dtype,
            Self.config.swizzle_mode,
            BM = Self.config.BM // 2,
            depth = Self.config.depth,
            group = Self.config.group,
            decoding=False,
            num_qk_stages = Self.config.num_qk_stages,
        ],
        k_tma_op: KVTMATile[
            Self.KVLUTType.dtype,
            Self.config.swizzle_mode,
            BN = Self.config.BN,
            BK = Self.config.BK0,
        ],
        v_tma_op: KVTMATile[
            Self.KVLUTType.dtype,
            Self.config.swizzle_mode,
            BN = Self.config.BN,
            BK = Self.config.padded_depth,
        ],
        ragged_tma_store: RaggedTMA3DTile[
            Self.output_type,
            Self.config.swizzle_mode,
            BM = Self.config.BM // 2,
            BN = Self.config.depth,
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
        comptime assert Self.MMA_M == 64 or Self.MMA_M == 128
        comptime assert _is_decoding[Self.MaxSeqLenType]() == False
        comptime assert Self.config.supported(), (
            "depth = "
            + String(Self.config.depth)
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
        mbar_base = (
            external_memory[
                SharedMemBarrier,
                address_space = AddressSpace.SHARED,
                alignment=128,
                name="mha_dynamic_shared_memory",
            ]()
            + Self.mbar_offset
        )

        var misc_mbars: Self.MiscMBarsType = {mbar_base}

        # https://github.com/NVIDIA/cutlass/blob/main/examples/77_blackwell_fmha/kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp
        comptime num_reg_softmax = 192
        comptime num_reg_correction = 88
        comptime num_reg_other = 40

        comptime assert not Self.PartitionType.do_partition, (
            "Neither partitioning nor decoding are supported by the 2-q"
            " implementation."
        )

        var warp_idx = UInt32(warp.broadcast(warp_id()))
        if warp_idx == 0:
            # Initialize all barriers (S/C/order/Q1Sync/K/V/O) in one call
            misc_mbars.init(lane_idx=Int32(thread_idx.x))
        elif warp_idx == 1:
            _ = Self.TmemAllocType.allocate(
                Self.get_tmem_addr_storage(misc_mbars)
            )
        elif warp_idx == 2:
            e = elect()
            if e != 0:
                q_tma_op.prefetch_descriptor()
            if e != 0:
                k_tma_op.prefetch_descriptor()
            if e != 0:
                v_tma_op.prefetch_descriptor()

        barrier()

        # warp group partitioning
        # Two QO:
        if warp_idx < 8:
            # softmax $warp_group_idx
            warpgroup_reg_alloc[num_reg_softmax]()
            var seq_info: SeqInfo = get_seq_info[Self.BM, Self.num_q_heads](
                batch_size, max_seq_len, valid_length, partition
            )

            if not seq_info.is_valid():
                return

            var pos: PositionSummary = PositionSummary.create[
                ragged = Self.ragged,
                _is_cache_length_accurate = Self._is_cache_length_accurate,
            ](kv_lut, seq_info, num_keys_arg, kv_input_row_offsets, max_seq_len)

            fa4_softmax[
                Self.KVLUTType,
                Self.output_type,
                Self.MaskType,
                Self.config,
                Self.ValidLengthType,
                Self.SinkType,
                Self._is_cache_length_accurate,
                Self.MaxSeqLenType,
            ](
                misc_mbars,
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

            var seq_info: SeqInfo = get_seq_info[Self.BM, Self.num_q_heads](
                batch_size, max_seq_len, valid_length, partition
            )
            if not seq_info.is_valid():
                return
            var pos: PositionSummary = PositionSummary.create[
                ragged = Self.ragged,
                _is_cache_length_accurate = Self._is_cache_length_accurate,
            ](kv_lut, seq_info, num_keys_arg, kv_input_row_offsets, max_seq_len)
            fa4_correction[
                Self.qkv_type,
                Self.MaskType,
                Self.config,
                Self.page_size,
            ](
                misc_mbars,
                pos.score_row,
                pos.num_keys,
                mask,
            )
        else:
            if warp_idx == 13:  # produce
                warpgroup_reg_dealloc[num_reg_other]()
                var seq_info: SeqInfo = get_seq_info[Self.BM, Self.num_q_heads](
                    batch_size, max_seq_len, valid_length, partition
                )

                if not seq_info.is_valid():
                    return
                var pos: PositionSummary = PositionSummary.create[
                    ragged = Self.ragged,
                    _is_cache_length_accurate = Self._is_cache_length_accurate,
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
                    misc_mbars,
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
                var seq_info: SeqInfo = get_seq_info[Self.BM, Self.num_q_heads](
                    batch_size, max_seq_len, valid_length, partition
                )

                if not seq_info.is_valid():
                    var tmem = Self.TmemAllocType.from_shared(
                        Self.get_tmem_addr_storage(misc_mbars)
                    )
                    tmem.release_lock()
                    tmem.deallocate()
                    return
                var pos: PositionSummary = PositionSummary.create[
                    ragged = Self.ragged,
                    _is_cache_length_accurate = Self._is_cache_length_accurate,
                ](
                    kv_lut,
                    seq_info,
                    num_keys_arg,
                    kv_input_row_offsets,
                    max_seq_len,
                )
                fa4_mma[
                    Self.qkv_type,
                    Self.MaskType,
                    Self.config,
                    Self.page_size,
                ](
                    misc_mbars,
                    pos.score_row,
                    pos.num_keys,
                    mask,
                )
            else:
                warpgroup_reg_dealloc[24]()

    @staticmethod
    @always_inline
    fn mask_status(
        mask: Self.MaskType, score_row: UInt32, kv_row: UInt32
    ) -> TileMaskStatus:
        return mask.status(
            Index[dtype = DType.int32](
                Int(score_row),
                Int(kv_row),
            ),
            Index[dtype = DType.int32](Self.BM, Self.BN),
        )

    @staticmethod
    @always_inline
    fn descriptor_q(
        q_smem: SharedMemPointer[Scalar[Self.qkv_type]],
    ) -> MMASmemDescriptorPair:
        return smem_descriptor[
            BMN = Self.config.BM // 2,
            BK = Self.config.BK0,
            swizzle_mode = Self.config.swizzle_mode,
            is_k_major=True,
        ](q_smem)
