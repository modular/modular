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

from sys import simd_width_of, size_of
from math import ceildiv, exp2, recip
from math.constants import log2e
from nn.mha_operand import MHAOperand
from nn.mha_score_mod import ScoreModTrait
from nn.mha_mask import MHAMask, TileMaskStatus, MaskStrategy
from nn.mha_tile_scheduler import (
    MHATileScheduler,
    SeqInfo,
    TransientScheduler,
)
from nn.mha_sm100_2q import (
    FA4Config,
    FA4MiscMBars,
    KVPipeline,
    SharedMemPointer,
    SharedMemTensor,
    SharedMemLT,
    elect,
    MBarType,
    TMemTile,
    ProducerPipeline,
)
from nn.mla_prefill_sm100_bf16 import (
    MLAPositionSummary,
    MLAKVProducerPipeline,
    split_smem,
)
from nn.mha import q_num_matrix_view_rows
from nn.mha_sm100_2q import (
    SM100TensorAccumulatorSS,
    SM100TensorAccumulatorTS,
    LocalTensor,
    apply_mask,
    STMatrixLayout,
    elect_mma_arrive,
    TMADestination,
    sub_ftz,
)
from nn.mha_fa3_utils import (
    get_seq_info,
    KVTMATile,
    kv_coord,
    MHAPosition,
    NonNullPointer,
    NullPointer,
    OptionalPointer,
    output_reg_to_smem_st_matrix,
    _LocalTT,
    _SharedMemTT,
    Pack,
    produce,
    q_coord,
    q_tma,
    QTMATile,
)
from layout.tma_async import (
    SharedMemBarrier,
    RaggedTMA3DTile,
    PipelineState,
)
from layout._layout import row_major
from layout._tile_tensor import stack_allocation as tt_stack_allocation
from layout.swizzle import make_swizzle, Swizzle
from layout.tensor_core_async import tile_layout_k_major, tile_layout_mn_major
from layout.layout import Layout
from layout.layout_tensor import LayoutTensor
import gpu.primitives.warp as warp
from gpu.sync import (
    named_barrier,
    cp_async_bulk_commit_group,
    cp_async_bulk_wait_group,
)
from gpu.globals import WARPGROUP_SIZE, WARP_SIZE
from gpu.memory import AddressSpace, external_memory, fence_async_view_proxy
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu.host import DeviceContext, FuncAttribute
from gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from gpu.primitives.warp import _vote_nvidia_helper
from gpu.compute.arch.mma_nvidia_sm100 import MMASmemDescriptorPair
from gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    barrier,
    thread_idx,
    block_idx,
    warp_id,
)
from nn.mha_utils import (
    MHAConfig,
    MHAPartitionScheme,
    NoPartition,
    OptionallyStaticInt,
    _is_decoding,
)
from gpu.compute.arch.tcgen05 import *
from linalg.arch.sm100.mma import smem_descriptor
from utils.static_tuple import StaticTuple
from utils.index import Index
from kv_cache.types import swizzle_granularity, padded_depth


struct TMAtoCvtPipeline[
    num_kv_stages: Int,
    num_producer: Int,
    num_consumer: Int,
](TrivialRegisterPassable):
    var consumer_mbars: MBarType
    var producer_mbars: MBarType
    var state: PipelineState[Self.num_kv_stages]

    @always_inline
    fn __init__(out self, consumer_mbars: MBarType, producer_mbars: MBarType):
        self.consumer_mbars = consumer_mbars
        self.producer_mbars = producer_mbars
        self.state = {}

    @always_inline
    fn init(self):
        comptime for i in range(Self.num_kv_stages):
            self.consumer_mbars[i].init(Int32(Self.num_consumer))
            self.producer_mbars[i].init(Int32(Self.num_producer))

    @always_inline
    fn producer_mbar(self) -> MBarType:
        var idx: UInt32 = self.state.index()
        return self.producer_mbars + idx

    @always_inline
    fn consumer_mbar(self) -> MBarType:
        var idx: UInt32 = self.state.index()
        return self.consumer_mbars + idx

    @always_inline
    fn producer_acquire(self):
        self.consumer_mbar()[].wait(self.state.phase())

    @always_inline
    fn consumer_wait(self):
        self.producer_mbar()[].wait(self.state.phase())

    @always_inline
    fn producer_commit(mut self):
        _ = self.producer_mbar()[].arrive()
        self.step()

    @always_inline
    fn consumer_release(mut self):
        _ = self.consumer_mbar()[].arrive()
        self.step()

    @always_inline
    fn step(mut self):
        self.state.step()


struct CvtToMMAPipline[
    num_stages: Int,
    num_producer: Int,
    num_consumer: Int,
](TrivialRegisterPassable):
    var producer_mbars: MBarType
    var consumer_mbars: MBarType
    var state: PipelineState[Self.num_stages]

    @always_inline
    fn __init__(out self, producer_mbars: MBarType, consumer_mbars: MBarType):
        self.producer_mbars = producer_mbars
        self.consumer_mbars = consumer_mbars
        self.state = {}

    @always_inline
    fn init(self):
        comptime for i in range(Self.num_stages):
            self.producer_mbars[i].init(Int32(Self.num_producer))
            self.consumer_mbars[i].init(Int32(Self.num_consumer))

    @always_inline
    fn producer_mbar(self) -> MBarType:
        var idx: UInt32 = self.state.index()
        return self.producer_mbars + idx

    @always_inline
    fn consumer_mbar(self) -> MBarType:
        var idx: UInt32 = self.state.index()
        return self.consumer_mbars + idx

    @always_inline
    fn producer_acquire(self):
        self.consumer_mbar()[].wait(self.state.phase())

    @always_inline
    fn consumer_wait(self):
        self.producer_mbar()[].wait(self.state.phase())

    @always_inline
    fn producer_commit(mut self):
        _ = self.producer_mbar()[].arrive()
        self.step()

    @always_inline
    fn consumer_release(mut self, elect: Int32):
        elect_mma_arrive(self.consumer_mbar(), elect)
        self.step()

    @always_inline
    fn step(mut self):
        self.state.step()


@always_inline
fn cvt_block_fp8_to_bf16[
    input_type: DType,
    output_type: DType,
    //,
    swizzle_fp8: Swizzle,
    swizzle_bf16: Swizzle,
](
    input: LayoutTensor[
        input_type, _, MutAnyOrigin, address_space = AddressSpace.SHARED, ...
    ],
    mut output: LayoutTensor[
        output_type, _, MutAnyOrigin, address_space = AddressSpace.SHARED, ...
    ],
    tid: UInt32,
):
    comptime assert (
        input_type == DType.float8_e4m3fn and output_type == DType.bfloat16
    ), "Only support float8_e4m3fn to bfloat16 conversion"

    comptime num_regs = input.layout.size() // WARP_SIZE
    comptime row_stride = type_of(input).stride[0]()

    var t_row = tid // 16
    var t_col = tid % 16

    var fp8_regs = SIMD[input_type, num_regs](0)

    comptime for i in range(num_regs // 4):
        var row = UInt32(i * 2) + t_row
        var col = t_col * 4
        var elem_offset = row * UInt32(row_stride) + col
        var fp8x4 = (input.ptr + Int(swizzle_fp8(elem_offset))).load[width=4]()
        fp8_regs = fp8_regs.insert[offset = i * 4](fp8x4)

    # make sure all the fp8_regs are loaded
    named_barrier[64](6)

    comptime for i in range(num_regs // 4):
        var row = UInt32(i * 2) + t_row
        var col = t_col * 4
        var elem_offset = row * UInt32(row_stride) + col
        var fp16x4 = fp8_regs.slice[4, offset = i * 4]().cast[output_type]()
        (output.ptr + Int(swizzle_bf16(elem_offset))).store[width=4](fp16x4)


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
    fn __eq__(self, other: Int) -> Bool:
        return self == Self(Int32(other))


fn warp_idx_to_role(warp_idx: UInt32) -> WarpRole:
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


struct MLASmemStorage[dtype: DType, num_mbars: Int, config: FA4Config]:
    comptime q_smem_size = Self.config.BM * Self.config.padded_depth
    comptime num_kv_stages = Self.config.num_kv_stages * Self.config.num_qk_stages
    comptime kv_smem_size = Self.config.padded_depth * Self.config.BN * Self.num_kv_stages
    comptime correction_smem_size = Self.config.correction_smem_elements()

    var q_smem: InlineArray[Scalar[Self.dtype], Self.q_smem_size]
    var kv_smem: InlineArray[Scalar[Self.dtype], Self.kv_smem_size]
    var correction_smem: InlineArray[Float32, Self.correction_smem_size]
    var mbar_base: InlineArray[SharedMemBarrier, Self.num_mbars]
    var tma_to_cvt_producer_mbars: InlineArray[
        SharedMemBarrier, Self.num_kv_stages
    ]
    var tma_to_cvt_consumer_mbars: InlineArray[
        SharedMemBarrier, Self.num_kv_stages
    ]
    var cvt_to_mma_producer_mbars: InlineArray[SharedMemBarrier, 2]
    var cvt_to_mma_consumer_mbars: InlineArray[SharedMemBarrier, 2]
    var tmem_addr: InlineArray[UInt32, 1]


struct SM100MLA[
    KVLUTType: MHAOperand,
    KRopeType: MHAOperand,
    output_type: DType,
    MaskType: MHAMask,
    ScoreModType: ScoreModTrait,
    SchedulerType: MHATileScheduler,
    config: FA4Config,
    use_score_mod: Bool,
    ValidLengthType: OptionalPointer,
    SinkType: OptionalPointer,
    KVRowOffsetsType: OptionalPointer,
    MaxSeqLenType: OptionallyStaticInt,
    PartitionType: MHAPartitionScheme,
    _ndbuffer_mha_operand: Bool,
](TrivialRegisterPassable):
    comptime qkv_type = Self.KVLUTType.dtype
    comptime accum_type = DType.float32
    comptime simd_size: Int = simd_width_of[Self.qkv_type]()

    comptime cta_group = 1  # TODO: support 2
    comptime BM = Self.config.BM
    comptime BN = Self.config.BN
    comptime depth = Self.config.depth  # 192
    comptime padded_depth = Self.config.padded_depth  # 192
    comptime num_q_heads = Self.config.num_q_heads
    comptime group = Self.config.group
    comptime page_size = Self.KVLUTType.page_size

    comptime k_rope_depth = 64
    comptime kv_depth = Self.config.depth - Self.k_rope_depth
    comptime cache_depth = 576

    comptime k_rope_swizzle_mode = (
        TensorMapSwizzle.SWIZZLE_64B if Self.KRopeType.dtype
        == DType.float8_e4m3fn else TensorMapSwizzle.SWIZZLE_128B
    )

    comptime num_m_mmas = 2
    comptime MMA_M = Self.config.BM // Self.num_m_mmas
    comptime qo_elements = Self.padded_depth * Self.MMA_M
    comptime qkv_dt_size = size_of[Self.qkv_type]()

    comptime num_qk_stages = Self.config.num_qk_stages

    # First MMA is Q@K' (can be staged by num_qk_stages)
    # (BM x depth) @ (BN x depth)' -> (BM x BN)
    comptime UMMA0Type = SM100TensorAccumulatorSS[
        Self.qkv_type,
        Self.accum_type,
        MMA_M = Self.MMA_M,  # generally 128
        MMA_N = Self.BN,
        BK = Self.depth,  # BK in memory depth
        swizzle_a = Self.config.swizzle_mode,
        swizzle_b = Self.config.swizzle_mode,
        transpose_b=True,
        num_stages = Self.num_qk_stages,
    ]
    # Second MMA is P@V
    # (BM x BN) @ (BN x depth) -> (BM x depth)
    comptime UMMA1Type = SM100TensorAccumulatorTS[
        Self.qkv_type,
        Self.accum_type,
        MMA_M = Self.MMA_M,
        MMA_N = Self.kv_depth,  # 128
        BK = Self.BN,
        swizzle_b = Self.config.swizzle_mode,
        transpose_b=False,
        num_stages = Self.num_qk_stages,
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

    comptime KVPipelineType = KVPipeline[
        Self.config.num_kv_stages, Self.config.num_qk_stages
    ]
    comptime PositionType = MHAPosition[
        Self.config.BM,
        Self.config.BN,
        Self.config.depth,
        Self.config.padded_depth,
        Self.config.num_q_heads,
        Self.config.group,
        _is_decoding[Self.MaxSeqLenType](),
    ]
    # Unified misc barriers type managing all barriers including KV/O pipelines
    comptime MiscMBarsType = FA4MiscMBars[
        num_qk_stages = Self.config.num_qk_stages,
        num_pv_stages = Self.config.num_pv_stages,
        num_kv_stages = Self.config.num_kv_stages,
        separate_kv=False,  # MLA uses unified KV pipeline
    ]

    @staticmethod
    @__llvm_arg_metadata(q_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(k_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(v_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(k_rope_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(ragged_tma_store, `nvvm.grid_constant`)
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.config.num_threads)
        )
    )
    fn mla_prefill_kernel(
        q_tma_op: QTMATile[
            Self.KVLUTType.dtype,
            Self.config.swizzle_mode,
            BM = Self.config.BM // 2,
            depth = Self.config.BK0,
            group = Self.config.group,
            decoding=False,
        ],
        k_tma_op: KVTMATile[
            Self.KVLUTType.dtype,
            Self.config.swizzle_mode,
            BN = Self.config.BN,
            BK = Self.kv_depth,
        ],
        k_rope_tma_op: KVTMATile[
            Self.KRopeType.dtype,
            Self.k_rope_swizzle_mode,
            BN = Self.config.BN,
            BK = Self.k_rope_depth,
        ],
        v_tma_op: KVTMATile[
            Self.KVLUTType.dtype,
            Self.config.swizzle_mode,
            BN = Self.config.BN,
            BK = Self.kv_depth,
        ],
        ragged_tma_store: RaggedTMA3DTile[
            Self.output_type,
            Self.config.swizzle_mode,
            BM = Self.config.BM // 2,
            BN = Self.kv_depth,
        ],
        kv_lut: Self.KVLUTType,
        k_rope_lut: Self.KRopeType,
        scale: Float32,
        batch_size: UInt32,
        pack: Pack[
            Self.MaskType,
            Self.ScoreModType,
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
        score_mod = pack.score_mod
        scheduler = pack.scheduler
        valid_length = pack.valid_length
        max_seq_len = pack.max_seq_len
        partition = pack.partition

        comptime num_qo = Self.config.num_qo()
        # TODO: We may want to support num_qo>2 for depth=64?
        comptime assert (
            num_qo == 1 or num_qo == 2
        ), "Currently only support num_qo == 1 or 2"
        smem_ptr = external_memory[
            Scalar[DType.uint8],
            address_space = AddressSpace.SHARED,
            alignment=128,
            name="mha_dynamic_shared_memory",
        ]()
        comptime SmemStorageType = MLASmemStorage[
            Self.qkv_type, Int(Self.MiscMBarsType.num_mbars()), Self.config
        ]
        ref smem_storage = smem_ptr.bitcast[SmemStorageType]()[]
        var q_smem = smem_storage.q_smem.unsafe_ptr()
        var kv_smem = smem_storage.kv_smem.unsafe_ptr()
        var correction_smem = smem_storage.correction_smem.unsafe_ptr()
        var mbar_base = smem_storage.mbar_base.unsafe_ptr()
        var tma_to_cvt_producer_mbars = (
            smem_storage.tma_to_cvt_producer_mbars.unsafe_ptr()
        )
        var tma_to_cvt_consumer_mbars = (
            smem_storage.tma_to_cvt_consumer_mbars.unsafe_ptr()
        )
        var cvt_to_mma_producer_mbars = (
            smem_storage.cvt_to_mma_producer_mbars.unsafe_ptr()
        )
        var cvt_to_mma_consumer_mbars = (
            smem_storage.cvt_to_mma_consumer_mbars.unsafe_ptr()
        )
        var ptr_tmem_addr = smem_storage.tmem_addr.unsafe_ptr()

        # All barriers are managed by misc_mbars (S/C/order/Q1Sync/KV/O)
        var misc_mbars: Self.MiscMBarsType = {mbar_base}

        var tma_to_cvt_pipeline = TMAtoCvtPipeline[
            Self.config.num_kv_stages,
            num_producer=1,
            num_consumer=64,
        ](tma_to_cvt_producer_mbars, tma_to_cvt_consumer_mbars)

        var cvt_to_mma_pipeline = CvtToMMAPipline[
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

        var warp_idx = UInt32(warp.broadcast(warp_id()))
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
                k_tma_op.prefetch_descriptor()
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
            var seq_info: SeqInfo = get_seq_info[Self.BM, Self.num_q_heads](
                batch_size, max_seq_len, valid_length, partition
            )

            if not seq_info.is_valid():
                return

            var pos: MLAPositionSummary = MLAPositionSummary.create[
                _ndbuffer_mha_operand = Self._ndbuffer_mha_operand,
            ](kv_lut, k_rope_lut, seq_info)

            Self.softmax(
                ptr_tmem_addr[0],
                warp_idx,
                misc_mbars,
                pos.score_row,
                seq_info,
                mask,
                pos.num_keys,
                scale.cast[Self.accum_type](),
                score_mod,
                max_seq_len.as_uint32(),
                ragged_tma_store,
                q_smem.bitcast[Scalar[Self.output_type]](),
                correction_smem,
            )

        elif role == WarpRole.Correction:
            # correction
            warpgroup_reg_dealloc[num_reg_correction]()

            var seq_info: SeqInfo = get_seq_info[Self.BM, Self.num_q_heads](
                batch_size, max_seq_len, valid_length, partition
            )
            if not seq_info.is_valid():
                return
            var pos: MLAPositionSummary = MLAPositionSummary.create[
                _ndbuffer_mha_operand = Self._ndbuffer_mha_operand,
            ](kv_lut, k_rope_lut, seq_info)
            Self.correction(
                ptr_tmem_addr[0],
                misc_mbars,
                pos.score_row,
                pos.num_keys,
                mask,
                correction_smem,
            )
        elif role == WarpRole.Load:
            warpgroup_reg_dealloc[num_reg_other]()
            var seq_info: SeqInfo = get_seq_info[Self.BM, Self.num_q_heads](
                batch_size, max_seq_len, valid_length, partition
            )

            if not seq_info.is_valid():
                return
            var pos: MLAPositionSummary = MLAPositionSummary.create[
                _ndbuffer_mha_operand = Self._ndbuffer_mha_operand,
            ](kv_lut, k_rope_lut, seq_info)

            Self.load(
                misc_mbars,
                tma_to_cvt_pipeline,
                pos.score_row,
                pos.num_keys,
                seq_info,
                max_seq_len,
                mask,
                q_tma_op,
                k_tma_op,
                k_rope_tma_op,
                v_tma_op,
                kv_lut,
                k_rope_lut,
                q_smem,
            )

        elif role == WarpRole.MMA:
            warpgroup_reg_dealloc[num_reg_other]()
            var seq_info: SeqInfo = get_seq_info[Self.BM, Self.num_q_heads](
                batch_size, max_seq_len, valid_length, partition
            )

            if not seq_info.is_valid():
                tcgen05_release_allocation_lock[Self.cta_group]()
                tcgen05_dealloc[Self.cta_group](
                    ptr_tmem_addr[0], Self.config.sm100_tmem_cols
                )
                return
            var pos: MLAPositionSummary = MLAPositionSummary.create[
                _ndbuffer_mha_operand = Self._ndbuffer_mha_operand,
            ](kv_lut, k_rope_lut, seq_info)
            Self.mma(
                ptr_tmem_addr[0],
                misc_mbars,
                cvt_to_mma_pipeline,
                pos.score_row,
                pos.num_keys,
                mask,
                q_smem,
            )
        elif role == WarpRole.CVTToBF16:
            warpgroup_reg_dealloc[num_reg_other]()

            var seq_info: SeqInfo = get_seq_info[Self.BM, Self.num_q_heads](
                batch_size, max_seq_len, valid_length, partition
            )

            if not seq_info.is_valid():
                return

            var pos: MLAPositionSummary = MLAPositionSummary.create[
                _ndbuffer_mha_operand = Self._ndbuffer_mha_operand,
            ](kv_lut, k_rope_lut, seq_info)

            var iter_count: UInt32 = (
                mask.last_masked_set_end[Self.BM, Self.BN, Self.page_size](
                    pos.score_row, pos.num_keys
                )
                - 1
            )

            var local_thread_idx = UInt32(thread_idx.x - 14 * UInt(WARP_SIZE))

            var kv_mem_ptr = q_smem + Self.config.BM * Self.config.padded_depth

            Self.convert_fp8_to_bf16(
                iter_count,
                tma_to_cvt_pipeline,
                cvt_to_mma_pipeline,
                kv_mem_ptr,
                local_thread_idx,
            )

        elif role == WarpRole.Empty:
            warpgroup_reg_dealloc[num_reg_empty]()

    @staticmethod
    @always_inline
    fn correction(
        tmem_addr: UInt32,
        mbars: Self.MiscMBarsType,
        score_row: UInt32,
        num_keys: UInt32,
        mask: Self.MaskType,
        correction_smem_arg: SharedMemPointer[Scalar[Self.accum_type]],
    ):
        comptime assert size_of[Self.accum_type]() == 4

        o0_tmem = tmem_addr + UInt32(Self.config.TMEM_O0)
        o1_tmem = tmem_addr + UInt32(Self.config.TMEM_O1)

        pipeline_c0 = mbars.consumer_c0()
        pipeline_c1 = mbars.consumer_c1()
        pipeline_o = mbars.consumer_o()

        var iter_count: UInt32 = (
            mask.total_iters[Self.BM, Self.BN, Self.page_size](
                score_row, num_keys
            )
            - 1
        )

        comptime batch_size = 16
        # output is BM x depth
        comptime load_iters = Self.kv_depth // (2 * batch_size)
        comptime load_remainder = Self.kv_depth % (2 * batch_size)
        var correction_smem_0 = correction_smem_arg + UInt32(thread_idx.x) % 128
        var correction_smem_1 = correction_smem_0 + (Self.BM // 2)

        while iter_count != 0:
            iter_count -= 1

            comptime for i in range(2):
                var c_scalar: Scalar[Self.accum_type]

                comptime if i == 0:
                    pipeline_c0.wait()
                    c_scalar = correction_smem_0[0]
                    pipeline_c0.release()
                else:
                    pipeline_c1.wait()
                    c_scalar = correction_smem_1[0]
                    pipeline_c1.release()

                change = _vote_nvidia_helper(c_scalar != 1) != 0
                pipeline_o.wait()
                if change:
                    # TODO: experiment with different batch sizes.
                    # The idea here is to both pipeline, and reduce peak register use.
                    comptime assert load_iters > 1
                    comptime assert Self.config.depth % batch_size == 0

                    var o_tmem: UInt32

                    comptime if i == 0:
                        o_tmem = o0_tmem
                    else:
                        o_tmem = o1_tmem

                    var o_b0: SIMD[Self.accum_type, batch_size]
                    var o_b1: SIMD[Self.accum_type, batch_size]
                    o_b0 = tcgen05_ld[
                        datapaths=32,
                        bits=32,
                        repeat=batch_size,
                        dtype = Self.accum_type,
                        pack=False,
                        width=batch_size,
                    ](o_tmem)

                    comptime for b in range(load_iters):
                        tcgen05_load_wait()  # ob0 loaded
                        # BN=64 or BN=80, load_iters=2
                        # b=0
                        # b0_offset0=0
                        # b1_offset =16
                        # b0_offset1=32
                        # b=1
                        # b0_offset0=32
                        # b1_offset =48
                        # b0_offset1=64
                        comptime b0_offset0 = 2 * b * batch_size
                        comptime b1_offset = b0_offset0 + batch_size
                        comptime b0_offset1 = b1_offset + batch_size
                        o_b1 = tcgen05_ld[  # 0b1 start
                            datapaths=32,
                            bits=32,
                            repeat=batch_size,
                            dtype = Self.accum_type,
                            pack=False,
                            width=batch_size,
                        ](o_tmem + UInt32(b1_offset))
                        tcgen05_st[  # 0b0*c_scalar store
                            datapaths=32,
                            bits=32,
                            repeat=batch_size,
                            pack=False,
                        ](o_tmem + UInt32(b0_offset0), o_b0 * c_scalar)
                        tcgen05_load_wait()  # ob1 loaded

                        comptime if b0_offset1 + batch_size <= Self.kv_depth:
                            o_b0 = tcgen05_ld[  # 0b0 start
                                datapaths=32,
                                bits=32,
                                repeat=batch_size,
                                dtype = Self.accum_type,
                                pack=False,
                                width=batch_size,
                            ](o_tmem + UInt32(b0_offset1))
                        tcgen05_st[  # 0b0*c_scalar store
                            datapaths=32,
                            bits=32,
                            repeat=batch_size,
                            pack=False,
                        ](o_tmem + UInt32(b1_offset), o_b1 * c_scalar)

                    comptime if load_remainder > 0:
                        tcgen05_load_wait()  # ob1 loaded
                        comptime offset = 2 * batch_size * load_iters
                        tcgen05_st[  # 0b0*c_scalar store
                            datapaths=32,
                            bits=32,
                            repeat=load_remainder,
                            pack=False,
                        ](o_tmem + UInt32(offset), o_b0 * c_scalar)
                    tcgen05_store_wait()
                    tcgen05_fence_before()
                pipeline_o.release()

    @staticmethod
    @always_inline
    fn softmax(
        tmem_addr: UInt32,
        warp_idx: UInt32,
        mbars: Self.MiscMBarsType,
        score_row: UInt32,
        seq_info: SeqInfo,
        mask: Self.MaskType,
        num_keys: UInt32,
        scale: Float32,
        score_mod: Self.ScoreModType,
        max_seq_len: UInt32,
        ragged_tma_store: RaggedTMA3DTile[
            Self.output_type,
            Self.config.swizzle_mode,
            BM = Self.config.BM // 2,
            BN = Self.kv_depth,
        ],
        o_smem: SharedMemPointer[Scalar[Self.output_type]],
        correction_smem_arg: SharedMemPointer[Scalar[Self.accum_type]],
    ):
        o_prod_mbar = mbars.mbar_base + Self.MiscMBarsType.O_producer_offset
        o_cons_mbar = mbars.mbar_base + Self.MiscMBarsType.O_consumer_offset
        # FIXME: for depth 256
        var s_tmem: UInt32 = tmem_addr + UInt32(Self.config.TMEM_S0)

        var warp_group_idx: UInt32 = warp_idx // 4

        comptime if Self.config.split_m:
            # split-M: second S is (+16 rows) in st-matrix space
            s_tmem += (16 << 16) * warp_group_idx
        else:
            # 2-Q path: S1 is at +BN columns
            s_tmem += UInt32(Self.config.BN) * warp_group_idx

        p_tmem = s_tmem
        s_tile = Self.UMMA0Type.CType(s_tmem)
        p_tile = Self.UMMA1Type.AType(p_tmem)

        pipeline_s = mbars.consumer_s(warp_group_idx)
        pipeline_c = mbars.producer_c(warp_group_idx)
        # TODO: order_s_wait/arrive
        order_s_wait = mbars.pipeline_order_wait(warp_group_idx)
        order_s_arrive = mbars.pipeline_order_arrive(warp_group_idx)
        var order_phase: UInt32 = 0

        var q_head_idx: UInt32 = seq_info.head_idx
        var tid = UInt32(thread_idx.x)
        var row = tid % 128
        var scale_log2e: Scalar[Self.accum_type] = scale
        var correction_smem = correction_smem_arg + tid

        comptime if not (
            Self.use_score_mod or Self.MaskType.apply_log2e_after_mask
        ):
            scale_log2e *= log2e

        @parameter
        @always_inline
        fn mask_row[
            BN: Int, //, mask_strategy: MaskStrategy
        ](s: LocalTensor[Self.accum_type, row_major[BN]()], kv_row: UInt32,):
            apply_mask[
                use_score_mod = Self.use_score_mod, mask_strategy=mask_strategy
            ](
                s,
                mask,
                score_mod,
                scale_log2e,
                prompt_idx=seq_info.prompt_idx,
                q_head_idx=q_head_idx,
                kv_tile_start_row=Int32(kv_row),
                max_seq_len=max_seq_len,
                num_keys=Int32(num_keys),
                score_row=Int32(score_row + tid),
            )

        # while waiting, offset output
        comptime splitBM = Self.BM // 2
        var num_output_rows = min(
            Int32(seq_info.seq_len)
            - Int32(seq_info.prompt_offset)
            - Int32(warp_group_idx) * Int32(splitBM),
            Int32(splitBM),
        )

        gmem_row = Self.PositionType.get_q_gmem_row[ragged=True](
            seq_info, max_seq_len
        )

        pipeline_s.wait()
        tcgen05_fence_after()
        s = tt_stack_allocation[
            dtype = Self.accum_type, address_space = AddressSpace.LOCAL
        ](row_major[Self.config.BN]())

        @parameter
        @always_inline
        fn load_mask_max[
            *, mask_strategy: MaskStrategy
        ](kv_row: UInt32) -> Scalar[Self.accum_type]:
            # break up into sets of 32
            # minimize wait time by using smallest first
            comptime assert Self.config.BN == 64, String(Self.config.BN)
            comptime BM = Self.config.BM // 2
            comptime batch_size = 32
            comptime has_remainder = (Self.config.BN % batch_size) != 0
            comptime first_cols = (
                Self.config.BN % batch_size
            ) if has_remainder else batch_size
            s0 = TMemTile[Self.accum_type, BM, first_cols](s_tmem).load_async()
            tcgen05_load_wait()

            s1 = TMemTile[Self.accum_type, BM, batch_size](
                s_tmem + UInt32(first_cols)
            ).load_async()
            mask_row[mask_strategy=mask_strategy](s0, kv_row)
            s0v = s0.ptr.load[width=first_cols]()
            vrow_max = s0v.reduce_max[size_out = Self.simd_size]()

            s.ptr.store(s0v)
            comptime cols = Self.config.BN - first_cols + batch_size

            comptime for i in range(cols // (2 * batch_size)):
                comptime offset0 = first_cols + batch_size * (2 * i)
                comptime offset1 = first_cols + batch_size * (2 * i + 1)
                comptime offset2 = first_cols + batch_size * (2 * i + 2)

                tcgen05_load_wait()

                comptime if offset1 >= Self.config.BN:
                    mask_row[mask_strategy=mask_strategy](
                        s1, kv_row + UInt32(offset0)
                    )
                    s1v = s1.ptr.load[width=batch_size]()
                    vrow_max = max(
                        s1v.reduce_max[size_out = Self.simd_size](), vrow_max
                    )
                    s.ptr.store(offset0, s1v)
                else:
                    s2 = TMemTile[Self.accum_type, BM, batch_size](
                        s_tmem + UInt32(offset1)
                    ).load_async()
                    mask_row[mask_strategy=mask_strategy](
                        s1, kv_row + UInt32(offset0)
                    )
                    s1v = s1.ptr.load[width=batch_size]()
                    vrow_max = max(
                        s1v.reduce_max[size_out = Self.simd_size](), vrow_max
                    )
                    s.ptr.store(offset0, s1v)
                    tcgen05_load_wait()

                    comptime if offset2 < Self.config.BN:
                        s1 = TMemTile[Self.accum_type, BM, batch_size](
                            s_tmem + UInt32(offset2)
                        ).load_async()
                    mask_row[mask_strategy=mask_strategy](
                        s2, kv_row + UInt32(offset1)
                    )
                    s2v = s2.ptr.load[width=batch_size]()
                    vrow_max = max(
                        s2v.reduce_max[size_out = Self.simd_size](), vrow_max
                    )
                    s.ptr.store(offset1, s2v)

            return vrow_max.reduce_max()

        var kv_row: UInt32 = mask.start_column[
            Self.BM, Self.BN, Self.page_size
        ](score_row)
        comptime mask_sets = Self.MaskType.nonfull_sets[Self.BM, Self.BN]()
        comptime mask_strategies = Self.MaskType.mask_strategies[
            Self.BM, Self.BN
        ]()
        comptime num_sets = len(mask_sets)
        var mask_iters: StaticTuple[UInt32, num_sets] = {}

        comptime if mask_sets[0] != TileMaskStatus.UNKNOWN_MASK:
            mask_ends = mask.masked_set_ends[
                BM = Self.BM, BN = Self.BN, page_size = Self.page_size
            ](score_row, num_keys)
            mask_iters[0] = mask_ends[0]

            comptime for i in range(1, num_sets):
                mask_iters[i] = mask_ends[i] - mask_ends[i - 1]

        comptime assert num_sets >= 1 and num_sets <= 3
        comptime assert (
            num_sets == 1 or mask_sets[0] != TileMaskStatus.UNKNOWN_MASK
        )

        comptime if num_sets == 1:
            row_max = load_mask_max[mask_strategy = mask_strategies[0]](kv_row)
            mask_iters[0] -= 1
        else:
            # find out which strategy to apply
            if mask_iters[0] > 0:
                row_max = load_mask_max[mask_strategy = mask_strategies[0]](
                    kv_row
                )
                mask_iters[0] -= 1
            else:
                comptime if num_sets == 2:
                    row_max = load_mask_max[mask_strategy = mask_strategies[1]](
                        kv_row
                    )
                    mask_iters[1] -= 1
                else:
                    if mask_iters[1] > 1:
                        row_max = load_mask_max[
                            mask_strategy = mask_strategies[1]
                        ](kv_row)
                        mask_iters[1] -= 1
                    else:
                        row_max = load_mask_max[
                            mask_strategy = mask_strategies[2]
                        ](kv_row)
                        mask_iters[2] -= 1

        @parameter
        @always_inline
        fn store_exp(
            row_max: Scalar[Self.accum_type],
        ) -> SIMD[Self.accum_type, 2]:
            comptime exp_simd = 2
            comptime vs_len = Self.config.BN // exp_simd  # 128 // 2 = 64
            comptime batch_size = 32
            comptime num_batch_iters = vs_len // batch_size
            comptime remainder = vs_len % batch_size
            comptime assert num_batch_iters > 0
            comptime BatchTileType = TMemTile[
                Self.qkv_type, Self.config.BM // 2, batch_size * exp_simd
            ]
            comptime RemainderTileType = TMemTile[
                Self.qkv_type, Self.config.BM // 2, remainder * exp_simd
            ]
            comptime assert (Self.config.BN % exp_simd) == 0

            vs = s.vectorize[exp_simd]()
            # We batch stores, e.g. use `tcgen_05.st.x32`.
            # If we have BN = 128, we would perform two such stores
            # (storing 64 elements as 32x bf16x2)
            #
            # Let `x` be the number of elements we add prior to storing.
            # If `x < 64`, with BN = 128, we have these live counts at
            # the two `tcgen_05.st.x32`:
            # 0. (BN - x) + 32
            # 1. (BN - x) + 32
            #
            # Thus, we can sum the first 32 elements, leaving the remaining 96
            # in registers until after we write.
            # The optimal solution for the number to do in advance is also
            # independent of the number of batches.
            comptime AccType = SIMD[Self.accum_type, exp_simd]
            var acc: AccType = exp2(rebind[AccType](vs[0]) - row_max)
            vs[0] = rebind[vs.ElementType](acc)

            comptime for i in range(1, batch_size // 2):
                vsi = exp2(rebind[AccType](vs[i]) - row_max)
                vs[i] = rebind[vs.ElementType](vsi)
                acc += vsi

            # at this point, we need 32 fewer fp32 registers but 16 more u32
            comptime for i in range(batch_size // 2, batch_size):
                vs[i] = exp2(vs[i] - row_max)

            BatchTileType(p_tmem).store(
                LocalTensor[
                    Self.accum_type, row_major[batch_size * exp_simd]()
                ](s.ptr, row_major[batch_size * exp_simd]())
            )

            comptime for b in range(1, num_batch_iters):
                comptime offset = batch_size * b

                comptime for i in range(offset, offset + batch_size):
                    vs[i] = exp2(vs[i] - row_max)

                comptime el_offset = offset * exp_simd
                comptime tmem_offset = (
                    el_offset * size_of[Self.qkv_type]()
                ) // size_of[Self.accum_type]()
                BatchTileType(p_tmem + UInt32(tmem_offset)).store(
                    LocalTensor[
                        Self.accum_type, row_major[batch_size * exp_simd]()
                    ](s.ptr + el_offset, row_major[batch_size * exp_simd]())
                )

            comptime if remainder > 0:
                comptime offset = batch_size * num_batch_iters

                comptime for i in range(offset, offset + remainder):
                    vs[i] = exp2(vs[i] - row_max)

                comptime el_offset = offset * exp_simd
                comptime tmem_offset = (
                    el_offset * size_of[Self.qkv_type]()
                ) // size_of[Self.accum_type]()
                RemainderTileType(p_tmem + UInt32(tmem_offset)).store(
                    LocalTensor[
                        Self.accum_type, row_major[remainder * exp_simd]()
                    ](s.ptr + el_offset, row_major[remainder * exp_simd]())
                )

            tcgen05_store_wait()
            tcgen05_fence_before()
            pipeline_s.release()
            # now we can sum the remaining elements of `acc`
            acc0 = vs[batch_size // 2]
            acc1 = vs[batch_size // 2 + 1]
            acc2 = vs[batch_size // 2 + 2] + vs[batch_size // 2 + 3]

            comptime for i in range(batch_size // 2 + 4, vs_len, 4):
                acc += rebind[AccType](vs[i])
                acc0 += vs[i + 1]
                acc1 += vs[i + 2]
                acc2 += vs[i + 3]
            return (acc + rebind[AccType](acc0)) + rebind[AccType](acc1 + acc2)

        var row_sum: SIMD[Self.accum_type, 2] = store_exp(row_max)

        var o_phase: UInt32 = 0  # initial wait is phase 0

        comptime rescale_threshold: Float32 = Float32(-8) if size_of[
            Self.qkv_type
        ]() >= 2 else Float32(0)

        # TODO: add ordering barriers to prevent overlap
        # between the two softmax warpgroups
        comptime if mask_sets[0] != TileMaskStatus.UNKNOWN_MASK:
            comptime for i in range(num_sets):
                comptime mask_status = mask_sets[i]
                comptime mask_strategy = mask_strategies[i]
                var iters: UInt32

                iters = mask_iters[i]
                while iters != 0:
                    iters -= 1
                    kv_row += UInt32(Self.config.BN)
                    pipeline_s.wait()
                    # calculate rowmax
                    old_max = row_max
                    var new_row_max: Scalar[Self.accum_type]

                    # last_iter == (i + 1 == num_sets) and (i == 0)
                    # `i == 0` is runtime; for now, we set to `True`
                    # as this number of iterations is small
                    comptime last_iter: Bool = i + 1 == num_sets
                    comptime masked: Bool = mask_status == TileMaskStatus.PARTIAL_MASK
                    new_row_max = load_mask_max[mask_strategy=mask_strategy](
                        kv_row
                    )
                    new_row_max = max(old_max, new_row_max)
                    diff = sub_ftz(old_max, new_row_max)
                    var correction: Float32

                    comptime if rescale_threshold < 0:
                        # old_max - new_row_max < -8
                        # 8 < new_row_max - old_max
                        if _vote_nvidia_helper(diff < rescale_threshold) != 0:
                            row_max = new_row_max
                            correction = exp2(diff)
                        else:
                            correction = 1
                    else:
                        row_max = new_row_max
                        correction = exp2(diff)
                    pipeline_c.acquire()
                    correction_smem[] = correction
                    pipeline_c.commit()
                    # update s->p
                    local_rowsum = store_exp(row_max)
                    row_sum = row_sum.fma(correction, local_rowsum)
                    o_phase ^= 1
        else:
            while True:
                kv_row += UInt32(Self.config.BN)
                if kv_row >= num_keys:
                    break
                mask_status = Self.mask_status(mask, score_row, kv_row)
                if mask_status == TileMaskStatus.FULL_MASK:
                    continue
                pipeline_s.wait()
                # calculate rowmax
                old_max = row_max
                var new_row_max: Scalar[Self.accum_type]
                if mask_status == TileMaskStatus.PARTIAL_MASK:
                    new_row_max = load_mask_max[
                        mask_strategy = MaskStrategy.COMPUTED
                        | MaskStrategy.OUT_OF_BOUNDS
                    ](kv_row)
                else:
                    new_row_max = load_mask_max[
                        mask_strategy = MaskStrategy.OUT_OF_BOUNDS
                    ](kv_row)
                new_row_max = max(old_max, new_row_max)
                diff = sub_ftz(old_max, new_row_max)
                var correction: Float32

                comptime if rescale_threshold < 0:
                    # old_max - new_row_max < -8
                    # 8 < new_row_max - old_max
                    if _vote_nvidia_helper(diff < rescale_threshold) != 0:
                        row_max = new_row_max
                        correction = exp2(diff)
                    else:
                        correction = 1
                else:
                    row_max = new_row_max
                    correction = exp2(diff)
                pipeline_c.acquire()
                correction_smem[] = correction
                pipeline_c.commit()
                # update s->p
                local_rowsum = store_exp(row_max)
                row_sum = row_sum.fma(correction, local_rowsum)
                o_phase ^= 1
        # Do the final correction and write
        inv_row_sum = recip(row_sum.reduce_add())
        o_tile = Self.UMMA1Type.CType(
            tmem_addr
            + UInt32(Self.config.TMEM_O0)
            + warp_group_idx * UInt32(Self.padded_depth)
        )
        # wait on the o_pipeline producer
        comptime assert size_of[Self.output_type]() == size_of[Self.qkv_type]()
        if num_output_rows > 0:
            o_prod_mbar[warp_group_idx].wait(o_phase)  # consumer wait
            tcgen05_fence_after()  # example 1
            # TODO: pass in a dedicated barrier that a q-writer can wait on in a persistent kernel?
            comptime HalfBM = Self.BM // 2

            Self.scale_write_output(
                row,
                warp_idx & 3,
                warp_group_idx,
                inv_row_sum,
                o_smem + warp_group_idx * UInt32(HalfBM * Self.kv_depth),
                o_tile,
                ragged_tma_store,
                o_cons_mbar + warp_group_idx,  # consumer arrive
                num_output_rows,
                q_head_idx,
                gmem_row + warp_group_idx * UInt32(HalfBM),
            )
        named_barrier[Int32(2 * WARPGROUP_SIZE)](2)
        if warp_idx == 0:
            tcgen05_release_allocation_lock[Self.cta_group]()
            tcgen05_dealloc[Self.cta_group](
                tmem_addr, Self.config.sm100_tmem_cols
            )

    @always_inline
    @staticmethod
    fn scale_write_output(
        local_row: UInt32,
        local_warp_idx: UInt32,
        warp_group_idx: UInt32,
        inv_row_sum: Scalar[Self.accum_type],
        o_smem_arg: SharedMemPointer[Scalar[Self.output_type]],
        o_tmem: TMemTile[Self.accum_type, Self.BM // 2, Self.kv_depth],
        ragged_tma_store: RaggedTMA3DTile[
            Self.output_type,
            Self.config.swizzle_mode,
            BM = Self.config.BM // 2,
            BN = Self.kv_depth,
        ],
        consumer_mbar: MBarType,
        num_output_rows: Int32,
        out_head_idx: UInt32,
        out_row_idx: UInt32,
    ):
        e = elect()
        if e != 0:
            ragged_tma_store.prefetch_descriptor()

        o = o_tmem.load_async_with_st_matrix_layout[
            num_threads=WARPGROUP_SIZE
        ]()

        comptime num_rows = o.layout[0].size()
        inv_row_sums = tt_stack_allocation[
            dtype = Self.accum_type, address_space = AddressSpace.LOCAL
        ](row_major[num_rows]())
        lane = local_row % 32
        lane_row = lane // 4

        #  0  1  2  3
        #  4  5  6  7
        #  8  9 10 11
        # 12 13 14 15
        # 16 17 18 19
        # 20 21 22 23
        # 24 25 26 27
        # 28 29 30 31
        # lane 0 needs to get
        comptime for i in range(num_rows):
            # lane // 4, lane // 4 + 8, lane // 4 + 16, lane // 4 + 24
            inv_row_sums[i] = warp.shuffle_idx(
                inv_row_sum, lane_row + UInt32(8 * i)
            )

        tcgen05_load_wait()
        tcgen05_fence_before()
        _ = consumer_mbar[].arrive()

        comptime for i in range(num_rows):
            irs = o.element_type(
                rebind[Scalar[Self.accum_type]](inv_row_sums[i])
            )

            comptime for j in range(o.layout[1].size()):
                o[i, j] *= irs

        comptime swizzle = make_swizzle[
            Self.output_type, Self.config.swizzle_mode
        ]()

        comptime ST = STMatrixLayout[
            Self.BM // 2, Self.kv_depth, num_threads=WARPGROUP_SIZE
        ]

        comptime swizzle_granularity = Self.config.swizzle_mode.bytes() // size_of[
            Self.output_type
        ]()
        comptime iters = Self.kv_depth // swizzle_granularity

        comptime swizzle_block_size: UInt32 = UInt32(
            WARP_SIZE * swizzle_granularity
        )
        o_smem = o_smem_arg + local_warp_idx * swizzle_block_size

        comptime for i in range(2):
            comptime datapath_offset: UInt32 = UInt32(
                16 * i * swizzle_granularity
            )

            comptime for j in range(iters):
                comptime ofs = i * ST.frag_size + j * (ST.frag_size // iters)
                comptime reg_layout = row_major[1, ST.frag_size // iters]()
                var rows_of_o_frags = _LocalTT[Self.accum_type, reg_layout](
                    o.ptr + ofs, reg_layout
                )  # all the repeats across n and m

                comptime warp_smem_offset: UInt32 = datapath_offset + UInt32(
                    j * (Self.BM // 2) * swizzle_granularity
                )
                comptime smem_layout = row_major[16, swizzle_granularity]()
                var accum_smem_warp_tile = _SharedMemTT[
                    Self.output_type, smem_layout
                ](o_smem + warp_smem_offset, smem_layout)

                output_reg_to_smem_st_matrix[
                    BM=16,
                    swizzle=swizzle,
                    num_consumer=1,
                ](
                    lane,
                    local_warp_group_idx=0,
                    output_reg_tile=rows_of_o_frags,
                    accum_smem_tile=accum_smem_warp_tile,
                )
        named_barrier[Int32(WARPGROUP_SIZE)](Int32(warp_group_idx))

        # # first thread of each warp_group
        if local_warp_idx == 0:
            if e != 0:
                fence_async_view_proxy()

            if e != 0:
                ragged_tma_store.async_copy_from(
                    o_smem,
                    ragged_idx=out_row_idx,
                    dynamic_dim=UInt32(num_output_rows),
                    middle_idx=out_head_idx,
                )
            if e != 0:
                cp_async_bulk_commit_group()
        cp_async_bulk_wait_group[0]()

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
    fn load(
        mbars: Self.MiscMBarsType,
        mut tma_to_cvt_pipeline: TMAtoCvtPipeline,
        score_row: UInt32,
        num_keys: UInt32,
        seq_info: SeqInfo,
        max_seq_len: Self.MaxSeqLenType,
        mask: Self.MaskType,
        q_tma_op: QTMATile[
            Self.KVLUTType.dtype,
            Self.config.swizzle_mode,
            BM = Self.config.BM // 2,
            depth = Self.config.BK0,  # padded depth -> 192
            group = Self.config.group,
            decoding=False,
        ],
        k_tma_op: KVTMATile[
            Self.KVLUTType.dtype,
            Self.config.swizzle_mode,
            BN = Self.config.BN,
            BK = Self.kv_depth,
        ],
        k_rope_tma_op: KVTMATile[
            Self.KRopeType.dtype,
            Self.k_rope_swizzle_mode,
            BN = Self.config.BN,
            BK = Self.k_rope_depth,
        ],
        v_tma_op: KVTMATile[
            Self.KVLUTType.dtype,
            Self.config.swizzle_mode,
            BN = Self.config.BN,
            BK = Self.kv_depth,
        ],
        kv_lut: Self.KVLUTType,
        k_rope_lut: Self.KRopeType,
        q_smem: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
    ):
        comptime KVPipeType = MLAKVProducerPipeline[
            Self.KVLUTType.dtype, Self.KRopeType.dtype, Self.config
        ]

        # If two-qo, we produce qkv in a pattern of
        # q0 & k0, q1, v0, k1, v1, k2, v2...
        comptime SMemTensorLT[layout: Layout] = SharedMemLT[
            Self.KVLUTType.dtype, layout
        ]
        comptime QType = SMemTensorLT[type_of(q_tma_op).layout]
        comptime KType = SMemTensorLT[type_of(k_tma_op).layout]
        comptime KRopeSMType = SMemTensorLT[type_of(k_rope_tma_op).layout]
        comptime VType = SMemTensorLT[type_of(v_tma_op).layout]

        var k_rope_head_idx: UInt32 = seq_info.head_idx // UInt32(Self.group)
        var kv_head_idx: UInt32 = seq_info.head_idx

        comptime q_elements = (Self.config.BM // 2) * Self.config.BK0
        comptime q_bytes = size_of[Self.qkv_type]() * q_elements

        kv_smem = q_smem + Self.config.BM * Self.config.padded_depth
        # Construct KV pipeline from unified barrier storage
        var kv_pipeline_arg = Self.KVPipelineType(mbars.get_kv_mbars())
        var pipeline_kv: KVPipeType = {kv_pipeline_arg, kv_smem}

        var mbark0: KVPipeType.KPairType

        mbark0 = pipeline_kv.get_k[qk_stage=0, expect=False]()  # no wait
        var q_gmem_row: UInt32 = Self.PositionType.get_q_gmem_row[ragged=True](
            seq_info, max_seq_len
        )
        var q_head_idx: UInt32 = seq_info.head_idx
        elect = elect() != 0

        # copy q0
        if elect:
            # Q0
            mbark0.mbar[].expect_bytes(
                Int32(pipeline_kv.k_nope_bytes + q_bytes)
            )
            q_tma_op.async_copy(
                QType(q_smem),
                mbark0.mbar[],
                q_coord[
                    depth = Self.depth,
                    swizzle_granularity = Self.swizzle_granularity,
                    decoding=False,
                ](q_gmem_row, q_head_idx),
            )
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
        # copy k0
        k_smem, k_rope_smem = split_smem[
            KVPipeType.k_nope_tma_layout,
            KVPipeType.k_rope_tma_layout,
            Self.KVLUTType.dtype,
            Self.KRopeType.dtype,
        ](mbark0.smem)
        if elect:
            # K0
            k_tma_op.async_copy(
                k_smem,
                mbark0.mbar[],
                kv_coord[
                    depth = Self.kv_depth,
                    swizzle_granularity = Self.swizzle_granularity,
                ](kv_gmem_row, kv_head_idx),
            )
            # K0 rope
            var k_rope_coord = kv_coord[
                depth = Self.k_rope_depth,
                swizzle_granularity = Self.swizzle_granularity,
            ](k_rope_gmem_row, k_rope_head_idx)
            k_rope_coord[0] = (
                Self.cache_depth - Self.k_rope_depth
            )  # only load last 64 head_dims

            tma_to_cvt_pipeline.producer_mbar()[].expect_bytes(
                Int32(pipeline_kv.k_rope_bytes)
            )

            k_rope_tma_op.async_copy(
                k_rope_smem,
                tma_to_cvt_pipeline.producer_mbar()[],
                k_rope_coord,
            )

        tma_to_cvt_pipeline.step()
        pipeline_kv.commit_kv_step()

        if elect:
            var q1_mbar = mbars.q1_wait_mbar()
            q1_mbar[0].expect_bytes(Int32(q_bytes))
            # Q1
            q_tma_op.async_copy(
                QType(q_smem + q_elements),
                q1_mbar[0],
                q_coord[
                    depth = Self.depth,
                    swizzle_granularity = Self.swizzle_granularity,
                    decoding=False,
                ](q_gmem_row + UInt32(Self.config.BM // 2), q_head_idx),
            )
        # copy v0
        if elect:
            mbarv0 = pipeline_kv.get_v[qk_stage=0]()
            v_tma_op.async_copy(
                mbarv0.smem,
                mbarv0.mbar[],
                kv_coord[
                    depth = Self.kv_depth,
                    swizzle_granularity = Self.swizzle_granularity,
                ](kv_gmem_row, kv_head_idx),
            )
        pipeline_kv.commit_kv_step()
        comptime check_mask = mask.nonfull_sets[Self.BM, Self.BN]()[
            0
        ] == TileMaskStatus.UNKNOWN_MASK

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
            k_rope_gmem_row = k_rope_lut.row_idx(seq_info.prompt_idx, kv_row)
            # produce k
            pipeline_kv.acquire_kv()

            if elect:
                mbarkn = pipeline_kv.get_k[qk_stage=0, expect=False]()
                k_smem_n, k_rope_smem_n = split_smem[
                    KVPipeType.k_nope_tma_layout,
                    KVPipeType.k_rope_tma_layout,
                    Self.KVLUTType.dtype,
                    Self.KRopeType.dtype,
                ](mbarkn.smem)

                mbarkn.mbar[].expect_bytes(Int32(pipeline_kv.k_nope_bytes))
                k_tma_op.async_copy(
                    k_smem_n,
                    mbarkn.mbar[],
                    kv_coord[
                        depth = Self.kv_depth,
                        swizzle_granularity = Self.swizzle_granularity,
                    ](kv_gmem_row, kv_head_idx),
                )
                # K rope
                tma_to_cvt_pipeline.producer_mbar()[].expect_bytes(
                    Int32(pipeline_kv.k_rope_bytes)
                )

                var k_rope_coord = kv_coord[
                    depth = Self.k_rope_depth,
                    swizzle_granularity = Self.swizzle_granularity,
                ](k_rope_gmem_row, k_rope_head_idx)
                k_rope_coord[0] = (
                    Self.cache_depth - Self.k_rope_depth
                )  # only load last 64 head_dims
                k_rope_tma_op.async_copy(
                    k_rope_smem_n,
                    tma_to_cvt_pipeline.producer_mbar()[],
                    k_rope_coord,
                )

            pipeline_kv.commit_kv_step()
            tma_to_cvt_pipeline.step()
            pipeline_kv.acquire_kv()

            if elect:
                mbarvn = pipeline_kv.get_v[qk_stage=0]()
                v_tma_op.async_copy(
                    mbarvn.smem,
                    mbarvn.mbar[],
                    kv_coord[
                        depth = Self.kv_depth,
                        swizzle_granularity = Self.swizzle_granularity,
                    ](kv_gmem_row, kv_head_idx),
                )
            pipeline_kv.commit_kv_step()

    @staticmethod
    @always_inline
    fn convert_fp8_to_bf16(
        mut iter_count: UInt32,
        mut tma_to_cvt_pipeline: TMAtoCvtPipeline,
        mut cvt_to_mma_pipeline: CvtToMMAPipline,
        kv_mem_ptr: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
        local_thread_idx: UInt32,
    ):
        var k_rope_smem_ptr = kv_mem_ptr + Self.config.BN * Self.kv_depth
        var local_warp_idx = local_thread_idx // UInt32(WARP_SIZE)
        var local_lane_idx = local_thread_idx % UInt32(WARP_SIZE)

        comptime swizzle_fp8 = make_swizzle[
            Self.KRopeType.dtype, TensorMapSwizzle.SWIZZLE_64B
        ]()
        comptime swizzle_bf16 = make_swizzle[
            Self.KVLUTType.dtype, TensorMapSwizzle.SWIZZLE_128B
        ]()

        var k_rope_tensor_fp8 = LayoutTensor[
            Self.KRopeType.dtype,
            Layout.row_major(Self.config.BN, Self.k_rope_depth),
            MutAnyOrigin,
            address_space = AddressSpace.SHARED,
        ](k_rope_smem_ptr.bitcast[Scalar[Self.KRopeType.dtype]]())

        var k_rope_tensor_bf16 = LayoutTensor[
            Self.KVLUTType.dtype,
            Layout.row_major(Self.config.BN, Self.k_rope_depth),
            MutAnyOrigin,
            address_space = AddressSpace.SHARED,
        ](k_rope_smem_ptr.bitcast[Scalar[Self.KVLUTType.dtype]]())

        # each warp do 32x64 tile
        k_rope_tile_fp8 = k_rope_tensor_fp8.tile[
            Self.config.BN // 2, Self.k_rope_depth
        ](Int(local_warp_idx), 0)
        k_rope_tile_bf16 = k_rope_tensor_bf16.tile[
            Self.config.BN // 2, Self.k_rope_depth
        ](Int(local_warp_idx), 0)

        tma_to_cvt_pipeline.consumer_wait()

        cvt_block_fp8_to_bf16[
            swizzle_fp8=swizzle_fp8,
            swizzle_bf16=swizzle_bf16,
        ](k_rope_tile_fp8, k_rope_tile_bf16, local_lane_idx)

        tma_to_cvt_pipeline.step()
        cvt_to_mma_pipeline.producer_commit()

        while iter_count != 0:
            iter_count -= 1
            tma_to_cvt_pipeline.consumer_wait()

            cvt_block_fp8_to_bf16[
                swizzle_fp8=swizzle_fp8,
                swizzle_bf16=swizzle_bf16,
            ](k_rope_tile_fp8, k_rope_tile_bf16, local_lane_idx)

            fence_async_view_proxy()

            tma_to_cvt_pipeline.step()
            cvt_to_mma_pipeline.producer_commit()

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

    @staticmethod
    @always_inline
    fn mma(
        tmem_addr: UInt32,
        mbars: Self.MiscMBarsType,
        mut cvt_to_mma_pipeline: CvtToMMAPipline,
        score_row: UInt32,
        num_keys: UInt32,
        mask: Self.MaskType,
        q_smem: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
    ):
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

        # O pipelines: o0 and o1 alternate with different initial phases
        var pipeline_o0 = mbars.producer_o0()
        var pipeline_o1 = mbars.producer_o1()
        # Keep consumer pointers for acquire operations (shared phase tracking)
        consumer_o0 = pipeline_o0.consumer_mbar_base
        consumer_o1 = pipeline_o1.consumer_mbar_base

        comptime q0_size = (Self.config.BM // 2) * Self.config.padded_depth
        comptime q0_bytes = UInt32(q0_size * size_of[Self.KVLUTType.dtype]())
        q0 = Self.descriptor_q(q_smem)
        q1 = q0 + q0_bytes
        kv_smem = q_smem + 2 * q0_size

        # MLA uses a shared KVPipeline where K and V alternate states.
        # Create K and V smem descriptors with MLA-specific dimensions.
        comptime full_kv_bytes: UInt32 = UInt32(
            Self.config.BN * Self.config.padded_depth * Self.qkv_dt_size
        )
        var k_smem_descriptor = smem_descriptor[
            BMN = Self.config.BN,
            BK = Self.config.BK0,
            swizzle_mode = Self.config.swizzle_mode,
            is_k_major=True,
        ](kv_smem)
        var v_smem_descriptor = smem_descriptor[
            BMN = Self.kv_depth,
            BK = Self.config.BK1,
            swizzle_mode = Self.config.swizzle_mode,
            is_k_major=False,
        ](kv_smem)

        # Construct KV pipeline from unified barrier storage (consumer starts phase=0)
        var pipeline = Self.KVPipelineType(mbars.get_kv_mbars())

        # We peel the first iteration, as we want to wait on q1
        var iter_count: UInt32 = (
            mask.total_iters[Self.BM, Self.BN, Self.page_size](
                score_row, num_keys
            )
            - 1
        )

        # Q_0 @ K_0'
        # wait for CVT for fp8 case, else wait for producer
        pipeline.consumer_wait[0]()  # wait K0
        cvt_to_mma_pipeline.consumer_wait()

        # wait for cvt producer barrier
        k0 = k_smem_descriptor + full_kv_bytes * pipeline.state.index()
        e = elect()

        Self.UMMA0Type.mma(q0, k0, s0_tmem, elect=e, c_scale=0)
        pipeline_s0.commit_mma(e)

        # Q_1 @ K_0'
        mbars.q1_wait_mbar()[0].wait()  # wait on Q1
        Self.UMMA0Type.mma(q1, k0, s1_tmem, elect=e, c_scale=0)
        pipeline_s1.commit_mma(e)

        pipeline.consumer_release[0](e)  # release K0, step to V state
        cvt_to_mma_pipeline.step()

        # Wait V0
        pipeline.consumer_wait[0]()  # wait V0
        vlatest = v_smem_descriptor + full_kv_bytes * pipeline.state.index()
        _ = consumer_s0[].wait(0)
        Self.UMMA1Type.mma(s0_tmem, vlatest, o0_tmem, elect=e, c_scale=0)
        pipeline_o0.commit_mma(e)
        var phase_s: UInt32 = 0
        var phase_o: UInt32 = 1

        var c_scale: UInt32 = 0

        while iter_count != 0:
            iter_count -= 1
            # Save V release index and step to next K position
            var v_release_index = pipeline.state.index()
            pipeline.state.step()

            # Q_0 @ K_n'
            pipeline.consumer_wait[0]()  # wait Kn
            cvt_to_mma_pipeline.consumer_wait()

            kn = k_smem_descriptor + full_kv_bytes * pipeline.state.index()
            Self.UMMA0Type.mma(q0, kn, s0_tmem, elect=e, c_scale=0)

            pipeline_s0.commit_mma(e)

            # O_1 + P_1 @ V_{n-1}
            _ = consumer_o1[].wait(phase_o)
            _ = consumer_s1[].wait(phase_s)
            Self.UMMA1Type.mma(
                s1_tmem, vlatest, o1_tmem, elect=e, c_scale=c_scale
            )
            pipeline_o1.commit_mma(e)
            phase_o = phase_s
            c_scale = 1
            # Release old V (deferred release, no step)
            elect_mma_arrive(pipeline.consumer_mbar[0](v_release_index), e)

            # Q_1 @ K_n'
            Self.UMMA0Type.mma(q1, kn, s1_tmem, elect=e, c_scale=0)
            pipeline_s1.commit_mma(e)
            phase_s ^= 1

            pipeline.consumer_release[0](e)  # release K, step to V state
            cvt_to_mma_pipeline.step()

            # O_0 + P_0 @ V_n
            pipeline.consumer_wait[0]()  # wait Vn
            vlatest = v_smem_descriptor + full_kv_bytes * pipeline.state.index()
            _ = consumer_o0[].wait(phase_o)
            _ = consumer_s0[].wait(phase_s)
            Self.UMMA1Type.mma(s0_tmem, vlatest, o0_tmem, elect=e, c_scale=1)
            pipeline_o0.commit_mma(e)

        _ = consumer_o1[].wait(phase_o)
        _ = consumer_s1[].wait(phase_s)
        Self.UMMA1Type.mma(s1_tmem, vlatest, o1_tmem, elect=e, c_scale=c_scale)
        pipeline_o1.commit_mma(e)


@always_inline
fn mla_sm100_prefill_fp8[
    output_type: DType,
    q_type: DType,
    KVType: MHAOperand,
    KRopeType: MHAOperand,
    MaskType: MHAMask,
    ScoreModType: ScoreModTrait,
    MaxPromptLenType: OptionallyStaticInt,
    //,
    config: MHAConfig,
    group: Int,
    q_depth: Int,
    cache_depth: Int,
    use_score_mod: Bool,
    _ndbuffer_mha_operand: Bool,
](
    output: LayoutTensor[
        output_type, address_space = AddressSpace.GENERIC, ...
    ],
    q: LayoutTensor[q_type, _, address_space = AddressSpace.GENERIC, ...],
    k: KVType,
    v: KVType,
    k_rope: KRopeType,
    mask_functor: MaskType,
    score_mod_functor: ScoreModType,
    valid_length: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    max_prompt_len: MaxPromptLenType,
    scale: Float32,
    batch_size: Int,
    ctx: DeviceContext,
) raises:
    comptime fa4_config = FA4Config(
        num_q_heads=Int(config.num_heads),
        group=group,
        depth=q_depth,
        dtype_size=size_of[q_type](),
        swizzle_mode=config.swizzle_mode,
        page_size=KVType.page_size,
        is_mla=True,
    )

    comptime k_rope_depth = 64
    comptime kv_depth = q_depth - k_rope_depth

    var num_rows_q = q_num_matrix_view_rows(q)

    comptime RaggedStoreType = RaggedTMA3DTile[
        output_type,
        fa4_config.swizzle_mode,
        BM = fa4_config.BM // 2,
        BN = fa4_config.depth - 64,
    ]

    var ragged_tma_store = RaggedStoreType.create(
        ctx, output.ptr, rows=num_rows_q, middle_dim=fa4_config.num_q_heads
    )

    q_tma_op = q_tma[
        fa4_config.swizzle_mode,
        BM = fa4_config.BM // 2,
        depth = fa4_config.depth,
        q_num_heads = fa4_config.num_q_heads,
        group = fa4_config.group,
        decoding=False,
    ](
        ctx,
        q.ptr.as_unsafe_pointer(),
        num_rows_q,
    )

    # [batch_size * num_keys, num_heads, kv_depth]
    k_tma_op = k.create_tma_tile[
        fa4_config.swizzle_mode,
        BN = fa4_config.BN,
        depth = fa4_config.depth - 64,
    ](ctx)

    comptime k_swizzle_mode = (
        TensorMapSwizzle.SWIZZLE_64B if KRopeType.dtype
        == DType.float8_e4m3fn else TensorMapSwizzle.SWIZZLE_128B
    )

    # [batch_size, num_keys, cache_num_heads, cache_depth]
    k_rope_tma_op = k_rope.create_tma_tile[
        k_swizzle_mode,
        BN = fa4_config.BN,
        depth=cache_depth,
        BK=k_rope_depth,
    ](ctx)

    # [batch_size * num_keys, num_heads, kv_depth]
    v_tma_op = v.create_tma_tile[
        fa4_config.swizzle_mode,
        BN = fa4_config.BN,
        depth = fa4_config.depth - 64,
    ](ctx)

    _mla_prefill_sm100_valid_length_dispatch[
        fa4_config=fa4_config,
        cache_depth=cache_depth,
        use_score_mod=use_score_mod,
        _ndbuffer_mha_operand=_ndbuffer_mha_operand,
    ](
        ragged_tma_store,
        q_tma_op,
        k_tma_op,
        v_tma_op,
        k_rope_tma_op,
        k,
        k_rope,
        mask_functor,
        score_mod_functor,
        valid_length,
        max_prompt_len,
        scale,
        batch_size,
        ctx,
    )


@always_inline
fn _mla_prefill_sm100_valid_length_dispatch[
    KVType: MHAOperand,
    output_type: DType,
    q_type: DType,
    k_swizzle_mode: TensorMapSwizzle,
    MaskType: MHAMask,
    ScoreModType: ScoreModTrait,
    KRopeType: MHAOperand,
    MaxPromptLenType: OptionallyStaticInt,
    //,
    fa4_config: FA4Config,
    cache_depth: Int,
    use_score_mod: Bool,
    _ndbuffer_mha_operand: Bool,
](
    ragged_tma_store: RaggedTMA3DTile[
        output_type,
        fa4_config.swizzle_mode,
        BM = fa4_config.BM // 2,
        BN = fa4_config.depth - 64,
    ],
    q_tma_op: QTMATile[
        q_type,
        fa4_config.swizzle_mode,
        BM = fa4_config.BM // 2,
        depth = fa4_config.depth,
        group = fa4_config.group,
        decoding=False,
    ],
    k_tma_op: KVTMATile[
        KVType.dtype,
        fa4_config.swizzle_mode,
        BN = fa4_config.BN,
        BK = padded_depth[
            KVType.dtype, fa4_config.swizzle_mode, fa4_config.depth - 64
        ](),
    ],
    v_tma_op: KVTMATile[
        KVType.dtype,
        fa4_config.swizzle_mode,
        BN = fa4_config.BN,
        BK = padded_depth[
            KVType.dtype, fa4_config.swizzle_mode, fa4_config.depth - 64
        ](),
    ],
    k_rope_tma_op: KVTMATile[
        KRopeType.dtype,
        k_swizzle_mode,
        BN = fa4_config.BN,
        BK=64,
    ],
    kv_lut: KVType,
    k_rope_lut: KRopeType,
    mask_functor: MaskType,
    score_mod_functor: ScoreModType,
    valid_length: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    max_prompt_len: MaxPromptLenType,
    scale: Float32,
    batch_size: Int,
    ctx: DeviceContext,
) raises:
    comptime SchedulerType = TransientScheduler[
        UInt32(fa4_config.BM), UInt32(fa4_config.num_q_heads)
    ]
    comptime ValidLengthType = NonNullPointer[DType.uint32]
    comptime SinkType = NullPointer[output_type]
    comptime KVRowOffsetsType = NullPointer[DType.uint32]
    comptime PartitionType = NoPartition[DType.float32]
    var valid_len: ValidLengthType = {
        rebind[UnsafePointer[UInt32, ImmutAnyOrigin]](valid_length.ptr)
    }

    comptime SM100MLAType = SM100MLA[
        KVType,
        KRopeType,
        output_type,
        MaskType,
        ScoreModType,
        SchedulerType,
        fa4_config,
        use_score_mod,
        ValidLengthType,
        SinkType,
        KVRowOffsetsType,
        MaxPromptLenType,
        PartitionType,
        _ndbuffer_mha_operand,
    ]

    comptime kernel = SM100MLAType.mla_prefill_kernel

    comptime out_depth = SM100MLAType.kv_depth

    comptime PackType = Pack[
        MaskType,
        ScoreModType,
        SchedulerType,
        ValidLengthType,
        SinkType,
        KVRowOffsetsType,
        MaxPromptLenType,
        PartitionType,
    ]

    var pack: PackType = {
        mask_functor,
        score_mod_functor,
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
    comptime smem_use = size_of[
        MLASmemStorage[
            SM100MLAType.qkv_type,
            Int(SM100MLAType.MiscMBarsType.num_mbars()),
            fa4_config,
        ]
    ]()

    ctx.enqueue_function[kernel, kernel](
        q_tma_op,
        k_tma_op,
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
