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

from sys import size_of
from math import ceildiv
from nn.mha_operand import MHAOperand
from nn.mha_mask import MHAMask, TileMaskStatus
from nn.mha_tile_scheduler import (
    SeqInfo,
    TransientScheduler,
)
from nn.mha_sm100_2q import (
    FA4Config,
    KVPipeline,
    SharedMemPointer,
    SharedMemLT,
    elect,
    MBarType,
    ProducerPipeline,
)
from nn.mha import q_num_matrix_view_rows
from nn.mha_sm100_2q import elect_mma_arrive
from nn.mha_fa3_utils import (
    get_seq_info,
    KVTMATile,
    get_seq_info,
    KVTMATile,
    kv_coord,
    NonNullPointer,
    NullPointer,
    Pack,
    produce,
    q_coord,
    q_tma,
    QTMATile,
)
from layout.tma_async import (
    SharedMemBarrier,
    RaggedTMA3DTile,
)
from layout.layout import Layout
from layout.layout_tensor import LayoutTensor
import gpu.primitives.warp as warp

from gpu.memory import AddressSpace, external_memory
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu.host import DeviceContext, FuncAttribute
from gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    barrier,
    thread_idx,
    warp_id,
)
from nn.mha_utils import (
    MHAConfig,
    NoPartition,
    OptionallyStaticInt,
    _is_decoding,
)
from gpu.compute.arch.tcgen05 import *
from linalg.arch.sm100.mma import smem_descriptor
from utils.static_tuple import StaticTuple
from utils.index import Index
from kv_cache.types import swizzle_granularity, padded_depth

from nn.mla_prefill_sm100_utils import (
    SM100MLA,
    MLAPositionSummary,
    MLAKVProducerPipeline,
    split_smem,
)


@fieldwise_init
struct WarpRole(Equatable, TrivialRegisterPassable):
    var _role: Int32
    comptime Softmax0 = Self(0)
    comptime Softmax1 = Self(1)
    comptime Correction = Self(2)
    comptime MMA = Self(3)
    comptime Load = Self(4)
    comptime Empty = Self(5)

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
    var tmem_addr: InlineArray[UInt32, 1]


__extension SM100MLA:
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
    @__llvm_metadata(`nvvm.minctasm`=Int(1))
    fn mla_prefill_kernel_bf16(
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
            TensorMapSwizzle.SWIZZLE_128B,
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
        var ptr_tmem_addr = smem_storage.tmem_addr.unsafe_ptr()

        # All barriers are managed by misc_mbars (S/C/order/Q1Sync/KV/O)
        var misc_mbars: Self.MiscMBarsType = {mbar_base}

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
            ](k_rope_lut, seq_info)

            Self.softmax(
                ptr_tmem_addr[0],
                warp_idx,
                misc_mbars,
                pos.score_row,
                seq_info,
                mask,
                pos.num_keys,
                scale.cast[Self.accum_type](),
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
            ](k_rope_lut, seq_info)
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
            ](k_rope_lut, seq_info)

            Self.load(
                misc_mbars,
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
            ](k_rope_lut, seq_info)
            Self.mma(
                ptr_tmem_addr[0],
                misc_mbars,
                pos.score_row,
                pos.num_keys,
                mask,
                q_smem,
            )
        elif role == WarpRole.Empty:
            warpgroup_reg_dealloc[num_reg_empty]()

    @staticmethod
    @always_inline
    fn load[
        KRopeType: MHAOperand
    ](
        mbars: Self.MiscMBarsType,
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
            KRopeType.dtype,
            TensorMapSwizzle.SWIZZLE_128B,
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
        k_rope_lut: KRopeType,
        q_smem: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
    ):
        comptime KVPipeType = MLAKVProducerPipeline[
            Self.KVLUTType.dtype, KRopeType.dtype, Self.config
        ]

        # If two-qo, we produce qkv in a pattern of
        # q0 & k0, q1, v0, k1, v1, k2, v2...
        comptime SMemTensorLT[layout: Layout] = SharedMemLT[
            Self.KVLUTType.dtype, layout
        ]
        comptime QType = SMemTensorLT[type_of(q_tma_op).layout]
        comptime KType = SMemTensorLT[type_of(k_tma_op).layout]
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
            mbark0.mbar[].expect_bytes(Int32(pipeline_kv.k_bytes + q_bytes))
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
            KRopeType.dtype,
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

            k_rope_tma_op.async_copy(
                k_rope_smem,
                mbark0.mbar[],
                k_rope_coord,
            )
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
                mbarkn = pipeline_kv.get_k[qk_stage=0]()
                k_smem_n, k_rope_smem_n = split_smem[
                    KVPipeType.k_nope_tma_layout,
                    KVPipeType.k_rope_tma_layout,
                    Self.KVLUTType.dtype,
                    KRopeType.dtype,
                ](mbarkn.smem)

                k_tma_op.async_copy(
                    k_smem_n,
                    mbarkn.mbar[],
                    kv_coord[
                        depth = Self.kv_depth,
                        swizzle_granularity = Self.swizzle_granularity,
                    ](kv_gmem_row, kv_head_idx),
                )
                # K rope
                var k_rope_coord = kv_coord[
                    depth = Self.k_rope_depth,
                    swizzle_granularity = Self.swizzle_granularity,
                ](k_rope_gmem_row, k_rope_head_idx)
                k_rope_coord[0] = (
                    Self.cache_depth - Self.k_rope_depth
                )  # only load last 64 head_dims
                k_rope_tma_op.async_copy(
                    k_rope_smem_n,
                    mbarkn.mbar[],
                    k_rope_coord,
                )

            pipeline_kv.commit_kv_step()
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
    fn mma(
        tmem_addr: UInt32,
        mbars: Self.MiscMBarsType,
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
        pipeline.consumer_wait[0]()  # wait K0
        k0 = k_smem_descriptor + full_kv_bytes * pipeline.state.index()
        e = elect()
        Self.UMMA0Type.mma(q0, k0, s0_tmem, elect=e, c_scale=0)
        pipeline_s0.commit_mma(e)

        # Q_1 @ K_0'
        mbars.q1_wait_mbar()[0].wait()  # wait on Q1
        Self.UMMA0Type.mma(q1, k0, s1_tmem, elect=e, c_scale=0)
        pipeline_s1.commit_mma(e)

        pipeline.consumer_release[0](e)  # release K0, step to V state

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
fn mla_sm100_prefill_bf16[
    output_type: DType,
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
](
    output: LayoutTensor[
        output_type, address_space = AddressSpace.GENERIC, ...
    ],
    q: LayoutTensor[q_type, _, address_space = AddressSpace.GENERIC, ...],
    k: KVType,
    v: KVType,
    k_rope: KRopeType,
    mask_functor: MaskType,
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

    # [batch_size, num_keys, cache_num_heads, cache_depth]
    k_rope_tma_op = k_rope.create_tma_tile[
        TensorMapSwizzle.SWIZZLE_128B,
        BN = fa4_config.BN,
        depth=cache_depth,
        BK=64,
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
    MaskType: MHAMask,
    KRopeType: MHAOperand,
    MaxPromptLenType: OptionallyStaticInt,
    //,
    fa4_config: FA4Config,
    cache_depth: Int,
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
        TensorMapSwizzle.SWIZZLE_128B,
        BN = fa4_config.BN,
        BK=64,
    ],
    kv_lut: KVType,
    k_rope_lut: KRopeType,
    mask_functor: MaskType,
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
        SchedulerType,
        fa4_config,
        ValidLengthType,
        SinkType,
        KVRowOffsetsType,
        MaxPromptLenType,
        PartitionType,
        _ndbuffer_mha_operand,
    ]

    comptime kernel = SM100MLAType.mla_prefill_kernel_bf16

    comptime out_depth = SM100MLAType.kv_depth

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
    comptime smem_use = fa4_config.smem_used

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
