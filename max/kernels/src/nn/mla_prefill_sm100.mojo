# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from sys import align_of, simd_width_of, size_of
from math import ceildiv, exp2, recip
from math.constants import log2e
from nn.mha_operand import MHAOperand
from nn.mha_score_mod import ScoreModTrait
from nn.mha_mask import MHAMask, TileMaskStatus, MASK_VALUE
from nn.mha_tile_scheduler import (
    MHASchedulerSynchronization,
    MHATileScheduler,
    MHATileState,
    MHATileSummary,
    SeqInfo,
    TransientScheduler,
)
from nn.mha_sm100_2q import (
    FA4Config,
    FA4MiscMBars,
    KVPipeline,
    SharedMemPointer,
    SharedMemTensor,
    elect,
    MBarType,
    ConsumerPipeline,
    TMemTile,
    KVConsumerPipeline,
    ProducerPipeline,
)
from nn.mha import q_num_matrix_view_rows
from nn.mha_sm100_2q import (
    SM100MHA2Q,
    MBarPipeline,
    SM100TensorAccumulatorSS,
    SM100TensorAccumulatorTS,
    LocalTensor,
    apply_mask,
    maximum,
    STMatrixLayout,
    elect_mma_arrive,
    TMADestination,
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
    Pack,
    PositionSummary,
    produce,
    q_coord,
    q_tma,
    QTMATile,
)
from layout.tma_async import (
    SharedMemBarrier,
    RaggedTMA3DTile,
)
from layout.swizzle import make_swizzle
from layout.tensor_core_async import tile_layout_k_major, tile_layout_mn_major
from layout.layout import Layout, blocked_product
from layout.layout_tensor import LayoutTensor
import gpu.warp as warp
from gpu.sync import (
    named_barrier,
    cp_async_bulk_commit_group,
    cp_async_bulk_wait_group,
    syncwarp,
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
    FlashAttentionAlgorithm,
    MHAConfig,
    MHAPartitionScheme,
    NoPartition,
    OptionallyStaticInt,
    _is_decoding,
)
from gpu.tcgen05 import *
from linalg.arch.sm100.mma import smem_descriptor
from utils.numerics import get_accum_type, min_or_neg_inf
from utils.static_tuple import StaticTuple
from utils.index import Index, IndexList
from kv_cache.types import swizzle_granularity


@register_passable("trivial")
struct MLAKVProducerPipeline[dtype: DType, config: FA4Config]:
    comptime k_layout = tile_layout_k_major[
        Self.dtype,
        Self.config.BN,
        128,
        Self.config.swizzle_mode,
    ]()
    comptime k_rope_layout = tile_layout_k_major[
        Self.dtype,
        Self.config.BN,
        64,
        Self.config.swizzle_mode,
    ]()

    comptime KType = SharedMemTensor[
        Self.dtype,
        tile_layout_k_major[
            Self.dtype,
            Self.config.BN,
            Self.config.BK0,
            Self.config.swizzle_mode,
        ](),
    ]

    comptime VType = SharedMemTensor[
        Self.dtype,
        tile_layout_mn_major[
            Self.dtype,
            128,
            Self.config.BK1,
            Self.config.swizzle_mode,
        ](),
    ]
    comptime KPairType = TMADestination[Self.dtype, Self.KType.layout]
    comptime VPairType = TMADestination[Self.dtype, Self.VType.layout]
    comptime k_elements = Self.KType.layout.size()
    comptime v_elements = Self.VType.layout.size()
    comptime k_bytes = Self.k_elements * size_of[Self.dtype]()
    comptime v_bytes = Self.v_elements * size_of[Self.dtype]()
    comptime SMemType = SharedMemPointer[Scalar[Self.dtype]]

    var kv_pipeline: KVPipeline[
        Self.config.num_kv_stages, Self.config.num_mma_stages
    ]
    var smem: Self.SMemType

    @always_inline
    fn __init__(
        out self,
        mbar: MBarType,
        smem: Self.SMemType,
    ):
        __comptime_assert (
            Self.config.padded_depth % Self.config.num_mma_stages == 0
        )
        __comptime_assert Self.config.BN % Self.config.num_mma_stages == 0
        self.kv_pipeline = {mbar}
        self.smem = smem
        self.kv_pipeline.state._phase = 1

    @always_inline
    fn __init__(
        out self,
        kv_pipeline: KVPipeline[
            Self.config.num_kv_stages, Self.config.num_mma_stages
        ],
        smem: Self.SMemType,
    ):
        __comptime_assert (
            Self.config.padded_depth % Self.config.num_mma_stages == 0
        )
        __comptime_assert Self.config.BN % Self.config.num_mma_stages == 0
        self.kv_pipeline = kv_pipeline
        self.smem = smem
        self.kv_pipeline.state._phase = 1

    @always_inline
    fn init(self):
        """
        Only one of the producer or consumer should call `init()`.
        """
        self.kv_pipeline.init()

    @always_inline
    fn get_kv_smem[*, mma_stage: Int](self) -> Self.SMemType:
        comptime stage_offset = mma_stage * Self.config.padded_depth * Self.config.BN
        var dyn_offset: UInt32 = (
            Self.k_elements * self.kv_pipeline.state.index()
        )
        return self.smem + stage_offset + dyn_offset

    @always_inline
    fn get_k[*, mma_stage: Int, expect: Bool = True](self) -> Self.KPairType:
        p_mbar = self.kv_pipeline.producer_mbar[mma_stage=mma_stage]()

        @parameter
        if expect:
            p_mbar[].expect_bytes(Self.k_bytes)
        return {p_mbar, {self.get_kv_smem[mma_stage=mma_stage]()}}

    @always_inline
    fn get_v[*, mma_stage: Int](self) -> Self.VPairType:
        p_mbar = self.kv_pipeline.producer_mbar[mma_stage=mma_stage]()
        p_mbar[].expect_bytes(Self.v_bytes)
        return {p_mbar, {self.get_kv_smem[mma_stage=mma_stage]()}}

    @always_inline
    fn acquire_kv[*, mma_stage: Int = Self.config.num_mma_stages - 1](self):
        self.kv_pipeline.producer_acquire[mma_stage]()

    @always_inline
    fn commit_kv_step(mut self):
        """
        Step the kv pipeline. The does not perform the commit on the mbars;
        that should be handled by the `tma_op.async_copy`.
        """
        self.kv_pipeline.state.step()


@register_passable("trivial")
struct SM100MLA[
    KVLUTType: MHAOperand,
    output_type: DType,
    MaskType: MHAMask,
    ScoreModType: ScoreModTrait,
    SchedulerType: MHATileScheduler,
    config: FA4Config,
    use_score_mod: Bool,
    ValidLengthType: OptionalPointer,
    SinkType: OptionalPointer,
    KVRowOffsetsType: OptionalPointer,
    _is_cache_length_accurate: Bool,
    MaxSeqLenType: OptionallyStaticInt,
    PartitionType: MHAPartitionScheme,
]:
    comptime qkv_type = Self.KVLUTType.dtype
    comptime accum_type = get_accum_type[Self.qkv_type]()
    comptime simd_size: Int = simd_width_of[Self.qkv_type]()

    comptime cta_group = 1  # TODO: support 2
    comptime BM = Self.config.BM
    comptime BN = Self.config.BN
    comptime depth = Self.config.depth  # 192
    comptime padded_depth = Self.config.padded_depth  # 192
    comptime num_q_heads = Self.config.num_q_heads
    comptime group = Self.config.group
    comptime ragged = not Self.ValidLengthType.is_null
    comptime page_size = Self.KVLUTType.page_size

    comptime k_rope_depth = 64
    comptime kv_depth = Self.config.depth - Self.k_rope_depth
    comptime cache_depth = 576

    comptime num_m_mmas = 2
    comptime MMA_M = Self.config.BM // Self.num_m_mmas
    comptime qo_elements = Self.padded_depth * Self.MMA_M
    comptime qkv_dt_size = size_of[Self.qkv_type]()

    comptime OPipelineType = MBarPipeline[2]  # x1 -> 4 barriers

    comptime num_mma_stages = Self.config.num_mma_stages

    # First MMA is
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
        num_stages = Self.num_mma_stages,
    ]
    # Second MMA is
    # (BM x BN) @ (BN x depth) -> (BM x depth)
    comptime UMMA1Type = SM100TensorAccumulatorTS[
        Self.qkv_type,
        Self.accum_type,
        MMA_M = Self.MMA_M,
        MMA_N = Self.kv_depth,  # 128
        BK = Self.BN,
        swizzle_b = Self.config.swizzle_mode,
        transpose_b=False,
        num_stages = Self.num_mma_stages,
    ]

    comptime swizzle_granularity = Self.config.swizzle_mode.bytes() // Self.qkv_dt_size
    comptime k_elements: UInt32 = Self.swizzle_granularity * Self.config.BN
    comptime qo_bytes: UInt32 = Self.qkv_dt_size * Self.qo_elements
    comptime k_bytes: UInt32 = Self.qkv_dt_size * Self.k_elements
    comptime MMA_K = 16
    comptime v_bytes_per_mma: UInt32 = Self.qkv_dt_size * Self.MMA_K * Self.config.padded_depth

    comptime KVPipelineType = KVPipeline[
        Self.config.num_kv_stages, Self.config.num_mma_stages
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

    @staticmethod
    @__llvm_arg_metadata(q_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(k_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(v_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(k_rope_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(ragged_tma_store, `nvvm.grid_constant`)
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Self.config.num_threads
        )
    )
    fn mla_prefill_kernel[
        KRopeType: MHAOperand
    ](
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
            Self.KVLUTType.dtype,
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
        scale: Float32,
        batch_size: UInt32,
        num_keys_arg: UInt32,
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
        __comptime_assert Self.MMA_M == 64 or Self.MMA_M == 128
        __comptime_assert _is_decoding[Self.MaxSeqLenType]() == False
        __comptime_assert Self.config.supported(), (
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
        __comptime_assert (
            not Self.SchedulerType.may_advance
        ), "Persistent kernels not yet supported with FA4"

        mask = pack.mask
        score_mod = pack.score_mod
        scheduler = pack.scheduler
        valid_length = pack.valid_length
        sink_weights = pack.sink_weights
        kv_input_row_offsets = pack.kv_input_row_offsets
        max_seq_len = pack.max_seq_len
        partition = pack.partition

        comptime num_qo = Self.config.num_qo()
        # TODO: We may want to support num_qo>2 for depth=64?
        __comptime_assert (
            num_qo == 1 or num_qo == 2
        ), "Currently only support num_qo == 1 or 2"
        q_smem = external_memory[
            Scalar[Self.qkv_type],
            address_space = AddressSpace.SHARED,
            alignment=128,
            name="mha_dynamic_shared_memory",
        ]()
        kv_smem = q_smem + Self.config.BM * Self.config.padded_depth
        comptime kv_total_stages = Self.config.num_kv_stages * Self.config.num_mma_stages
        comptime kv_smem_total_bytes = Self.config.padded_depth * Self.config.BN * kv_total_stages
        var correction_smem: SharedMemPointer[Scalar[Self.accum_type]] = (
            kv_smem + kv_smem_total_bytes
        ).bitcast[Scalar[Self.accum_type]]()
        var mbar_base: MBarType

        @parameter
        if Self.config.use_tmem_for_correction():
            mbar_base = correction_smem.bitcast[SharedMemBarrier]()
        else:
            mbar_base = (
                correction_smem + Self.config.correction_smem_elements()
            ).bitcast[SharedMemBarrier]()

        kv_pipeline = Self.KVPipelineType(mbar_base)
        mbar_base += Self.KVPipelineType.num_mbars()
        # O += P@V -> correction
        o_mbar = mbar_base  # 2, UMMA
        mbar_base += Self.OPipelineType.num_mbars()
        var misc_mbars: FA4MiscMBars = {mbar_base}
        # S = Q@K' -> softmax 0/1
        # softmax 0/1 -> correction
        # 4s (2 consumer, 2 producer)
        # 4c (2 consumer, 2 producer)
        # 2 softmax-order
        ptr_tmem_addr = misc_mbars.end().bitcast[UInt32]()

        # https://github.com/NVIDIA/cutlass/blob/main/examples/77_blackwell_fmha/kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp
        comptime num_reg_softmax = 200
        comptime num_reg_correction = 80
        comptime num_reg_other = 32

        __comptime_assert not Self.PartitionType.do_partition, (
            "Neither partitioning nor decoding are supported by the 2-q"
            " implementation."
        )

        var warp_idx: UInt32 = warp.broadcast(warp_id())
        if warp_idx == 0:
            if elect() != 0:
                kv_pipeline.init()

                # o produced by 1 MMA, consumed by 128 correction
                @parameter
                for i in range(2):
                    o_mbar[i].init(1)  # producer
                    o_mbar[i + 2].init(WARPGROUP_SIZE)  # consumer
                misc_mbars.init()
        elif warp_idx == 1:
            tcgen05_alloc[Self.cta_group](
                ptr_tmem_addr, Self.config.sm100_tmem_cols
            )

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

            Self.softmax(
                ptr_tmem_addr[0],
                warp_idx,
                misc_mbars,
                o_mbar,
                pos.score_row,
                seq_info,
                mask,
                pos.num_keys,
                scale.cast[Self.accum_type](),
                score_mod,
                max_seq_len.as_uint32(),
                ragged_tma_store,
                q_smem.bitcast[Scalar[Self.output_type]](),
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
            Self.correction(
                ptr_tmem_addr[0],
                misc_mbars,
                o_mbar,
                pos.score_row,
                pos.num_keys,
                mask,
            )
        else:
            warpgroup_reg_dealloc[num_reg_other]()
            if warp_idx == 13:  # produce
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
                Self.load(
                    misc_mbars,
                    kv_pipeline,
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
                    q_smem,
                )

            elif warp_idx == 12:  # Q @ K', P @ V
                var seq_info: SeqInfo = get_seq_info[Self.BM, Self.num_q_heads](
                    batch_size, max_seq_len, valid_length, partition
                )

                if not seq_info.is_valid():
                    tcgen05_release_allocation_lock[Self.cta_group]()
                    tcgen05_dealloc[Self.cta_group](
                        ptr_tmem_addr[0], Self.config.sm100_tmem_cols
                    )
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
                Self.mma(
                    ptr_tmem_addr[0],
                    misc_mbars,
                    kv_pipeline,
                    o_mbar,
                    pos.score_row,
                    pos.num_keys,
                    mask,
                    q_smem,
                )

    @staticmethod
    @always_inline
    fn correction(
        tmem_addr: UInt32,
        mbars: FA4MiscMBars,
        o_mbar: MBarType,
        score_row: UInt32,
        num_keys: UInt32,
        mask: Self.MaskType,
    ):
        __comptime_assert size_of[Self.accum_type]() == 4

        o0_tmem = tmem_addr + Self.config.TMEM_O0
        o1_tmem = tmem_addr + Self.config.TMEM_O1
        c0_tmem = tmem_addr + Self.config.TMEM_C0
        c1_tmem = tmem_addr + Self.config.TMEM_C1

        pipeline_c0 = mbars.consumer_c0()
        pipeline_c1 = mbars.consumer_c1()
        pipeline_o = ConsumerPipeline[2](o_mbar)

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

        while iter_count != 0:
            iter_count -= 1

            @parameter
            for i in range(2):
                var c_tmem: UInt32

                @parameter
                if i == 0:
                    c_tmem = c0_tmem
                    pipeline_c0.wait()
                else:
                    c_tmem = c1_tmem
                    pipeline_c1.wait()

                # correct
                c_scalar = tcgen05_ld[
                    datapaths=32,
                    bits=32,
                    repeat=1,
                    dtype = Self.accum_type,
                    pack=False,
                    width=1,
                ](c_tmem)
                tcgen05_load_wait()

                @parameter
                if i == 0:
                    pipeline_c0.release()
                else:
                    pipeline_c1.release()

                change = _vote_nvidia_helper(c_scalar != 1) != 0
                pipeline_o.wait()
                if change:
                    # TODO: experiment with different batch sizes.
                    # The idea here is to both pipeline, and reduce peak register use.
                    __comptime_assert load_iters > 1
                    __comptime_assert Self.config.depth % batch_size == 0

                    var o_tmem: UInt32

                    @parameter
                    if i == 0:
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

                    @parameter
                    for b in range(load_iters):
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
                        ](o_tmem + b1_offset)
                        tcgen05_st[  # 0b0*c_scalar store
                            datapaths=32,
                            bits=32,
                            repeat=batch_size,
                            pack=False,
                        ](o_tmem + b0_offset0, o_b0 * c_scalar)
                        tcgen05_load_wait()  # ob1 loaded

                        @parameter
                        if b0_offset1 + batch_size <= Self.kv_depth:
                            o_b0 = tcgen05_ld[  # 0b0 start
                                datapaths=32,
                                bits=32,
                                repeat=batch_size,
                                dtype = Self.accum_type,
                                pack=False,
                                width=batch_size,
                            ](o_tmem + b0_offset1)
                        tcgen05_st[  # 0b0*c_scalar store
                            datapaths=32,
                            bits=32,
                            repeat=batch_size,
                            pack=False,
                        ](o_tmem + b1_offset, o_b1 * c_scalar)

                    @parameter
                    if load_remainder > 0:
                        tcgen05_load_wait()  # ob1 loaded
                        comptime offset = 2 * batch_size * load_iters
                        tcgen05_st[  # 0b0*c_scalar store
                            datapaths=32,
                            bits=32,
                            repeat=load_remainder,
                            pack=False,
                        ](o_tmem + offset, o_b0 * c_scalar)
                    tcgen05_store_wait()
                    tcgen05_fence_before()
                pipeline_o.release()

    @staticmethod
    @always_inline
    fn softmax(
        tmem_addr: UInt32,
        warp_idx: UInt32,
        mbars: FA4MiscMBars,
        o_mbar: MBarType,
        score_row: UInt32,
        seq_info: SeqInfo,
        mask: Self.MaskType,
        num_keys: UInt32,
        scale: Scalar[Self.accum_type],
        score_mod: Self.ScoreModType,
        max_seq_len: UInt32,
        ragged_tma_store: RaggedTMA3DTile[
            Self.output_type,
            Self.config.swizzle_mode,
            BM = Self.config.BM // 2,
            BN = Self.kv_depth,
        ],
        o_smem: SharedMemPointer[Scalar[Self.output_type]],
        sink_weights: Self.SinkType,
    ):
        # FIXME: for depth 256
        var s_tmem: UInt32 = tmem_addr + Self.config.TMEM_S0

        var warp_group_idx: UInt32 = warp_idx // 4

        @parameter
        if Self.config.split_m:
            # split-M: second S is (+16 rows) in st-matrix space
            s_tmem += (16 << 16) * warp_group_idx
        else:
            # 2-Q path: S1 is at +BN columns
            s_tmem += Self.config.BN * warp_group_idx

        p_tmem = s_tmem
        c_tmem = p_tmem + Self.config.BN // 2
        s_tile = Self.UMMA0Type.CType(s_tmem)
        p_tile = Self.UMMA1Type.AType(p_tmem)

        pipeline_s = mbars.consumer_s(warp_group_idx)
        pipeline_c = mbars.producer_c(warp_group_idx)
        # TODO: order_s_wait/arrive
        order_s_wait = mbars.pipeline_order_wait(warp_group_idx)
        order_s_arrive = mbars.pipeline_order_arrive(warp_group_idx)
        var order_phase: UInt32 = 0

        var q_head_idx: UInt32 = seq_info.head_idx
        var tid: UInt32 = thread_idx.x
        var row: UInt32 = tid % 128
        var scale_log2e: Scalar[Self.accum_type] = scale

        @parameter
        if not (Self.use_score_mod or Self.MaskType.apply_log2e_after_mask):
            scale_log2e *= log2e

        @parameter
        @always_inline
        fn mask_row[
            BN: Int, //, masked: Bool, last_iter: Bool
        ](
            s: LocalTensor[Self.accum_type, Layout.row_major(BN)],
            kv_row: UInt32,
        ):
            apply_mask[
                decoding=False,
                use_score_mod = Self.use_score_mod,
                masked=masked,
                last_iter=last_iter,
            ](
                s,
                mask,
                score_mod,
                scale_log2e,
                prompt_idx=seq_info.prompt_idx,
                q_head_idx=q_head_idx,
                kv_tile_start_row=kv_row,
                max_seq_len=max_seq_len,
                num_keys=num_keys,
                score_row=score_row + tid,
            )

        # while waiting, offset output
        comptime splitBM = Self.BM // 2
        var num_output_rows = min(
            splitBM,
            Int32(seq_info.seq_len)
            - Int32(seq_info.prompt_offset)
            - Int32(warp_group_idx) * splitBM,
        )

        gmem_row = Self.PositionType.get_q_gmem_row[ragged = Self.ragged](
            seq_info, max_seq_len
        )

        pipeline_s.wait()
        tcgen05_fence_after()
        s = LocalTensor[
            Self.accum_type, Layout.row_major(Self.config.BN)
        ].stack_allocation()

        @parameter
        @always_inline
        fn load_mask_max[
            *, masked: Bool, last_iter: Bool
        ](kv_row: UInt32) -> Scalar[Self.accum_type]:
            # break up into sets of 32
            # minimize wait time by using smallest first
            __comptime_assert Self.config.BN == 64, String(Self.config.BN)
            comptime BM = Self.config.BM // 2
            comptime batch_size = 32
            comptime has_remainder = (Self.config.BN % batch_size) != 0
            comptime first_cols = (
                Self.config.BN % batch_size
            ) if has_remainder else batch_size
            s0 = TMemTile[Self.accum_type, BM, first_cols](s_tmem).load_async()
            tcgen05_load_wait()

            s1 = TMemTile[Self.accum_type, BM, batch_size](
                s_tmem + first_cols
            ).load_async()
            mask_row[masked=masked, last_iter=last_iter](s0, kv_row)
            vrow_max = maximum[width = Self.simd_size](s0)

            s.ptr.store(s0.ptr.load[width=first_cols]())
            comptime cols = Self.config.BN - first_cols + batch_size

            @parameter
            for i in range(cols // (2 * batch_size)):
                comptime offset0 = first_cols + batch_size * (2 * i)
                comptime offset1 = first_cols + batch_size * (2 * i + 1)
                comptime offset2 = first_cols + batch_size * (2 * i + 2)

                tcgen05_load_wait()

                @parameter
                if offset1 >= Self.config.BN:
                    mask_row[masked=masked, last_iter=last_iter](
                        s1, kv_row + offset0
                    )
                    vrow_max = maximum(s1, vrow_max)
                    s.ptr.store(offset0, s1.ptr.load[width=batch_size]())
                else:
                    s2 = TMemTile[Self.accum_type, BM, batch_size](
                        s_tmem + offset1
                    ).load_async()
                    mask_row[masked=masked, last_iter=last_iter](
                        s1, kv_row + offset0
                    )
                    vrow_max = maximum(s1, vrow_max)
                    s.ptr.store(offset0, s1.ptr.load[width=batch_size]())
                    tcgen05_load_wait()

                    @parameter
                    if offset2 < Self.config.BN:
                        s1 = TMemTile[Self.accum_type, BM, batch_size](
                            s_tmem + offset2
                        ).load_async()
                    mask_row[masked=masked, last_iter=last_iter](
                        s2, kv_row + offset1
                    )
                    vrow_max = maximum(s2, vrow_max)
                    s.ptr.store(offset1, s2.ptr.load[width=batch_size]())

            return vrow_max.reduce_max()

        var kv_row: UInt32 = mask.start_column[
            Self.BM, Self.BN, Self.page_size
        ](score_row)
        comptime mask_sets = Self.MaskType.nonfull_sets[Self.BM, Self.BN]()
        comptime num_sets = len(mask_sets)
        var row_max: Scalar[Self.accum_type] = load_mask_max[
            masked=True, last_iter=True
        ](kv_row)
        var sink_weights_ptr = UnsafePointer[Scalar[Self.qkv_type]]()
        var sink_weight: Scalar[Self.accum_type]

        @parameter
        if not Self.SinkType.is_null:
            sink_weights_ptr = rebind[UnsafePointer[Scalar[Self.qkv_type]]](
                sink_weights.value()
            )
            var head_idx: UInt32 = seq_info.head_idx
            sink_weight = (
                sink_weights_ptr[head_idx].cast[Self.accum_type]() * log2e
            )
            row_max = max(row_max, sink_weight)
        else:
            sink_weights_ptr = UnsafePointer[Scalar[Self.qkv_type]]()
            sink_weight = 0.0

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
            __comptime_assert num_batch_iters > 0
            comptime BatchTileType = TMemTile[
                Self.qkv_type, Self.config.BM // 2, batch_size * exp_simd
            ]
            comptime RemainderTileType = TMemTile[
                Self.qkv_type, Self.config.BM // 2, remainder * exp_simd
            ]
            __comptime_assert (Self.config.BN % exp_simd) == 0

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
            vs[0] = rebind[vs.element_type](acc)

            @parameter
            for i in range(1, batch_size // 2):
                vsi = exp2(rebind[AccType](vs[i]) - row_max)
                vs[i] = rebind[vs.element_type](vsi)
                acc += vsi

            # at this point, we need 32 fewer fp32 registers but 16 more u32
            @parameter
            for i in range(batch_size // 2, batch_size):
                vs[i] = exp2(vs[i] - row_max)

            BatchTileType(p_tmem).store(
                LocalTensor[
                    Self.accum_type, Layout.row_major(batch_size * exp_simd)
                ](s.ptr)
            )

            @parameter
            for b in range(1, num_batch_iters):
                comptime offset = batch_size * b

                @parameter
                for i in range(offset, offset + batch_size):
                    vs[i] = exp2(vs[i] - row_max)

                comptime el_offset = offset * exp_simd
                comptime tmem_offset = (
                    el_offset * size_of[Self.qkv_type]()
                ) // size_of[Self.accum_type]()
                BatchTileType(p_tmem + tmem_offset).store(
                    LocalTensor[
                        Self.accum_type, Layout.row_major(batch_size * exp_simd)
                    ](s.ptr + el_offset)
                )

            @parameter
            if remainder > 0:
                comptime offset = batch_size * num_batch_iters

                @parameter
                for i in range(offset, offset + remainder):
                    vs[i] = exp2(vs[i] - row_max)

                comptime el_offset = offset * exp_simd
                comptime tmem_offset = (
                    el_offset * size_of[Self.qkv_type]()
                ) // size_of[Self.accum_type]()
                RemainderTileType(p_tmem + tmem_offset).store(
                    LocalTensor[
                        Self.accum_type, Layout.row_major(remainder * exp_simd)
                    ](s.ptr + el_offset)
                )

            tcgen05_store_wait()
            tcgen05_fence_before()
            pipeline_s.release()
            # now we can sum the remaining elements of `acc`
            acc0 = vs[batch_size // 2]
            acc1 = vs[batch_size // 2 + 1]
            acc2 = vs[batch_size // 2 + 2] + vs[batch_size // 2 + 3]

            @parameter
            for i in range(batch_size // 2 + 4, vs_len, 4):
                acc += rebind[AccType](vs[i])
                acc0 += vs[i + 1]
                acc1 += vs[i + 2]
                acc2 += vs[i + 3]
            return (acc + rebind[AccType](acc0)) + rebind[AccType](acc1 + acc2)

        var row_sum: SIMD[Self.accum_type, 2] = store_exp(row_max)

        var o_phase: UInt32 = 0  # initial wait is phase 0

        @parameter
        if not Self.SinkType.is_null:
            row_sum[0] += exp2(sink_weight - row_max)

        # TODO: add ordering barriers to prevent overlap
        # between the two softmax warpgroups
        @parameter
        if mask_sets[0] != TileMaskStatus.UNKNOWN_MASK:
            mask_ends = mask.masked_set_ends[
                BM = Self.BM, BN = Self.BN, page_size = Self.page_size
            ](score_row, num_keys)
            var decrement: Bool = True

            @parameter
            for i in range(num_sets):
                comptime mask_status = mask_sets[i]
                var iters: UInt32

                @parameter
                if i == 0:
                    iters = mask_ends[i]
                else:
                    iters = mask_ends[i] - mask_ends[i - 1]
                if decrement and iters > 0:
                    iters -= 1
                    decrement = False
                while iters != 0:
                    iters -= 1
                    kv_row += Self.config.BN
                    pipeline_s.wait()
                    # calculate rowmax
                    old_max = row_max
                    var new_row_max: Scalar[Self.accum_type]

                    # last_iter == (i + 1 == num_sets) and (i == 0)
                    # `i == 0` is runtime; for now, we set to `True`
                    # as this number of iterations is small
                    comptime last_iter: Bool = i + 1 == num_sets
                    comptime masked: Bool = mask_status == TileMaskStatus.PARTIAL_MASK
                    new_row_max = load_mask_max[
                        masked=masked, last_iter=last_iter
                    ](kv_row)
                    row_max = max(old_max, new_row_max)
                    correction = exp2(old_max - row_max)
                    pipeline_c.acquire()
                    tcgen05_st[
                        datapaths=32,
                        bits=32,
                        repeat=1,
                        pack=False,
                    ](c_tmem, correction)
                    pipeline_c.commit()
                    # update s->p
                    local_rowsum = store_exp(row_max)
                    row_sum = row_sum.fma(correction, local_rowsum)
                    o_phase ^= 1
        else:
            while True:
                kv_row += Self.config.BN
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
                    new_row_max = load_mask_max[masked=True, last_iter=True](
                        kv_row
                    )
                else:
                    new_row_max = load_mask_max[masked=False, last_iter=True](
                        kv_row
                    )
                row_max = max(old_max, new_row_max)
                correction = exp2(old_max - row_max)
                pipeline_c.acquire()
                tcgen05_st[
                    datapaths=32,
                    bits=32,
                    repeat=1,
                    pack=False,
                ](c_tmem, correction)
                pipeline_c.commit()
                # update s->p
                local_rowsum = store_exp(row_max)
                row_sum = row_sum.fma(correction, local_rowsum)
                o_phase ^= 1
        # Do the final correction and write
        inv_row_sum = recip(row_sum.reduce_add())
        o_tile = Self.UMMA1Type.CType(
            tmem_addr + Self.config.TMEM_O0 + warp_group_idx * Self.padded_depth
        )
        # wait on the o_pipeline producer
        __comptime_assert (
            size_of[Self.output_type]() == size_of[Self.qkv_type]()
        )
        if num_output_rows > 0:
            o_mbar[warp_group_idx].wait(o_phase)  # consumer wait
            tcgen05_fence_after()  # example 1
            # TODO: pass in a dedicated barrier that a q-writer can wait on in a persistent kernel?
            comptime HalfBM = Self.BM // 2

            Self.scale_write_output(
                row,
                warp_idx & 3,
                warp_group_idx,
                inv_row_sum,
                o_smem + warp_group_idx * (HalfBM * Self.kv_depth),
                o_tile,
                ragged_tma_store,
                o_mbar + 2 + warp_group_idx,  # consumer arrive
                num_output_rows,
                q_head_idx,
                gmem_row + warp_group_idx * HalfBM,
            )
        named_barrier[2 * WARPGROUP_SIZE](2)
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
        inv_row_sums = LocalTensor[
            Self.accum_type, Layout.row_major(num_rows)
        ].stack_allocation()
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
        @parameter
        for i in range(num_rows):
            # lane // 4, lane // 4 + 8, lane // 4 + 16, lane // 4 + 24
            inv_row_sums[i] = warp.shuffle_idx(inv_row_sum, lane_row + 8 * i)

        tcgen05_load_wait()
        tcgen05_fence_before()
        _ = consumer_mbar[].arrive()

        @parameter
        for i in range(num_rows):
            irs = o.element_type(
                rebind[Scalar[Self.accum_type]](inv_row_sums[i])
            )

            @parameter
            for j in range(o.layout[1].size()):
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

        comptime swizzle_block_size: UInt32 = WARP_SIZE * swizzle_granularity
        o_smem = o_smem_arg + local_warp_idx * swizzle_block_size

        @parameter
        for i in range(2):
            comptime datapath_offset: UInt32 = 16 * i * swizzle_granularity

            @parameter
            for j in range(iters):
                comptime ofs = i * ST.frag_size + j * (ST.frag_size // iters)
                var rows_of_o_frags = LocalTensor[
                    Self.accum_type,
                    layout = Layout.row_major(1, ST.frag_size // iters),
                ](
                    o.ptr + ofs
                )  # all the repeats across n and m

                comptime warp_smem_offset: UInt32 = datapath_offset + j * (
                    Self.BM // 2
                ) * swizzle_granularity
                accum_smem_warp_tile = LayoutTensor[
                    Self.output_type,
                    Layout.row_major(16, swizzle_granularity),
                    MutAnyOrigin,
                    address_space = AddressSpace.SHARED,
                ](o_smem + warp_smem_offset)

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
        named_barrier[WARPGROUP_SIZE](Int32(warp_group_idx))

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
            Index[dtype = DType.int32](Int(Self.BM), Int(Self.BN)),
        )

    @staticmethod
    @always_inline
    fn load(
        mbars: FA4MiscMBars,
        kv_pipeline_arg: Self.KVPipelineType,
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
            Self.KVLUTType.dtype,
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
        q_smem: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
    ):
        comptime KVPipeType = MLAKVProducerPipeline[
            Self.KVLUTType.dtype, Self.config
        ]

        # If two-qo, we produce qkv in a pattern of
        # q0 & k0, q1, v0, k1, v1, k2, v2...
        comptime SMemTensor[layout: Layout] = SharedMemTensor[
            Self.KVLUTType.dtype, layout
        ]
        comptime QType = SMemTensor[type_of(q_tma_op).layout]
        comptime KType = SMemTensor[type_of(k_tma_op).layout]
        comptime KRopeSMType = SMemTensor[type_of(k_rope_tma_op).layout]
        comptime VType = SMemTensor[type_of(v_tma_op).layout]

        var k_rope_head_idx: UInt32 = seq_info.head_idx // Self.group
        var kv_head_idx: UInt32 = seq_info.head_idx

        comptime q_elements = (Self.config.BM // 2) * Self.config.BK0
        comptime q_bytes = size_of[Self.qkv_type]() * q_elements

        kv_smem = q_smem + Self.config.BM * Self.config.padded_depth
        var pipeline_kv: KVPipeType = {kv_pipeline_arg, kv_smem}

        var mbark0: KVPipeType.KPairType

        mbark0 = pipeline_kv.get_k[mma_stage=0, expect=False]()  # no wait
        var q_gmem_row: UInt32 = Self.PositionType.get_q_gmem_row[
            ragged = Self.ragged
        ](seq_info, max_seq_len)
        var q_head_idx: UInt32 = seq_info.head_idx
        elect = elect() != 0

        # copy q0
        if elect:
            # Q0
            mbark0.mbar[].expect_bytes(pipeline_kv.k_bytes + q_bytes)
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
        var iter_count: UInt32 = (
            mask.last_masked_set_end[Self.BM, Self.BN, Self.page_size](
                score_row, num_keys
            )
            - 1
        )
        # copy k0
        k_smem, k_rope_smem = mbark0.split_smem[
            KVPipeType.k_layout, KVPipeType.k_rope_layout
        ]()
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
            ](kv_gmem_row, k_rope_head_idx)
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
            ref q1_mbar = mbars.q1_wait_mbar()
            q1_mbar.expect_bytes(q_bytes)
            # Q1
            q_tma_op.async_copy(
                QType(q_smem + q_elements),
                q1_mbar,
                q_coord[
                    depth = Self.depth,
                    swizzle_granularity = Self.swizzle_granularity,
                    decoding=False,
                ](q_gmem_row + Self.config.BM // 2, q_head_idx),
            )
        # copy v0
        if elect:
            mbarv0 = pipeline_kv.get_v[mma_stage=0]()
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
            kv_row += Self.config.BN

            @parameter
            if check_mask:
                if (
                    Self.mask_status(mask, score_row, kv_row)
                    == TileMaskStatus.FULL_MASK
                ):
                    continue
            kv_gmem_row = kv_lut.row_idx(seq_info.prompt_idx, kv_row)
            # produce k
            pipeline_kv.acquire_kv()
            if elect:
                mbarkn = pipeline_kv.get_k[mma_stage=0]()
                k_smem_n, k_rope_smem_n = mbarkn.split_smem[
                    KVPipeType.k_layout, KVPipeType.k_rope_layout
                ]()

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
                ](kv_gmem_row, k_rope_head_idx)
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
                mbarvn = pipeline_kv.get_v[mma_stage=0]()
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
        mbars: FA4MiscMBars,
        kv_pipeline_arg: Self.KVPipelineType,
        o_mbar: MBarType,
        score_row: UInt32,
        num_keys: UInt32,
        mask: Self.MaskType,
        q_smem: SharedMemPointer[Scalar[Self.KVLUTType.dtype]],
    ):
        comptime KVPipeType = KVConsumerPipeline[
            Self.KVLUTType.dtype, Self.config
        ]

        s0_tmem = tmem_addr + Self.config.TMEM_S0
        s1_tmem = tmem_addr + Self.config.TMEM_S1
        o0_tmem = tmem_addr + Self.config.TMEM_O0
        o1_tmem = tmem_addr + Self.config.TMEM_O1

        producer_s0 = mbars.producer_s0().mbar  # phase = 1
        consumer_s0 = producer_s0 + 1
        producer_s1 = mbars.producer_s1().mbar  # phase = 1
        consumer_s1 = producer_s1 + 1
        pipeline_o_initial = ProducerPipeline[2](o_mbar)  # phase = 1
        producer_o0 = pipeline_o_initial.mbar
        producer_o1 = producer_o0 + 1
        consumer_o0 = producer_o1 + 1
        consumer_o1 = consumer_o0 + 1

        comptime q0_size = (Self.config.BM // 2) * Self.config.padded_depth
        comptime q0_bytes = q0_size * size_of[Self.KVLUTType.dtype]()
        q0 = Self.descriptor_q(q_smem)
        q1 = q0 + q0_bytes
        kv_smem = q_smem + 2 * q0_size

        var pipeline_kv: KVPipeType = {kv_pipeline_arg, kv_smem}

        # We peel the first iteration, as we want to wait on q1
        var iter_count: UInt32 = (
            mask.total_iters[Self.BM, Self.BN, Self.page_size](
                score_row, num_keys
            )
            - 1
        )

        # Q_0 @ K_0'
        k0 = pipeline_kv.wait_k[mma_stage=0, pre_increment=False]()  # [kv0]
        e = elect()
        Self.UMMA0Type.mma(q0, k0, s0_tmem, elect=e, c_scale=0)
        elect_mma_arrive(producer_s0, e)
        # pipeline_s0.step()  # pipline_s0.phase = 0

        # Q_1 @ K_0'
        # pipeline_s1.producer_acquire()
        mbars.q1_wait_mbar().wait()  # wait on Q1
        # we don't need to wait on s1
        Self.UMMA0Type.mma(q1, k0, s1_tmem, elect=e, c_scale=0)
        elect_mma_arrive(producer_s1, e)

        pipeline_kv.release_k(e)  # [kv0]->kv1

        vlatest = pipeline_kv.wait_v[mma_stage=0]()  # [kv1]
        # For the first V tile in the current KV stage buffer:
        # Use the SAME base pointer you used for K (no manual offset).
        _ = consumer_s0[].wait(0)
        Self.UMMA1Type.mma(s0_tmem, vlatest, o0_tmem, elect=e, c_scale=0)
        elect_mma_arrive(producer_o0, e)
        var phase_s: UInt32 = 0
        var phase_o: UInt32 = 1

        var c_scale: UInt32 = 0
        # wait order
        # s0.wait(1)              # Q0@K0'
        # s1.wait(1)              # Q1@K0'
        # s0.wait(0), o0.wait(1)  # P0@V0
        # s1.wait(0), o1.wait(1)  # P1@V0

        while iter_count != 0:
            iter_count -= 1
            # Q_0 @ K_n'
            kn = pipeline_kv.wait_k[mma_stage=0]()  # kv_{2n-1}->[kv_{2n}]
            Self.UMMA0Type.mma(q0, kn, s0_tmem, elect=e, c_scale=0)
            elect_mma_arrive(producer_s0, e)

            # O_1 + P_1 @ V_{n-1}
            _ = consumer_o1[].wait(phase_o)
            # pipeline_o.acquire()
            _ = consumer_s1[].wait(phase_s)
            # pipeline_s1.acquire()
            Self.UMMA1Type.mma(
                s1_tmem, vlatest, o1_tmem, elect=e, c_scale=c_scale
            )
            elect_mma_arrive(producer_o1, e)
            # pipeline_o.step()
            phase_o = phase_s
            c_scale = 1
            pipeline_kv.release_v(e)  # [kv_{2n-1}]

            # Q_1 @ K_n'
            Self.UMMA0Type.mma(q1, kn, s1_tmem, elect=e, c_scale=0)
            elect_mma_arrive(producer_s1, e)
            phase_s ^= 1

            pipeline_kv.release_k(e)  # [kv_{2n}]->kv_{2n+1}

            # O_0 + P_0 @ V_n
            vlatest = pipeline_kv.wait_v[mma_stage=0]()  # [kv_{2n+1}]
            _ = consumer_o0[].wait(phase_o)
            # pipeline_o.acquire()
            _ = consumer_s0[].wait(phase_s)
            # pipeline_s0.acquire()
            Self.UMMA1Type.mma(s0_tmem, vlatest, o0_tmem, elect=e, c_scale=1)
            elect_mma_arrive(producer_o0, e)

        _ = consumer_o1[].wait(phase_o)
        _ = consumer_s1[].wait(phase_s)
        Self.UMMA1Type.mma(s1_tmem, vlatest, o1_tmem, elect=e, c_scale=c_scale)
        elect_mma_arrive(producer_o1, e)


@always_inline
fn mla_sm100_prefill[
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
    _is_cache_length_accurate: Bool,
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
        depth=Int(q_depth),
        dtype_size=size_of[q_type](),
        swizzle_mode=config.swizzle_mode,
        page_size=KVType.page_size,
        is_mla=True,
    )

    comptime SchedulerType = TransientScheduler[
        fa4_config.BM, fa4_config.num_q_heads
    ]
    comptime ValidLengthType = NonNullPointer[DType.uint32]
    comptime SinkType = NullPointer[output_type]
    comptime KVRowOffsetsNull = NullPointer[DType.uint32]
    comptime PartitionType = NoPartition[get_accum_type[q.dtype]()]
    var valid_len: ValidLengthType = {valid_length.ptr}

    comptime SM100MLAType = SM100MLA[
        KVType,
        output.dtype,
        MaskType,
        ScoreModType,
        SchedulerType,
        fa4_config,
        use_score_mod,
        ValidLengthType,
        SinkType,
        KVRowOffsetsNull,
        _is_cache_length_accurate,
        MaxPromptLenType,
        PartitionType,
    ]

    comptime RaggedStoreType = RaggedTMA3DTile[
        output_type,
        fa4_config.swizzle_mode,
        BM = fa4_config.BM // 2,
        BN = SM100MLAType.kv_depth,
    ]

    comptime kernel = SM100MLAType.mla_prefill_kernel[KRopeType]

    var num_rows_q = q_num_matrix_view_rows(q)

    q_tma_op = q_tma[
        fa4_config.swizzle_mode,
        BM = fa4_config.BM // 2,
        depth = fa4_config.depth,
        q_num_heads = fa4_config.num_q_heads,
        group = fa4_config.group,
        decoding=False,
    ](ctx, q.ptr, num_rows_q)

    # [batch_size * num_keys, num_heads, kv_depth]
    k_tma_op = k.create_tma_tile[
        fa4_config.swizzle_mode,
        BN = fa4_config.BN,
        depth = SM100MLAType.kv_depth,
    ](ctx)

    # [batch_size, num_keys, cache_num_heads, cache_depth]
    k_rope_tma_op = k_rope.create_tma_tile[
        TensorMapSwizzle.SWIZZLE_128B,
        BN = fa4_config.BN,
        depth=cache_depth,
        BK = SM100MLAType.k_rope_depth,
    ](ctx)

    # [batch_size * num_keys, num_heads, kv_depth]
    v_tma_op = v.create_tma_tile[
        fa4_config.swizzle_mode,
        BN = fa4_config.BN,
        depth = SM100MLAType.kv_depth,
    ](ctx)

    comptime out_depth = SM100MLAType.kv_depth

    var ragged_tma_store = RaggedStoreType.create(
        ctx, output.ptr, rows=num_rows_q, middle_dim=fa4_config.num_q_heads
    )

    comptime PackType = Pack[
        MaskType,
        ScoreModType,
        SchedulerType,
        ValidLengthType,
        SinkType,
        KVRowOffsetsNull,
        MaxPromptLenType,
        PartitionType,
    ]

    var pack: PackType = {
        mask_functor,
        score_mod_functor,
        SchedulerType(),
        valid_len,
        SinkType(),
        KVRowOffsetsNull(),
        max_prompt_len,
        PartitionType(),
    }

    var max_num_prompt_tiles: UInt32 = ceildiv(
        max_prompt_len.as_uint32(), fa4_config.BM
    )
    var num_key_blocks: UInt32 = (
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
        k,
        scale,
        UInt32(batch_size),
        UInt32(cache_depth),
        pack,
        grid_dim=SchedulerType.grid_dim(batch_size, num_key_blocks),
        block_dim=(Int(num_threads), 1, 1),
        shared_mem_bytes=Int(smem_use),
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(smem_use),
    )
