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
from math.constants import log2e
from math import exp2, recip

from nn.mha_operand import MHAOperand
from nn.mha_mask import MHAMask, TileMaskStatus, MaskStrategy
from nn.mha_tile_scheduler import MHATileScheduler, SeqInfo
from nn.mha_sm100_2q import (
    FA4Config,
    SM100TensorAccumulatorSS,
    SM100TensorAccumulatorTS,
    LocalTensor,
    KVPipeline,
    FA4MiscMBars,
    SharedMemPointer,
    SharedMemLT,
    TMemTile,
    STMatrixLayout,
    TMADestination,
    MBarType,
    apply_mask,
    sub_ftz,
    elect,
    elect_mma_arrive,
)
from nn.mha_fa3_utils import (
    OptionalPointer,
    MHAPosition,
    output_reg_to_smem_st_matrix,
    _LocalTT,
    _SharedMemTT,
)
from nn.mha_utils import OptionallyStaticInt, MHAPartitionScheme

from layout.layout import Layout
from layout.layout_tensor import LayoutTensor
from layout.tma_async import RaggedTMA3DTile, PipelineState
from layout.tile_tensor import stack_allocation as tt_stack_allocation
from layout._layout import row_major
from layout.swizzle import make_swizzle, Swizzle
from layout.tensor_core_async import tile_layout_k_major, tile_layout_mn_major

from linalg.arch.sm100.mma import smem_descriptor

from gpu.globals import WARPGROUP_SIZE, WARP_SIZE
from gpu.memory import fence_async_view_proxy
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu import thread_idx
from gpu.compute.arch.tcgen05 import *
from gpu.compute.arch.mma_nvidia_sm100 import MMASmemDescriptorPair
from gpu.primitives.warp import _vote_nvidia_helper
import gpu.primitives.warp as warp
from gpu.sync import (
    named_barrier,
    cp_async_bulk_commit_group,
    cp_async_bulk_wait_group,
)

from utils.static_tuple import StaticTuple
from utils.index import Index


@always_inline
fn split_smem[
    first: Layout, second: Layout, first_dtype: DType, second_dtype: DType
](tensor: SharedMemLT) -> Tuple[
    SharedMemLT[first_dtype, first], SharedMemLT[second_dtype, second]
]:
    comptime first_size = first.size()
    var ptr = tensor.ptr.bitcast[Scalar[first_dtype]]()
    return {
        SharedMemLT[first_dtype, first](ptr),
        SharedMemLT[second_dtype, second](
            (ptr + first_size).bitcast[Scalar[second_dtype]]()
        ),
    }


struct MLAPositionSummary(TrivialRegisterPassable):
    var num_keys: UInt32
    var score_row: UInt32

    @always_inline
    fn __init__(out self, num_keys: UInt32, score_row: UInt32):
        self.num_keys = num_keys
        self.score_row = score_row

    @staticmethod
    @always_inline
    fn get_num_keys_and_start_pos[
        KRopeType: MHAOperand,
        //,
        _ndbuffer_mha_operand: Bool,
    ](k_rope_lut: KRopeType, seq_info: SeqInfo) -> Tuple[UInt32, UInt32]:
        comptime if _ndbuffer_mha_operand:
            num_keys = UInt32(
                warp.broadcast(
                    k_rope_lut.cache_length(Int(seq_info.prompt_idx))
                )
            )
            start_pos = UInt32(num_keys - warp.broadcast(seq_info.seq_len))
        else:
            start_pos = UInt32(
                warp.broadcast(
                    k_rope_lut.cache_length(Int(seq_info.prompt_idx))
                )
            )
            num_keys = start_pos + warp.broadcast(seq_info.seq_len)
        return {num_keys, start_pos}

    @staticmethod
    @always_inline
    fn get_score_row(seq_info: SeqInfo, start_pos: UInt32) -> UInt32:
        return start_pos + warp.broadcast(seq_info.prompt_offset)

    @staticmethod
    @always_inline
    fn create[
        KRopeType: MHAOperand,
        //,
        _ndbuffer_mha_operand: Bool,
    ](k_rope_lut: KRopeType, seq_info: SeqInfo,) -> MLAPositionSummary:
        num_keys, start_pos = Self.get_num_keys_and_start_pos[
            _ndbuffer_mha_operand=_ndbuffer_mha_operand,
        ](k_rope_lut, seq_info)
        score_row = Self.get_score_row(seq_info, start_pos)
        return {num_keys, score_row}


struct MLAKVProducerPipeline[
    k_nope_dtype: DType, k_rope_dtype: DType, config: FA4Config
](TrivialRegisterPassable):
    comptime k_rope_swizzle_mode = (
        TensorMapSwizzle.SWIZZLE_64B if Self.k_rope_dtype
        == DType.float8_e4m3fn else TensorMapSwizzle.SWIZZLE_128B
    )
    comptime k_nope_tma_layout = tile_layout_k_major[
        Self.k_nope_dtype,
        Self.config.BN,
        128,
        Self.config.swizzle_mode,
    ]()
    comptime k_rope_tma_layout = tile_layout_k_major[
        Self.k_rope_dtype,
        Self.config.BN,
        64,
        Self.k_rope_swizzle_mode,
    ]()
    comptime k_tma_layout = tile_layout_k_major[
        Self.k_nope_dtype,
        Self.config.BN,
        Self.config.BK0,
        Self.config.swizzle_mode,
    ]()
    comptime v_tma_layout = tile_layout_mn_major[
        Self.k_nope_dtype,
        128,
        Self.config.BK1,
        Self.config.swizzle_mode,
    ]()

    comptime KType = SharedMemLT[Self.k_nope_dtype, Self.k_tma_layout]
    comptime VType = SharedMemLT[Self.k_nope_dtype, Self.v_tma_layout]
    comptime KPairType = TMADestination[Self.k_nope_dtype, Self.k_tma_layout]
    comptime VPairType = TMADestination[Self.k_nope_dtype, Self.v_tma_layout]
    comptime k_nope_elements = Self.k_nope_tma_layout.size()
    comptime k_rope_elements = Self.k_rope_tma_layout.size()
    comptime k_elements = Self.k_nope_elements + Self.k_rope_elements
    comptime v_elements = Self.v_tma_layout.size()
    comptime k_nope_bytes = Self.k_nope_elements * size_of[Self.k_nope_dtype]()
    comptime k_rope_bytes = Self.k_rope_elements * size_of[Self.k_rope_dtype]()
    comptime k_bytes = Self.k_nope_bytes + Self.k_rope_bytes
    comptime v_bytes = Self.v_elements * size_of[Self.k_nope_dtype]()
    comptime SMemType = SharedMemPointer[Scalar[Self.k_nope_dtype]]

    var kv_pipeline: KVPipeline[
        Self.config.num_kv_stages, Self.config.num_qk_stages
    ]
    var smem: Self.SMemType

    @always_inline
    fn __init__(
        out self,
        mbar: MBarType,
        smem: Self.SMemType,
    ):
        comptime assert (
            Self.config.padded_depth % Self.config.num_qk_stages == 0
        )
        comptime assert Self.config.BN % Self.config.num_qk_stages == 0
        self.kv_pipeline = {mbar}
        self.smem = smem
        self.kv_pipeline.state._phase = 1

    @always_inline
    fn __init__(
        out self,
        kv_pipeline: KVPipeline[
            Self.config.num_kv_stages, Self.config.num_qk_stages
        ],
        smem: Self.SMemType,
    ):
        comptime assert (
            Self.config.padded_depth % Self.config.num_qk_stages == 0
        )
        comptime assert Self.config.BN % Self.config.num_qk_stages == 0
        self.kv_pipeline = kv_pipeline
        self.smem = smem
        self.kv_pipeline.state._phase = 1

    @always_inline
    fn get_kv_smem[*, qk_stage: Int](self) -> Self.SMemType:
        comptime stage_offset = qk_stage * Self.config.padded_depth * Self.config.BN
        var dyn_offset: UInt32 = (
            UInt32(Self.k_elements) * self.kv_pipeline.state.index()
        )
        return self.smem + stage_offset + dyn_offset

    @always_inline
    fn get_k[*, qk_stage: Int, expect: Bool = True](self) -> Self.KPairType:
        p_mbar = self.kv_pipeline.producer_mbar[qk_stage=qk_stage]()

        comptime if expect:
            p_mbar[].expect_bytes(Int32(Self.k_bytes))
        return {p_mbar, {self.get_kv_smem[qk_stage=qk_stage]()}}

    @always_inline
    fn get_v[*, qk_stage: Int](self) -> Self.VPairType:
        p_mbar = self.kv_pipeline.producer_mbar[qk_stage=qk_stage]()
        p_mbar[].expect_bytes(Int32(Self.v_bytes))
        return {p_mbar, {self.get_kv_smem[qk_stage=qk_stage]()}}

    @always_inline
    fn acquire_kv[*, qk_stage: Int = Self.config.num_qk_stages - 1](self):
        self.kv_pipeline.producer_acquire[qk_stage]()

    @always_inline
    fn commit_kv_step(mut self):
        """
        Step the kv pipeline. The does not perform the commit on the mbars;
        that should be handled by the `tma_op.async_copy`.
        """
        self.kv_pipeline.state.step()


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
fn cvt_block_fp8_to_bf16_with_scale[
    input_type: DType,
    output_type: DType,
    KRopeType: MHAOperand,
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
    k_rope_lut: KRopeType,
    seq_info: SeqInfo,
    kv_start_row: UInt32,
    num_keys: UInt32,
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

        comptime if KRopeType.quantization_enabled:
            var tok_idx = kv_start_row + row
            if tok_idx < num_keys:
                scale = k_rope_lut.load_scale[width=1](
                    batch_idx=Int(seq_info.prompt_idx),
                    start_tok_idx=Int(tok_idx),
                    head_idx=0,
                    head_dim_idx=576 - 64,
                )
            else:
                scale = SIMD[KRopeType.scale_dtype, 1](1)

            var fp32x4 = fp8_regs.slice[4, offset = i * 4]().cast[
                KRopeType.scale_dtype
            ]()
            fp32x4 = fp32x4 * scale
            (output.ptr + Int(swizzle_bf16(elem_offset))).store[width=4](
                fp32x4.cast[output_type]()
            )
        else:
            var fp16x4 = fp8_regs.slice[4, offset = i * 4]().cast[output_type]()
            (output.ptr + Int(swizzle_bf16(elem_offset))).store[width=4](fp16x4)

    fence_async_view_proxy()


struct SM100MLA[
    KVLUTType: MHAOperand,
    KRopeType: MHAOperand,
    output_type: DType,
    MaskType: MHAMask,
    SchedulerType: MHATileScheduler,
    config: FA4Config,
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
        False,
    ]
    # Unified misc barriers type managing all barriers including KV/O pipelines
    comptime MiscMBarsType = FA4MiscMBars[
        num_qk_stages = Self.config.num_qk_stages,
        num_pv_stages = Self.config.num_pv_stages,
        num_kv_stages = Self.config.num_kv_stages,
        separate_kv=False,  # MLA uses unified KV pipeline
    ]

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

        comptime if not Self.MaskType.apply_log2e_after_mask:
            scale_log2e *= log2e

        @parameter
        @always_inline
        fn mask_row[
            BN: Int, //, mask_strategy: MaskStrategy
        ](s: LocalTensor[Self.accum_type, row_major[BN]()], kv_row: UInt32,):
            apply_mask[mask_strategy=mask_strategy](
                s,
                mask,
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

            BatchTileType(p_tmem).store_async(
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
                BatchTileType(p_tmem + UInt32(tmem_offset)).store_async(
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
                RemainderTileType(p_tmem + UInt32(tmem_offset)).store_async(
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
    fn descriptor_q(
        q_smem: SharedMemPointer[Scalar[Self.qkv_type]],
    ) -> MMASmemDescriptorPair:
        return smem_descriptor[
            BMN = Self.config.BM // 2,
            BK = Self.config.BK0,
            swizzle_mode = Self.config.swizzle_mode,
            is_k_major=True,
        ](q_smem)
