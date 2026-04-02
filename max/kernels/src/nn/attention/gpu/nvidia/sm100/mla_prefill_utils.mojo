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

from std.sys import simd_width_of, size_of

from nn.attention.mha_operand import MHAOperand
from nn.attention.mha_mask import MHAMask, TileMaskStatus
from nn.attention.gpu.nvidia.mha_tile_scheduler import MHATileScheduler, SeqInfo
from nn.attention.gpu.nvidia.sm100.attention import (
    FA4Config,
    EnableForcedOrdering,
)
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    SM100TensorAccumulatorSS,
    SM100TensorAccumulatorTS,
    FA4MiscMBars,
    SharedMemPointer,
    TMADestination,
    MBarType,
    elect,
    elect_mma_arrive,
)
from nn.attention.gpu.nvidia.sm90.attention import (
    OptionalPointer,
    MHAPosition,
)
from nn.attention.mha_utils import OptionallyStaticInt, MHAPartitionScheme

from layout import TileTensor
from layout.tma_async import PipelineState
from layout.swizzle import Swizzle
from layout.tile_layout import row_major as tt_row_major
from layout.tensor_core_async import (
    tile_layout_k_major_typed,
    tile_layout_mn_major_typed,
)

from linalg.arch.sm100.mma import smem_descriptor

from std.gpu.host.info import B200
from std.gpu.globals import WARP_SIZE
from std.gpu.memory import fence_async_view_proxy
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from std.gpu.compute.arch.mma_nvidia_sm100 import (
    MMASmemDescriptorPair,
    UMMAKind,
)
import std.gpu.primitives.warp as warp
from std.gpu.sync import named_barrier

from std.utils.index import Index


# rope_dtype is the dtype of the MMA!
struct MLAConfig[
    qkv_dtype: DType,
    *,
    rope_gmem_dtype: DType,
    rope_mma_dtype: DType,
    scale_dtype: DType = DType.invalid,
](TrivialRegisterPassable):
    var fa4_config: FA4Config[
        Self.qkv_dtype,
        rope_dtype=Self.rope_mma_dtype,
        scale_dtype=Self.scale_dtype,
    ]
    var MMA_M: Int
    var BM: Int
    var BN: Int
    var BK0: Int  # BK for MMA0
    var BK1: Int  # BK for MMA1
    var qk_depth: Int
    var rope_depth: Int
    var nope_depth: Int
    var cache_depth: Int
    var padded_qk_depth: Int  # align_up(k_depth, swizzle_elems)
    var group: Int
    var num_q_heads: Int
    var num_kv_heads: Int
    comptime TMEM_S0: Int = 0
    var TMEM_S1: Int
    var TMEM_O0: Int
    var TMEM_O1: Int
    var TMEM_P0: Int
    var TMEM_P1: Int
    var TMEM_C0: Int
    var TMEM_C1: Int
    var tmem_used: Int
    var num_kv_stages: Int
    var num_qk_stages: Int  # Stages for Q@K' (K loading pipelining)
    var num_pv_stages: Int  # Stages for P@V (P writing pipelining)
    var smem_used: Int
    comptime num_threads: Int = 512  # 2x softmax, 1x correction, 1x other
    var split_m: Bool
    var qkv_swizzle_mode: TensorMapSwizzle
    var rope_mma_swizzle_mode: TensorMapSwizzle
    var rope_gmem_swizzle_mode: TensorMapSwizzle
    var output_swizzle_mode: TensorMapSwizzle

    comptime qkv_dtype_size: Int = size_of[Self.qkv_dtype]()
    comptime rope_mma_dtype_size: Int = size_of[Self.rope_mma_dtype]()
    comptime rope_gmem_dtype_size: Int = size_of[Self.rope_gmem_dtype]()
    comptime sm100_smem_carveout = B200.shared_memory_per_multiprocessor - 1024
    comptime sm100_tmem_cols = 512
    comptime mbar_size = size_of[DType.int64]()
    comptime num_correction_cols = 1

    def __init__(
        out self,
        *,
        num_q_heads: Int,
        group: Int,
        depth: Int,
        page_size: Int,
    ):
        comptime if Self.qkv_dtype_size == 1:
            self.qkv_swizzle_mode = TensorMapSwizzle.SWIZZLE_64B
        else:
            self.qkv_swizzle_mode = TensorMapSwizzle.SWIZZLE_128B

        comptime if Self.rope_mma_dtype_size == 1:
            self.rope_mma_swizzle_mode = TensorMapSwizzle.SWIZZLE_64B
        else:
            self.rope_mma_swizzle_mode = TensorMapSwizzle.SWIZZLE_128B

        comptime if Self.rope_gmem_dtype_size == 1:
            self.rope_gmem_swizzle_mode = TensorMapSwizzle.SWIZZLE_64B
        else:
            self.rope_gmem_swizzle_mode = TensorMapSwizzle.SWIZZLE_128B

        self.output_swizzle_mode = TensorMapSwizzle.SWIZZLE_128B

        self.fa4_config = {
            num_q_heads = num_q_heads,
            group = group,
            qk_depth = depth,
            ov_depth = depth - 64,
            swizzle_mode = self.qkv_swizzle_mode,
            page_size = page_size,
            is_mla = True,
        }

        self.MMA_M = self.fa4_config.MMA_M
        self.BM = self.fa4_config.BM
        self.BN = self.fa4_config.BN
        self.BK0 = self.fa4_config.BK0
        self.BK1 = self.fa4_config.BK1
        self.qk_depth = self.fa4_config.qk_depth
        self.rope_depth = 64
        self.nope_depth = self.qk_depth - self.rope_depth
        self.cache_depth = 576
        self.padded_qk_depth = self.fa4_config.padded_qk_depth
        self.tmem_used = self.fa4_config.tmem_used
        self.num_kv_stages = self.fa4_config.num_kv_stages
        self.num_qk_stages = self.fa4_config.num_qk_stages
        self.num_pv_stages = self.fa4_config.num_pv_stages
        self.smem_used = self.fa4_config.smem_used
        self.split_m = self.fa4_config.split_m
        self.group = self.fa4_config.group
        self.num_q_heads = self.fa4_config.num_q_heads
        self.num_kv_heads = self.fa4_config.num_kv_heads
        self.TMEM_S1 = self.fa4_config.TMEM_S1
        self.TMEM_O0 = self.fa4_config.TMEM_O0
        self.TMEM_O1 = self.fa4_config.TMEM_O1
        self.TMEM_P0 = self.fa4_config.TMEM_P0
        self.TMEM_P1 = self.fa4_config.TMEM_P1
        self.TMEM_C0 = self.fa4_config.TMEM_C0
        self.TMEM_C1 = self.fa4_config.TMEM_C1

    @always_inline
    def num_qo(self) -> Int:
        return 2

    @always_inline
    def num_rope_buffers(self) -> Int:
        return self.fa4_config.num_rope_buffers()

    def supported(self) -> Bool:
        return (
            self.qk_depth >= 64
            and self.BN >= 64
            and self.num_kv_stages >= 2
            and self.tmem_used <= Self.sm100_tmem_cols
            and self.smem_used <= Self.sm100_smem_carveout
        )

    def correction_smem_elements(self) -> Int:
        return self.BM * Self.num_correction_cols

    def num_active_warps_per_group(self) -> Int:
        return 4

    def num_active_threads_per_group(self) -> Int:
        return WARP_SIZE * self.num_active_warps_per_group()


@always_inline
def split_smem[
    first_size: Int, second_size: Int, first_dtype: DType, second_dtype: DType
](tensor: TileTensor[address_space=AddressSpace.SHARED, ...]) -> Tuple[
    TileTensor[
        first_dtype,
        type_of(tt_row_major[first_size]()),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ],
    TileTensor[
        second_dtype,
        type_of(tt_row_major[second_size]()),
        MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ],
]:
    """Split a shared memory tensor into two TileTensors at the boundary
    of `first_size` elements.

    TMA only uses .ptr — flat row_major layout avoids needing
    InternalLayout equivalents of swizzled layouts.
    """
    comptime SmemPtr[dt: DType] = UnsafePointer[
        Scalar[dt], MutAnyOrigin, address_space=AddressSpace.SHARED
    ]
    var ptr = rebind[SmemPtr[first_dtype]](tensor.ptr)
    comptime first_layout = tt_row_major[first_size]()
    comptime second_layout = tt_row_major[second_size]()
    return {
        TileTensor[
            first_dtype,
            type_of(first_layout),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ](ptr, first_layout),
        TileTensor[
            second_dtype,
            type_of(second_layout),
            MutAnyOrigin,
            address_space=AddressSpace.SHARED,
        ](
            rebind[SmemPtr[second_dtype]](ptr + first_size),
            second_layout,
        ),
    }


struct MLAPositionSummary(TrivialRegisterPassable):
    var num_keys: UInt32
    var score_row: UInt32

    @always_inline
    def __init__(out self, num_keys: UInt32, score_row: UInt32):
        self.num_keys = num_keys
        self.score_row = score_row

    @staticmethod
    @always_inline
    def get_num_keys_and_start_pos[
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
    def get_score_row(seq_info: SeqInfo, start_pos: UInt32) -> UInt32:
        return start_pos + warp.broadcast(seq_info.prompt_offset)

    @staticmethod
    @always_inline
    def create[
        KRopeType: MHAOperand,
        //,
        _ndbuffer_mha_operand: Bool,
    ](k_rope_lut: KRopeType, seq_info: SeqInfo,) -> MLAPositionSummary:
        num_keys, start_pos = Self.get_num_keys_and_start_pos[
            _ndbuffer_mha_operand=_ndbuffer_mha_operand,
        ](k_rope_lut, seq_info)
        score_row = Self.get_score_row(seq_info, start_pos)
        return {num_keys, score_row}


struct MLAKVLayouts[
    k_nope_dtype: DType,
    k_rope_dtype: DType,
    kv_scale_dtype: DType,
    config: MLAConfig,
]:
    """Comptime layout and size metadata for MLA K/V tiles."""

    comptime k_nope_tma_layout = tile_layout_k_major_typed[
        Self.k_nope_dtype,
        Self.config.BN,
        128,
        Self.config.qkv_swizzle_mode,
    ].static_product
    comptime k_rope_tma_layout = tile_layout_k_major_typed[
        Self.k_rope_dtype,
        Self.config.BN,
        64,
        Self.config.rope_gmem_swizzle_mode,
    ].static_product
    comptime k_tma_layout = tile_layout_k_major_typed[
        Self.k_nope_dtype,
        Self.config.BN,
        Self.config.BK0,
        Self.config.qkv_swizzle_mode,
    ].static_product
    comptime v_tma_layout = tile_layout_mn_major_typed[
        Self.k_nope_dtype,
        128,
        Self.config.BK1,
        Self.config.qkv_swizzle_mode,
    ].static_product

    comptime KPairType = TMADestination[Self.k_nope_dtype, Self.k_tma_layout]
    comptime VPairType = TMADestination[Self.k_nope_dtype, Self.v_tma_layout]
    comptime k_elements = Self.k_nope_tma_layout + Self.k_rope_tma_layout
    comptime v_elements = Self.v_tma_layout
    comptime k_nope_bytes = Self.k_nope_tma_layout * size_of[
        Self.k_nope_dtype
    ]()
    comptime k_rope_bytes = Self.k_rope_tma_layout * size_of[
        Self.k_rope_dtype
    ]()
    comptime k_bytes = Self.k_nope_bytes + Self.k_rope_bytes
    comptime v_bytes = Self.v_elements * size_of[Self.k_nope_dtype]()
    comptime SMemType = SharedMemPointer[Scalar[Self.k_nope_dtype]]


struct TMAtoCvtPipeline[
    num_kv_stages: Int,
    num_producer: Int,
    num_consumer: Int,
](TrivialRegisterPassable):
    var consumer_mbars: MBarType
    var producer_mbars: MBarType
    var state: PipelineState[Self.num_kv_stages]

    @always_inline
    def __init__(out self, consumer_mbars: MBarType, producer_mbars: MBarType):
        self.consumer_mbars = consumer_mbars
        self.producer_mbars = producer_mbars
        self.state = {}

    @always_inline
    def init(self):
        comptime for i in range(Self.num_kv_stages):
            self.consumer_mbars[i].init(Int32(Self.num_consumer))
            self.producer_mbars[i].init(Int32(Self.num_producer))

    @always_inline
    def producer_mbar(self) -> MBarType:
        var idx: UInt32 = self.state.index()
        return self.producer_mbars + idx

    @always_inline
    def consumer_mbar(self) -> MBarType:
        var idx: UInt32 = self.state.index()
        return self.consumer_mbars + idx

    @always_inline
    def producer_acquire(self):
        self.consumer_mbar()[].wait(self.state.phase())

    @always_inline
    def consumer_wait(self):
        self.producer_mbar()[].wait(self.state.phase())

    @always_inline
    def producer_commit(mut self):
        _ = self.producer_mbar()[].arrive()
        self.step()

    @always_inline
    def consumer_release(mut self):
        _ = self.consumer_mbar()[].arrive()
        self.step()

    @always_inline
    def step(mut self):
        self.state.step()


struct CvtToMMAPipeline[
    num_stages: Int,
    num_producer: Int,
    num_consumer: Int,
](TrivialRegisterPassable):
    var producer_mbars: MBarType
    var consumer_mbars: MBarType
    var state: PipelineState[Self.num_stages]

    @always_inline
    def __init__(out self, producer_mbars: MBarType, consumer_mbars: MBarType):
        self.producer_mbars = producer_mbars
        self.consumer_mbars = consumer_mbars
        self.state = {}

    @always_inline
    def init(self):
        comptime for i in range(Self.num_stages):
            self.producer_mbars[i].init(Int32(Self.num_producer))
            self.consumer_mbars[i].init(Int32(Self.num_consumer))

    @always_inline
    def producer_mbar(self) -> MBarType:
        var idx: UInt32 = self.state.index()
        return self.producer_mbars + idx

    @always_inline
    def consumer_mbar(self) -> MBarType:
        var idx: UInt32 = self.state.index()
        return self.consumer_mbars + idx

    @always_inline
    def producer_acquire(self):
        self.consumer_mbar()[].wait(self.state.phase())

    @always_inline
    def consumer_wait(self):
        self.producer_mbar()[].wait(self.state.phase())

    @always_inline
    def producer_commit(mut self):
        _ = self.producer_mbar()[].arrive()
        self.step()

    @always_inline
    def consumer_release(mut self, elect: Int32):
        elect_mma_arrive(self.consumer_mbar(), elect)
        self.step()

    @always_inline
    def step(mut self):
        self.state.step()


@always_inline
def cvt_block_fp8_to_bf16_with_scale[
    input_type: DType,
    output_dtype: DType,
    KRopeType: MHAOperand,
    //,
    swizzle_fp8: Swizzle,
    swizzle_bf16: Swizzle,
](
    input: TileTensor[input_type, _, address_space=AddressSpace.SHARED, ...],
    mut output: TileTensor[
        mut=True, output_dtype, _, address_space=AddressSpace.SHARED, ...
    ],
    k_rope_lut: KRopeType,
    seq_info: SeqInfo,
    kv_start_row: UInt32,
    num_keys: UInt32,
    tid: UInt32,
):
    """TileTensor overload — standalone implementation using `.ptr` and
    comptime `static_shape`/`static_stride` directly."""
    comptime assert (
        input_type == DType.float8_e4m3fn and output_dtype == DType.bfloat16
    ), "Only support float8_e4m3fn to bfloat16 conversion"

    comptime num_regs = (
        type_of(input).static_shape[0] * type_of(input).static_shape[1]
    ) // WARP_SIZE
    comptime row_stride = type_of(input).static_stride[0]

    var t_row, t_col = divmod(tid, 16)

    var fp8_regs = SIMD[input_type, num_regs](0)

    comptime for i in range(num_regs // 4):
        var row = UInt32(i * 2) + t_row
        var col = t_col * 4
        var elem_offset = row * UInt32(row_stride) + col
        var fp8x4 = (input.ptr + Int(swizzle_fp8(elem_offset))).load[width=4]()
        fp8_regs = fp8_regs.insert[offset=i * 4](fp8x4)

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

            var fp32x4 = fp8_regs.slice[4, offset=i * 4]().cast[
                KRopeType.scale_dtype
            ]()
            fp32x4 = fp32x4 * scale
            (output.ptr + Int(swizzle_bf16(elem_offset))).store[width=4](
                fp32x4.cast[output_dtype]()
            )
        else:
            var fp16x4 = fp8_regs.slice[4, offset=i * 4]().cast[output_dtype]()
            (output.ptr + Int(swizzle_bf16(elem_offset))).store[width=4](fp16x4)

    fence_async_view_proxy()


struct SM100MLA[
    KVLUTType: MHAOperand,
    KRopeType: MHAOperand,
    output_dtype: DType,
    MaskType: MHAMask,
    SchedulerType: MHATileScheduler,
    config: MLAConfig,
    ValidLengthType: OptionalPointer,
    SinkType: OptionalPointer,
    KVRowOffsetsType: OptionalPointer,
    MaxSeqLenType: OptionallyStaticInt,
    PartitionType: MHAPartitionScheme,
    _ndbuffer_mha_operand: Bool,
](TrivialRegisterPassable):
    comptime qkv_dtype = Self.KVLUTType.dtype
    comptime rope_mma_dtype = Self.config.rope_mma_dtype
    comptime rope_gmem_dtype = Self.KRopeType.dtype
    comptime accum_dtype = DType.float32
    comptime simd_size: Int = simd_width_of[Self.qkv_dtype]()

    comptime cta_group = 1  # TODO: support 2
    comptime BM = Self.config.BM
    comptime BN = Self.config.BN
    comptime qk_depth = Self.config.qk_depth  # 192
    comptime padded_depth = Self.config.padded_qk_depth  # 192
    comptime num_q_heads = Self.config.num_q_heads
    comptime group = Self.config.group
    comptime page_size = Self.KVLUTType.page_size

    comptime rope_depth = Self.config.rope_depth
    comptime nope_depth = Self.config.nope_depth
    comptime cache_depth = Self.config.cache_depth

    comptime num_m_mmas = 2
    comptime MMA_M = Self.config.BM // Self.num_m_mmas
    comptime qkv_dt_size = size_of[Self.qkv_dtype]()

    comptime num_qk_stages = Self.config.num_qk_stages
    comptime num_pv_stages = Self.config.num_pv_stages

    comptime nope_mma_kind = (
        UMMAKind.KIND_F16 if Self.qkv_dtype.is_half_float() else UMMAKind.KIND_F8F6F4
    )
    comptime rope_mma_kind = (
        UMMAKind.KIND_F16 if Self.rope_mma_dtype.is_half_float() else UMMAKind.KIND_F8F6F4
    )
    # use_fused_kv means we use a fused kv pipeline in shared memory
    # that forces us to put the k nope and rope in separate regions of smem
    # preventing us from fusing the nope and rope parts of UMMA0
    comptime fused_umma0 = (Self.qkv_dtype == Self.rope_mma_dtype) and (
        not Self.config.fa4_config.use_fused_kv
    )
    comptime BK0 = Self.qk_depth if Self.fused_umma0 else Self.nope_depth

    # First MMA is Q@K' (can be staged by num_qk_stages)
    # (BM x depth) @ (BN x depth)' -> (BM x BN)
    comptime UMMA0Type = SM100TensorAccumulatorSS[
        Self.qkv_dtype,
        Self.accum_dtype,
        MMA_M=Self.MMA_M,  # generally 128
        MMA_N=Self.BN,
        BK=Self.BK0,  # BK in memory depth
        mma_kind=Self.nope_mma_kind,
        swizzle_a=Self.config.qkv_swizzle_mode,
        swizzle_b=Self.config.qkv_swizzle_mode,
        transpose_b=True,
        num_stages=Self.num_qk_stages,
    ]
    comptime UMMA0RopeType = SM100TensorAccumulatorSS[
        Self.rope_mma_dtype,
        Self.accum_dtype,
        MMA_M=Self.MMA_M,
        MMA_N=Self.BN,
        BK=Self.rope_depth,
        mma_kind=Self.rope_mma_kind,
        swizzle_a=Self.config.rope_mma_swizzle_mode,
        swizzle_b=Self.config.rope_mma_swizzle_mode,
        transpose_b=True,
        num_stages=Self.num_qk_stages,
    ]
    # Second MMA is P@V
    # (BM x BN) @ (BN x depth) -> (BM x depth)
    comptime UMMA1Type = SM100TensorAccumulatorTS[
        Self.qkv_dtype,
        Self.accum_dtype,
        MMA_M=Self.MMA_M,
        MMA_N=Self.nope_depth,  # 128
        BK=Self.BN,
        mma_kind=Self.nope_mma_kind,
        swizzle_b=Self.config.qkv_swizzle_mode,
        transpose_b=False,
        num_stages=Self.num_pv_stages,
    ]

    # Byte offset within Q's smem tile where Q_rope columns begin.
    # Q is stored as tile_layout_k_major(BM/2, BK0), column-major atoms.
    # Q_nope occupies (BM/2) * padded_v_depth elements, then Q_rope follows.
    comptime q_rope_byte_offset: Int = (
        Self.MMA_M
        * Self.config.fa4_config.padded_ov_depth
        * size_of[Self.qkv_dtype]()
    )

    comptime PositionType = MHAPosition[
        Self.config.BM,
        Self.config.BN,
        Self.config.qk_depth,
        Self.config.padded_qk_depth,
        Self.config.num_q_heads,
        Self.config.group,
        False,
    ]
    # Unified misc barriers type managing all barriers including KV/O pipelines.
    # Use fa4_config fields so that the type expression matches
    # SM100AttentionSMem[config.fa4_config, ...].MiscMBarsType.
    comptime MiscMBarsType = FA4MiscMBars[
        num_qk_stages=Self.config.fa4_config.num_qk_stages,
        num_pv_stages=Self.config.fa4_config.num_pv_stages,
        num_kv_stages=Self.config.fa4_config.num_kv_stages,
        use_order_barriers=EnableForcedOrdering,
        use_fused_kv=Self.config.fa4_config.use_fused_kv,
    ]

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
            Index[dtype=DType.int32](Self.BM, Self.BN),
        )

    @staticmethod
    @always_inline
    def descriptor_q(
        q_smem: SharedMemPointer[Scalar[Self.qkv_dtype]],
    ) -> MMASmemDescriptorPair:
        return smem_descriptor[
            BMN=Self.config.BM // 2,
            BK=Self.config.nope_depth,
            swizzle_mode=Self.config.qkv_swizzle_mode,
            is_k_major=True,
        ](q_smem)

    @always_inline
    @staticmethod
    def descriptor_q_rope(
        q_smem: SharedMemPointer[Scalar[Self.rope_mma_dtype]],
    ) -> MMASmemDescriptorPair:
        return smem_descriptor[
            BMN=Self.config.BM // 2,
            BK=Self.config.rope_depth,
            swizzle_mode=Self.config.rope_mma_swizzle_mode,
            is_k_major=True,
        ](q_smem)
