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

from math import exp2, recip, align_up, log2, ceildiv
from math.constants import log2e
from sys import size_of, _RegisterPackType
import gpu.primitives.warp as warp
from gpu import (
    barrier,
    thread_idx,
    block_idx,
    warp_id,
)
from gpu.globals import WARPGROUP_SIZE
from gpu.host import DeviceContext
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu.host.info import B200
from gpu.memory import AddressSpace, external_memory, fence_async_view_proxy
from gpu.primitives.grid_controls import launch_dependent_grids
from gpu.compute.arch.mma_nvidia_sm100 import (
    UMMAInsDescriptor,
    UMMAKind,
)

from gpu.sync import (
    named_barrier,
)
from gpu.compute.arch.tcgen05 import (
    tcgen05_fence_after,
    tcgen05_ld,
    tcgen05_load_wait,
    tcgen05_st,
)
from gpu.primitives.warp import _vote_nvidia_helper
from gpu.compute.arch.mma_nvidia_sm100 import MMASmemDescriptorPair
from layout.int_tuple import IntTuple, UNKNOWN_VALUE
from layout.layout import (
    Layout,
)
from logger import Logger

from layout.layout_tensor import (
    LayoutTensor,
)
from layout.swizzle import make_ldmatrix_swizzle
from layout.tensor_core_async import (
    tile_layout_k_major,
    tile_layout_mn_major,
)
from layout.tma_async import (
    create_tensor_tile,
    PipelineState,
    SharedMemBarrier,
    _tma_desc_tile_layout,
    TMATensorTile,
)
from layout.runtime_layout import RuntimeLayout
from memory import bitcast
from nn.mha_fa3_utils import (
    OptionalPointer,
    Pack,
)
from nn.mha_mask import MHAMask
from nn.mha_operand import MHAOperand
from nn.mha_score_mod import ScoreModTrait
from gpu.host.nvidia.tma import TensorMapSwizzle
from utils.index import Index, IndexList
from utils.numerics import get_accum_type, min_or_neg_inf
from utils.static_tuple import StaticTuple
from linalg.arch.sm100.mma import smem_descriptor

from nn.mha_sm100_2q import (
    elect,
    KVPipeline,
    TMemTile,
    LocalTensor,
    elect_mma_arrive,
    ProducerPipeline,
    ConsumerPipeline,
    MBarPipeline,
    sub_ftz,
)
from layout._layout import row_major
from layout._tile_tensor import stack_allocation as tt_stack_allocation
from nn.mha_fa3_utils import KVTMATile
from builtin.device_passable import DevicePassable
from sys._assembly import inlined_assembly

comptime logger = Logger()

# ------------------------------------------------------------------------------
# Helper functions for MLA decoding TMA tiles
# ------------------------------------------------------------------------------

comptime QOTMATile[
    dtype: DType, BM: Int, BK: Int, swizzle_mode: TensorMapSwizzle
] = TMATensorTile[
    dtype,
    tile_layout_k_major[dtype, BM, BK, swizzle_mode=swizzle_mode](),
    _tma_desc_tile_layout[dtype, 2, IndexList[2](BM, BK), swizzle_mode](),
    is_k_major=True,
]


@always_inline
fn tma_tile_qo[
    dtype: DType,
    //,
    swizzle_mode: TensorMapSwizzle,
    *,
    BM: Int,
    BK: Int,
    depth: Int,
](
    ctx: DeviceContext,
    ptr: UnsafePointer[Scalar[dtype]],
    rows: Int,
    out res: QOTMATile[dtype, BM, BK, swizzle_mode],
) raises:
    comptime layout = Layout.row_major(UNKNOWN_VALUE, depth)
    var rt_layout = RuntimeLayout[layout].row_major(IndexList[2](rows, depth))
    var tensor = LayoutTensor[dtype, layout, MutAnyOrigin](ptr, rt_layout)

    res = rebind[QOTMATile[dtype, BM, BK, swizzle_mode]](
        create_tensor_tile[
            IndexList[2](BM, BK),
            swizzle_mode=swizzle_mode,
        ](ctx, tensor)
    )


# ------------------------------------------------------------------------------
# Helper functions for MLA decoding pack
# ------------------------------------------------------------------------------


struct MLA_Decode_Pack[
    ValidLengthType: OptionalPointer,
    MaskType: MHAMask,
    ScoreModType: ScoreModTrait,
    SplitAccumType: OptionalPointer,
](Copyable, DevicePassable, TrivialRegisterPassable):
    var mask: Self.MaskType
    var score_mod: Self.ScoreModType
    var valid_length: Self.ValidLengthType
    var lse_accum_split_ptr: Self.SplitAccumType
    comptime device_type: AnyType = Self

    fn _to_device_type(self, target: MutOpaquePointer[_]):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        return "Pack"

    @staticmethod
    fn get_device_type_name() -> String:
        return Self.get_type_name()

    @always_inline
    fn __init__(
        out self,
        mask: Self.MaskType,
        score_mod: Self.ScoreModType,
        valid_length: Self.ValidLengthType,
        lse_accum_split_ptr: Self.SplitAccumType,
    ):
        self.mask = mask
        self.score_mod = score_mod
        self.valid_length = valid_length
        self.lse_accum_split_ptr = lse_accum_split_ptr


# ------------------------------------------------------------------------------
# MLA decoding implementation for SM100
# ------------------------------------------------------------------------------


@always_inline
fn num_matrix_view_rows_decode[
    dtype: DType,
    //,
](q: LayoutTensor[dtype, ...]) -> Int:
    # q and output are (batch x seq_len x num_heads , depth)
    # output when split-k is used are (split_k x batch x seq_len x num_heads , depth)
    var num_rows: Int = q.dim[0]()

    @parameter
    for i in range(1, q.rank - 1):
        num_rows *= q.dim[i]()
    return num_rows


# ------------------------------------------------------------------------------
# Shared memory types for SM100
# ------------------------------------------------------------------------------
comptime SharedMemPointer[type: AnyType] = UnsafePointer[
    type, address_space = AddressSpace.SHARED, origin=MutAnyOrigin
]

comptime MBarType = SharedMemPointer[SharedMemBarrier]

comptime SharedMemTensor[dtype: DType, layout: Layout] = LayoutTensor[
    dtype,
    layout,
    MutAnyOrigin,
    address_space = AddressSpace.SHARED,
    layout_int_type = DType.int32,
    linear_idx_type = DType.int32,
    alignment=128,
]


# ------------------------------------------------------------------------------
# MLA decoding configuration for SM100
# ------------------------------------------------------------------------------
struct MLA_SM100_Decode_Config:
    var MMA_M: Int
    var MMA_PV_N: Int
    var MMA_QK_N: Int
    var BM: Int
    var BN: Int
    var BK0: Int  # BK for MMA0
    var BK1: Int  # BK for MMA1
    var q_depth: Int
    var depth: Int  # this is V depth
    var padded_depth: Int
    var padded_q_depth: Int
    var rope_depth: Int  # this is Q depth - V depth
    var group: Int
    var num_q_heads: Int
    var num_kv_heads: Int
    comptime TMEM_O: Int = 0
    comptime TMEM_S0: Int = Self.TMEM_O + 256
    comptime TMEM_S1: Int = Self.TMEM_S0 + 32
    comptime TMEM_CORR_SCALE: Int = Self.TMEM_S1 + 32
    comptime TMEM_CORR_LI: Int = Self.TMEM_CORR_SCALE + 1
    var tmem_used: Int
    var num_kv_stages: Int
    var smem_used: Int
    var dtype_size: Int
    var num_threads: Int  # 1x softmax, 1x correction, 1x other
    var swizzle_mode: TensorMapSwizzle
    var kv_mma_swizzle_mode: TensorMapSwizzle
    var kv_tma_swizzle_mode: TensorMapSwizzle
    comptime MMA_K = 16
    comptime sm100_smem_carveout = B200.shared_memory_per_multiprocessor - 1024
    comptime sm100_tmem_cols = 512
    comptime mbar_size = size_of[DType.int64]()  # 8
    comptime cta_group = 1  # TODO: support 2
    var decoding_warp_split_k: Bool
    var out_rows: Int

    fn __init__(
        out self,
        *,
        num_q_heads: Int,
        group: Int,
        depth: Int,
        q_depth: Int,
        dtype_size: Int,
        kv_type_size: Int,
        swizzle_mode: TensorMapSwizzle,
        kv_mma_swizzle_mode: TensorMapSwizzle,
        page_size: Int,
        decoding_warp_split_k: Bool,
    ):
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_q_heads // group
        self.group = group
        self.depth = depth
        self.q_depth = q_depth
        self.rope_depth = q_depth - depth

        self.BM = 64
        self.MMA_M = 64
        self.MMA_PV_N = 256
        self.MMA_QK_N = 64
        self.BN = 64  # This can be increased since we are not doing sparse

        self.dtype_size = dtype_size
        self.swizzle_mode = swizzle_mode
        self.kv_mma_swizzle_mode = kv_mma_swizzle_mode
        swizzle_elems = swizzle_mode.bytes() // dtype_size
        self.padded_depth = align_up(depth, swizzle_elems)
        self.padded_q_depth = align_up(q_depth, swizzle_elems)

        self.kv_tma_swizzle_mode = (
            TensorMapSwizzle.SWIZZLE_64B if kv_type_size
            == 1 else TensorMapSwizzle.SWIZZLE_128B
        )
        self.num_threads = 128 * Int(4 if kv_type_size == 1 else 3)

        # 4 bytes for the TMEM base pointer
        var smem_use = 4
        self.tmem_used = self.TMEM_S0 + 32
        self.decoding_warp_split_k = decoding_warp_split_k
        self.BK0 = self.padded_q_depth
        self.BK1 = self.BN
        #  Here we can replace the GCD with MIN as the num_q_heads is always power of two.
        # self.out_rows = gcd(self.BM, self.num_q_heads)
        self.out_rows = min(self.BM, self.num_q_heads)
        # to store Q we need(64x576x2)  = 73728 bytes
        smem_use += self.BM * self.padded_q_depth * dtype_size
        # we have two scratch buffer float 32 for max and li
        # 128 is the warpgroup size and 4 bytes for float 32
        #  (128 * 1 * 4 * 2) =1024 bytes
        # Since I have 102k extra slot here for correction I will create another slot to let
        # softmax smoothly use its internal state max and li
        comptime smem_for_max_and_li = 128 * 1 * 4 * 2
        # 4 + (64x576x2) + (128 * 1 * 4 * 2) = 74756 bytes
        smem_use += smem_for_max_and_li
        # we need BMxBN x dtype bites for storing the P matrix in smem
        # (64x64x2) = 8192 bytes
        var smem_for_out = self.BM * self.BN * dtype_size
        # 4 + (64x576x2) + (64x64x2) + (128 * 1 * 4 * 2) = 82948 bytes
        smem_use += smem_for_out
        # to store K/V we need the bigger size which is K for storing here
        # so we have (64x576x2) = 73728 bytes
        var smem_per_kv = (
            self.BN * self.padded_q_depth * dtype_size
        )  # (two slot buffer for k/v)
        # now we need to calcuate howmany slot per K /V we can fit in the remaining memory
        # so far we have
        # 4 + (64x576x2) + (64x64x2) + (128 * 1 * 4 * 2) = 82948 bytes
        # the curveout require 1k for l1 cache so
        # for b200 we have sm100_smem_carveout 233472 - 1024 =  232448 bytes
        # remaining smem  = 232448 - 82948 = 149500 bytes
        # so we can fit 149500 // 73728 = 2 slots per K/V and still have 2044
        # bytes left for  barriers and other stuff
        self.num_kv_stages = (
            Self.sm100_smem_carveout - smem_use
        ) // smem_per_kv
        smem_use += self.num_kv_stages * (smem_per_kv)
        # We have the following resources that need smem barriers:
        # num_kv_stages = 2, so:

        # bar_write_done[1] → 8  producer pipeline- softmax
        # bar_write_ready[1] → 8  consumer pipeline - write
        var num_out_barrier = (self.depth // self.BN) * 2
        # total number of barriers is fixed_transaction_barriers + num_out_barrier
        # bar_q → 1           producer pipeline - load consumer - mma
        # bar_kv_reay[2] → 2  consumer pipeline - mma
        # bar_kv_free[2] → 2   producer pipeline - load
        # bar_s_done[2] → 2  producer pipeline - mma
        # bar_s_ready[2] → 2  consumer pipeline - softmax
        # bar_p_done[2] → 2  producer pipeline- softmax
        # bar_p_ready[2] → 2  consumer pipeline - mma
        # bar_correction_done[1] → 1  producer pipeline- softmax
        # bar_correction_ready[1] → 1  consumer pipeline - correction
        # bar_o_done[1] → 2  producer pipeline- MMA PV
        # bar_o_ready[1] → 2  consumer pipeline - Correction
        # corr_done_bars[2] → 2  producer pipeline- correction consumer softmax

        # Hence fixed_transaction_barriers: 1 + 2 + 2 + 2 + 2  +2 + 2  + 2 + 2 + 2 + 2 + 2 + 1 = 23.
        # If fp8, then add 4 more barriers for the convert fp8 to bf16 pipeline
        # bar_convert_done[2] → 2  producer pipeline- Convert fp8 to bf16
        # bar_convert_ready[2] → 2  consumer pipeline - mma
        # Hence fixed_transaction_barriers +2 + 2 = 27.
        smem_use += (
            (27 if kv_type_size == 1 else 23) + num_out_barrier
        ) * Self.mbar_size + (
            ((self.depth // self.BN) - 1) * 2 * Self.mbar_size
        )

        # for convinence we will add the smem for max and li to the smem_used for
        # correction separately (128 * 1 * 4 * 2) =1024 bytes
        # comptime smem_for_max_and_li_correction = 128 * 1 * 4 * 2
        # 4 + (64x576x2) + (128 * 1 * 4 * 2) = 74756 bytes
        # smem_use += smem_for_max_and_li_correction

        # 4 + (64x576x2) + (64x576x2x2) + (64x64x2) + (128 * 1 * 4 * 2) + (19x8)  = 230556 bytes
        # we need to add ((Depth/BN) -1) x2
        # more barriers. for the splitk the two added is enough for now.
        # 230556 + ((512/64) -1) x2 x8 = 230556 + 15 x16 = 230556 + 240 = 230796 bytes
        # hence the remaining memory is 232448 - 230796 = 1652 bytes
        # in split K the barrier number reduced to 22
        # hence the remaining memory is 232448 - 230556 = 1892 bytes
        self.smem_used = smem_use

    fn supported(self) -> Bool:
        return (
            self.q_depth == 576
            and self.BN == 64
            and self.BM == 64
            and self.depth == 512
            and self.num_kv_stages == 2
            and self.tmem_used <= Self.sm100_tmem_cols
            and self.smem_used <= Self.sm100_smem_carveout
        )


# ------------------------------------------------------------------------------
# Offset position struct
# ------------------------------------------------------------------------------
struct OffsetPosition[
    config: MLA_SM100_Decode_Config,
    KVLUTType: MHAOperand,
    ragged: Bool,
    is_cache_length_accurate: Bool,
    ValidLengthType: OptionalPointer,
    decoding_warp_split_k: Bool = False,
](TrivialRegisterPassable):
    var seq_len: Int
    var max_seq_len: Int  # q_max_seq_len (padded seq dimension for all batches)
    var num_keys: Int  # Total keys for this batch (full KV cache length)
    var q_row_offset: Int  # Row offset for Q tensor (no split dimension)
    var out_row_offset: Int  # Row offset for output tensor (includes split dimension)
    var split_idx: Int  # Which split partition this CTA handles
    var batch_idx: Int  # Which batch this CTA handles
    var kv_start_row: Int  # Starting KV row for this split
    var num_keys_this_split: Int  # Number of keys this split processes

    @always_inline
    fn __init__(
        out self,
        k: Self.KVLUTType,
        valid_length: UnsafePointer[
            Scalar[Self.ValidLengthType.dtype], origin=ImmutAnyOrigin
        ],
        max_seq_len: Int,
        num_partitions: Int,
        batch_size: Int,
    ):
        self.seq_len = 0
        self.max_seq_len = max_seq_len
        self.num_keys = 0
        self.q_row_offset = 0
        self.out_row_offset = 0
        self.split_idx = 0
        self.batch_idx = 0
        self.kv_start_row = 0
        self.num_keys_this_split = 0

        # Decode block_idx.z into split_idx and batch_idx
        # Grid layout: block_z = batch_size * num_partitions
        # block_idx.z = batch_idx * num_partitions + split_idx
        @parameter
        if Self.decoding_warp_split_k:
            self.batch_idx = Int(block_idx.z) // num_partitions
            self.split_idx = Int(block_idx.z) % num_partitions
        else:
            self.batch_idx = Int(block_idx.z)
            self.split_idx = 0

        @parameter
        if Self.ragged:
            # treat valid_lengths as input_row_offsets
            # Use batch_idx (not block_idx.z) to index into valid_length
            var start_of_seq = Int(valid_length[self.batch_idx])
            var end_of_seq = Int(valid_length[self.batch_idx + 1])
            self.seq_len = end_of_seq - start_of_seq

            # Q row offset: no split dimension
            # Q shape: (total_tokens * num_heads, depth)
            self.q_row_offset = (
                start_of_seq * Self.config.num_q_heads
                + Int(block_idx.x) * Self.config.BM
                + Int(block_idx.y) * Self.config.num_q_heads
            )

            # Output row offset: includes split dimension for split-K
            @parameter
            if Self.decoding_warp_split_k:
                # For ragged with split-K, o_accum_split uses PADDED layout:
                # Shape: (num_partitions, batch_size, max_seq_len, num_heads, depth)
                # This must match the combine kernel's read pattern which uses
                # batch_idx * max_seq_len * num_heads as the stride per batch.
                var rows_per_split = (
                    batch_size * max_seq_len * Self.config.num_q_heads
                )
                self.out_row_offset = (
                    self.split_idx * rows_per_split
                    + self.batch_idx * max_seq_len * Self.config.num_q_heads
                    + Int(block_idx.y) * Self.config.num_q_heads
                    + Int(block_idx.x) * Self.config.BM
                )
            else:
                self.out_row_offset = self.q_row_offset

        # This is when the sequence length is Fixed
        else:
            self.seq_len = max_seq_len

            # Q row offset: (batch * seq_len * num_heads, depth)
            # Row = batch_idx * (seq_len * num_heads) + seq_idx * num_heads + head_block * BM
            self.q_row_offset = (
                Self.config.num_q_heads * self.seq_len * self.batch_idx
                + Int(block_idx.x) * Self.config.BM
                + Int(block_idx.y) * Self.config.num_q_heads
            )

            # Output row offset for split-K:
            # Out shape: (split_k * batch * seq_len * num_heads, depth)
            # Row = split_idx * (batch * seq_len * num_heads) + q_row_offset
            @parameter
            if Self.decoding_warp_split_k:
                var rows_per_split = (
                    batch_size * self.seq_len * Self.config.num_q_heads
                )
                self.out_row_offset = (
                    self.split_idx * rows_per_split + self.q_row_offset
                )
            else:
                self.out_row_offset = self.q_row_offset

        # Get num_keys from KV cache for this batch
        # Use batch_idx (not block_idx.z) to get the correct cache length
        self.num_keys = k.cache_length(self.batch_idx)

        @parameter
        if not Self.is_cache_length_accurate:
            self.num_keys += self.seq_len

        # Compute KV range for this split
        # Each split handles a portion of the KV cache: [kv_start_row, kv_start_row + num_keys_this_split)
        @parameter
        if Self.decoding_warp_split_k:
            # Page-aligned split strategy: only last CTA handles ragged remainder
            # All other CTAs process complete 128-element pages
            comptime page_size = 128
            var total_pages = (self.num_keys + page_size - 1) // page_size
            var pages_per_split = (
                total_pages + num_partitions - 1
            ) // num_partitions

            # Split boundaries are page-aligned
            var start_page = self.split_idx * pages_per_split
            var end_page = min(
                (self.split_idx + 1) * pages_per_split, total_pages
            )

            self.kv_start_row = start_page * page_size
            var kv_end_row = min(end_page * page_size, self.num_keys)
            self.num_keys_this_split = max(kv_end_row - self.kv_start_row, 0)
        else:
            # No split: process all keys starting from row 0
            self.kv_start_row = 0
            self.num_keys_this_split = self.num_keys

    @always_inline
    fn cache_len(self) -> Int:
        # num_keys is total keys, seq_len is chunk length
        return max(self.num_keys - self.seq_len, 0)

    @always_inline
    fn start_pos(self, cache_start_pos: UInt32) -> UInt32:
        # start_pos is the base absolute Q index for this chunk (plus any external base)
        return UInt32(self.cache_len()) + cache_start_pos


# ------------------------------------------------------------------------------
# MLA decoding Load fp8 to bf16 ProducerKVPipeline
# ------------------------------------------------------------------------------
struct KVLoad2CvtProducer[dtype: DType, config: MLA_SM100_Decode_Config](
    TrivialRegisterPassable
):
    comptime KVPipeType = KVPipelineGeneric[
        Self.config.num_kv_stages, 1, 1, WARPGROUP_SIZE + 2
    ]

    # BF16-stage element count (64*576 = 36864)
    comptime bf16_stage_elems = Self.config.BN * Self.config.q_depth

    # FP8 overlay stride in FP8 elements:
    # lower-half(fp8) + upper-half(fp8) = 2 * bf16_stage_elems
    comptime fp8_stage_stride_elems = 2 * Self.bf16_stage_elems

    var pipe: Self.KVPipeType
    # IMPORTANT: this pointer must already point to the UPPER HALF (1:fp8) of stage0
    var smem_upper_fp8: SharedMemPointer[Scalar[Self.dtype]]

    @always_inline
    fn __init__(
        out self,
        pipe: Self.KVPipeType,
        smem_upper_fp8: SharedMemPointer[Scalar[Self.dtype]],
    ):
        self.pipe = pipe
        self.smem_upper_fp8 = smem_upper_fp8
        self.pipe.state._phase = 1

    @always_inline
    fn init(self):
        self.pipe.init()

    @always_inline
    fn stage_base_ptr[
        *, qk_stage: Int = 0
    ](self) -> SharedMemPointer[Scalar[Self.dtype]]:
        var stage_idx: UInt32 = self.pipe.state.index()
        return self.smem_upper_fp8 + stage_idx * UInt32(
            Self.fp8_stage_stride_elems
        )

    @always_inline
    fn producer_mbar[*, qk_stage: Int = 0](self) -> MBarType:
        return self.pipe.producer_mbar[qk_stage]()

    @always_inline("nodebug")
    fn acquire[*, qk_stage: Int = 0](self):
        self.pipe.producer_acquire[qk_stage]()

    @always_inline("nodebug")
    fn commit_step(mut self):
        self.pipe.state.step()


# ------------------------------------------------------------------------------
# MLA decoding Convert fp8 to bf16 ConsumerKVPipeline
# ------------------------------------------------------------------------------


struct KVLoad2CvtConsumer[dtype: DType, config: MLA_SM100_Decode_Config](
    TrivialRegisterPassable
):
    comptime PipeT = KVPipelineGeneric[
        Self.config.num_kv_stages, 1, 1, WARPGROUP_SIZE + 2
    ]

    comptime bf16_stage_elems = Self.config.BN * Self.config.q_depth
    comptime fp8_stage_stride_elems = 2 * Self.bf16_stage_elems

    var pipe: Self.PipeT
    # points to UPPER HALF (1:fp8) of stage0
    var smem_upper_fp8: SharedMemPointer[Scalar[Self.dtype]]

    @always_inline
    fn __init__(
        out self,
        pipe: Self.PipeT,
        smem_upper_fp8: SharedMemPointer[Scalar[Self.dtype]],
    ):
        self.pipe = pipe
        self.smem_upper_fp8 = smem_upper_fp8

    @always_inline
    fn stage_base_ptr(self) -> SharedMemPointer[Scalar[Self.dtype]]:
        var idx: UInt32 = self.pipe.state.index()
        return self.smem_upper_fp8 + idx * UInt32(Self.fp8_stage_stride_elems)

    @always_inline("nodebug")
    fn wait(self):
        self.pipe.consumer_wait[0]()

    @always_inline("nodebug")
    fn release_all(mut self):
        _ = self.pipe.consumer_mbar[0]()[].arrive()
        self.pipe.state.step()


# ------------------------------------------------------------------------------
# MLA decoding produce bf16 for MMAConsumerKVPipeline
# ------------------------------------------------------------------------------


struct KVCvt2MmaProducer[dtype: DType, config: MLA_SM100_Decode_Config](
    TrivialRegisterPassable
):
    comptime PipeT = KVPipelineGeneric[
        Self.config.num_kv_stages, 1, WARPGROUP_SIZE, 2
    ]
    comptime kv_stage_elems = Self.config.BN * Self.config.q_depth

    var pipe: Self.PipeT
    var smem: SharedMemPointer[Scalar[Self.dtype]]

    @always_inline
    fn __init__(
        out self, pipe: Self.PipeT, smem: SharedMemPointer[Scalar[Self.dtype]]
    ):
        self.pipe = pipe
        self.smem = smem
        self.pipe.state._phase = 1

    @always_inline("nodebug")
    fn acquire(self):
        # waits until MMA (2 consumers) released this stage
        self.pipe.producer_acquire[0]()

    @always_inline
    fn stage_index(self) -> UInt32:
        return self.pipe.state.index()

    @always_inline
    fn stage_base_ptr(self) -> SharedMemPointer[Scalar[Self.dtype]]:
        var idx = self.pipe.state.index()
        return self.smem + idx * UInt32(Self.kv_stage_elems)

    @always_inline("nodebug")
    fn commit_all(mut self):
        # 128 threads arrive on producer mbar
        _ = self.pipe.producer_mbar[0]()[].arrive()
        self.pipe.state.step()


# ------------------------------------------------------------------------------
# MLA decoding consume bf16  for MMAConsumerKVPipeline
# ------------------------------------------------------------------------------
struct KVCvt2MmaConsumer[dtype: DType, config: MLA_SM100_Decode_Config](
    TrivialRegisterPassable
):
    comptime KVPipeType = KVPipelineGeneric[
        Self.config.num_kv_stages, 1, WARPGROUP_SIZE, 2
    ]
    comptime kv_stage_elems = Self.config.BN * Self.config.q_depth

    var pipe: Self.KVPipeType
    var smem: SharedMemPointer[Scalar[Self.dtype]]

    @always_inline
    fn __init__(
        out self,
        pipe: Self.KVPipeType,
        smem: SharedMemPointer[Scalar[Self.dtype]],
    ):
        self.pipe = pipe
        self.smem = smem

    @always_inline
    fn stage_base_ptr[
        *, qk_stage: Int = 0
    ](self) -> SharedMemPointer[Scalar[Self.dtype]]:
        var stage_idx: UInt32 = self.pipe.state.index()
        var stage_offset: UInt32 = stage_idx * UInt32(Self.kv_stage_elems)
        return self.smem + stage_offset

    @always_inline
    fn stage_index[*, qk_stage: Int = 0](self) -> UInt32:
        return self.pipe.state.index()

    @always_inline("nodebug")
    fn wait[*, qk_stage: Int = 0](self):
        # Wait on producer mbar for (current index, current phase)
        self.pipe.consumer_wait[qk_stage]()

    @always_inline("nodebug")
    fn release[*, qk_stage: Int = 0](mut self, e: Int32):
        # Signal "stage consumed" to the producer via consumer mbar
        self.pipe.consumer_release[qk_stage](e)


# ------------------------------------------------------------------------------
# MLA decoding ProducerKVPipeline
# ------------------------------------------------------------------------------


struct DecodeKVProducer[dtype: DType, config: MLA_SM100_Decode_Config](
    TrivialRegisterPassable
):
    comptime KVPipeType = KVPipelineGeneric[Self.config.num_kv_stages, 1, 1, 2]

    # One KV "stage" = whole 64 x 576 logical K tile (loaded as 9 x 64x64)
    comptime kv_stage_elems = Self.config.BN * Self.config.q_depth
    comptime kv_stage_bytes = Self.kv_stage_elems * size_of[Self.dtype]()

    var pipe: Self.KVPipeType
    var smem: SharedMemPointer[Scalar[Self.dtype]]

    @always_inline
    fn __init__(
        out self,
        pipe: Self.KVPipeType,
        smem: SharedMemPointer[Scalar[Self.dtype]],
    ):
        self.pipe = pipe
        self.smem = smem

        # IMPORTANT: producer starts at phase 1, like FA4
        self.pipe.state._phase = 1

    @always_inline
    fn init(self):
        self.pipe.init()

    @always_inline
    fn stage_base_ptr[
        *, qk_stage: Int = 0
    ](self) -> SharedMemPointer[Scalar[Self.dtype]]:
        var stage_idx: UInt32 = self.pipe.state.index()
        var stage_offset: UInt32 = stage_idx * UInt32(Self.kv_stage_elems)
        return self.smem + stage_offset

    @always_inline
    fn stage_index[*, qk_stage: Int = 0](self) -> UInt32:
        return self.pipe.state.index()

    @always_inline
    fn producer_mbar[*, qk_stage: Int = 0](self) -> MBarType:
        return self.pipe.producer_mbar[qk_stage]()

    @always_inline("nodebug")
    fn acquire[*, qk_stage: Int = 0](self):
        # Block until consumer has released this stage
        self.pipe.producer_acquire[qk_stage]()

    @always_inline("nodebug")
    fn commit_step(mut self):
        # After we have launched TMA copies for this stage
        # we advance producer's logical stage index.
        self.pipe.state.step()


# ------------------------------------------------------------------------------
# MLA decoding ConsumerKVPipeline
# ------------------------------------------------------------------------------
struct DecodeKVConsumer[dtype: DType, config: MLA_SM100_Decode_Config](
    TrivialRegisterPassable
):
    comptime KVPipeType = KVPipelineGeneric[Self.config.num_kv_stages, 1, 1, 2]
    comptime kv_stage_elems = Self.config.BN * Self.config.q_depth

    var pipe: Self.KVPipeType
    var smem: SharedMemPointer[Scalar[Self.dtype]]

    @always_inline
    fn __init__(
        out self,
        pipe: Self.KVPipeType,
        smem: SharedMemPointer[Scalar[Self.dtype]],
    ):
        # NOTE: we copy the KVPipeline value – that's how FA4 does it.
        # Both sides keep their own PipelineState; the *barriers* do the real sync.
        self.pipe = pipe
        self.smem = smem

    @always_inline
    fn stage_base_ptr[
        *, qk_stage: Int = 0
    ](self) -> SharedMemPointer[Scalar[Self.dtype]]:
        var stage_idx: UInt32 = self.pipe.state.index()
        var stage_offset: UInt32 = stage_idx * UInt32(Self.kv_stage_elems)
        return self.smem + stage_offset

    @always_inline
    fn stage_index[*, qk_stage: Int = 0](self) -> UInt32:
        return self.pipe.state.index()

    @always_inline("nodebug")
    fn wait[*, qk_stage: Int = 0](self):
        # Wait on producer mbar for (current index, current phase)
        self.pipe.consumer_wait[qk_stage]()

    @always_inline("nodebug")
    fn release[*, qk_stage: Int = 0](mut self, e: Int32):
        # Signal "stage consumed" to the producer via consumer mbar
        self.pipe.consumer_release[qk_stage](e)


# ------------------------------------------------------------------------------
# MLA decoding KVPipelineGeneric
# ------------------------------------------------------------------------------
struct KVPipelineGeneric[
    num_kv_stages: Int,
    num_qk_stages: Int,
    num_producer: Int,
    num_consumer: Int,
](TrivialRegisterPassable):
    """
    KVPipeline has `num_kv_stages * num_qk_stages` stages.
    `num_kv_stages` refers to how many `K` and `V` tiles we pipeline
    for performing the `S = Q@K'` and `O += P@V` MMAs.
    Each of these MMAs is broken up into `num_qk_stages` pipelined
    MMAs. We set `step=False` for all but the last MMA that completes
    the operation.
    An alternative implementation would separate the two, and potentially
    allow for more overall stages at the cost of slightly more bookkeeping.
    """

    comptime num_stages: Int = Self.num_kv_stages * Self.num_qk_stages

    # mbars are ordered in {producer, consumer} pairs
    var mbar: MBarType
    var state: PipelineState[Self.num_kv_stages]

    @always_inline
    fn __init__(out self, mbar: MBarType):
        self.mbar = mbar
        self.state = {}

    @always_inline
    fn init(self):
        # Consumer & Producer mbars: arrived by 1 thread performing TMA/mma
        @parameter
        for i in range(Self.num_stages):
            self.mbar[i].init(Int32(Self.num_producer))

        @parameter
        for i in range(Self.num_stages, Self.num_stages * 2):
            self.mbar[i].init(Int32(Self.num_consumer))

    @always_inline
    fn producer_mbar[qk_stage: Int](self) -> MBarType:
        var idx: UInt32 = self.state.index()
        return self.mbar + UInt32(Self.num_qk_stages) * idx + qk_stage

    @always_inline
    fn consumer_mbar[qk_stage: Int](self, idx: UInt32) -> MBarType:
        comptime const_offset = qk_stage + Self.num_stages
        return self.mbar + UInt32(Self.num_qk_stages) * idx + const_offset

    @always_inline
    fn consumer_mbar[qk_stage: Int](self) -> MBarType:
        return self.consumer_mbar[qk_stage](self.state.index())

    @always_inline("nodebug")
    fn producer_acquire[qk_stage: Int = Self.num_qk_stages - 1](self):
        """
        Returns the dynamic pipe idx.
        """
        self.consumer_mbar[qk_stage]()[].wait(self.state.phase())

    @always_inline("nodebug")
    fn consumer_wait[qk_stage: Int = Self.num_qk_stages - 1](self):
        self.producer_mbar[qk_stage]()[].wait(self.state.phase())

    @always_inline("nodebug")
    fn consumer_release[
        qk_stage: Int = Self.num_qk_stages - 1
    ](mut self, e: Int32):
        elect_mma_arrive(self.consumer_mbar[qk_stage](), e)

        @parameter
        if qk_stage == Self.num_qk_stages - 1:
            self.state.step()

    @staticmethod
    @always_inline
    fn num_mbars() -> UInt32:
        return UInt32(2 * Self.num_qk_stages * Self.num_kv_stages)


# ------------------------------------------------------------------------------
# MLA decoding MiscMBars for producer and consumer
# ------------------------------------------------------------------------------
struct DecodeSM100MiscMBars[
    num_stages: Int, num_producer: Int, num_consumer: Int
](TrivialRegisterPassable):
    var mbar_base: MBarType

    # 2 S slots (S0, S1)

    @always_inline
    fn __init__(out self, mbar_base: MBarType):
        self.mbar_base = mbar_base

    @always_inline
    fn init(self):
        # Layout: [S_prod[0..1], S_cons[0..1]]
        var s_pipe = MBarPipeline[Self.num_stages](self.mbar_base)
        # for S 1 producer thread (elect in MMA warpgroup), 128 consumer threads (softmax warpgroup)
        # for P 128 producer threads (softmax warpgroup), 1 consumer thread (elect in MMA warpgroup)
        s_pipe.init[
            num_producer = UInt32(Self.num_producer),
            num_consumer = UInt32(Self.num_consumer),
        ]()

    @always_inline
    fn producer(self) -> ProducerPipeline[Self.num_stages]:
        return {self.mbar_base, self.mbar_base + Self.num_stages}

    @always_inline
    fn consumer(self) -> ConsumerPipeline[Self.num_stages]:
        return {self.mbar_base, self.mbar_base + Self.num_stages}

    @always_inline
    fn end(self) -> MBarType:
        # We consumed 2 * s_num_stages mbars: prod[2] + cons[2]
        return self.mbar_base + 2 * Self.num_stages


# ------------------------------------------------------------------------------
# MLA decoding S pipeline betweeen MMA and Softmax
# ------------------------------------------------------------------------------
########## Producer of the S slot ##########
struct DecodeSProducer(TrivialRegisterPassable):
    comptime SNumStages = 2
    var pipe: ProducerPipeline[Self.SNumStages]

    @always_inline
    fn __init__(out self, pipe: ProducerPipeline[Self.SNumStages]):
        # Copy initialized pipeline (state: index=0, phase=1)
        self.pipe = pipe

    @always_inline
    fn acquire(self):
        # Wait for softmax to mark this S slot "free"
        self.pipe.acquire()

    @always_inline
    fn slot_index(self) -> UInt32:
        return self.pipe.state.index()

    @always_inline
    fn commit_mma(mut self, elect: Int32):
        # Signal "S slot is filled" to softmax
        self.pipe.commit_mma(elect)
        # Advance producer's stage/phase bookkeeping
        self.pipe.step()


########## Consumer of the S slot ##########
struct DecodeSConsumer(TrivialRegisterPassable):
    comptime SNumStages = 2
    var pipe: ConsumerPipeline[Self.SNumStages]

    @always_inline
    fn __init__(out self, pipe: ConsumerPipeline[Self.SNumStages]):
        self.pipe = pipe

    @always_inline
    fn wait(self) -> UInt32:
        # Block until MMA has filled the current S slot
        self.pipe.wait()
        return self.pipe.state.index()

    @always_inline
    fn release(mut self):
        # Mark this S slot as "consumed" so MMA can reuse it
        self.pipe.release()


# ------------------------------------------------------------------------------
# MLA decoding P Pipeline betweeen Softmax and MMA
# ------------------------------------------------------------------------------
########## Producer of the P slot ##########
struct DecodePProducer(TrivialRegisterPassable):
    comptime PNumStages = 2
    var pipe: ProducerPipeline[Self.PNumStages]

    @always_inline
    fn __init__(out self, pipe: ProducerPipeline[Self.PNumStages]):
        self.pipe = pipe

    # Softmax threads collectively wait until MMA has released P
    @always_inline
    fn acquire(self):
        self.pipe.acquire()
        # -> consumer_mbar.wait(phase), all 128 threads see the same phase

    # After writing P, all 128 threads call commit()
    @always_inline("nodebug")
    fn commit(mut self):
        self.pipe.commit()
        # -> producer_mbar.arrive() (128 arrivals total)
        # -> state.step() (phase toggles for next iteration)

    # optional helper
    @always_inline
    fn stage_index(self) -> UInt32:
        return self.pipe.state.index()


########## Consumer of the P slot ##########
struct DecodePConsumer(TrivialRegisterPassable):
    comptime PNumStages = 2
    var pipe: ConsumerPipeline[Self.PNumStages]

    @always_inline
    fn __init__(out self, pipe: ConsumerPipeline[Self.PNumStages]):
        self.pipe = pipe

    # Should be called by MMA elect thread only
    @always_inline("nodebug")
    fn wait(self) -> UInt32:
        self.pipe.wait()
        return self.pipe.state.index()
        # -> producer_mbar.wait(phase)
        # blocks until 128 Softmax commits complete

    # Also called by MMA elect thread only

    @always_inline("nodebug")
    fn release_mma(mut self, elect: Int32):
        # Like KVPipeline.consumer_release but for generic pipeline
        var mbar = self.pipe.consumer_mbar()
        elect_mma_arrive(mbar, elect)
        self.pipe.step()


# ------------------------------------------------------------------------------
# MLA decoding Opipeline betweeen MMA and Correction
# ------------------------------------------------------------------------------
########## Producer of the O slot ##########
struct DecodeOProducer(TrivialRegisterPassable):
    comptime ONumStages = 2
    var pipe: ProducerPipeline[Self.ONumStages]

    @always_inline
    fn __init__(out self, pipe: ProducerPipeline[Self.ONumStages]):
        # Copy initialized pipeline (state: index=0, phase=1)
        self.pipe = pipe

    @always_inline
    fn acquire(self):
        # Wait for softmax to mark this S slot "free"
        self.pipe.acquire()

    @always_inline
    fn slot_index(self) -> UInt32:
        return self.pipe.state.index()

    @always_inline
    fn commit_mma(mut self, elect: Int32):
        # Signal "S slot is filled" to softmax
        self.pipe.commit_mma(elect)
        # Advance producer's stage/phase bookkeeping
        self.pipe.step()


########## Consumer of the O slot ##########
struct DecodeOConsumer(TrivialRegisterPassable):
    comptime ONumStages = 2
    var pipe: ConsumerPipeline[Self.ONumStages]

    @always_inline
    fn __init__(out self, pipe: ConsumerPipeline[Self.ONumStages]):
        self.pipe = pipe

    @always_inline
    fn wait(self):
        # Block until MMA has filled the current S slot
        self.pipe.wait()
        _ = self.pipe.state.index()

    @always_inline
    fn release(mut self):
        # Mark this S slot as "consumed" so MMA can reuse it
        self.pipe.release()


# ------------------------------------------------------------------------------
# MLA decoding C Pipeline between Softmax and Correction
# ------------------------------------------------------------------------------
struct DecodeCProducer(TrivialRegisterPassable):
    comptime CNumStages = 1
    var pipe: ProducerPipeline[Self.CNumStages]

    @always_inline
    fn __init__(out self, pipe: ProducerPipeline[Self.CNumStages]):
        self.pipe = pipe

    # Softmax warpgroup: all 128 threads call acquire() before writing corr scalars
    @always_inline("nodebug")
    fn acquire(self):
        self.pipe.acquire()
        # -> consumer_mbar.wait(phase) on correction side (prev iteration)

    # After writing correction scalars for this O:
    @always_inline("nodebug")
    fn commit(mut self):
        self.pipe.commit()
        # producer_mbar.arrive() from 128 threads + state.step()


struct DecodeCConsumer(TrivialRegisterPassable):
    comptime CNumStages = 1
    var pipe: ConsumerPipeline[Self.CNumStages]

    # Correction warpgroup: all 128 threads wait until correction scalars are ready
    @always_inline
    fn __init__(out self, pipe: ConsumerPipeline[Self.CNumStages]):
        self.pipe = pipe

    @always_inline("nodebug")
    fn wait(self):
        # perform producer_mbar.wait(phase)
        self.pipe.wait()

    @always_inline("nodebug")
    fn release(mut self):
        # perform consumer_mbar.arrive() from 128 threads + state.step()
        self.pipe.release()


# ------------------------------------------------------------------------------
# MLA decoding  pipeline correction is the producer and write is the consumer
# ------------------------------------------------------------------------------


struct OutPipeline[num_out_stages: Int, num_producer: Int, num_consumer: Int](
    TrivialRegisterPassable
):
    """
    OutPipeline has `num_out_stages` stages.
    `num_out_stages` refers to how many output stages we pipeline
    for performing the output store.
    """

    comptime num_stages: Int = Self.num_out_stages

    # mbars are ordered in {producer, consumer} pairs
    var mbar: MBarType
    var state: PipelineState[Self.num_stages]

    @always_inline
    fn __init__(out self, mbar: MBarType):
        self.mbar = mbar
        self.state = {}

    @always_inline
    fn init(self):
        # Consumer & Producer mbars: arrived by num_producer and num_consumer threads
        @parameter
        for i in range(Self.num_stages):
            self.mbar[i].init(Int32(Self.num_producer))

        @parameter
        for i in range(Self.num_stages):
            (self.mbar + Self.num_stages)[i].init(Int32(Self.num_consumer))

    @always_inline
    fn producer_mbar(self) -> MBarType:
        return self.mbar

    @always_inline
    fn consumer_mbar(self) -> MBarType:
        return self.mbar + Self.num_stages

    @always_inline("nodebug")
    fn producer_acquire(self):
        """
        Returns the dynamic pipe idx.
        """
        var idx = self.state.index()
        self.consumer_mbar()[idx].wait(self.state.phase())

    @always_inline("nodebug")
    fn consumer_wait(self):
        var idx = self.state.index()
        self.producer_mbar()[idx].wait(self.state.phase())

    @always_inline("nodebug")
    fn consumer_release[](mut self, e: Int32):
        var idx = self.state.index()
        elect_mma_arrive(self.consumer_mbar() + idx, e)
        self.state.step()

    @always_inline("nodebug")
    fn producer_commit(mut self):
        # All 128 producer threads should call this.
        # mbar was initialized with num_producer = WARPGROUP_SIZE,
        # so producer_mbar()[].arrive() must be called by each producer thread.
        var idx = self.state.index()
        _ = self.producer_mbar()[idx].arrive()
        self.state.step()

    @staticmethod
    @always_inline
    fn num_mbars() -> UInt32:
        return UInt32(2 * Self.num_stages)


struct DecodeOutProducer[dtype: DType, config: MLA_SM100_Decode_Config](
    TrivialRegisterPassable
):
    # mma.ws split BN elements across even/odd warps
    comptime col_per_warp = Self.config.MMA_PV_N // 2
    comptime num_out_blocks: Int = Self.config.depth // Self.config.BN
    comptime block_per_warp = Self.col_per_warp // Self.config.BN
    comptime blocks_per_stage = 2 if Self.block_per_warp != 0 else 1
    comptime num_out_stages: Int = Self.num_out_blocks // Self.blocks_per_stage
    comptime OutPipeType = OutPipeline[Self.num_out_stages, WARPGROUP_SIZE, 1]

    # One KV "stage" = whole 64 x 576 logical K tile (loaded as 9 x 64x64)
    comptime out_stage_elems = Self.config.BM * Self.config.BN
    comptime out_stage_bytes = Self.out_stage_elems * size_of[Self.dtype]()

    var pipe: Self.OutPipeType
    var smem: SharedMemPointer[Scalar[Self.dtype]]

    @always_inline
    fn __init__(
        out self,
        pipe: Self.OutPipeType,
        smem: SharedMemPointer[Scalar[Self.dtype]],
    ):
        self.pipe = pipe
        self.smem = smem

        # IMPORTANT: producer starts at phase 1, like FA4
        self.pipe.state._phase = 1

    @always_inline
    fn init(self):
        # Only producer OR consumer should call init(), not both.
        self.pipe.init()

    @always_inline
    fn stage_base_ptr(
        self, half_idx: Int
    ) -> SharedMemPointer[Scalar[Self.dtype]]:
        var stage_idx: UInt32 = self.pipe.state.index()
        var stage_offset: UInt32 = stage_idx * UInt32(
            Self.out_stage_elems
        ) * UInt32(Self.blocks_per_stage) + UInt32(
            half_idx * Self.out_stage_elems
        )
        return self.smem + stage_offset

    @always_inline
    fn producer_mbar(self) -> MBarType:
        return self.pipe.producer_mbar()

    @always_inline("nodebug")
    fn acquire(self):
        # Block until consumer has released this stage
        self.pipe.producer_acquire()

    @always_inline("nodebug")
    fn commit_step(mut self):
        # After we have launched TMA copies for this stage
        # we advance producer's logical stage index.

        self.pipe.producer_commit()


struct DecodeOutConsumer[dtype: DType, config: MLA_SM100_Decode_Config](
    TrivialRegisterPassable
):
    # mma.ws split BN elements across even/odd warps
    comptime col_per_warp = Self.config.MMA_PV_N // 2
    comptime num_out_blocks: Int = Self.config.depth // Self.config.BN
    comptime block_per_warp = Self.col_per_warp // Self.config.BN
    comptime blocks_per_stage = 2 if Self.block_per_warp != 0 else 1
    comptime num_out_stages: Int = Self.num_out_blocks // Self.blocks_per_stage
    comptime OutPipeType = OutPipeline[Self.num_out_stages, WARPGROUP_SIZE, 1]
    comptime out_stage_elems = Self.config.BM * Self.config.BN

    var pipe: Self.OutPipeType
    var smem: SharedMemPointer[Scalar[Self.dtype]]

    @always_inline
    fn __init__(
        out self,
        pipe: Self.OutPipeType,
        smem: SharedMemPointer[Scalar[Self.dtype]],
    ):
        self.pipe = pipe
        self.smem = smem

    @always_inline
    fn stage_base_ptr(
        self, half_idx: Int
    ) -> SharedMemPointer[Scalar[Self.dtype]]:
        var stage_idx: UInt32 = self.pipe.state.index()
        var stage_offset: UInt32 = stage_idx * UInt32(
            Self.out_stage_elems
        ) * UInt32(Self.blocks_per_stage) + UInt32(
            half_idx * Self.out_stage_elems
        )
        return self.smem + stage_offset

    @always_inline("nodebug")
    fn wait(self):
        # Wait on producer mbar for (current index, current phase)
        self.pipe.consumer_wait()

    @always_inline("nodebug")
    fn release(mut self, e: Int32):
        # Signal "stage consumed" to the producer via consumer mbar
        self.pipe.consumer_release(e)


# ------------------------------------------------------------------------------
# MLA decoding build_ss for ws
# ------------------------------------------------------------------------------


@always_inline
fn build_mma_ss_ws(
    kind: String,
    layout_a: Layout,
    layout_b: Layout,
    *,
    operand_size: Int,
    num_k_mmas: Int,
    tcgen05_mma_type: String,
) -> String:
    # rda and rdb are the 64-bit smem descriptors.
    # %pj: jump predicate (elect==0 -> skip)
    # %ps: enable-input-d predicate (c_scale != 0).
    mma = """{
.reg .b64 %rda;
.reg .b64 %rdb;
.reg .s32 %ra;
.reg .s32 %rb;
.reg .pred %pj;
.reg .pred %ps;
setp.eq.s32 %pj, $6, 0;
"""
    tcgen05_mma = tcgen05_mma_type + kind

    for k in range(num_k_mmas):
        if k == 0:
            # rda/rdb from the base descriptors
            mma += "mov.b64 %rda, {$7, $8};\n"
            mma += "mov.b64 %rdb, {$4, $5};\n"
            # %ps = (c_scale != 0)
            mma += "setp.ne.b32 %ps, $3, 0;\n"
        else:
            # rda = a_desc + a_offset
            var a_offset = (layout_a(IntTuple(0, 16 * k)) * operand_size) >> 4
            mma += String("add.s32 %ra, $7, ", a_offset, ";\n")
            mma += "mov.b64 %rda, {%ra, $8};\n"

            # rdb = b_desc + b_offset
            var b_offset = (layout_b(IntTuple(0, 16 * k)) * operand_size) >> 4
            mma += String("add.s32 %rb, $4, ", b_offset, ";\n")
            mma += "mov.b64 %rdb, {%rb, $5};\n"

            if k == 1:
                # after the first K-slice we always accumulate: enable-input-d = true
                mma += "setp.ne.b32 %ps, 1, 0;\n"

        # tcgen05.mma.ws:
        # [d-tmem], a-desc, b-desc, idesc, enable-input-d , {, zero-column-mask-desc};
        mma += String("@%pj bra skip", k, ";")
        mma += tcgen05_mma + " [$0], %rda, %rdb, $2, %ps;\n"

        mma += String("skip", k, ":\n")
    return mma + "}"


@always_inline
fn bulk_mma_ws[
    kind: UMMAKind,
    //,
    layout_a: Layout,
    layout_b: Layout,
    *,
    num_k_mmas: Int,
    operand_size: Int,
    tcgen05_mma_type: String,
](
    idesc: UMMAInsDescriptor[kind],
    a: MMASmemDescriptorPair,
    b: MMASmemDescriptorPair,
    c_tmem: UInt32,
    c_scale: UInt32,
    elect: Int32,
):
    comptime mma_string = build_mma_ss_ws(
        String(kind),
        layout_a,
        layout_b,
        operand_size=operand_size,
        num_k_mmas=num_k_mmas,
        tcgen05_mma_type=tcgen05_mma_type,
    )

    inlined_assembly[mma_string, NoneType, constraints="r,r,r,r,r,r,r,r,r"](
        c_tmem, 0, idesc, c_scale, b.lo, b.hi, elect, a.lo, a.hi
    )


# ------------------------------------------------------------------------------
# MLA decoding Tensor AccumulatorSS for QKT
# ------------------------------------------------------------------------------
struct DecodeSM100QKTSS[
    operand_type: DType,
    accum_type: DType,
    *,
    config: MLA_SM100_Decode_Config,
](TrivialRegisterPassable):
    comptime MMA_M = Self.config.MMA_M  # 64 rows
    comptime MMA_N = Self.config.MMA_QK_N  # 64 cols
    comptime MMA_K = Self.config.MMA_K  # 16
    comptime BK = Self.config.BK0  # 576
    comptime num_k_mmas = Self.BK // Self.MMA_K
    comptime operand_size = size_of[Self.operand_type]()

    # ----- A (Q) tile layout -----
    comptime ALayout = tile_layout_k_major[
        Self.operand_type,
        Self.config.BM,  # 64 rows
        Self.BK,  # 576 cols
        Self.config.swizzle_mode,
    ]()

    # ----- B (K) tile layout -----
    comptime BLayout = tile_layout_k_major[
        Self.operand_type,
        Self.config.BN,  # 64 rows
        Self.BK,  # 576 cols
        Self.config.kv_mma_swizzle_mode,
    ]()

    # ----- Instruction descriptor -----
    comptime UMMAInstDesc = UMMAInsDescriptor[UMMAKind.KIND_F16].create[
        Self.accum_type,
        Self.operand_type,
        Self.operand_type,
        Index[dtype = DType.uint32](Self.MMA_M, Self.MMA_N),
        transpose_b=True,  # QKᵀ
    ]()

    @staticmethod
    @always_inline
    fn descriptor_q_block(
        q_smem: SharedMemPointer[Scalar[Self.operand_type]],
    ) -> MMASmemDescriptorPair:
        # Q: 64 x 64, k-major, same swizzle as TMA
        var base = q_smem
        return smem_descriptor[
            BMN = Self.config.BM,  # 64 rows
            BK = Self.BK,  # 576 (padded_q_depth)
            swizzle_mode = Self.config.swizzle_mode,
            is_k_major=True,
        ](base)

    @staticmethod
    @always_inline
    fn descriptor_k_block(
        kv_smem: SharedMemPointer[Scalar[Self.operand_type]],
    ) -> MMASmemDescriptorPair:
        var base = kv_smem
        # Layout is 64 x 64, k-major, same swizzle as k_tma
        return smem_descriptor[
            BMN = Self.config.BN,  # 64 rows
            BK = Self.BK,  # 576 columns
            swizzle_mode = Self.config.kv_mma_swizzle_mode,
            is_k_major=True,
        ](base)

    @staticmethod
    @always_inline
    fn mma[
        *, stage_idx: Int = 0
    ](
        a: MMASmemDescriptorPair,
        b: MMASmemDescriptorPair,
        c: UInt32,
        *,
        c_scale: UInt32,
        elect: Int32,
    ):
        comptime assert stage_idx == 0, "stage_idx should be 0"
        bulk_mma_ws[
            kind = UMMAKind.KIND_F16,
            layout_a = Self.ALayout,
            layout_b = Self.BLayout,
            num_k_mmas = Self.num_k_mmas,
            operand_size = Self.operand_size,
            tcgen05_mma_type="tcgen05.mma.ws.cta_group::1.",
        ](Self.UMMAInstDesc, a, b, c, c_scale, elect)


struct DecodeSM100PVSS[
    operand_type: DType,
    accum_type: DType,
    *,
    config: MLA_SM100_Decode_Config,
](TrivialRegisterPassable):
    comptime MMA_M = Self.config.MMA_M  # 64 rows
    comptime MMA_N = Self.config.MMA_PV_N
    comptime MMA_K = Self.config.MMA_K  # 16
    comptime BM = Self.config.BM  # 64
    comptime BN = Self.MMA_N  # 64
    comptime BK = Self.config.BK1  # 64
    comptime num_k_mmas = Self.BK // Self.MMA_K
    comptime operand_size = size_of[Self.operand_type]()

    # ----- A (P) tile layout -----
    # P tiles are treated as mn-major in smem
    comptime ALayout = tile_layout_k_major[
        Self.operand_type,
        Self.BM,  # 64
        Self.BK,  # 64
        Self.config.swizzle_mode,
    ]()

    # ----- B (V) tile layout -----
    # V tiles are mn-major (is_k_major = False in descriptor_v_block)
    comptime BLayout = tile_layout_mn_major[
        Self.operand_type,
        Self.BN,  # 256 as this is the max number of column accepted for mma
        Self.BK,  # 64
        Self.config.kv_mma_swizzle_mode,
    ]()

    # ----- Instruction descriptor -----
    comptime UMMAPVSS = UMMAInsDescriptor[UMMAKind.KIND_F16].create[
        Self.accum_type,
        Self.operand_type,
        Self.operand_type,
        Index[dtype = DType.uint32](Self.MMA_M, Self.MMA_N),
        transpose_b=False,  # P (k-major) * V (mn-major) = no transpose
    ]()

    @staticmethod
    @always_inline
    fn descriptor_v_block(
        kv_smem: SharedMemPointer[Scalar[Self.operand_type]],
    ) -> MMASmemDescriptorPair:
        var base = kv_smem
        # Layout is BDepth_max x 64, mn-major, same swizzle as k_tma
        return smem_descriptor[
            BMN = Self.BN,
            BK = Self.BK,  # 64 rows
            swizzle_mode = Self.config.kv_mma_swizzle_mode,
            is_k_major=False,
        ](base)

    @staticmethod
    @always_inline
    fn descriptor_p_block(
        p_smem: SharedMemPointer[Scalar[Self.operand_type]],
    ) -> MMASmemDescriptorPair:
        var base = p_smem
        # P: 64 x 64, k-major, same swizzle as Q/K
        return smem_descriptor[
            BMN = Self.BM,  # 64 rows
            BK = Self.BK,  # 64 columns
            swizzle_mode = Self.config.swizzle_mode,
            is_k_major=True,  # P is k-major
        ](base)

    @staticmethod
    @always_inline
    fn mma[
        *, stage_idx: Int = 0
    ](
        a: MMASmemDescriptorPair,
        b: MMASmemDescriptorPair,
        c: UInt32,
        *,
        c_scale: UInt32,
        elect: Int32,
    ):
        comptime assert stage_idx == 0, "stage_idx should be 0"
        bulk_mma_ws[
            kind = UMMAKind.KIND_F16,
            layout_a = Self.ALayout,
            layout_b = Self.BLayout,
            num_k_mmas = Self.num_k_mmas,
            operand_size = Self.operand_size,
            tcgen05_mma_type="tcgen05.mma.ws.cta_group::1.",
        ](Self.UMMAPVSS, a, b, c, c_scale, elect)


# ------------------------------------------------------------------------------
# Helper functions for writing from local memory to shared memory using swizzle
# ------------------------------------------------------------------------------


@always_inline
fn write_bf16x2_row_to_smem_chunked[
    local_tile_size: Int,
    *,
    out_dtype: DType,
    in_dtype: DType,
    config: MLA_SM100_Decode_Config,
    chunk_size: Int = 16,
    scale_needed: Bool = False,
](
    shared_mem: UnsafePointer[
        Scalar[out_dtype], MutAnyOrigin, address_space = AddressSpace.SHARED
    ],
    local_mem: LocalTensor[in_dtype, row_major[local_tile_size]()],
    col_start: Int,
    row_start: Int,
    scale: Scalar[in_dtype] = 1.0,
):
    """Chunked write with optional scaling. Reduces register pressure."""
    comptime num_chunks = local_tile_size // chunk_size
    comptime groups_per_chunk = chunk_size // 8
    comptime total_groups = num_chunks * groups_per_chunk

    # Precompute swizzle function once
    comptime swz = make_ldmatrix_swizzle[
        dtype=out_dtype,
        row_size = config.BN,
        log2_vector_width=3,
    ]()

    # Precompute all swizzle offsets before the loop
    var phys_offsets = StaticTuple[Int, total_groups]()

    @parameter
    for i in range(total_groups):
        comptime chunk_idx = i // groups_per_chunk
        comptime group_idx = i % groups_per_chunk
        comptime col_offset = chunk_idx * chunk_size + group_idx * 8
        var logical_elem = row_start * config.BN + col_start + col_offset
        phys_offsets[i] = swz(logical_elem)

    var lmv = local_mem.vectorize[8]()

    @parameter
    for chunk in range(0, num_chunks):

        @parameter
        for g in range(0, groups_per_chunk):
            # Compute the correct index into the vectorized view
            # vec_idx accounts for both chunk offset and position within chunk
            comptime vec_idx = chunk * groups_per_chunk + g

            var vec_val = lmv[vec_idx]

            @parameter
            if scale_needed:
                vec_val *= scale

            var bf16_vec = vec_val.cast[out_dtype]()
            var packed = bitcast[DType.uint32, 4](bf16_vec)
            st_shared_v4_b32_at_bf16_elem_off[out_dtype=out_dtype](
                shared_mem,
                phys_offsets[vec_idx],
                packed,
            )


@always_inline
fn ld_shared_v4_u32(
    src_u8: UnsafePointer[
        Scalar[DType.uint8], MutAnyOrigin, address_space = AddressSpace.SHARED
    ],
    byte_off: Int,
) -> SIMD[DType.uint32, 4]:
    var addr = src_u8 + byte_off
    var result = inlined_assembly[
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        _RegisterPackType[UInt32, UInt32, UInt32, UInt32],
        # 4 outputs (return struct has 4 elems) + 1 input (addr)
        constraints="=r,=r,=r,=r,l",
        has_side_effect=False,
    ](addr)
    return SIMD[DType.uint32, 4](result[0], result[1], result[2], result[3])


@always_inline
fn cvt_fp8x8_from_2xu32_to_bf16x8_packed_u32x4[
    *,
    fp8_dtype: DType,
    out_dtype: DType,
](w0: UInt32, w1: UInt32,) -> SIMD[DType.uint32, 4]:
    var u32x2: SIMD[DType.uint32, 2] = SIMD[DType.uint32, 2](w0, w1)
    var fp8x8: SIMD[fp8_dtype, 8] = bitcast[fp8_dtype, 8](u32x2)
    var bf16x8: SIMD[out_dtype, 8] = fp8x8.cast[out_dtype]()
    return bitcast[DType.uint32, 4](bf16x8)


@always_inline
fn st_shared_v4_b32_at_bf16_elem_off[
    out_dtype: DType
](
    dst_bf16: UnsafePointer[
        Scalar[out_dtype], MutAnyOrigin, address_space = AddressSpace.SHARED
    ],
    elem_off: Int,  # bf16 element offset
    packed: SIMD[DType.uint32, 4],
):
    var dst_ptr = dst_bf16 + elem_off
    _ = inlined_assembly[
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        NoneType,
        constraints="l,r,r,r,r",
        has_side_effect=True,
    ](dst_ptr, packed[0], packed[1], packed[2], packed[3])


# --------------------------------------------------------------------------
# MLA decoding softmax Pipeline
# --------------------------------------------------------------------------
@always_inline
fn clamped_index_coordinate(
    var prompt_idx: UInt32,
    var q_head_idx: UInt32,
    var q_idx_abs: UInt32,
    var col: UInt32,
    var tile_key_base: UInt32,
    var num_keys: Int,
    var cache_start_pos: UInt32,
) -> IndexList[4, element_type = DType.uint32]:
    # Global key index (column) for this element
    var score_col: UInt32 = tile_key_base + col
    var k_idx_abs: UInt32 = score_col + cache_start_pos
    # Clamp k to last valid key so MaterializedMask never reads OOB.
    var last_k_abs: UInt32 = cache_start_pos + UInt32(max(num_keys - 1, 0))
    var k_idx_abs_safe: UInt32 = min(k_idx_abs, last_k_abs)
    return IndexList[4, element_type = DType.uint32](
        Int(prompt_idx),
        Int(q_head_idx),
        Int(q_idx_abs),
        Int(k_idx_abs_safe),
    )


struct MLA_SM100_Decode_Common[
    q_type: DType,
    KVLUTType: MHAOperand,
    output_type: DType,
    SplitAccumType: OptionalPointer,
    MaskType: MHAMask,
    ScoreModType: ScoreModTrait,
    config: MLA_SM100_Decode_Config,
    use_score_mod: Bool,
    ValidLengthType: OptionalPointer,
    _is_cache_length_accurate: Bool = False,
    ragged: Bool = False,
](TrivialRegisterPassable):
    comptime kv_type = Self.KVLUTType.dtype
    comptime AccumType = get_accum_type[Self.q_type]()
    # 576 / 64 = 9
    comptime NumQKBlocks = Self.config.padded_q_depth // Self.config.BN
    # 512 / 64 = 8
    comptime NumVOBlocks = Self.config.padded_depth // Self.config.BN
    # 64 * 64 = 4096
    comptime BlockElems = Self.config.BM * Self.config.BN
    # 2 bytes for float16
    comptime bytes_per_element = size_of[Self.q_type]()
    # the stage element is the same for both K and V
    comptime KVStageElems = Self.NumQKBlocks * Self.BlockElems
    comptime output_tile_width = (Self.config.BN // 2) * (
        4 // size_of[Self.output_type]()
    )
    # O: 128 x 256
    comptime O_M = Self.config.BM * 2  # 128
    comptime O_N = Self.config.padded_depth // 2  # 256

    # S: 128 x 32
    comptime S_M = Self.config.BM * 2  # 128
    comptime S_N = Self.config.BN // 2  # 32
    comptime OTMemTile = TMemTile[Self.AccumType, Self.O_M, Self.O_N]
    comptime STMemTile = TMemTile[Self.AccumType, Self.S_M, Self.S_N]
    comptime UMMAQKTSS = DecodeSM100QKTSS[
        operand_type = Self.q_type,
        accum_type = Self.AccumType,
        config = Self.config,
    ]
    comptime UMMAPVSS = DecodeSM100PVSS[
        operand_type = Self.q_type,
        accum_type = Self.AccumType,
        config = Self.config,
    ]

    # --------------------------------------------------------------------------
    # PDL early exit cleanup for split-K CTAs with no work.
    # Writes -inf to LSE, zeros o_accum_split via shared-memory + TMA store,
    # then calls barrier() + launch_dependent_grids().
    # --------------------------------------------------------------------------
    @staticmethod
    @always_inline
    fn pdl_early_exit(
        split_idx: Int,
        batch_idx: Int,
        max_seq_len: Int,
        out_row_offset: Int,
        batch_size: Int,
        lse_accum_split_ptr: Self.SplitAccumType,
        o_tma: QOTMATile[
            dtype = Self.output_type,
            BM = Self.config.out_rows,
            BK = Self.config.BN,
            swizzle_mode = Self.config.swizzle_mode,
        ],
    ):
        var tid = Int(thread_idx.x)

        # -- 1. Write -inf to LSE so combine gives this split zero weight --
        # LSE layout: (num_splits, batch_size, max_seq_len, num_heads)
        # Use max_seq_len (not per-batch seq_len) for strides to match
        # the PADDED buffer layout and the combine kernel's read pattern.
        var head_start = Int(block_idx.x) * Self.config.BM
        var seq_idx = Int(block_idx.y)
        var stride_seq = Self.config.num_q_heads
        var stride_batch = max_seq_len * stride_seq
        var stride_split = batch_size * stride_batch
        var neg_inf_val = min_or_neg_inf[Self.AccumType]()

        # First BM threads each write one head's LSE
        if tid < Self.config.BM:
            var head_idx = head_start + tid
            if head_idx < Self.config.num_q_heads:
                var lse_offset = (
                    split_idx * stride_split
                    + batch_idx * stride_batch
                    + seq_idx * stride_seq
                    + head_idx
                )
                var lse_ptr = rebind[
                    UnsafePointer[
                        Scalar[Self.AccumType],
                        origin=MutAnyOrigin,
                    ]
                ](lse_accum_split_ptr.value())
                lse_ptr[lse_offset] = neg_inf_val

        # -- 2. Zero o_accum_split via shared-memory + TMA store --
        # Get shared memory (always allocated at kernel launch)
        var smem_ptr = external_memory[
            Scalar[Self.output_type],
            address_space = AddressSpace.SHARED,
            alignment=128,
            name="mha_dynamic_shared_memory",
        ]()
        # TMA output tile: (out_rows, BN) elements
        comptime tile_elems = Self.config.out_rows * Self.config.BN
        # Zero shared memory cooperatively with all threads
        var idx = tid
        while idx < tile_elems:
            smem_ptr[idx] = Scalar[Self.output_type](0)
            idx += Self.config.num_threads
        barrier()

        # Thread 0 issues TMA stores to write zeros to the
        # o_accum_split region for every column tile
        if tid == 0:
            var smem_tensor = SharedMemTensor[
                Self.output_type,
                type_of(o_tma).layout,
            ](smem_ptr)
            var row = UInt(out_row_offset)
            comptime num_col_tiles = Self.config.depth // Self.config.BN

            @parameter
            for c in range(num_col_tiles):
                fence_async_view_proxy()
                o_tma.async_store(
                    smem_tensor,
                    (UInt(c * Self.config.BN), row),
                )
            o_tma.commit_group()
        o_tma.wait_group[0]()

        # -- 3. Barrier + PDL signal, then return --
        barrier()
        launch_dependent_grids()

    # --------------------------------------------------------------------------
    # MLA decoding load_q and load_kv function
    # --------------------------------------------------------------------------
    @staticmethod
    @always_inline
    fn load_kv(
        tma: KVTMATile[
            dtype = Self.kv_type,
            swizzle_mode = Self.config.kv_tma_swizzle_mode,
            BN = Self.config.BK1,  # tile_m =64
            BK = Self.config.BK0,  # tile_n =576
        ],
        smem: SharedMemPointer[Scalar[Self.kv_type]],
        mbar: MBarType,
        col_start: UInt,
        row_start: UInt,
    ):
        var block_smem = smem
        var smem_tensor = SharedMemTensor[
            Self.kv_type, type_of(tma).layout  # 64x576 swizzled tile
        ](block_smem)
        tma.async_copy_3d(
            smem_tensor, mbar[], (Int(col_start), Int(0), Int(row_start))
        )

    @staticmethod
    @always_inline
    fn load_q(
        tma: QOTMATile[
            dtype = Self.q_type,
            BM = Self.config.BM,  # tile_m =64
            BK = Self.config.BK0,  # tile_n =576
            swizzle_mode = Self.config.swizzle_mode,
        ],
        smem: SharedMemPointer[Scalar[Self.q_type]],
        mbar: MBarType,
        col_start: UInt,
        row_start: UInt,
    ):
        var block_smem = smem
        var smem_tensor = SharedMemTensor[
            Self.q_type, type_of(tma).layout  # 64x576 swizzled tile
        ](block_smem)

        tma.async_copy(smem_tensor, mbar[], (Int(col_start), Int(row_start)))

    @staticmethod
    @always_inline
    fn apply_mask[
        half_load: Int, NonCausalMask: Bool, CausalMask: Bool
    ](
        tiles_done: Int,
        col0: Int,
        num_keys: Int,
        s_row: LocalTensor[Self.AccumType, row_major[half_load]()],
        mask: Self.MaskType,
        score_mod: Self.ScoreModType,
        prompt_idx: UInt32,
        q_head_idx: UInt32,
        score_row: UInt32,
        max_seq_len: UInt32,
        cache_len: Int,
        start_pos: UInt32,
        cache_start_pos: UInt32,
        kv_start_row: Int = 0,  # Starting KV row for split-K (0 for non-split)
    ) -> Scalar[Self.AccumType]:
        # Tile / column base this thread covers in num_keys in global KV cache
        # For split-K: kv_start_row + tiles_done * BN gives global position
        # For non-split: kv_start_row=0, so this is just tiles_done * BN
        var tile_key_base: Int = kv_start_row + tiles_done * Self.config.BN
        # first key index for this thread
        var col_base: Int = tile_key_base + col0

        # Per-row causal masking for chunked decode
        # Allowed keys for this query row in a chunked causal decode:
        # cache_len + score_row + 1
        var causal_limit: Int

        @parameter
        if CausalMask:
            causal_limit = cache_len + Int(score_row) + 1
        else:
            causal_limit = num_keys
        var keys_remaining = causal_limit - col_base
        var n_valid = max(min(keys_remaining, half_load), 0)
        # Build mask_bits with lowest n_valid bits = 1
        var mask_bits_64: UInt64 = (UInt64(1) << UInt64(n_valid)) - UInt64(1)
        var mask_bits: UInt32 = UInt32(mask_bits_64 & UInt64(0xFFFF_FFFF))

        var current_max: Scalar[Self.AccumType] = min_or_neg_inf[
            Self.AccumType
        ]()

        @parameter
        for i in range(0, half_load):
            # rank1-style mask_r2p: turn bit into predicate and use it to select
            var bit: UInt32 = (mask_bits >> UInt32(i)) & UInt32(1)
            var in_bound: Bool = bit != UInt32(0)
            # masked_val = s_row[i]      if in_bound
            #            = -inf          otherwise
            var val: Scalar[Self.AccumType] = s_row[i][0]
            var masked_val = val if in_bound else min_or_neg_inf[
                Self.AccumType
            ]()

            @parameter
            if NonCausalMask:
                var v: SIMD[Self.AccumType, 1] = masked_val
                var coord = clamped_index_coordinate(
                    prompt_idx,
                    q_head_idx,
                    score_row + start_pos + cache_start_pos,
                    UInt32(col0 + i),
                    UInt32(tile_key_base),
                    num_keys,
                    cache_start_pos,
                )
                v = mask.mask(coord, v)
                masked_val = v[0]

            @parameter
            if Self.use_score_mod:
                var v2: SIMD[Self.AccumType, 1] = masked_val
                var coord = clamped_index_coordinate(
                    prompt_idx,
                    q_head_idx,
                    score_row + start_pos + cache_start_pos,
                    UInt32(col0 + i),
                    UInt32(tile_key_base),
                    num_keys,
                    cache_start_pos,
                )
                v2 = score_mod.score_mod(coord, v2, Int(max_seq_len))
                masked_val = v2[0]

            s_row[i][0] = masked_val
            current_max = max(current_max, masked_val)

        return current_max

    @staticmethod
    @always_inline
    fn Softmax(
        tmem_addr: UInt32,
        s_bars: DecodeSM100MiscMBars[
            num_stages=2, num_producer=1, num_consumer=WARPGROUP_SIZE
        ],
        p_bars: DecodeSM100MiscMBars[
            num_stages=2, num_producer=WARPGROUP_SIZE, num_consumer=1
        ],
        kv_smem: SharedMemPointer[Scalar[Self.q_type]],
        max_smem: SharedMemPointer[Scalar[Self.AccumType]],  # 128x1 buffer
        li_smem: SharedMemPointer[Scalar[Self.AccumType]],  # 128x1 buffer
        out_smem: SharedMemPointer[Scalar[Self.output_type]],
        c_bars: DecodeSM100MiscMBars[
            num_stages=1,
            num_producer=WARPGROUP_SIZE,
            num_consumer=WARPGROUP_SIZE,
        ],
        corr_done_bars: DecodeSM100MiscMBars[
            num_stages=2,
            num_producer=WARPGROUP_SIZE,
            num_consumer=WARPGROUP_SIZE,
        ],
        out_pipeline: OutPipeline[
            num_out_stages = DecodeOutProducer[
                Self.output_type, Self.config
            ].num_out_stages,
            num_producer=WARPGROUP_SIZE,
            num_consumer=1,
        ],
        offset_position: OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
        ],
        scale: Float32,
        mask: Self.MaskType,
        score_mod: Self.ScoreModType,
        prompt_idx: UInt32,  # batch index
        max_seq_len: UInt32,  # for score_mod
        lse_accum_split_ptr: Self.SplitAccumType,
        batch_size: Int,
    ):
        comptime MaskName: String = Self.MaskType.name()
        comptime assert Self.AccumType.is_floating_point()

        comptime NoMask: Bool = (MaskName == "NullMask")
        comptime CausalMask: Bool = (MaskName == "CausalMask")
        comptime NeedLog2eAfter: Bool = Self.MaskType.apply_log2e_after_mask or Self.use_score_mod
        comptime CheckDuringDecoding: Bool = Self.MaskType.check_mask_during_decoding

        # Same S base / stride as in mma()
        var s0_tmem = tmem_addr + UInt32(Self.config.TMEM_S0)
        var s_stride = UInt32(Self.config.TMEM_S1 - Self.config.TMEM_S0)
        comptime TileLayout = Layout.row_major(WARPGROUP_SIZE)  # 128x1
        var max_Smem_Tensor = SharedMemTensor[Self.AccumType, TileLayout](
            max_smem
        )
        var li_Smem_Tensor = SharedMemTensor[Self.AccumType, TileLayout](
            li_smem
        )

        var corr_scale_tmem = tmem_addr + UInt32(Self.config.TMEM_CORR_SCALE)
        # For split-K: use num_keys_this_split for loop bounds
        # but keep num_keys (total) for masking with global KV positions
        var num_keys = offset_position.num_keys  # Total keys for masking
        var num_keys_this_split = (
            offset_position.num_keys_this_split
        )  # Keys for this split
        var kv_start_row = (
            offset_position.kv_start_row
        )  # Starting KV position for this split
        var cache_start_pos: UInt32 = 0
        var cache_len: Int = offset_position.cache_len()
        var start_pos: UInt32 = offset_position.start_pos(cache_start_pos)

        # S consumer wrapper
        var s_cons = DecodeSConsumer(s_bars.consumer())
        var p_prod = DecodePProducer(p_bars.producer())
        var c_prod = DecodeCProducer(c_bars.producer())
        var warp_idx = warp.broadcast(warp_id())
        # 0..127 inside the softmax WG
        var lane_id = Int(thread_idx.x)
        # Lane mapping inside the softmax warpgroup
        var row: Int = lane_id & 0x3F  # 0..63
        var half: Int = lane_id >> 6  # 0 or 1
        # Column range this thread owns in P
        var col0: Int = half * Self.config.BN >> 1  # 0 or 32

        var q_head_idx: UInt32 = UInt32(block_idx.x) * UInt32(
            Self.config.BM
        ) + UInt32(row)
        var score_row = UInt32(block_idx.y)  # decode: single token per batch

        var mi: Scalar[Self.AccumType] = min_or_neg_inf[Self.AccumType]()
        var li: Scalar[Self.AccumType] = 0.0
        comptime log2e_f32 = Scalar[Self.AccumType](log2e)
        comptime half_load = (Self.config.BN >> 1)
        var scale_log2e = scale.cast[Self.AccumType]()

        var tiles_done: Int = 0
        # Use num_keys_this_split for loop bounds (each split processes its portion)
        var num_k_tiles = ceildiv(num_keys_this_split, Self.config.BN)
        while tiles_done < num_k_tiles:
            # Wait for an S slot to become ready
            var slot_idx: UInt32 = s_cons.wait()
            var s_tmem_slot = s0_tmem + slot_idx * s_stride

            tcgen05_fence_after()

            # Each thread reads one full 32-element row (128 rows x 32 columns)
            var s_row = tt_stack_allocation[
                dtype = Self.AccumType, address_space = AddressSpace.LOCAL
            ](row_major[half_load]())
            var s_row_val = tcgen05_ld[
                datapaths=32,
                bits=32,
                repeat=32,
                dtype = Self.AccumType,
                pack=False,
            ](s_tmem_slot)

            s_row.ptr.store(0, s_row_val)
            tcgen05_load_wait()

            s_cons.release()

            var s_row_val_vectorized = s_row.vectorize[2]()
            comptime vs_count = (half_load + 2 - 1) // 2

            @parameter
            for _vi in range(vs_count):
                s_row_val_vectorized[_vi] = (
                    s_row_val_vectorized[_vi] * scale_log2e
                )

            @parameter
            if NoMask or CausalMask:
                current_max = Self.apply_mask[
                    half_load, NonCausalMask=False, CausalMask=CausalMask
                ](
                    tiles_done,
                    col0,
                    num_keys,
                    s_row,
                    mask,
                    score_mod,
                    prompt_idx,
                    q_head_idx,
                    score_row,
                    max_seq_len,
                    cache_len,
                    start_pos,
                    cache_start_pos,
                    kv_start_row,  # Pass kv_start_row for split-K global position
                )
            else:
                current_max = Self.apply_mask[
                    half_load, NonCausalMask=True, CausalMask=False
                ](
                    tiles_done,
                    col0,
                    num_keys,
                    s_row,
                    mask,
                    score_mod,
                    prompt_idx,
                    q_head_idx,
                    score_row,
                    max_seq_len,
                    cache_len,
                    start_pos,
                    cache_start_pos,
                    kv_start_row,  # Pass kv_start_row for split-K global position
                )
            current_max *= log2e_f32

            # every softmax thread signals arrival on the shared-mem barrier
            comptime rescale_threshold: Float32 = Float32(
                -8 if size_of[Self.q_type]() >= 2 else 0
            )
            max_Smem_Tensor[lane_id] = current_max
            named_barrier[Int32(WARPGROUP_SIZE)](2)
            # 0 ^ 64 = 64
            # 1 ^ 64 = 65
            # 2 ^ 64 = 66
            # ...
            # 63 ^ 64 = 127
            # 64 ^ 64 = 0
            # 65 ^ 64 = 1
            # ...
            # 127 ^ 64 = 63
            var other_half_max = max_Smem_Tensor[lane_id ^ 64][0]
            current_max = max(current_max, other_half_max)
            var new_max: Scalar[Self.AccumType] = max(mi, current_max)
            var diff = sub_ftz(rebind[Float32](mi), rebind[Float32](new_max))
            var scale_for_old_max: Scalar[Self.AccumType]
            if _vote_nvidia_helper(diff < rescale_threshold) != 0:
                scale_for_old_max = rebind[Scalar[Self.AccumType]](exp2(diff))
            else:
                scale_for_old_max = 1.0
                new_max = mi
            var float2_register = s_row.vectorize[2]()
            var float2_current_sum: SIMD[Self.AccumType, 2] = 0.0

            @parameter
            for i in range(0, half_load // 2):
                var element = float2_register[i]
                float2_register[i] = exp2(element.fma(log2e_f32, -new_max))
                float2_current_sum += rebind[SIMD[Self.AccumType, 2]](
                    float2_register[i]
                )

            # compute softmax using S_tmem_slot -> produce probabilities in regs
            # Expose correction scalars in SMEM for Correction warpgroup
            if tiles_done > 0:
                c_prod.acquire()
                # write back the exp2f(mi - new_max); to the correction_max_smem
                # corr_max_Smem_Tensor[lane_id] = scale_for_old
                # Issue the TMEM store: 32 datapaths × 32 bits × repeat=1
                tcgen05_st[
                    datapaths=32,
                    bits=32,
                    repeat=1,
                    pack=False,
                ](corr_scale_tmem, scale_for_old_max)
                #  signal to the correction warpgroup:
                c_prod.commit()

            # wait until MMA has released P (consumer_mbar.phase matches)
            p_prod.acquire()
            var p_stage = p_prod.stage_index()  # 0 or 1
            var p_smem = kv_smem + (
                p_stage * UInt32(Self.KVStageElems)
                + UInt32(Self.NumVOBlocks * Self.BlockElems)
            )

            # Write P to shared memory (no scaling needed)
            write_bf16x2_row_to_smem_chunked[
                half_load,
                out_dtype = Self.q_type,
                in_dtype = Self.AccumType,
                config = Self.config,
            ](p_smem, s_row, col0, row)

            fence_async_view_proxy()
            # 128 threads call -> producer_mbar.arrive() (128 arrivals) + state.step()
            p_prod.commit()
            mi = new_max
            li = li.fma(
                scale_for_old_max, float2_current_sum[0] + float2_current_sum[1]
            )
            # now update the li scale for the next tile
            tiles_done += 1

        li_Smem_Tensor[lane_id] = li
        named_barrier[Int32(WARPGROUP_SIZE)](2)
        li += li_Smem_Tensor[lane_id ^ 64][0]

        # --------------------------------------------------------------------------
        # Split-K: Store partial LSE to lse_accum_split for combine kernel
        # --------------------------------------------------------------------------
        # LSE (Log-Sum-Exp) in log2 format: lse = log2(li) + mi
        # This allows the combine kernel to merge partial results:
        #   global_lse = log2(sum(exp2(lse_i - max_lse))) + max_lse
        #   scale_i = exp2(lse_i - global_lse)
        #   final_output = sum(scale_i * partial_output_i)
        #
        # LSE accumulator shape: (num_splits, batch_size, seq_len, num_heads)
        # Strides: stride_split = batch_size * seq_len * num_heads
        #          stride_batch = seq_len * num_heads
        #          stride_seq = num_heads
        @parameter
        if Self.config.decoding_warp_split_k:
            # Only threads with valid heads should write LSE
            # head_idx = block_idx.x * BM + row (where row is 0-63 for each half)
            # Each thread in the warpgroup handles one row (one head)
            # row = lane_id & 0x3F gives 0-63 for both halves
            # half = lane_id >> 6 gives 0 or 1
            # We only need one write per head, so half=0 threads write
            var head_idx = Int(block_idx.x) * Self.config.BM + row
            var half_idx = lane_id >> 6  # 0 for first half, 1 for second half

            if half_idx == 0 and head_idx < Self.config.num_q_heads:
                # Compute LSE in log2 format: log2(li) + mi
                # li is already the sum of exp2 values, mi is already in log2 scale
                var partial_lse = log2(li[0]) + mi

                # LSE offset calculation:
                # lse_accum_split shape: (num_splits, batch_size, max_seq_len, num_heads)
                # Use max_seq_len (not per-batch seq_len) for strides to match
                # the PADDED buffer layout and the combine kernel's read pattern.
                var seq_idx = Int(block_idx.y)
                var stride_batch = (
                    offset_position.max_seq_len * Self.config.num_q_heads
                )
                var stride_split = batch_size * stride_batch
                var stride_seq = Self.config.num_q_heads

                var lse_offset = (
                    offset_position.split_idx * stride_split
                    + offset_position.batch_idx * stride_batch
                    + seq_idx * stride_seq
                    + head_idx
                )
                # need to rebind the pointer to mutable pointer for write access
                var lse_ptr = rebind[
                    UnsafePointer[Scalar[Self.AccumType], origin=MutAnyOrigin]
                ](lse_accum_split_ptr.value())
                lse_ptr[lse_offset] = partial_lse

        # --------------------------------------------------------------------------
        # Epilogue: scale output by recip(li) and write to shared memory as bf16
        # --------------------------------------------------------------------------
        comptime assert (
            Self.AccumType == DType.float32
        ), "accumulator type should be float32"
        comptime assert (
            Self.output_type == DType.bfloat16
        ), "output type should be bfloat16"

        comptime DecodeOutProducerType = DecodeOutProducer[
            Self.output_type, Self.config
        ]
        comptime blocks_per_stage = DecodeOutProducerType.blocks_per_stage
        var o_tmem = tmem_addr + UInt32(Self.config.TMEM_O)

        # By the time we reach to epilogue the KV is free.
        # So we can safely use the KV buffer for writing the output and have more async write.
        # howere the write function massively change based on the mma_n size
        # so when mma_n is 256 we have the first 128 columns with waro0/1 and the
        # next 128 column with warp2/3 tiles and so on for the next 256 columns
        # it is 256/32 which is equivalent of 512/64

        comptime num_store_tiles = Self.config.depth // Self.config.BN
        comptime epi_half_load: UInt32 = UInt32(Self.config.BN >> 1)
        comptime chunk_size: Int = 16
        comptime total_elems: Int = Int(epi_half_load) * blocks_per_stage
        var out_prod = DecodeOutProducer[Self.output_type, Self.config](
            out_pipeline, out_smem
        )

        # Pre-compute scale factor
        var o_scale_li: Scalar[Self.AccumType] = (
            recip(li)[0] if li[0] != 0 else 0
        )

        var warp_pair = UInt32(warp_idx >> 1)
        var epi_col0: Int = Int(
            warp_pair * epi_half_load * UInt32((blocks_per_stage >> 1) ^ 1)
        )

        # Number of MMA PV rounds (outer loop) and iterations within each round (inner loop)
        # MMA_PV_N=256 processes 4 blocks (256/64=4) at a time
        # depth=512 has 8 blocks total, so 2 MMA PV rounds (512/256=2)
        # Each round has (MMA_PV_N/BN)/blocks_per_stage = (256/64)/2 = 2 iterations
        # corr_done_bars has 2 slots matching the 2 MMA PV rounds
        #   0       64     128     192      256      320      384     448     512
        #   |-------|-------|-------|--------|--------|--------|-------|-------|
        #     w0/1    w0/1     w2/3    w2/3     w0/1     w0/1     w2/3    w2/3
        # The pattern repeats every MMA_PV_N (256) columns
        comptime num_mma_pv_rounds = Self.config.depth // Self.config.MMA_PV_N
        comptime iters_per_mma_round = (
            Self.config.MMA_PV_N // Self.config.BN
        ) // blocks_per_stage

        @parameter
        for mma_round in range(num_mma_pv_rounds):
            # Wait for Correction to finish corrections for this MMA PV round
            corr_done_bars.mbar_base[mma_round].wait(0)

            # Fence to ensure all MMA writes to O TMEM are visible before we read
            tcgen05_fence_after()

            @parameter
            for slot in range(iters_per_mma_round):
                # Global iteration index combining mma_round and slot
                comptime i = mma_round * iters_per_mma_round + slot

                var o_tmem_base: UInt32 = o_tmem + UInt32(
                    i
                ) * epi_half_load * UInt32(blocks_per_stage)

                # Load all data for this tile into a LocalTensor
                var o_row_subtile = tt_stack_allocation[
                    dtype = Self.AccumType, address_space = AddressSpace.LOCAL
                ](row_major[total_elems]())
                o_row_subtile.ptr.store(
                    0,
                    tcgen05_ld[
                        datapaths=32,
                        bits=32,
                        repeat=total_elems,
                        dtype = Self.AccumType,
                        pack=False,
                    ](o_tmem_base),
                )
                tcgen05_load_wait()

                out_prod.acquire()
                var stage_ptr = out_prod.stage_base_ptr(
                    Int(warp_pair * UInt32(blocks_per_stage >> 1))
                )

                # Write O to shared memory with scaling
                write_bf16x2_row_to_smem_chunked[
                    total_elems,
                    out_dtype = Self.output_type,
                    in_dtype = Self.AccumType,
                    config = Self.config,
                    chunk_size=chunk_size,
                    scale_needed=True,
                ](stage_ptr, o_row_subtile, epi_col0, row, o_scale_li)

                out_prod.commit_step()

    # --------------------------------------------------------------------------
    # MLA decoding Correction kernel
    # --------------------------------------------------------------------------
    @staticmethod
    @always_inline
    fn Correction(
        tmem_addr: UInt32,
        o_bars: DecodeSM100MiscMBars[
            num_stages=2, num_producer=1, num_consumer=WARPGROUP_SIZE
        ],
        c_bars: DecodeSM100MiscMBars[
            num_stages=1,
            num_producer=WARPGROUP_SIZE,
            num_consumer=WARPGROUP_SIZE,
        ],
        corr_done_bars: DecodeSM100MiscMBars[
            num_stages=2,
            num_producer=WARPGROUP_SIZE,
            num_consumer=WARPGROUP_SIZE,
        ],
        offset_position: OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
        ],
    ):
        var o_tmem = tmem_addr + UInt32(Self.config.TMEM_O)
        var corr_scale_tmem = tmem_addr + UInt32(Self.config.TMEM_CORR_SCALE)
        var o_cons = DecodeOConsumer(o_bars.consumer())
        var c_cons = DecodeCConsumer(c_bars.consumer())
        var tiles_done: Int = 1

        var num_k_tiles = ceildiv(
            offset_position.num_keys_this_split, Self.config.BN
        )

        while tiles_done < num_k_tiles:
            # after computing per-row c_scalar from max/li:
            c_cons.wait()
            # 2) Issue TMEM load: 32 datapaths × 32 bits × repeat=1
            var scale_value = tcgen05_ld[
                datapaths=32,
                bits=32,
                repeat=1,
                dtype = Self.AccumType,
                pack=False,
            ](corr_scale_tmem)
            tcgen05_load_wait()
            c_cons.release()
            change = _vote_nvidia_helper(scale_value < 1.0) != 0
            if change:
                comptime num_o_tiles = Self.config.MMA_PV_N // (
                    Self.output_tile_width * 2
                )
                comptime o_range = Self.config.depth // Self.config.MMA_PV_N
                # the MNNA.ws split the output accross two warps
                comptime o_stride = Self.config.MMA_PV_N // 2

                @parameter
                for slot_idx in range(o_range):
                    o_cons.wait()

                    @parameter
                    for i in range(0, num_o_tiles):
                        # Here we load from o_tmem. it is 32 bit float and we load 64 fp32 element per tile
                        var o_tmem_subtile: UInt32 = (
                            o_tmem
                            + UInt32(i) * UInt32(Self.config.BN)
                            + UInt32(slot_idx) * UInt32(o_stride)
                        )
                        var o_row_subtile = tt_stack_allocation[
                            dtype = Self.AccumType,
                            address_space = AddressSpace.LOCAL,
                        ](row_major[Self.config.BN]())
                        o_row_subtile.ptr.store(
                            0,
                            tcgen05_ld[
                                datapaths=32,
                                bits=32,
                                repeat = Self.config.BN,
                                dtype = Self.AccumType,
                                pack=False,
                            ](o_tmem_subtile),
                        )
                        tcgen05_load_wait()

                        var float2_register = o_row_subtile.vectorize[2]()

                        @parameter
                        for j in range(0, Self.config.BN // 2):
                            var element = rebind[SIMD[Self.AccumType, 2]](
                                float2_register[j]
                            )
                            float2_register[j] = rebind[
                                type_of(float2_register[j])
                            ](element * SIMD[Self.AccumType, 2](scale_value[0]))
                        tcgen05_st[
                            datapaths=32,
                            bits=32,
                            repeat = Self.config.BN,
                            pack=False,
                        ](
                            o_tmem_subtile,
                            o_row_subtile.ptr.load[width = Self.config.BN](),
                        )
                    o_cons.release()
            else:
                o_cons.release()
                o_cons.release()
            tiles_done += 1

        # Wait on the final O from MMA before signaling Softmax
        o_cons.wait()
        # Signal to Softmax that first 4 blocks are ready (slot 0)
        _ = corr_done_bars.mbar_base[0].arrive()
        o_cons.release()
        # second stage of the correction pipeline
        o_cons.wait()
        # Signal to Softmax that all corrections are done and O is ready (slot 1)
        _ = corr_done_bars.mbar_base[1].arrive()
        # Release the final O barrier
        o_cons.release()

    # --------------------------------------------------------------------------
    # MLA decoding store kernel
    # --------------------------------------------------------------------------
    # If it goes to the batch loop remember correction is out of sync with MMA on
    # O_tmem wait and release as it starts one stage before MMA
    @staticmethod
    @always_inline
    fn store(
        out_pipeline: OutPipeline[
            num_out_stages = DecodeOutProducer[
                Self.output_type, Self.config
            ].num_out_stages,
            num_producer=WARPGROUP_SIZE,
            num_consumer=1,
        ],
        out_smem: SharedMemPointer[Scalar[Self.output_type]],
        o_tma: QOTMATile[
            dtype = Self.output_type,
            BM = Self.config.out_rows,
            BK = Self.config.BN,
            swizzle_mode = Self.config.swizzle_mode,
        ],
        offset_position: OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
            Self.config.decoding_warp_split_k,
        ],
    ):
        comptime DecodeOutConsumerType = DecodeOutConsumer[
            Self.output_type, Self.config
        ]
        comptime col_per_warp = DecodeOutConsumerType.col_per_warp
        comptime blocks_per_stage = DecodeOutConsumerType.blocks_per_stage
        comptime num_out_stages = DecodeOutConsumerType.num_out_stages
        comptime num_out_stages_per_mma = num_out_stages // num_mma_pv
        comptime num_store_tiles = Self.config.depth // Self.output_tile_width
        comptime num_mma_pv = Self.config.padded_depth // Self.config.MMA_PV_N
        var out_cons = DecodeOutConsumer[Self.output_type, Self.config](
            out_pipeline, out_smem
        )
        elect_mask = elect()
        var is_leader = elect_mask != 0
        var row: UInt = UInt(offset_position.out_row_offset)
        # The code work with the assumption that the num_q_heads is power of two.
        var tma_phase: UInt32 = 0

        #   0       64     128     192      256      320      384     448     512
        #   |-------|-------|-------|--------|--------|--------|-------|-------|
        #     w0/1    w0/1     w2/3    w2/3     w0/1     w0/1     w2/3    w2/3

        @parameter
        for n in range(0, num_mma_pv):

            @parameter
            for m in range(0, num_out_stages_per_mma):
                out_cons.wait()

                @parameter
                for k in range(0, blocks_per_stage):
                    var stage_ptr = out_cons.stage_base_ptr(k)
                    var col: UInt = UInt(
                        n * Self.config.MMA_PV_N
                        + m * Self.config.BN
                        + k * col_per_warp
                    )
                    var smem_tensor = SharedMemTensor[
                        Self.output_type,
                        type_of(o_tma).layout,
                    ](stage_ptr)
                    if is_leader:
                        fence_async_view_proxy()
                        o_tma.async_store(smem_tensor, (col, row))
                out_cons.release(elect_mask)
        if is_leader:
            o_tma.commit_group()
        o_tma.wait_group[0]()
