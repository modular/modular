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

from collections import OptionalReg
from math import ceildiv, exp2, recip, align_up, align_down, gcd
from math.constants import log2e
from sys import align_of, simd_width_of, size_of, env_get_int
import gpu.warp as warp
from algorithm.functional import unswitch
from bit import prev_power_of_two, pop_count
from buffer import NDBuffer
from collections import OptionalReg
from gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    barrier,
    thread_idx,
    block_idx,
    warp_id,
)
from nn.mha_utils import DynamicInt, NoPartition
from gpu.globals import WARPGROUP_SIZE, WARP_SIZE
from gpu.cluster import elect_one_sync
from gpu.host import DeviceContext, FuncAttribute, DeviceBuffer
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu.host.info import B200
from gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc, Scope
from gpu.memory import AddressSpace, external_memory, fence_async_view_proxy
from gpu.mma import MMAOperandDescriptor
from gpu.mma_sm100 import (
    MMASmemDescriptor,
    UMMAInsDescriptor,
    UMMAKind,
    mma,
    mma_arrive,
)

from gpu.sync import (
    named_barrier,
    mbarrier_arrive,
    mbarrier_try_wait_parity_shared,
    mbarrier_init,
    cp_async_bulk_commit_group,
    cp_async_bulk_wait_group,
)
from gpu.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_fence_after,
    tcgen05_fence_before,
    tcgen05_ld,
    tcgen05_load_wait,
    tcgen05_release_allocation_lock,
    tcgen05_st,
    tcgen05_store_wait,
)
from gpu.primitives.warp import _vote_nvidia_helper
from gpu.compute.arch.mma_nvidia_sm100 import MMASmemDescriptorPair
from layout.int_tuple import IntTuple, UNKNOWN_VALUE
from layout.layout import (
    Layout,
    blocked_product,
    composition,
    logical_divide,
    logical_product,
    make_layout,
)
from logger import Logger

from layout.layout_tensor import (
    LayoutTensor,
    LayoutTensorIter,
    copy_local_to_shared,
    copy_sram_to_dram,
    ThreadScope,
)
from layout.swizzle import make_swizzle
from layout.tensor_core_async import (
    tile_layout_k_major,
    tile_layout_mn_major,
    _CM_NUM_ROWS,
    _CM_ROW_BYTES,
    tile_to_descriptor,
)
from layout.tma_async import (
    PipelineState,
    SharedMemBarrier,
    SplitLastDimTMATensorTile,
    create_split_tma,
    create_tma_tile,
    _tma_desc_tile_layout,
    TMATensorTile,
)
from layout.runtime_layout import RuntimeLayout
from memory import bitcast
from nn.mha_fa3_utils import (
    _get_position,
    MHAPosition,
    NonNullPointer,
    NullPointer,
    OptionalPointer,
    output_reg_to_smem,
    output_reg_to_smem_st_matrix,
    Pack,
    produce,
    QTMATile,
)
from nn.mha_mask import MHAMask, TileMaskStatus
from nn.mha_operand import MHAOperand
from nn.mha_score_mod import ScoreModTrait
from nn.mha_tile_scheduler import (
    MHASchedulerSynchronization,
    MHATileScheduler,
    MHATileState,
    MHATileSummary,
    SeqInfo,
    TransientScheduler,
)
from gpu.host.nvidia.tma import TensorMapSwizzle
from nn.mha_utils import (
    FlashAttentionAlgorithm,
    MHAConfig,
    MHAPartitionScheme,
    OptionallyStaticInt,
    _is_decoding,
    _kernel_mask,
    get_start_and_end_for_partitions,
)
from nn.softmax import (
    _online_softmax_correction,
    _rowmax_online_softmax,
    _rowsum,
)
from utils.index import Index, IndexList
from utils.numerics import get_accum_type, min_or_neg_inf
from utils.static_tuple import StaticTuple
from linalg.arch.sm100.mma import smem_descriptor, _create_mma_desc_pair

from pathlib import Path
from nn.mha_sm100_2q import (
    FA4MiscMBars,
    elect,
    KVPipeline,
    TMemTile,
    bulk_mma,
    LocalTensor,
    elect_mma_arrive,
    ProducerPipeline,
    ConsumerPipeline,
    MBarPipeline,
)
from nn.mha_fa3_utils import q_smem_shape, q_gmem_shape, KVTMATile
from nn.mha import q_num_matrix_view_rows
from builtin.device_passable import DevicePassable
from sys._assembly import inlined_assembly
from nn.mha_mask import NullMask, MaterializedMask, CausalMask

comptime logger = Logger()


# ------------------------------------------------------------------------------
# MLA decoding configuration for SM100
# ------------------------------------------------------------------------------
struct MLA_SM100_Decode_Config:
    var MMA_M: Int
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
    comptime num_threads: Int = 384  # 1x softmax, 1x correction, 1x other
    var swizzle_mode: TensorMapSwizzle
    var kv_swizzle_mode: TensorMapSwizzle
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
        swizzle_mode: TensorMapSwizzle,
        kv_swizzle_mode: TensorMapSwizzle,
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
        self.dtype_size = dtype_size
        self.swizzle_mode = swizzle_mode
        self.kv_swizzle_mode = kv_swizzle_mode
        swizzle_elems = swizzle_mode.bytes() // dtype_size
        self.padded_depth = align_up(depth, swizzle_elems)
        self.padded_q_depth = align_up(q_depth, swizzle_elems)

        # 4 bytes for the TMEM base pointer
        var smem_use = 4
        self.BN = 64  # This can be increased since we are not doing sparse
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
        # bar_q → 1           producer pipeline - load consumer - mma
        # bar_kv_reay[2] → 2  consumer pipeline - mma
        # bar_kv_free[2] → 2   producer pipeline - load
        # bar_s_done[2] → 2  producer pipeline - mma
        # bar_s_ready[2] → 2  consumer pipeline - softmax
        # bar_p_done[2] → 2  producer pipeline- softmax
        # bar_p_ready[2] → 2  consumer pipeline - mma
        # bar_correction_done[1] → 1  producer pipeline- softmax
        # bar_correction_ready[1] → 1  consumer pipeline - mma
        # bar_o_done[1] → 1  producer pipeline- MMA
        # bar_o_ready[1] → 1  consumer pipeline - Correction
        # bar_li_done[1] → 1  producer pipeline- softmax
        # bar_li_ready[1] → 1  consumer pipeline - correction
        # Total: 1 + 2 + 2 + 2 + 2  +2 + 2  + 1 + 1 + 1 + 1  +1 + 1 = 19 transaction barriers.
        # If decoding_warp_split_k is True, we need to add 2 more barriers for the splitk.
        # bar_write_done[1] → 1 or  8  producer pipeline- Correction
        # bar_write_ready[1] → 1 or 8  consumer pipeline - write
        # However if it is false, we need to add ((Depth/BN) -1) x2  more barriers.
        # the splitk the two added is enough for now. if decoding_warp_split_k
        # is False, we need to add ((Depth/BN) -1) x2  more barriers.
        # for the splitk the two added is enough for now. because in that case
        # we can increase the pipeline number between epilogue and wite by reusing
        # the KV SMEM for output.
        var num_out_barrier = (
            1 if decoding_warp_split_k else self.depth // self.BN
        ) * 2
        # total number of barriers is 19 + num_out_barrier
        smem_use += ((19 + num_out_barrier) * Self.mbar_size) + Int(
            not decoding_warp_split_k
        ) * (((self.depth // self.BN) - 1) * 2 * Self.mbar_size)

        # for convinence we will add the smem for max and li to the smem_used for
        # correction separately (128 * 1 * 4 * 2) =1024 bytes
        # comptime smem_for_max_and_li_correction = 128 * 1 * 4 * 2
        # 4 + (64x576x2) + (128 * 1 * 4 * 2) = 74756 bytes
        # smem_use += smem_for_max_and_li_correction

        # 4 + (64x576x2) + (64x576x2x2) + (64x64x2) + (128 * 1 * 4 * 2) + (19x8)  = 230556 bytes
        # if decoding_warp_split_k is False, we need to add ((Depth/BN) -1) x2
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
        create_tma_tile[
            IndexList[2](BM, BK),
            swizzle_mode=swizzle_mode,
        ](ctx, tensor)
    )


# ------------------------------------------------------------------------------
# Helper functions for MLA decoding pack
# ------------------------------------------------------------------------------


@register_passable("trivial")
struct MLA_Decode_Pack[
    ValidLengthType: OptionalPointer,
    MaskType: MHAMask,
    ScoreModType: ScoreModTrait,
](Copyable, DevicePassable):
    var mask: Self.MaskType
    var score_mod: Self.ScoreModType
    var valid_length: Self.ValidLengthType

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
    ):
        self.mask = mask
        self.score_mod = score_mod
        self.valid_length = valid_length


# ------------------------------------------------------------------------------
# MLA decoding implementation for SM100
# ------------------------------------------------------------------------------


@always_inline
fn num_matrix_view_rows_decode[
    dtype: DType,
    //,
](q: LayoutTensor[dtype, ...]) -> Int:
    # q and out are (batch x seq_len=1 x num_heads , depth)
    var num_rows: Int = q.dim[0]()

    @parameter
    for i in range(1, q.rank - 1):
        num_rows *= q.dim[i]()
    return num_rows


fn mla_decode_sm100[
    q_type: DType,
    q_layout: Layout,
    k_t: MHAOperand,
    output_type: DType,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    valid_layout: Layout,
    config: MHAConfig,
    depth: Int,
    num_heads: Int,
    group: Int = 1,
    *,
    use_score_mod: Bool = False,
    ragged: Bool = False,
    _use_valid_length: Bool = False,
    _is_cache_length_accurate: Bool = False,
    decoding_warp_split_k: Bool = False,
](
    q: LayoutTensor[
        q_type, q_layout, address_space = AddressSpace.GENERIC, ...
    ],
    k: k_t,
    output: LayoutTensor[address_space = AddressSpace.GENERIC, ...],
    scale: Float32,
    batch_size: Int,
    num_partitions: Int,
    max_cache_valid_length: Int,  # longest KV cache entry
    valid_length: LayoutTensor[
        DType.uint32, address_space = AddressSpace.GENERIC, ...
    ],
    mask: mask_t,
    score_mod: score_mod_t,
    ctx: DeviceContext,
) raises:
    comptime mla_config = MLA_SM100_Decode_Config(
        num_q_heads=num_heads,
        group=group,  # num_q_heads/h_k(1)
        depth=Int(depth - 64),  # 512
        q_depth=Int(depth),  # 576
        dtype_size=size_of[q_type](),
        swizzle_mode=config.swizzle_mode,
        kv_swizzle_mode=TensorMapSwizzle.SWIZZLE_128B,
        page_size=k_t.page_size,
        decoding_warp_split_k=decoding_warp_split_k,
    )
    var num_rows_qo = num_matrix_view_rows_decode(q)
    q_ptr = rebind[UnsafePointer[Scalar[k_t.dtype], origin=MutAnyOrigin]](
        q.to_device_buffer(ctx).unsafe_ptr()
    )
    q_tma_op = tma_tile_qo[
        swizzle_mode = mla_config.swizzle_mode,
        BM = mla_config.BM,
        BK = mla_config.BN,
        depth = mla_config.q_depth,
    ](ctx, q_ptr, num_rows_qo)

    k_tma_op = k.create_tma_tile[
        BN = mla_config.BM,  # tile_m =64
        depth = mla_config.q_depth,
        BK = mla_config.BN,  # tile_n =64
        swizzle_mode = mla_config.kv_swizzle_mode,
    ](ctx)
    comptime output_tile_width = (mla_config.BN // 2) * (
        4 // size_of[output_type]()
    )
    o_ptr = rebind[UnsafePointer[Scalar[output_type], origin=MutAnyOrigin]](
        output.to_device_buffer(ctx).unsafe_ptr()
    )
    o_tma_op = tma_tile_qo[
        swizzle_mode = mla_config.swizzle_mode,
        BM = mla_config.out_rows,
        BK = mla_config.BN,
        depth = mla_config.depth,
    ](ctx, o_ptr, num_rows_qo)

    if ragged:
        comptime ValidLengthType = NonNullPointer[DType.uint32]
        var valid_len: ValidLengthType = {
            valid_length.to_device_buffer(ctx).unsafe_ptr()
        }
        launch_mla_sm100_decode_enqueue_kernel[
            KVLUTType=k_t,
            output_type=output_type,
            MaskType=mask_t,
            ScoreModType=score_mod_t,
            config=mla_config,
            use_score_mod=use_score_mod,
            ValidLengthType=ValidLengthType,
            ragged=True,
            _use_valid_length=_use_valid_length,
            _is_cache_length_accurate=_is_cache_length_accurate,
        ](
            q_tma_op,
            k_tma_op,
            o_tma_op,
            k,
            scale,
            batch_size,
            num_partitions,
            max_cache_valid_length,
            valid_len,
            mask,
            score_mod,
            ctx,
        )
    else:
        comptime ValidLengthType = NullPointer[DType.uint32]
        var valid_len: ValidLengthType = {}
        launch_mla_sm100_decode_enqueue_kernel[
            KVLUTType=k_t,
            output_type=output_type,
            MaskType=mask_t,
            ScoreModType=score_mod_t,
            config=mla_config,
            use_score_mod=use_score_mod,
            ValidLengthType=ValidLengthType,
            ragged=False,
            _use_valid_length=_use_valid_length,
            _is_cache_length_accurate=_is_cache_length_accurate,
        ](
            q_tma_op,
            k_tma_op,
            o_tma_op,
            k,
            scale,
            batch_size,
            num_partitions,
            max_cache_valid_length,
            valid_len,
            mask,
            score_mod,
            ctx,
        )


@always_inline
fn launch_mla_sm100_decode_enqueue_kernel[
    KVLUTType: MHAOperand,
    output_type: DType,
    MaskType: MHAMask,
    ScoreModType: ScoreModTrait,
    config: MLA_SM100_Decode_Config,
    use_score_mod: Bool,
    ValidLengthType: OptionalPointer,
    _use_valid_length: Bool = False,
    _is_cache_length_accurate: Bool = False,
    ragged: Bool = False,
](
    q_tma: QOTMATile[
        dtype = KVLUTType.dtype,
        BM = config.BM,
        BK = config.BN,
        swizzle_mode = config.swizzle_mode,
    ],
    k_tma: KVTMATile[
        dtype = KVLUTType.dtype,
        swizzle_mode = config.kv_swizzle_mode,
        BN = config.BM,
        BK = config.BN,
    ],
    o_tma: QOTMATile[
        dtype=output_type,
        BM = config.out_rows,
        BK = config.BN,
        swizzle_mode = config.swizzle_mode,
    ],
    kv_lut: KVLUTType,
    scale: Float32,
    batch_size: Int,
    num_partitions: Int,
    max_cache_valid_length: Int,  # longest KV cache entry,
    valid_len: ValidLengthType,
    mask: MaskType,
    score_mod: ScoreModType,
    ctx: DeviceContext,
) raises:
    var mla_decode_pack = MLA_Decode_Pack[
        ValidLengthType=ValidLengthType,
        MaskType=MaskType,
        ScoreModType=ScoreModType,
    ](mask, score_mod, valid_len)
    var block_x = ceildiv(config.num_q_heads, config.BM)
    # TODO: this should be seq_len and batch to be distributed across the grid
    var block_y = batch_size
    var block_z = 1  # num_partitions
    var grid_dim = (block_x, block_y, block_z)
    # we have 3 warp groups:
    # - one for load/store/2xMMA
    # - one for compute softmax
    # - one for compute correction
    var num_threads = 128 * 3
    var block_dim = (num_threads, 1, 1)
    logger.info(
        "block_dim:",
        block_dim[0],
        block_dim[1],
        block_dim[2],
        "grid_dim:",
        grid_dim[0],
        grid_dim[1],
        grid_dim[2],
        "config.smem_used:",
        config.smem_used,
        "config.num_q_heads:",
        config.num_q_heads,
        "config.num_kv_heads:",
        config.num_kv_heads,
        "config.num_threads:",
        num_threads,
        "config.num_kv_stages:",
        config.num_kv_stages,
        "config.BM:",
        config.BM,
        "config.BN:",
        config.BN,
        "config.BK0:",
        config.BK0,
        "config.BK1:",
        config.BK1,
        "config.q_depth:",
        config.q_depth,
        "config.depth:",
        config.depth,
        "config.padded_depth:",
        config.padded_depth,
        "config.padded_q_depth:",
        config.padded_q_depth,
        "config.rope_depth:",
        config.rope_depth,
        "config.swizzle_mode:",
        config.swizzle_mode,
        "max_cache_valid_length:",
        max_cache_valid_length,
        "output_tile_width:",
        (config.BN // 2) * (4 // size_of[output_type]()),
    )

    logger.info("------ Dispatching to SM100 MLA-SM100-DECODE ------")
    logger.info(
        "QK Type:",
        KVLUTType.dtype,
        "Q Depth:",
        config.q_depth,
        "Number of Q // KV Heads:",
        config.num_q_heads,
        "//",
        config.num_kv_heads,
        "Batch Size:",
        batch_size,
        "Num Partitions:",
        num_partitions,
        "Max Cache Valid Length:",
        max_cache_valid_length,
    )
    comptime kernel = MLA_SM100_Decode[
        KVLUTType=KVLUTType,
        output_type=output_type,
        MaskType=MaskType,
        ScoreModType=ScoreModType,
        config=config,
        use_score_mod=use_score_mod,
        ValidLengthType=ValidLengthType,
        _is_cache_length_accurate=_is_cache_length_accurate,
        _use_valid_length=_use_valid_length,
        ragged=ragged,
    ].kernel
    ctx.enqueue_function[kernel, kernel](
        q_tma,
        k_tma,
        o_tma,
        kv_lut,
        scale,
        batch_size,
        num_partitions,
        max_cache_valid_length,
        mla_decode_pack,
        grid_dim=grid_dim,
        block_dim=block_dim,
        shared_mem_bytes=config.smem_used,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            config.smem_used
        ),
    )


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
# Offset position struct
# ------------------------------------------------------------------------------
@register_passable("trivial")
struct OffsetPosition[
    config: MLA_SM100_Decode_Config,
    KVLUTType: MHAOperand,
    ragged: Bool,
    use_valid_length: Bool,
    is_cache_length_accurate: Bool,
    ValidLengthType: OptionalPointer,
]:
    var start_of_seq: Int
    var end_of_seq: Int
    var seq_len: Int
    var q_batch_offset: Int
    var num_keys: Int

    @always_inline
    fn __init__(
        out self,
        k: Self.KVLUTType,
        valid_length: UnsafePointer[
            Scalar[Self.ValidLengthType.dtype], origin=MutAnyOrigin
        ],
    ):
        self.start_of_seq = 0
        self.end_of_seq = 0
        self.seq_len = 0
        self.q_batch_offset = 0
        self.num_keys = 0

        if Self.ragged:
            # treat valid_lengths as a input_row_offsets
            self.start_of_seq = Int(valid_length[Int(block_idx.y)])
            self.end_of_seq = Int(valid_length[Int(block_idx.y) + 1])
            self.start_of_seq = self.start_of_seq
            self.end_of_seq = self.end_of_seq
            self.seq_len = self.end_of_seq - self.start_of_seq
            self.q_batch_offset = (
                self.start_of_seq
                * Int(Self.config.q_depth)
                * Int(Self.config.num_q_heads)
            )
        elif Self.use_valid_length:
            # treat valid_lengths as valid lengths
            self.q_batch_offset = Int(
                Self.config.q_depth * Self.config.num_q_heads * Int(block_idx.y)
            )
            self.seq_len = Int(valid_length[block_idx.y])
        else:
            self.seq_len = 1
            self.q_batch_offset = Int(
                Self.config.q_depth * Self.config.num_q_heads * Int(block_idx.y)
            )

        self.num_keys = k.cache_length(Int(block_idx.y))

        @parameter
        if not Self.is_cache_length_accurate:
            self.num_keys += self.seq_len


# ------------------------------------------------------------------------------
# MLA decoding ProducerKVPipeline
# ------------------------------------------------------------------------------


@register_passable("trivial")
struct DecodeKVProducer[dtype: DType, config: MLA_SM100_Decode_Config]:
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
        # Only producer OR consumer should call init(), not both.
        self.pipe.init()

    @always_inline
    fn stage_base_ptr[
        *, mma_stage: Int = 0
    ](self) -> SharedMemPointer[Scalar[Self.dtype]]:
        # Which KV stage (0..num_kv_stages-1)?
        var stage_idx: UInt32 = self.pipe.state.index()
        var stage_offset: UInt32 = stage_idx * Self.kv_stage_elems
        return self.smem + stage_offset

    @always_inline
    fn stage_index[*, mma_stage: Int = 0](self) -> UInt32:
        return self.pipe.state.index()

    @always_inline
    fn producer_mbar[*, mma_stage: Int = 0](self) -> MBarType:
        return self.pipe.producer_mbar[mma_stage]()

    @always_inline("nodebug")
    fn acquire[*, mma_stage: Int = 0](self):
        # Block until consumer has released this stage
        self.pipe.producer_acquire[mma_stage]()

    @always_inline("nodebug")
    fn commit_step(mut self):
        # After we have launched TMA copies for this stage
        # we advance producer's logical stage index.
        self.pipe.state.step()


# ------------------------------------------------------------------------------
# MLA decoding ConsumerKVPipeline
# ------------------------------------------------------------------------------


@register_passable("trivial")
struct DecodeKVConsumer[dtype: DType, config: MLA_SM100_Decode_Config]:
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
        *, mma_stage: Int = 0
    ](self) -> SharedMemPointer[Scalar[Self.dtype]]:
        var stage_idx: UInt32 = self.pipe.state.index()
        var stage_offset: UInt32 = stage_idx * Self.kv_stage_elems
        return self.smem + stage_offset

    @always_inline
    fn stage_index[*, mma_stage: Int = 0](self) -> UInt32:
        return self.pipe.state.index()

    @always_inline("nodebug")
    fn wait[*, mma_stage: Int = 0](self):
        # Wait on producer mbar for (current index, current phase)
        self.pipe.consumer_wait[mma_stage]()

    @always_inline("nodebug")
    fn release[*, mma_stage: Int = 0](mut self, e: Int32):
        # Signal "stage consumed" to the producer via consumer mbar
        self.pipe.consumer_release[mma_stage](e)


# ------------------------------------------------------------------------------
# MLA decoding ProducerKVPipeline
# ------------------------------------------------------------------------------


@register_passable("trivial")
struct KVPipelineGeneric[
    num_kv_stages: Int,
    num_mma_stages: Int,
    num_producer: Int,
    num_consumer: Int,
]:
    """
    KVPipeline has `num_kv_stages * num_mma_stages` stages.
    `num_kv_stages` refers to how many `K` and `V` tiles we pipeline
    for performing the `S = Q@K'` and `O += P@V` MMAs.
    Each of these MMAs is broken up into `num_mma_stages` pipelined
    MMAs. We set `step=False` for all but the last MMA that completes
    the operation.
    An alternative implementation would separate the two, and potentially
    allow for more overall stages at the cost of slightly more bookkeeping.
    """

    comptime num_stages: Int = Self.num_kv_stages * Self.num_mma_stages

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
            self.mbar[i].init(Self.num_producer)

        @parameter
        for i in range(Self.num_stages, Self.num_stages * 2):
            self.mbar[i].init(Self.num_consumer)

    @always_inline
    fn producer_mbar[mma_stage: Int](self) -> MBarType:
        var idx: UInt32 = self.state.index()
        return self.mbar + Self.num_mma_stages * idx + mma_stage

    @always_inline
    fn consumer_mbar[mma_stage: Int](self, idx: UInt32) -> MBarType:
        comptime const_offset = mma_stage + Self.num_stages
        return self.mbar + Self.num_mma_stages * idx + const_offset

    @always_inline
    fn consumer_mbar[mma_stage: Int](self) -> MBarType:
        return self.consumer_mbar[mma_stage](self.state.index())

    @always_inline("nodebug")
    fn producer_acquire[mma_stage: Int = Self.num_mma_stages - 1](self):
        """
        Returns the dynamic pipe idx.
        """
        self.consumer_mbar[mma_stage]()[].wait(self.state.phase())

    @always_inline("nodebug")
    fn consumer_wait[mma_stage: Int = Self.num_mma_stages - 1](self):
        self.producer_mbar[mma_stage]()[].wait(self.state.phase())

    @always_inline("nodebug")
    fn consumer_release[
        mma_stage: Int = Self.num_mma_stages - 1
    ](mut self, e: Int32):
        elect_mma_arrive(self.consumer_mbar[mma_stage](), e)

        @parameter
        if mma_stage == Self.num_mma_stages - 1:
            self.state.step()

    @staticmethod
    @always_inline
    fn num_mbars() -> UInt32:
        return 2 * Self.num_mma_stages * Self.num_kv_stages


@register_passable("trivial")
struct TMADestination[dtype: DType, layout: Layout]:
    var mbar: MBarType
    var smem: SharedMemTensor[Self.dtype, Self.layout]

    @always_inline
    fn __init__(
        out self, mbar: MBarType, smem: SharedMemTensor[Self.dtype, Self.layout]
    ):
        self.mbar = mbar
        self.smem = smem


# ------------------------------------------------------------------------------
# MLA decoding MiscMBars for producer and consumer
# ------------------------------------------------------------------------------
@register_passable("trivial")
struct DecodeSM100MiscMBars[
    num_stages: Int, num_producer: Int, num_consumer: Int
]:
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
            num_producer = Self.num_producer, num_consumer = Self.num_consumer
        ]()

    @always_inline
    fn producer(self) -> ProducerPipeline[Self.num_stages]:
        # ProducerPipeline assumes layout [prod0..prodN-1][cons0..consN-1]
        return {self.mbar_base}

    @always_inline
    fn consumer(self) -> ConsumerPipeline[Self.num_stages]:
        return {self.mbar_base}

    @always_inline
    fn end(self) -> MBarType:
        # We consumed 2 * s_num_stages mbars: prod[2] + cons[2]
        return self.mbar_base + 2 * Self.num_stages


# ------------------------------------------------------------------------------
# MLA decoding S pipeline betweeen MMA and Softmax
# ------------------------------------------------------------------------------
########## Producer of the S slot ##########
@register_passable("trivial")
struct DecodeSProducer:
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
@register_passable("trivial")
struct DecodeSConsumer:
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
@register_passable("trivial")
struct DecodePProducer:
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
@register_passable("trivial")
struct DecodePConsumer:
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
@register_passable("trivial")
struct DecodeOProducer:
    comptime ONumStages = 1
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
@register_passable("trivial")
struct DecodeOConsumer:
    comptime ONumStages = 1
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
@register_passable("trivial")
struct DecodeCProducer:
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


@register_passable("trivial")
struct DecodeCConsumer:
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


@register_passable("trivial")
struct OutPipeline[num_out_stages: Int, num_producer: Int, num_consumer: Int]:
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
            self.mbar[i].init(Self.num_producer)

        @parameter
        for i in range(Self.num_stages):
            (self.mbar + Self.num_stages)[i].init(Self.num_consumer)

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
        return 2 * Self.num_stages


@register_passable("trivial")
struct DecodeOutProducer[dtype: DType, config: MLA_SM100_Decode_Config]:
    comptime num_out_stages: Int = 1 if Self.config.decoding_warp_split_k else Self.config.depth // Self.config.BN
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
    fn stage_base_ptr(self) -> SharedMemPointer[Scalar[Self.dtype]]:
        var stage_idx: UInt32 = self.pipe.state.index()
        var stage_offset: UInt32 = stage_idx * Self.out_stage_elems
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


@register_passable("trivial")
struct DecodeOutConsumer[dtype: DType, config: MLA_SM100_Decode_Config]:
    comptime num_out_stages: Int = 1 if Self.config.decoding_warp_split_k else Self.config.depth // Self.config.BN
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
    fn stage_base_ptr(self) -> SharedMemPointer[Scalar[Self.dtype]]:
        var stage_idx: UInt32 = self.pipe.state.index()
        var stage_offset: UInt32 = stage_idx * Self.out_stage_elems
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
@register_passable("trivial")
struct DecodeSM100TensorAccumulatorSS[
    operand_type: DType,
    accum_type: DType,
    *,
    config: MLA_SM100_Decode_Config,
]:
    # Common geometry
    comptime BM = Self.config.BM  # 64
    comptime BN = Self.config.BN  # 64

    # MMA shapes (same for QK and PV)
    comptime MMA_M = Self.config.BM  # 64 rows
    comptime MMA_N = Self.config.BN  # 64 cols
    comptime MMA_K = Self.config.MMA_K  # 16

    # K-depth per logical block
    comptime BK = Self.config.BN  # 64
    comptime num_k_mmas = Self.BK // Self.MMA_K

    # Datatype / layout common bits
    comptime operand_size = size_of[Self.operand_type]()
    comptime a_swizzle = Self.config.swizzle_mode
    comptime b_swizzle = Self.config.kv_swizzle_mode

    # S tile (QKᵀ result) geometry – only QK uses these
    comptime S_M = Self.config.BM * 2  # 128
    comptime S_N = Self.config.BN // 2  # 32

    # O tile (PV result) geometry – only PV uses these
    comptime O_M = Self.config.BM * 2  # 128
    comptime O_N = Self.config.padded_depth // 2  # 256


@register_passable("trivial")
struct DecodeSM100QKTSS[
    operand_type: DType,
    accum_type: DType,
    *,
    config: MLA_SM100_Decode_Config,
]:
    # Bring in common geometry / constants
    comptime TensorAccumulatorSS = DecodeSM100TensorAccumulatorSS[
        Self.operand_type,
        Self.accum_type,
        config = Self.config,
    ]

    comptime CTileType = TMemTile[
        Self.accum_type,
        Self.TensorAccumulatorSS.S_M,
        Self.TensorAccumulatorSS.S_N,
    ]

    # ----- A (Q) tile layout -----
    comptime ALayout = tile_layout_k_major[
        Self.operand_type,
        Self.TensorAccumulatorSS.BM,  # 64 rows
        Self.TensorAccumulatorSS.BN,  # 64 cols
        Self.TensorAccumulatorSS.a_swizzle,
    ]()

    # ----- B (K) tile layout -----
    comptime BLayout = tile_layout_k_major[
        Self.operand_type,
        Self.TensorAccumulatorSS.BM,  # 64 rows
        Self.TensorAccumulatorSS.BN,  # 64 cols
        Self.TensorAccumulatorSS.b_swizzle,
    ]()

    # ----- Instruction descriptor -----
    comptime UMMAInstDesc = UMMAInsDescriptor[UMMAKind.KIND_F16].create[
        Self.accum_type,
        Self.operand_type,
        Self.operand_type,
        Index[dtype = DType.uint32](
            Self.TensorAccumulatorSS.MMA_M, Self.TensorAccumulatorSS.MMA_N
        ),
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
            BMN = Self.TensorAccumulatorSS.BM,  # 64 rows
            BK = Self.TensorAccumulatorSS.BN,  # 64 (padded_q_depth)
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
            BMN = Self.config.BM,  # 64 rows
            BK = Self.config.BN,  # 64 columns
            swizzle_mode = Self.config.kv_swizzle_mode,
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
        __comptime_assert stage_idx == 0, "stage_idx should be 0"
        bulk_mma_ws[
            kind = UMMAKind.KIND_F16,
            layout_a = Self.ALayout,
            layout_b = Self.BLayout,
            num_k_mmas = Self.TensorAccumulatorSS.num_k_mmas,
            operand_size = Self.TensorAccumulatorSS.operand_size,
            tcgen05_mma_type="tcgen05.mma.ws.cta_group::1.",
        ](Self.UMMAInstDesc, a, b, c, c_scale, elect)


@register_passable("trivial")
struct DecodeSM100PVSS[
    operand_type: DType,
    accum_type: DType,
    *,
    config: MLA_SM100_Decode_Config,
]:
    # Shared base
    comptime TensorAccumulatorSS = DecodeSM100TensorAccumulatorSS[
        Self.operand_type,
        Self.accum_type,
        config = Self.config,
    ]

    comptime CTileType = TMemTile[
        Self.accum_type,
        Self.TensorAccumulatorSS.O_M,
        Self.TensorAccumulatorSS.O_N,
    ]

    # ----- A (P) tile layout -----
    # P tiles are treated as mn-major in smem
    comptime ALayout = tile_layout_k_major[
        Self.operand_type,
        Self.TensorAccumulatorSS.BM,  # 64
        Self.TensorAccumulatorSS.BN,  # 64
        Self.TensorAccumulatorSS.a_swizzle,
    ]()

    # ----- B (V) tile layout -----
    # V tiles are mn-major (is_k_major = False in descriptor_v_block)

    comptime BLayout = tile_layout_mn_major[
        Self.operand_type,
        Self.TensorAccumulatorSS.BM,  # 64
        Self.TensorAccumulatorSS.BN,  # 64
        Self.TensorAccumulatorSS.a_swizzle,
    ]()
    # ----- Instruction descriptor -----
    comptime UMMAPVSS = UMMAInsDescriptor[UMMAKind.KIND_F16].create[
        Self.accum_type,
        Self.operand_type,
        Self.operand_type,
        Index[dtype = DType.uint32](
            Self.TensorAccumulatorSS.MMA_M, Self.TensorAccumulatorSS.MMA_N
        ),
        transpose_b=False,  # P (k-major) * V (mn-major) = no transpose
    ]()

    @staticmethod
    @always_inline
    fn descriptor_v_block(
        kv_smem: SharedMemPointer[Scalar[Self.operand_type]],
    ) -> MMASmemDescriptorPair:
        var base = kv_smem
        # Layout is 64 x 64, mn-major, same swizzle as k_tma
        return smem_descriptor[
            BMN = Self.TensorAccumulatorSS.BM,  # 64 rows
            BK = Self.TensorAccumulatorSS.BN,  # 64 columns
            swizzle_mode = Self.config.kv_swizzle_mode,
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
            BMN = Self.TensorAccumulatorSS.BM,  # 64 rows
            BK = Self.TensorAccumulatorSS.BN,  # 64 columns
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
        __comptime_assert stage_idx == 0, "stage_idx should be 0"
        bulk_mma_ws[
            kind = UMMAKind.KIND_F16,
            layout_a = Self.ALayout,
            layout_b = Self.BLayout,
            num_k_mmas = Self.TensorAccumulatorSS.num_k_mmas,
            operand_size = Self.TensorAccumulatorSS.operand_size,
            tcgen05_mma_type="tcgen05.mma.ws.cta_group::1.",
        ](Self.UMMAPVSS, a, b, c, c_scale, elect)


# ------------------------------------------------------------------------------
# Helper functions for writing from local memory to shared memory using swizzle
# ------------------------------------------------------------------------------


@always_inline("nodebug")
fn write_bf16x2_row_to_smem_fast[
    layout: Layout,
    *,
    out_dtype: DType,
    in_dtype: DType,
    config: MLA_SM100_Decode_Config,
    local_tile_size: Int,
](
    shared_mem: SharedMemPointer[Scalar[out_dtype]],
    local_mem: LocalTensor[in_dtype, layout],
    col_start: Int,
    row_start: Int,
):
    # Swizzle at 128B mode – operates on 16B (8-element) chunk indices
    comptime swz = make_swizzle[out_dtype, config.swizzle_mode]()
    comptime element_per_write = 8
    comptime tile_number = local_tile_size // element_per_write
    var vetorized_local_mem = local_mem.vectorize[element_per_write]()

    # For each group of 8 F32 -> 16 BF16 (4×bf16x2) = 128 bits
    @parameter
    for g in range(0, tile_number):  # 4 * 8 = 32 elems
        # Columns in P for this 128-bit chunk
        var col_base: Int = (
            col_start + g * element_per_write
        )  # 0,8,16,24 or 32,40,48,56
        # todo: this one always assume row major
        var logical_elem: Int = row_start * config.BN + col_base
        physical_offset = swz(logical_elem)

        shared_mem.store(
            physical_offset, (vetorized_local_mem[g].cast[out_dtype]())
        )


# ------------------------------------------------------------------------------
# MLA decoding kernel struct for SM100
# ------------------------------------------------------------------------------
@register_passable("trivial")
struct MLA_SM100_Decode[
    KVLUTType: MHAOperand,
    output_type: DType,
    MaskType: MHAMask,
    ScoreModType: ScoreModTrait,
    config: MLA_SM100_Decode_Config,
    use_score_mod: Bool,
    ValidLengthType: OptionalPointer,
    _is_cache_length_accurate: Bool = False,
    _use_valid_length: Bool = False,
    ragged: Bool = False,
]:
    comptime qkv_type = Self.KVLUTType.dtype
    comptime AccumType = get_accum_type[Self.qkv_type]()
    # 576 / 64 = 9
    comptime NumQKBlocks = Self.config.padded_q_depth // Self.config.BN
    # 512 / 64 = 8
    comptime NumVOBlocks = Self.config.padded_depth // Self.config.BN
    # 64 * 64 = 4096
    comptime BlockElems = Self.config.BM * Self.config.BN
    # 2 bytes for float16
    comptime bytes_per_element = size_of[Self.qkv_type]()
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
        operand_type = Self.qkv_type,
        accum_type = Self.AccumType,
        config = Self.config,
    ]
    comptime UMMAPVSS = DecodeSM100PVSS[
        operand_type = Self.qkv_type,
        accum_type = Self.AccumType,
        config = Self.config,
    ]

    # --------------------------------------------------------------------------
    # MLA decoding main kernel function
    # --------------------------------------------------------------------------
    #    KSlot0 (tile 0)        KSlot1 (tile 1)
    #          |                    |
    #          V                    V
    #    UMMA WS → S0         UMMA WS → S1
    #          |                    |
    #   arrive mbar_s0        arrive mbar_s1
    #          |                    |
    #          |---- Softmax Warpgroup ----|
    #          |                           |
    #          V                           V
    #       wait_s0                      wait_s1
    #       S0 → P0                      S1 → P1
    #           |                          |
    #        UMMA WS → O                   |
    #           |                          |
    #           |                          |
    #       arrive mbar_0                  |
    #           |                          |
    #           |                          |
    #           V---- Coorection WG  ------|
    #                                      |
    #                                  UMMA WP → O
    #                               arrive mbar_0
    #                                      |
    #                                 Coorection WG1 → O
    #                                      |
    #                                    wait_O_filled
    #                                     C_WG
    #                                      |
    #                                     wair_out
    #                                       |
    #                                    Write WG
    #                                      |
    #                                     W_WG
    #
    #

    # --------------------------------------------------------------------------
    # MLA decoding SMEMDescriptors for Q, K, V, P
    # --------------------------------------------------------------------------

    @staticmethod
    @__llvm_arg_metadata(q_tma, `nvvm.grid_constant`)
    @__llvm_arg_metadata(k_tma, `nvvm.grid_constant`)
    @__llvm_arg_metadata(o_tma, `nvvm.grid_constant`)
    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Self.config.num_threads
        )
    )
    fn kernel(
        q_tma: QOTMATile[
            dtype = Self.qkv_type,
            BM = Self.config.BM,
            BK = Self.config.BN,
            swizzle_mode = Self.config.swizzle_mode,
        ],
        k_tma: KVTMATile[
            dtype = Self.qkv_type,
            swizzle_mode = Self.config.kv_swizzle_mode,
            BN = Self.config.BM,
            BK = Self.config.BN,
        ],
        o_tma: QOTMATile[
            dtype = Self.output_type,
            BM = Self.config.out_rows,
            BK = Self.config.BN,
            swizzle_mode = Self.config.swizzle_mode,
        ],
        kv_lut: Self.KVLUTType,
        scale: Float32,
        batch_size: Int,
        num_partitions: Int,
        max_cache_valid_length: Int,  # longest KV cache entry,
        mla_decode_pack: MLA_Decode_Pack[
            ValidLengthType = Self.ValidLengthType,
            MaskType = Self.MaskType,
            ScoreModType = Self.ScoreModType,
        ],
    ):
        comptime num_reg_softmax = 192
        comptime num_reg_correction = 192
        comptime num_reg_other = 112
        mask = mla_decode_pack.mask
        score_mod = mla_decode_pack.score_mod
        valid_length = mla_decode_pack.valid_length
        q_smem = external_memory[
            Scalar[Self.qkv_type],
            address_space = AddressSpace.SHARED,
            alignment=128,
            name="mha_dynamic_shared_memory",
        ]()
        var kv_smem = q_smem + Self.BlockElems * Self.NumQKBlocks
        comptime kv_total_stages = Self.config.num_kv_stages
        # to reuse the K for V as well, we break KV as 9 stages of 64x64 to cover 64x576
        comptime kv_smem_total = Self.BlockElems * Self.NumQKBlocks * kv_total_stages

        # we need to use the KSmem for out pointer
        # We move P to the last slot of KV pipeline SO now we have tile of 64x
        # 32 of flot or 64x64 of FP16 to save output into
        # tiles in SMEM and smooth the pipeline for the next batch if we use splitk
        var out_smem_start = (
            kv_smem
            + kv_smem_total if Self.config.decoding_warp_split_k else kv_smem
        )
        # there is potential to have two Tmem for S, because we have two K so we can
        # unblock the MMA while loading S to reg for softrmax
        # if it was splitk we need to use the extra P slot. If not we need
        # to clear the KV slot before statting the max because KV slot is used by
        # MMA/load when max is valid.
        var out_smem_total = (
            Self.BlockElems if Self.config.decoding_warp_split_k else kv_smem_total
        )

        var out_smem = out_smem_start.bitcast[Scalar[Self.output_type]]()

        var max_smem = (out_smem + out_smem_total).bitcast[
            Scalar[Self.AccumType]
        ]()

        var li_smem = (
            max_smem + WARPGROUP_SIZE
        )  # 128 x1 for SMEM correction for Softmax
        #  Now we have to define MBARS for the kernel
        var mbar_base: MBarType = (li_smem + WARPGROUP_SIZE).bitcast[
            SharedMemBarrier
        ]()

        var mbar_q: MBarType = mbar_base  # q uses 0
        var mbar_kv_base: MBarType = mbar_base + 1  # barrier total[1]

        var kv_pipeline = KVPipelineGeneric[
            num_kv_stages = Self.config.num_kv_stages,  # 2
            num_mma_stages=1,
            num_producer=1,
            num_consumer=2,
        ](mbar_kv_base)

        # Move mbar_base to the first free barrier *after* KV:
        mbar_base = mbar_kv_base + kv_pipeline.num_mbars()  # kv uses 1..4
        # Move mbar_base to the first free barrier *after* k done:
        var s_bars = DecodeSM100MiscMBars[
            num_stages=2, num_producer=1, num_consumer=WARPGROUP_SIZE
        ](
            mbar_base
        )  # S uses 5..8
        mbar_base = s_bars.end()  # barrier total[9]
        var p_bars = DecodeSM100MiscMBars[
            num_stages=2, num_producer=WARPGROUP_SIZE, num_consumer=1
        ](
            mbar_base
        )  # P uses 9 .. 12
        mbar_base = p_bars.end()  # barrier total [13]
        var o_bars = DecodeSM100MiscMBars[
            num_stages=1, num_producer=1, num_consumer=WARPGROUP_SIZE
        ](
            mbar_base
        )  # O uses 13 and 14
        mbar_base = o_bars.end()  # barrier total [15]
        # C pipeline, Softmax -> Correction
        var c_bars = DecodeSM100MiscMBars[
            num_stages=1,
            num_producer=WARPGROUP_SIZE,
            num_consumer=WARPGROUP_SIZE,
        ](
            mbar_base
        )  # C uses 15 and 16
        mbar_base = c_bars.end()  # barrier total [17]
        var li_bars = DecodeSM100MiscMBars[
            num_stages=1,
            num_producer=WARPGROUP_SIZE,
            num_consumer=WARPGROUP_SIZE,
        ](
            mbar_base
        )  # C uses 17 and 18
        mbar_base = li_bars.end()  # barrier total [19]
        # if decoding_warp_split_k is False, we need to add ((Depth/BN) -1) x2
        # more barriers. for the splitk the two added is enough for now.
        comptime num_out_stages = 1 if Self.config.decoding_warp_split_k else Self.config.depth // Self.config.BN
        var out_pipeline = OutPipeline[
            num_out_stages=num_out_stages,
            num_producer=WARPGROUP_SIZE,
            num_consumer=1,
        ](
            mbar_base
        )  # Write uses 19 and 20 + (num_out_stages-1)*2
        mbar_base += (
            out_pipeline.num_mbars()
        )  # barrier total [21 + (num_out_stages-1)*2]
        var warp_idx: UInt32 = warp.broadcast(warp_id())
        var ptr_tmem_addr = (mbar_base).bitcast[UInt32]()
        is_leader = elect() != 0
        if warp_idx == 8:
            if is_leader:
                mbar_q[].init(1)
                # only one thread will load the Q
                kv_pipeline.init()
                s_bars.init()
                p_bars.init()
                o_bars.init()
                c_bars.init()
                out_pipeline.init()
                li_bars.init()
        elif warp_idx == 9:
            tcgen05_alloc[Self.config.cta_group](
                ptr_tmem_addr, Self.config.sm100_tmem_cols
            )
        barrier()
        var kv_prod = DecodeKVProducer[Self.qkv_type, Self.config](
            kv_pipeline, kv_smem
        )
        var kv_cons = DecodeKVConsumer[Self.qkv_type, Self.config](
            kv_pipeline, kv_smem
        )

        var out_prod = DecodeOutProducer[Self.output_type, Self.config](
            out_pipeline, out_smem
        )
        var out_cons = DecodeOutConsumer[Self.output_type, Self.config](
            out_pipeline, out_smem
        )

        var offset_position = OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._use_valid_length,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
        ](kv_lut, valid_length.value())
        var num_k_tiles = ceildiv(offset_position.num_keys, Self.config.BN)
        if warp_idx < 4:  # softmax warpgroup
            warpgroup_reg_alloc[num_reg_softmax]()
            Self.Softmax(
                ptr_tmem_addr[0],
                s_bars,
                p_bars,
                kv_smem,
                max_smem,
                li_smem,
                c_bars,
                li_bars,
                num_k_tiles,
                offset_position.num_keys,
                scale,
                mask,
                score_mod,
                prompt_idx=UInt32(block_idx.y),
                max_seq_len=UInt32(1),
            )
        elif warp_idx >= 4 and warp_idx < 8:  # correction warpgroup
            warpgroup_reg_alloc[num_reg_correction]()
            Self.Correction(
                ptr_tmem_addr[0], o_bars, out_prod, c_bars, li_bars, num_k_tiles
            )
        else:
            warpgroup_reg_dealloc[num_reg_other]()
            if warp_idx == 8:
                Self.load(
                    q_tma,
                    k_tma,
                    kv_lut,
                    q_smem,
                    kv_smem,
                    mbar_q,
                    kv_prod,
                    offset_position,
                )
            elif warp_idx == 9:
                Self.mmaQK(
                    ptr_tmem_addr[0],
                    q_tma,
                    k_tma,
                    kv_lut,
                    q_smem,
                    kv_smem,
                    mbar_q,
                    s_bars,
                    kv_cons,
                    offset_position,
                )
            elif warp_idx == 10:
                Self.mmaPV(
                    ptr_tmem_addr[0],
                    k_tma,
                    kv_lut,
                    kv_smem,
                    p_bars,
                    o_bars,
                    kv_cons,
                    offset_position,
                )
            elif warp_idx == 11:
                Self.store(out_cons, o_tma)
        barrier()
        if warp_idx == 9:
            tcgen05_release_allocation_lock[Self.config.cta_group]()
            tcgen05_dealloc[Self.config.cta_group](
                ptr_tmem_addr[0], Self.config.sm100_tmem_cols
            )

    # --------------------------------------------------------------------------
    # MLA decoding load_q and load_kv function
    # --------------------------------------------------------------------------
    @staticmethod
    @always_inline
    fn load_kv(
        tma: KVTMATile[
            dtype = Self.qkv_type,
            swizzle_mode = Self.config.kv_swizzle_mode,
            BN = Self.config.BM,
            BK = Self.config.BN,
        ],
        smem: SharedMemPointer[Scalar[Self.qkv_type]],
        mbar: MBarType,
        col_start: UInt,
        row_start: UInt,
    ):
        @parameter
        for block in range(0, Self.NumQKBlocks):
            var block_smem = smem + block * Self.BlockElems
            var smem_tensor = SharedMemTensor[
                Self.qkv_type, type_of(tma).layout  # 64x64 swizzled tile
            ](block_smem)

            tma.async_copy_3d(
                smem_tensor,
                mbar[],
                (
                    col_start + UInt(block * Self.config.BN),
                    UInt(0),
                    row_start,
                ),  # 0, 64, 128, ...
            )

    @staticmethod
    @always_inline
    fn load_q(
        tma: QOTMATile[
            dtype = Self.qkv_type,
            BM = Self.config.BM,
            BK = Self.config.BN,
            swizzle_mode = Self.config.swizzle_mode,
        ],
        smem: SharedMemPointer[Scalar[Self.qkv_type]],
        mbar: MBarType,
        col_start: UInt,
        row_start: UInt,
    ):
        @parameter
        for block in range(0, Self.NumQKBlocks):
            var block_smem = smem + block * Self.BlockElems
            var smem_tensor = SharedMemTensor[
                Self.KVLUTType.dtype, type_of(tma).layout  # 64x64 swizzled tile
            ](block_smem)

            tma.async_copy(
                smem_tensor,
                mbar[],
                (
                    col_start + UInt(block * Self.config.BN),
                    row_start,
                ),  # 0, 64, 128, ...
            )

    @staticmethod
    @always_inline
    fn load(
        q_tma: QOTMATile[
            dtype = Self.qkv_type,
            BM = Self.config.BM,
            BK = Self.config.BN,
            swizzle_mode = Self.config.swizzle_mode,
        ],
        k_tma: KVTMATile[
            dtype = Self.qkv_type,
            swizzle_mode = Self.config.kv_swizzle_mode,
            BN = Self.config.BM,
            BK = Self.config.BN,
        ],
        kv_lut: Self.KVLUTType,
        q_smem: SharedMemPointer[Scalar[Self.qkv_type]],
        kv_smem: SharedMemPointer[Scalar[Self.qkv_type]],
        mbar_q: MBarType,
        mut kv_prod: DecodeKVProducer[Self.qkv_type, Self.config],
        offset_position: OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._use_valid_length,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
        ],
    ):
        # assuming seq_len is 1 for now
        var elect_mask = elect()
        var is_leader = elect_mask != 0
        var row: UInt = UInt(
            block_idx.x * UInt(Self.config.BM)
            + block_idx.y * UInt(Self.config.num_q_heads)
        )
        var pipe_qk = PipelineState[num_stages=1]()
        var kv_row: UInt32 = 0  # it seems it goes to the contiuse page as

        # it requires the kv_row to be zero to be corrected
        var kv_gmem_row: UInt32 = kv_lut.row_idx(block_idx.y, kv_row)
        if is_leader:
            # this is the total bytes expected to be transferred to the mbar for Q and K0
            mbar_q[].expect_bytes(
                Self.config.BM * Self.config.q_depth * size_of[Self.qkv_type]()
            )
            Self.load_q(q_tma, q_smem, mbar_q, UInt(0), row)

        var k0_bar: MBarType = kv_prod.producer_mbar[mma_stage=0]()

        if is_leader:
            k0_bar[].expect_bytes(
                Self.config.BN * Self.config.q_depth * size_of[Self.qkv_type]()
            )
            var stage_ptr = kv_prod.stage_base_ptr[mma_stage=0]()
            Self.load_kv(k_tma, stage_ptr, k0_bar, UInt(0), UInt(kv_gmem_row))

        kv_prod.commit_step()

        kv_row += Self.config.BN

        # We already primed tile 0 into stage 0 (kv_smem base).
        # Now pipeline the remaining K tiles: tile_idx in [1, num_k_tiles)
        var tile_idx: Int = 1
        num_k_tiles = ceildiv(offset_position.num_keys, Self.config.BN)
        while tile_idx < num_k_tiles:
            kv_prod.acquire[mma_stage=0]()
            # Current pipeline stage index (0 or 1 for 2-stage KV)
            var stage_ptr = kv_prod.stage_base_ptr[mma_stage=0]()

            var k_mbar = kv_prod.producer_mbar[mma_stage=0]()

            # Base pointer for this KV stage in shared memory
            # Producer-side barrier for this stage (already init'ed in kv_pipeline.init())
            var kv_gmem_row: UInt32 = kv_lut.row_idx(block_idx.y, kv_row)

            if is_leader:
                k_mbar[].expect_bytes(
                    Self.config.BN
                    * Self.config.q_depth
                    * size_of[Self.qkv_type]()
                )
                Self.load_kv(
                    k_tma, stage_ptr, k_mbar, UInt(0), UInt(kv_gmem_row)
                )

            kv_row += Self.config.BN
            kv_prod.commit_step()

            # this way we can make sure that the k wont be overwritten before mma consume it as V
            tile_idx += 1

    # --------------------------------------------------------------------------
    # MLA decoding MMA for Q, K, V, P blocks
    # --------------------------------------------------------------------------

    # -------------------------------------------------
    # PIPELINE LOOP:
    #   loop over tiles 1..num_k_tiles-1
    #   each iteration does:
    #     - PV(tile_idx-1) with prev_stage_idx  (then release its KV stage)
    #     - QK(tile_idx) with the next KV stage
    # -------------------------------------------------
    # QK process the Numkey veritcally, meaning the C Scale for the first
    # block of all tiles is going to be zero the PV multiply the P horizontally
    # to V meaning only the C scale for prev tile for is going to be Zero for all
    # 9 block and after that it is going to be 1
    #                Q                                              KV0/1
    #   ___ ___ ___ ___ ___ ___ ___ ___ ___       ___ ___ ___ ___ ___ ___ ___ ___ ___
    #  |___|___|___|___|___|___|___|___|___|  T0 |___|___|___|___|___|___|___|___|___|
    #                                         T1 |___|___|___|___|___|___|___|___|___|
    #                                         T2 |___|___|___|___|___|___|___|___|___|
    #                                         T3 |___|___|___|___|___|___|___|___|___|
    #     S0     S1     S0    S1
    #   ______ ______ ______ ______
    #  |__T0__|__T1__|__T2__|__T3__|
    #
    #     P0     P0     P0    P0
    #   ______ ______ ______ ______
    #  |__T0__|__T1__|__T2__|__T3__|

    # We move it to It might be possible to create two P slot and put it at the
    # last slot of KV pipeline, Need to verify if that gives better performance.
    # QK process the Numkey veritcally, meaning the C Scale for the first block
    # of all tiles is going to be zero the PV multiply the P horisontally to V
    # meaning only the C scale for prev tile for is going to be Zero for all 9 block
    # and after that it is going to be 1
    #                Q                                              KV0/1
    #   ___ ___ ___ ___ ___ ___ ___ ___ ___       ___ ___ ___ ___ ___ ___ ___ ___ _______
    #  |___|___|___|___|___|___|___|___|___|  T0 |___|___|___|___|___|___|___|___|__P0/1_|
    #                                         T1 |___|___|___|___|___|___|___|___|__P0/1_|
    #                                         T2 |___|___|___|___|___|___|___|___|__P0/1_|
    #                                         T3 |___|___|___|___|___|___|___|___|__P0/1_|
    #     S0    S1    S0    S1
    #   ______ ______ ______ ______
    #  |__T0__|__T1__|__T2__|__T3__|
    #
    #   P0     P1    P0    P1
    #  ______ ______ ______ ______
    # |__T0__|__T1__|__T2__|__T3__|

    @staticmethod
    @always_inline
    fn mmaQK(
        tmem_addr: UInt32,
        q_tma: QOTMATile[
            dtype = Self.qkv_type,
            BM = Self.config.BM,
            BK = Self.config.BN,
            swizzle_mode = Self.config.swizzle_mode,
        ],
        k_tma: KVTMATile[
            dtype = Self.qkv_type,
            swizzle_mode = Self.config.kv_swizzle_mode,
            BN = Self.config.BM,
            BK = Self.config.BN,
        ],
        kv_lut: Self.KVLUTType,
        q_smem: SharedMemPointer[Scalar[Self.qkv_type]],
        kv_smem: SharedMemPointer[Scalar[Self.qkv_type]],
        mbar_q: MBarType,
        s_bars: DecodeSM100MiscMBars[
            num_stages=2, num_producer=1, num_consumer=WARPGROUP_SIZE
        ],
        mut kv_cons: DecodeKVConsumer[Self.qkv_type, Self.config],
        offset_position: OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._use_valid_length,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
        ],
    ):
        var s0_tmem = tmem_addr + UInt32(Self.config.TMEM_S0)
        var o_tmem = tmem_addr + UInt32(Self.config.TMEM_O)
        var elect_mask = elect()
        # c_scale = 0 for the very first MMA (overwrite),
        #           1 afterwards (accumulate)
        # Number of K-tiles we have in this row
        num_k_tiles = ceildiv(offset_position.num_keys, Self.config.BN)

        # Early exit if there are no K tiles
        if num_k_tiles == 0:
            return

        # ---  S producer wrapper (2-stage pipeline) ---
        var s_prod = DecodeSProducer(s_bars.producer())
        comptime s_stride = UInt32(Self.config.TMEM_S1 - Self.config.TMEM_S0)

        var q_descriptor = Self.UMMAQKTSS.descriptor_q_block(q_smem)
        var k_descriptor = Self.UMMAQKTSS.descriptor_k_block(kv_smem)
        comptime stage_stride_in_bytes = Self.KVStageElems * Self.bytes_per_element
        comptime block_stride_in_bytes = Self.BlockElems * Self.bytes_per_element

        mbar_q[].wait(0)
        var tile_idx: Int = 0

        while tile_idx < num_k_tiles:
            # wait until the corresponding consumer has freed a slot
            s_prod.acquire()

            # Which S slot (0 or 1) are we producing into this time?
            var slot_idx: UInt32 = s_prod.slot_index()
            var s_tmem_slot = s0_tmem + UInt32(slot_idx) * s_stride

            kv_cons.wait[mma_stage=0]()
            # wait for stage 0
            # the stage_ptr is the pointer to the ready block of the first stage
            # and already has the correct stage pointer
            # this will let QK0 goes to SO_tmem and QK1 goes to S1_tmem and so on.
            k_slot_index = kv_cons.stage_index[mma_stage=0]()

            @parameter
            for block in range(0, Self.NumQKBlocks):
                Self.UMMAQKTSS.mma[stage_idx=0](
                    a=q_descriptor + block * block_stride_in_bytes,
                    b=k_descriptor
                    + k_slot_index * stage_stride_in_bytes
                    + block * block_stride_in_bytes,
                    c=s_tmem_slot,
                    c_scale=UInt32(
                        block != 0
                    ),  # only the first block of the tile it's 0
                    elect=elect_mask,
                )
            tcgen05_fence_before()
            s_prod.commit_mma(elect_mask)
            # Here we release the kV for ther QK but Load should not load it
            # as the V has not released yet
            kv_cons.release[mma_stage=0](elect_mask)
            tile_idx += 1

    @staticmethod
    @always_inline
    fn mmaPV(
        tmem_addr: UInt32,
        k_tma: KVTMATile[
            dtype = Self.qkv_type,
            swizzle_mode = Self.config.kv_swizzle_mode,
            BN = Self.config.BM,
            BK = Self.config.BN,
        ],
        kv_lut: Self.KVLUTType,
        kv_smem: SharedMemPointer[Scalar[Self.qkv_type]],
        p_bars: DecodeSM100MiscMBars[
            num_stages=2, num_producer=WARPGROUP_SIZE, num_consumer=1
        ],
        o_bars: DecodeSM100MiscMBars[
            num_stages=1, num_producer=1, num_consumer=WARPGROUP_SIZE
        ],
        mut kv_cons: DecodeKVConsumer[Self.qkv_type, Self.config],
        offset_position: OffsetPosition[
            Self.config,
            Self.KVLUTType,
            Self.ragged,
            Self._use_valid_length,
            Self._is_cache_length_accurate,
            Self.ValidLengthType,
        ],
    ):
        var o_tmem = tmem_addr + UInt32(Self.config.TMEM_O)
        var elect_mask = elect()
        # c_scale = 0 for the very first MMA (overwrite),
        #           1 afterwards (accumulate)
        # Number of K-tiles we have in this row
        num_k_tiles = ceildiv(offset_position.num_keys, Self.config.BN)

        # Early exit if there are no K tiles
        if num_k_tiles == 0:
            return

        # ---  S producer wrapper (2-stage pipeline) ---
        comptime s_stride = UInt32(Self.config.TMEM_S1 - Self.config.TMEM_S0)

        var p_cons = DecodePConsumer(p_bars.consumer())
        var o_prod = DecodeOProducer(o_bars.producer())
        var p_smem_base = kv_smem + Self.NumVOBlocks * Self.BlockElems
        var p_descriptor = Self.UMMAPVSS.descriptor_p_block(p_smem_base)
        var v_descriptor = Self.UMMAPVSS.descriptor_v_block(kv_smem)
        comptime stage_stride_in_bytes = Self.KVStageElems * Self.bytes_per_element
        comptime block_stride_in_bytes = Self.BlockElems * Self.bytes_per_element

        var tile_idx: Int = 0
        var c_scale: UInt32 = 0
        while tile_idx < num_k_tiles:
            kv_cons.wait[mma_stage=0]()
            var p_slot_index = p_cons.wait()
            var v_slot_index = kv_cons.stage_index[mma_stage=0]()

            o_prod.acquire()

            @parameter
            for block in range(
                0, Self.NumVOBlocks
            ):  # PV does not have the k-rope so we don't need to do the last block
                Self.UMMAPVSS.mma[stage_idx=0](
                    a=p_descriptor + p_slot_index * stage_stride_in_bytes,
                    b=v_descriptor
                    + v_slot_index * stage_stride_in_bytes
                    + block * block_stride_in_bytes,
                    c=o_tmem + UInt32(block) * UInt32(Self.config.BN // 2),
                    c_scale=c_scale,
                    elect=elect_mask,
                )
            # Signal P-consumer mbar and advance P pipeline state
            p_cons.release_mma(elect_mask)

            kv_cons.release[mma_stage=0](elect_mask)
            tcgen05_fence_before()
            o_prod.commit_mma(elect_mask)
            if tile_idx == 0:
                c_scale = 1
            tile_idx += 1

    # --------------------------------------------------------------------------
    # MLA decoding softmax Pipeline
    # --------------------------------------------------------------------------
    @staticmethod
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
        var score_col: UInt32 = UInt32(tile_key_base + col)
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

    @staticmethod
    @always_inline
    fn apply_mask[
        half_load: Int, masked: Bool
    ](
        tiles_done: Int,
        col0: Int,
        num_keys: Int,
        s_row: LocalTensor[Self.AccumType, Layout.row_major(half_load)],
        mask: Self.MaskType,
        score_mod: Self.ScoreModType,
        prompt_idx: UInt32,
        q_head_idx: UInt32,
        score_row: UInt32,
        max_seq_len: UInt32,
        start_pos: UInt32,
        cache_start_pos: UInt32,
    ) -> Scalar[Self.AccumType]:
        # Tile / column base this thread covers in num_keys in globalse
        # 64 * tile_index
        var tile_key_base: Int = tiles_done * Self.config.BN
        # first key index for this thread
        var col_base: Int = tile_key_base + col0

        # For now: global padding only.  because the seq len is set to 1
        # TODO: per-row + causal for seq_len > 1.
        # clamp num_keys - col_base into [0, half_load]
        var keys_remaining: Int = num_keys - col_base
        var n_valid: Int = max(0, min(keys_remaining, half_load))  # 0..32

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
            if masked:
                var v = SIMD[Self.AccumType, 1](masked_val)
                var coord = Self.clamped_index_coordinate(
                    prompt_idx,
                    q_head_idx,
                    score_row + start_pos,
                    col0 + i,
                    tile_key_base,
                    num_keys,
                    cache_start_pos,
                )
                v = mask.mask(coord, v)
                masked_val = v[0]

            @parameter
            if Self.use_score_mod:
                var v2 = SIMD[Self.AccumType, 1](masked_val)
                var coord = Self.clamped_index_coordinate(
                    prompt_idx,
                    q_head_idx,
                    score_row + start_pos,
                    col0 + i,
                    tile_key_base,
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
        kv_smem: SharedMemPointer[Scalar[Self.qkv_type]],
        max_smem: SharedMemPointer[Scalar[Self.AccumType]],  # 128x1 buffer
        li_smem: SharedMemPointer[Scalar[Self.AccumType]],  # 128x1 buffer
        c_bars: DecodeSM100MiscMBars[
            num_stages=1,
            num_producer=WARPGROUP_SIZE,
            num_consumer=WARPGROUP_SIZE,
        ],
        li_bars: DecodeSM100MiscMBars[
            num_stages=1,
            num_producer=WARPGROUP_SIZE,
            num_consumer=WARPGROUP_SIZE,
        ],
        num_k_tiles: Int,
        num_keys: Int,
        scale: Float32,
        mask: Self.MaskType,
        score_mod: Self.ScoreModType,
        prompt_idx: UInt32,  # batch index
        max_seq_len: UInt32,  # for score_mod
    ):
        comptime MaskName: String = Self.MaskType.name()
        comptime NoMask: Bool = (MaskName == "NullMask")
        comptime IsMaterializedMask: Bool = (MaskName == "MaterializedMask")
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
        var corr_li_tmem = tmem_addr + UInt32(Self.config.TMEM_CORR_LI)

        # NEW: S consumer wrapper
        var s_cons = DecodeSConsumer(s_bars.consumer())
        var p_prod = DecodePProducer(p_bars.producer())
        var c_prod = DecodeCProducer(c_bars.producer())
        var li_prod = DecodeCProducer(li_bars.producer())
        var warp_idx = warp.broadcast(warp_id())
        var warp_group_idx: Int32 = warp_idx >> 2
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
        var score_row: UInt32 = 0  # decode: single token per batch

        var mi: Scalar[Self.AccumType] = min_or_neg_inf[Self.AccumType]()
        var li: Scalar[Self.AccumType] = 0.0
        comptime log2e_f32 = Scalar[Self.AccumType](log2e)
        comptime half_load = (Self.config.BN >> 1)
        var scale_log2e = scale.cast[Self.AccumType]()

        var start_pos: UInt32 = UInt32(num_keys - 1)
        var cache_start_pos: UInt32 = 0

        var tiles_done: Int = 0
        while tiles_done < num_k_tiles:
            # Wait for an S slot to become ready
            var slot_idx: UInt32 = s_cons.wait()
            var s_tmem_slot = s0_tmem + slot_idx * s_stride

            tcgen05_fence_after()

            # Each thread reads one full 32-element row (128 rows x 32 columns)
            var s_row = LocalTensor[
                Self.AccumType, Layout.row_major(half_load)
            ].stack_allocation()
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
            s_row_val_vectorized *= scale_log2e

            # At this stage the causal mask is trivial because seqlen is 1
            # TODO: add per row causal mask once seqlen is > 1
            @parameter
            if NoMask or CausalMask:
                current_max = Self.apply_mask[half_load, masked=False](
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
                    start_pos,
                    cache_start_pos,
                )
            else:
                current_max = Self.apply_mask[half_load, masked=True](
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
                    start_pos,
                    cache_start_pos,
                )
            current_max *= log2e_f32

            # every softmax thread signals arrival on the shared-mem barrier
            max_Smem_Tensor[lane_id] = current_max
            named_barrier[WARPGROUP_SIZE](2)
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
            var scale_for_old_max: Scalar[Self.AccumType] = exp2(mi - new_max)
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

            # Before first tile or each tile:
            # wait until MMA has released P (consumer_mbar.phase matches)
            p_prod.acquire()
            var p_stage = p_prod.stage_index()  # 0 or 1
            var p_smem = kv_smem + (
                p_stage * Self.KVStageElems
                + (Self.NumVOBlocks) * Self.BlockElems
            )

            write_bf16x2_row_to_smem_fast[
                out_dtype = Self.qkv_type,
                in_dtype = Self.AccumType,
                config = Self.config,
                local_tile_size=half_load,
            ](p_smem, s_row, col_start=col0, row_start=row)

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
        named_barrier[WARPGROUP_SIZE](2)
        li += li_Smem_Tensor[lane_id ^ 64][0]
        li_prod.acquire()

        # here we write back the Li to the correction_li_smem buffer
        tcgen05_st[
            datapaths=32,
            bits=32,
            repeat=1,
            pack=False,
        ](corr_li_tmem, li)
        # signal to the correction warpgroup:
        li_prod.commit()

    # --------------------------------------------------------------------------
    # MLA decoding Correction and Epilogue kernel
    # --------------------------------------------------------------------------
    @staticmethod
    @always_inline
    fn Correction(
        tmem_addr: UInt32,
        o_bars: DecodeSM100MiscMBars[
            num_stages=1, num_producer=1, num_consumer=WARPGROUP_SIZE
        ],
        mut out_prod: DecodeOutProducer[Self.output_type, Self.config],
        c_bars: DecodeSM100MiscMBars[
            num_stages=1,
            num_producer=WARPGROUP_SIZE,
            num_consumer=WARPGROUP_SIZE,
        ],
        li_bars: DecodeSM100MiscMBars[
            num_stages=1,
            num_producer=WARPGROUP_SIZE,
            num_consumer=WARPGROUP_SIZE,
        ],
        num_k_tiles: Int,
    ):
        var o_tmem = tmem_addr + UInt32(Self.config.TMEM_O)
        var corr_scale_tmem = tmem_addr + UInt32(Self.config.TMEM_CORR_SCALE)
        var corr_li_tmem = tmem_addr + UInt32(Self.config.TMEM_CORR_LI)
        var o_cons = DecodeOConsumer(o_bars.consumer())
        var c_cons = DecodeCConsumer(c_bars.consumer())
        var li_cons = DecodeCConsumer(li_bars.consumer())
        var tiles_done: Int = 1

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
            # 3) Ensure the loads are complete before using li/scale
            tcgen05_load_wait()
            c_cons.release()
            o_cons.wait()
            change = _vote_nvidia_helper(scale_value != 1) != 0
            if change:
                # the rows are symmetric distributed across two warps however
                # the correction value for both warps containing the same row are the same
                # so when a row load 64 elements it is actually  the first half
                # and the third half of the row is here as the second half
                # and the forth half of the row is in the other warp
                comptime num_o_tiles = Self.config.depth // (
                    Self.output_tile_width * 2
                )

                @parameter
                for i in range(0, num_o_tiles):
                    # Here we load from o_tmem. it is 32 bit float and we load 64 fp32 element per tile
                    var o_tmem_subtile: UInt32 = o_tmem + UInt32(i) * UInt32(
                        Self.config.BN
                    )
                    var o_row_subtile = LocalTensor[
                        Self.AccumType, Layout.row_major(Self.config.BN)
                    ].stack_allocation()
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

            tiles_done += 1

        __comptime_assert (
            Self.AccumType == DType.float32
        ), "accumulator type should be float32"
        __comptime_assert (
            Self.output_type == DType.bfloat16
        ), "output type should be bfloat16"

        # Each warp covers 32 rows in the 64x64 tile
        # var row: Int = (corr_warp & 1) * 32 + lane    # 0..63
        li_cons.wait()
        var li = tcgen05_ld[
            datapaths=32,
            bits=32,
            repeat=1,
            dtype = Self.AccumType,
            pack=False,
        ](corr_li_tmem)
        # Wait until the async load completes before using the registers
        tcgen05_load_wait()
        li_cons.release()
        o_cons.wait()

        # By the time we reach to epilogue the KV is free when we dont have the SplitK algorithm.
        # So we can safely use the KV buffer for writing the output and have more async write.

        # it is 256/32 which is equivalent of 512/64
        comptime num_store_tiles = Self.config.depth // Self.output_tile_width
        comptime half_load: UInt32 = Self.config.BN >> 1

        @parameter
        for i in range(0, num_store_tiles):
            # 2. Compute TMEM base for this subtile t
            var o_tmem_subtile: UInt32 = o_tmem + UInt32(i) * UInt32(half_load)
            var o_row_subtile = LocalTensor[
                Self.AccumType, Layout.row_major(Int(half_load))
            ].stack_allocation()
            o_row_subtile.ptr.store(
                0,
                tcgen05_ld[
                    datapaths=32,
                    bits=32,
                    repeat = Int(half_load),
                    dtype = Self.AccumType,
                    pack=False,
                ](o_tmem_subtile),
            )
            tcgen05_load_wait()

            var float2_register = o_row_subtile.vectorize[2]()

            var o_scale_li: SIMD[Self.AccumType, 2] = SIMD[Self.AccumType, 2](
                recip(li)[0]
            )
            if li[0] == 0.0:
                o_scale_li = SIMD[Self.AccumType, 2](0.0)

            @parameter
            for j in range(0, half_load // 2):
                var element = rebind[SIMD[Self.AccumType, 2]](
                    float2_register[j]
                )
                float2_register[j] = rebind[type_of(float2_register[j])](
                    element * o_scale_li
                )

            out_prod.acquire()
            warp_idx = warp.broadcast(warp_id() - 4)
            var lane: UInt32 = thread_idx.x & 0x7F  # 0..127
            var row: UInt32 = lane & 0x3F  # lan % Config.BN 0..63
            var warp_pair: UInt32 = (
                warp_idx >> 1
            )  # 0..3 inside correction group # 0 or 1
            # Column range this thread owns in P
            var col0: UInt32 = warp_pair * half_load  # 0 or 32

            var stage_ptr = out_prod.stage_base_ptr()

            write_bf16x2_row_to_smem_fast[
                out_dtype = Self.output_type,
                in_dtype = Self.AccumType,
                config = Self.config,
                local_tile_size = Int(half_load),
            ](stage_ptr, o_row_subtile, col_start=Int(col0), row_start=Int(row))
            # The fence_async_view_proxy() here is not needed as the store puts this fence.
            out_prod.commit_step()

        o_cons.release()

    # --------------------------------------------------------------------------
    # MLA decoding store kernel
    # --------------------------------------------------------------------------
    # If it goes to the batch loop remember correction is out of sync with MMA on
    # O_tmem wait and release as it starts one stage before MMA
    @staticmethod
    @always_inline
    fn store(
        mut out_cons: DecodeOutConsumer[Self.output_type, Self.config],
        o_tma: QOTMATile[
            dtype = Self.output_type,
            BM = Self.config.out_rows,
            BK = Self.config.BN,
            swizzle_mode = Self.config.swizzle_mode,
        ],
    ):
        elect_mask = elect()
        var is_leader = elect_mask != 0
        comptime num_store_tiles = Self.config.depth // Self.output_tile_width
        var row: UInt = UInt(
            block_idx.x * UInt(Self.config.BM)
            + block_idx.y * UInt(Self.config.num_q_heads)
        )
        # The code work with the assumption that the num_q_heads is always power of two.
        var tma_phase: UInt32 = 0

        @parameter
        for i in range(0, num_store_tiles):
            var col: UInt = UInt(i * Self.config.BN)
            out_cons.wait()
            var stage_ptr = out_cons.stage_base_ptr()
            var smem_tensor = SharedMemTensor[
                Self.output_type,
                type_of(o_tma).layout,
            ](stage_ptr)
            if is_leader:
                fence_async_view_proxy()
                o_tma.async_store(smem_tensor, (col, row))

                @parameter
                if Self.config.decoding_warp_split_k:
                    o_tma.commit_group()
                    o_tma.wait_group[0]()
            out_cons.release(elect_mask)
        if is_leader and not Self.config.decoding_warp_split_k:
            o_tma.commit_group()
        o_tma.wait_group[0]()
