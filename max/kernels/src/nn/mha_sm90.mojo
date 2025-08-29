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
from math import ceildiv, exp2, recip
from math.constants import log2e
from sys import align_of, env_get_int, simd_width_of, size_of

import gpu.warp as warp
from buffer import NDBuffer
from gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    barrier,
    block_dim,
    lane_id,
    thread_idx,
)
from gpu.globals import WARPGROUP_SIZE
from gpu.host import DeviceContext, FuncAttribute
from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.host.info import H100
from gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from gpu.memory import AddressSpace, external_memory
from gpu.sync import named_barrier
from layout.int_tuple import IntTuple
from layout.layout import Layout
from layout.layout_tensor import (
    LayoutTensor,
    LayoutTensorIter,
    copy_local_to_shared,
    copy_sram_to_dram,
    cp_async_k_major,
)
from layout.swizzle import make_swizzle
from layout.tensor_core_async import (
    TensorCoreAsync,
    tile_layout_k_major,
    tile_layout_mn_major,
    warpgroup_fence,
)
from layout.tma_async import (
    PipelineState,
    SharedMemBarrier,
    TMANestedTensorTile,
)
from nn.mha_mask import MHAMask, TileMaskStatus
from nn.mha_operand import MHAOperand
from nn.mha_score_mod import ScoreModTrait
from nn.mha_tile_scheduler import (
    MHASchedulerSynchronization,
    MHATileScheduler,
    MHATileState,
    MHATileSummary,
    QueuedTileScheduler,
    SeqInfo,
    TileScheduler,
    TransientScheduler,
)
from nn.mha_utils import (
    FlashAttentionAlgorithm,
    MHAConfig,
    MHAPartitionScheme,
    OptionallyStaticInt,
    _is_decoding,
    get_start_and_end_for_partitions,
)
from nn.mha_fa3_utils import (
    MHAPosition,
    _apply_mask,
    _get_position,
    produce,
    q_out_tma,
    valid_length_managed_tensor_slice_to_ndbuffer,
    output_reg_to_smem,
)
from nn.softmax import (
    _online_softmax_correction,
    _rowmax_online_softmax,
    _rowsum,
)
from tensor_internal import ManagedTensorSlice

from utils.index import Index
from utils.numerics import get_accum_type, min_or_neg_inf
from utils.static_tuple import StaticTuple


@always_inline
fn mha_sm90_dispatch[
    q_type: DType,
    KVType: MHAOperand,
    MaskType: MHAMask,
    ScoreModType: ScoreModTrait,
    output_type: DType,
    MaxPromptLenType: OptionallyStaticInt,
    PartitionType: MHAPartitionScheme, //,
    config: MHAConfig,
    group: Int,
    use_score_mod: Bool,
    ragged: Bool,
    sink: Bool,
    _is_cache_length_accurate: Bool,
](
    output: UnsafePointer[Scalar[output_type]],
    q_arg: UnsafePointer[Scalar[q_type]],
    k: KVType,
    v: KVType,
    num_rows_q: Int,
    mask_functor: MaskType,
    score_mod_functor: ScoreModType,
    valid_length: ManagedTensorSlice[dtype = DType.uint32, rank=1],
    max_prompt_len_arg: MaxPromptLenType,
    max_cache_valid_length_arg: Int,
    scale: Float32,
    kv_input_row_offsets: OptionalReg[
        NDBuffer[DType.uint32, 1, MutableAnyOrigin]
    ],
    batch_size_arg: Int,
    partition: PartitionType,
    ctx: DeviceContext,
    sink_weights: OptionalReg[NDBuffer[q_type, 1, MutableAnyOrigin]],
) raises:
    constrained[
        config.type == KVType.dtype and config.type == q_type,
        "config, kv, and q types must all match for FA3.",
    ]()
    q = rebind[UnsafePointer[Scalar[KVType.dtype]]](q_arg)
    alias decoding: Bool = MaxPromptLenType.static_value.or_else(0) == 1
    alias new_config = MHAConfig(
        config.type,
        config.num_heads,
        config.depth,
        num_queries_per_block=OptionalReg[UInt](64),
        num_keys_per_block=OptionalReg[UInt](config.num_keys_per_block),
        BK=OptionalReg[UInt](config.BK),
    ) if decoding else config
    alias BM = new_config.block_m()
    alias BK = new_config.block_k()
    constrained[
        BM % 64 == 0,
        "SM90 requires BM%64==0, but BM==",
        String(BM),
    ]()
    constrained[
        BK == 64,
        "H100 requires BK%64==0 as it uses 128B swizzles, but BK==",
        String(BK),
    ]()
    alias BN = new_config.block_n()
    # we add smem use for SharedMemBarrier synchronization
    alias smem_use = new_config.shared_mem_bytes[True, sm_90=True]()
    # add the number of producer threads (i.e. 1 WARP_GROUP_SIZE)
    alias num_threads = new_config.num_threads[True]()
    constrained[num_threads % 128 == 0]()

    # Persistent kernels not currently supported with partitioning
    # This doesn't seem useful: we partition to make SMs more busy,
    # implying we don't have enough to make them persistent.
    # This also requires some tricky control flow handling to support,
    # which we haven't added yet.
    alias persistent = 0 if PartitionType.do_partition else env_get_int[
        "USE_EXPERIMENTAL_KERNELS", 0
    ]()
    constrained[new_config.algorithm == FlashAttentionAlgorithm(3)]()

    var max_cache_valid_length: UInt32 = UInt32(max_cache_valid_length_arg)
    var batch_size: UInt32 = UInt32(batch_size_arg)
    var max_prompt_len: UInt32 = max_prompt_len_arg.as_uint32()
    var max_num_prompt_tiles: UInt32 = ceildiv(max_prompt_len, BM)
    var block_x: UInt32 = max_num_prompt_tiles * partition.num_partitions()

    alias q_num_heads: Int = config.num_heads
    alias num_scheduler_heads = q_num_heads // group if decoding else q_num_heads
    # if decoding,
    alias scheduler_tile_shape = 1 if decoding else BM
    alias swizzle_mode = TensorMapSwizzle.SWIZZLE_128B
    q_tma = q_out_tma[
        group if decoding else Int(BM),
        config.depth,
        swizzle_mode,
        q_num_heads=q_num_heads,
        decoding=decoding,
    ](ctx, q, num_rows_q)
    k_tma = k.create_tma_tile[BN, config.depth, swizzle_mode, is_k_major=True](
        ctx
    )
    v_tma = v.create_tma_tile[BN, config.depth, swizzle_mode, is_k_major=False](
        ctx
    )

    @parameter
    if persistent == 0:
        alias SchedulerType = TransientScheduler[
            scheduler_tile_shape, num_scheduler_heads
        ]
        alias kernel_sm90 = _mha_sm90[
            KVType,
            output_type,
            MaskType,
            ScoreModType,
            SchedulerType,
            new_config,
            group=group,
            use_score_mod=use_score_mod,
            ragged=ragged,
            sink=sink,
            _is_cache_length_accurate=_is_cache_length_accurate,
            MaxSeqLenType=MaxPromptLenType,
            PartitionType=PartitionType,
            swizzle_mode=swizzle_mode,
        ]
        var scheduler: SchedulerType = SchedulerType()
        gd = SchedulerType.grid_dim(batch_size, block_x)

        @parameter
        if MaxPromptLenType.static_value:

            @parameter
            if PartitionType.do_partition:
                ctx.enqueue_function[kernel_sm90](
                    q_tma,
                    k_tma,
                    v_tma,
                    output,
                    k,
                    scale,
                    batch_size,
                    max_cache_valid_length,
                    valid_length_managed_tensor_slice_to_ndbuffer(valid_length),
                    kv_input_row_offsets,
                    sink_weights,
                    partition,
                    mask_functor,
                    score_mod_functor,
                    grid_dim=SchedulerType.grid_dim(batch_size, block_x),
                    block_dim=(Int(num_threads), 1, 1),
                    shared_mem_bytes=Int(smem_use),
                    func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                        smem_use
                    ),
                )
            else:
                ctx.enqueue_function[kernel_sm90](
                    q_tma,
                    k_tma,
                    v_tma,
                    output,
                    k,
                    scale,
                    batch_size,
                    max_cache_valid_length,
                    valid_length_managed_tensor_slice_to_ndbuffer(valid_length),
                    kv_input_row_offsets,
                    sink_weights,
                    mask_functor,
                    score_mod_functor,
                    grid_dim=SchedulerType.grid_dim(batch_size, block_x),
                    block_dim=(Int(num_threads), 1, 1),
                    shared_mem_bytes=Int(smem_use),
                    func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                        smem_use
                    ),
                )

        else:

            @parameter
            if PartitionType.do_partition:
                ctx.enqueue_function[kernel_sm90](
                    q_tma,
                    k_tma,
                    v_tma,
                    output,
                    k,
                    scale,
                    batch_size,
                    max_prompt_len,
                    max_cache_valid_length,
                    valid_length_managed_tensor_slice_to_ndbuffer(valid_length),
                    kv_input_row_offsets,
                    sink_weights,
                    partition,
                    mask_functor,
                    score_mod_functor,
                    grid_dim=SchedulerType.grid_dim(batch_size, block_x),
                    block_dim=(Int(num_threads), 1, 1),
                    shared_mem_bytes=Int(smem_use),
                    func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                        smem_use
                    ),
                )
            else:
                ctx.enqueue_function[kernel_sm90](
                    q_tma,
                    k_tma,
                    v_tma,
                    output,
                    k,
                    scale,
                    batch_size,
                    max_prompt_len,
                    max_cache_valid_length,
                    valid_length_managed_tensor_slice_to_ndbuffer(valid_length),
                    kv_input_row_offsets,
                    sink_weights,
                    mask_functor,
                    score_mod_functor,
                    grid_dim=SchedulerType.grid_dim(batch_size, block_x),
                    block_dim=(Int(num_threads), 1, 1),
                    shared_mem_bytes=Int(smem_use),
                    func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                        smem_use
                    ),
                )
    elif persistent == 2:
        alias SchedulerType = TileScheduler[
            scheduler_tile_shape, num_scheduler_heads
        ]
        alias kernel_sm90 = _mha_sm90[
            KVType,
            output_type,
            MaskType,
            ScoreModType,
            SchedulerType,
            new_config,
            group=group,
            use_score_mod=use_score_mod,
            ragged=ragged,
            _is_cache_length_accurate=_is_cache_length_accurate,
            MaxSeqLenType=MaxPromptLenType,
            PartitionType=PartitionType,
            swizzle_mode=swizzle_mode,
            sink=sink,
        ]
        var scheduler: SchedulerType = SchedulerType()

        @parameter
        if MaxPromptLenType.static_value:

            @parameter
            if PartitionType.do_partition:
                ctx.enqueue_function[kernel_sm90](
                    q_tma,
                    k_tma,
                    v_tma,
                    output,
                    k,
                    scale,
                    batch_size,
                    max_cache_valid_length,
                    valid_length_managed_tensor_slice_to_ndbuffer(valid_length),
                    kv_input_row_offsets,
                    sink_weights,
                    partition,
                    mask_functor,
                    score_mod_functor,
                    scheduler,
                    grid_dim=SchedulerType.grid_dim(batch_size, block_x),
                    block_dim=(Int(num_threads), 1, 1),
                    shared_mem_bytes=Int(smem_use),
                    func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                        smem_use
                    ),
                )
            else:
                ctx.enqueue_function[kernel_sm90](
                    q_tma,
                    k_tma,
                    v_tma,
                    output,
                    k,
                    scale,
                    batch_size,
                    max_cache_valid_length,
                    valid_length_managed_tensor_slice_to_ndbuffer(valid_length),
                    kv_input_row_offsets,
                    sink_weights,
                    mask_functor,
                    score_mod_functor,
                    scheduler,
                    grid_dim=SchedulerType.grid_dim(batch_size, block_x),
                    block_dim=(Int(num_threads), 1, 1),
                    shared_mem_bytes=Int(smem_use),
                    func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                        smem_use
                    ),
                )
        else:

            @parameter
            if PartitionType.do_partition:
                ctx.enqueue_function[kernel_sm90](
                    q_tma,
                    k_tma,
                    v_tma,
                    output,
                    k,
                    scale,
                    batch_size,
                    max_prompt_len,
                    max_cache_valid_length,
                    valid_length_managed_tensor_slice_to_ndbuffer(valid_length),
                    kv_input_row_offsets,
                    sink_weights,
                    partition,
                    mask_functor,
                    score_mod_functor,
                    scheduler,
                    grid_dim=SchedulerType.grid_dim(batch_size, block_x),
                    block_dim=(Int(num_threads), 1, 1),
                    shared_mem_bytes=Int(smem_use),
                    func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                        smem_use
                    ),
                )
            else:
                ctx.enqueue_function[kernel_sm90](
                    q_tma,
                    k_tma,
                    v_tma,
                    output,
                    k,
                    scale,
                    batch_size,
                    max_prompt_len,
                    max_cache_valid_length,
                    valid_length_managed_tensor_slice_to_ndbuffer(valid_length),
                    kv_input_row_offsets,
                    sink_weights,
                    mask_functor,
                    score_mod_functor,
                    scheduler,
                    grid_dim=SchedulerType.grid_dim(batch_size, block_x),
                    block_dim=(Int(num_threads), 1, 1),
                    shared_mem_bytes=Int(smem_use),
                    func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                        smem_use
                    ),
                )
    else:
        alias SchedulerType = QueuedTileScheduler[
            scheduler_tile_shape, num_scheduler_heads, decoding=decoding
        ]
        alias kernel_sm90 = _mha_sm90[
            KVType,
            output_type,
            MaskType,
            ScoreModType,
            SchedulerType,
            new_config,
            group=group,
            use_score_mod=use_score_mod,
            ragged=ragged,
            _is_cache_length_accurate=_is_cache_length_accurate,
            MaxSeqLenType=MaxPromptLenType,
            PartitionType=PartitionType,
            swizzle_mode=swizzle_mode,
            sink=sink,
        ]
        var schedule = ctx.enqueue_create_buffer[DType.uint32](1).enqueue_fill(
            UInt32(H100.sm_count)
        )
        ctx.synchronize()
        var scheduler: SchedulerType = SchedulerType(schedule.unsafe_ptr())

        # these nested branches are to reduce risk of memory corruption
        # when passing 0-sized arguments, which is currently not handled
        # correctly in Mojo
        # TODO: Remove and simplify when KERN-1753 is fixed
        @parameter
        if MaxPromptLenType.static_value:

            @parameter
            if PartitionType.do_partition:
                ctx.enqueue_function[kernel_sm90](
                    rebind[SchedulerType](scheduler),
                    q_tma,
                    k_tma,
                    v_tma,
                    output,
                    k,
                    scale,
                    batch_size,
                    max_cache_valid_length,
                    valid_length_managed_tensor_slice_to_ndbuffer(valid_length),
                    rebind[
                        OptionalReg[NDBuffer[DType.uint32, 1, MutableAnyOrigin]]
                    ](kv_input_row_offsets),
                    sink_weights,
                    partition,
                    rebind[MaskType](mask_functor),
                    rebind[ScoreModType](score_mod_functor),
                    grid_dim=SchedulerType.grid_dim(batch_size, block_x),
                    block_dim=(Int(num_threads), 1, 1),
                    shared_mem_bytes=Int(smem_use),
                    func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                        smem_use
                    ),
                )
            else:
                ctx.enqueue_function[kernel_sm90](
                    rebind[SchedulerType](scheduler),
                    q_tma,
                    k_tma,
                    v_tma,
                    output,
                    k,
                    scale,
                    batch_size,
                    max_cache_valid_length,
                    valid_length_managed_tensor_slice_to_ndbuffer(valid_length),
                    rebind[
                        OptionalReg[NDBuffer[DType.uint32, 1, MutableAnyOrigin]]
                    ](kv_input_row_offsets),
                    sink_weights,
                    rebind[MaskType](mask_functor),
                    rebind[ScoreModType](score_mod_functor),
                    grid_dim=SchedulerType.grid_dim(batch_size, block_x),
                    block_dim=(Int(num_threads), 1, 1),
                    shared_mem_bytes=Int(smem_use),
                    func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                        smem_use
                    ),
                )
        else:

            @parameter
            if PartitionType.do_partition:
                ctx.enqueue_function[kernel_sm90](
                    rebind[SchedulerType](scheduler),
                    q_tma,
                    k_tma,
                    v_tma,
                    output,
                    k,
                    scale,
                    batch_size,
                    max_prompt_len,
                    max_cache_valid_length,
                    valid_length_managed_tensor_slice_to_ndbuffer(valid_length),
                    rebind[
                        OptionalReg[NDBuffer[DType.uint32, 1, MutableAnyOrigin]]
                    ](kv_input_row_offsets),
                    sink_weights,
                    partition,
                    rebind[MaskType](mask_functor),
                    rebind[ScoreModType](score_mod_functor),
                    grid_dim=SchedulerType.grid_dim(batch_size, block_x),
                    block_dim=(Int(num_threads), 1, 1),
                    shared_mem_bytes=Int(smem_use),
                    func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                        smem_use
                    ),
                )
            else:
                ctx.enqueue_function[kernel_sm90](
                    rebind[SchedulerType](scheduler),
                    q_tma,
                    k_tma,
                    v_tma,
                    output,
                    k,
                    scale,
                    batch_size,
                    max_prompt_len,
                    max_cache_valid_length,
                    valid_length_managed_tensor_slice_to_ndbuffer(valid_length),
                    rebind[
                        OptionalReg[NDBuffer[DType.uint32, 1, MutableAnyOrigin]]
                    ](kv_input_row_offsets),
                    sink_weights,
                    rebind[MaskType](mask_functor),
                    rebind[ScoreModType](score_mod_functor),
                    grid_dim=SchedulerType.grid_dim(batch_size, block_x),
                    block_dim=(Int(num_threads), 1, 1),
                    shared_mem_bytes=Int(smem_use),
                    func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                        smem_use
                    ),
                )
        _ = schedule


@__llvm_arg_metadata(q_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(k_tma_op, `nvvm.grid_constant`)
@__llvm_arg_metadata(v_tma_op, `nvvm.grid_constant`)
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        config.num_threads[True]()
    )
)
fn _mha_sm90[
    KVLUTType: MHAOperand,
    output_type: DType,
    MaskType: MHAMask,
    ScoreModType: ScoreModTrait,
    SchedulerType: MHATileScheduler,
    config: MHAConfig,
    group: Int,
    use_score_mod: Bool,
    ragged: Bool,
    sink: Bool,
    _is_cache_length_accurate: Bool,
    MaxSeqLenType: OptionallyStaticInt,
    PartitionType: MHAPartitionScheme,
    swizzle_mode: TensorMapSwizzle,
](
    scheduler: SchedulerType,
    q_tma_op: TMANestedTensorTile[
        KVLUTType.dtype,
        max(group, 8) if _is_decoding[MaxSeqLenType]() else Int(
            config.block_m()
        ),
        64 if _is_decoding[MaxSeqLenType]() else config.depth,
        swizzle_mode,
        is_k_major=True,
    ],
    k_tma_op: TMANestedTensorTile[
        KVLUTType.dtype,
        config.block_n(),
        config.depth,
        swizzle_mode,
        is_k_major=True,
    ],
    v_tma_op: TMANestedTensorTile[
        KVLUTType.dtype,
        config.block_n(),
        config.depth,
        swizzle_mode,
        is_k_major=False,
    ],
    o_ptr_arg: UnsafePointer[Scalar[output_type]],
    kv_lut: KVLUTType,
    scale: Float32,
    batch_size: UInt32,
    max_seq_len: MaxSeqLenType,  # sequence length after padding.
    num_keys_arg: UInt32,
    valid_length: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
    kv_input_row_offsets: OptionalReg[
        NDBuffer[DType.uint32, 1, MutableAnyOrigin]
    ],
    sink_weights: OptionalReg[NDBuffer[KVLUTType.dtype, 1, MutableAnyOrigin]],
    partition: PartitionType,
    mask: MaskType,
    score_mod: ScoreModType,
):
    """MHA for token gen where seqlen = 1 and num_keys >= 1.

    The general data layout and steps conform to flash attention. Two exceptions:

    1 Partition across B, H, and num_keys (TODO).  The last one is split-K and
      will need a separate reduction kernel at the end.

    2 First bmm becomes gemv and second bmm becomes gevm.
      TODO: use more optimized kernels for them

    """
    alias kv_type = KVLUTType.dtype
    alias decoding: Bool = _is_decoding[MaxSeqLenType]()

    alias simd_size = simd_width_of[kv_type]()

    alias num_warps_m = config.num_warps_m()
    alias num_consumer_threads = config.num_consumer_threads()
    alias BM = config.block_m()
    alias BN = config.block_n()
    alias num_heads = config.num_heads
    alias depth = config.depth
    # num_consumer_threads ignores the producers
    # actual number of threads is num_consumer_threads + 128
    alias num_consumer = num_consumer_threads // WARPGROUP_SIZE
    alias pipeline_stages = Int(config.num_pipeline_stages)
    var tid: UInt32 = thread_idx.x
    var warp_group_idx: UInt32 = warp.broadcast(tid // WARPGROUP_SIZE)

    constrained[
        num_warps_m == (num_consumer_threads // WARP_SIZE),
        "Number of warps doesn't match warp tile sizes.",
    ]()

    var warp_id: UInt32 = warp.broadcast((tid - WARPGROUP_SIZE) // WARP_SIZE)
    var lane: UInt32 = lane_id()

    # Coordinates of the current warp.
    var warp_y: UInt32 = warp_id  # // num_warps_n
    alias warp_x: UInt32 = 0  # warp_id % num_warps_n

    alias q_smem_layout_consumer = tile_layout_k_major[
        DType.bfloat16, BM, depth, swizzle_mode=swizzle_mode
    ]()
    alias k_smem_layout = k_tma_op.layout
    alias v_smem_layout = v_tma_op.layout
    # for wgmma_0, we mutliply BM x depth @ depth x BN -> BM x BN
    # for wgmma_1, we multiply BM x BN @ BN x depth -> BM x depth
    # For wgmma_0, we iterate over (depth//BK) tiles of size BKxBN
    # For wgmma_1, we iterate over (BN//BK) tiles of size BKxdepth
    alias persistent = SchedulerType.may_advance

    # The entire query block (BM x depth) is tiled in shared memory.
    alias q_size = q_smem_layout_consumer.size()
    alias q_smem_size = 2 * q_size if persistent else q_size
    q_smem = external_memory[
        Scalar[kv_type],
        address_space = AddressSpace.SHARED,
        alignment=128,
        name="mha_dynamic_shared_memory",
    ]()
    # We have `num_pipeline_stages` instances of each
    alias kv_smem_size = config.kv_smem_size(True)
    kv_smem = q_smem + q_smem_size

    # var head_idx: UInt32 = block_idx.y
    # var q_tile_idx: UInt32 = block_idx.x

    # q tile has valid shape q_tile_num_rows x depth
    # q_tile_num_rows could be less than BM when seqlen % BM != 0

    alias MMA_M = 16  # per warp
    alias MMA_N0 = BN
    alias MMA_N1 = depth
    alias MMA_K = 16
    alias WM = config.WM
    alias num_m_mmas = WM // MMA_M
    constrained[num_m_mmas == 1, "FIXME: life this constraint"]()
    # alias WN = config.WN
    # alias num_n_mmas = WN // MMA_N
    alias num_n_mmas = 1
    # alias num_k_mmas = BK // MMA_K

    alias accum_type = get_accum_type[kv_type]()
    alias p_frag_size = MMA_M * MMA_N0 // WARP_SIZE
    alias o_frag_size = MMA_M * MMA_N1 // WARP_SIZE
    alias frag_simdwidth = 2

    alias a_frag_size = MMA_M * MMA_K // WARP_SIZE
    # MMA_N0 // MMA_K
    alias frag_ratio = p_frag_size // a_frag_size

    # the first mma is BMxdepth @ depthxBN
    alias wgmma_0 = TensorCoreAsync[
        accum_type,
        kv_type,
        kv_type,
        Index(4 * MMA_M, MMA_N0, 16),
        a_swizzle=swizzle_mode,
        b_swizzle=swizzle_mode,
        transpose_b=True,
    ]()
    # the second mma is BMxBN @ BNxdepth
    alias wgmma_1 = TensorCoreAsync[
        accum_type,
        kv_type,
        kv_type,
        Index(4 * MMA_M, MMA_N1, 16),
        a_swizzle = TensorMapSwizzle.SWIZZLE_NONE,
        b_swizzle=swizzle_mode,
        transpose_b=False,
    ]()

    alias num_row_blocks_per_mma = 2
    # a wgmma.m64n32k16 `D` fragment looks like
    #
    # 0,1  4,5   8, 9  12,13
    # 2,3  6,7  10,11  14,15
    #
    # Each row/column has `p_frag_simdwidth`-sized vectors
    # (e.g. `4,5` is of size 2 = p_frag_simdwidth)
    # We have `num_row_blocks_per_mma` rows.
    # The total number of elements (16) equals `p_frag_size`.
    # The number of columns equals
    # `p_frag_size // (num_row_blocks_per_mma * p_frag_simdwidth)`
    #
    # This gives us the layout:
    #
    # Note the ordering of strides:
    # ((1, 3), (0, 2, 4))
    # alias output_layout = Layout(
    #     IntTuple(
    #         IntTuple(num_row_blocks_per_mma, num_m_mmas),
    #         IntTuple(
    #             p_frag_simdwidth,
    #             p_frag_size // (num_row_blocks_per_mma * p_frag_simdwidth),
    #             num_n_mmas,
    #         ),
    #     ),
    #     IntTuple(
    #         IntTuple(p_frag_simdwidth, p_frag_size),
    #         IntTuple(1, 2 * p_frag_simdwidth, num_m_mmas * p_frag_size),
    #     ),
    # )
    # Vectorizing the layout:
    alias element_layout = Layout.row_major(1, frag_simdwidth)
    alias vec_output_row_shape = IntTuple(num_row_blocks_per_mma, num_m_mmas)
    alias p_vec_output_layout = Layout(
        IntTuple(
            vec_output_row_shape,
            IntTuple(
                p_frag_size // (num_row_blocks_per_mma * frag_simdwidth),
                num_n_mmas,
            ),
        ),
        IntTuple(
            IntTuple(frag_simdwidth, p_frag_size),
            IntTuple(
                num_row_blocks_per_mma * frag_simdwidth,
                num_m_mmas * p_frag_size,
            ),
        ),
    )
    alias o_vec_output_layout = Layout(
        IntTuple(
            vec_output_row_shape,
            IntTuple(
                o_frag_size // (num_row_blocks_per_mma * frag_simdwidth),
                num_n_mmas,
            ),
        ),
        IntTuple(
            IntTuple(frag_simdwidth, o_frag_size),
            IntTuple(
                num_row_blocks_per_mma * frag_simdwidth,
                num_m_mmas * o_frag_size,
            ),
        ),
    )
    alias num_rows_per_warp = p_vec_output_layout[0].size()
    alias num_cols_p = p_vec_output_layout[1].size()
    alias num_cols_output = o_vec_output_layout[1].size()

    # Rowwise max and sum for online softmax
    alias accum_simd_width = simd_width_of[accum_type]()
    alias row_alignment = align_of[SIMD[accum_type, accum_simd_width]]()
    # Account for group query.
    alias kv_num_heads = num_heads // group

    alias mma_thread_layout = Layout.row_major(8, 4)

    # Handle sink_weights
    var sink_weights_ptr = UnsafePointer[Scalar[kv_type]]()

    @parameter
    if sink:
        debug_assert(
            Bool(sink_weights),
            "expect sink_weights to be non-null when sink=true",
        )
        sink_weights_ptr = sink_weights.value().data

    produced_mbar_kv = (kv_smem + kv_smem_size).bitcast[SharedMemBarrier]()
    consumed_mbar_kv = produced_mbar_kv + pipeline_stages
    produced_mbar_q = consumed_mbar_kv + pipeline_stages
    consumed_mbar_q = produced_mbar_q + 2
    block_idx_ptr = (consumed_mbar_q + 2).bitcast[UInt32]()

    alias USE_TMA = True
    # https://github.com/Dao-AILab/flash-attention/blob/3b5047d2ce742848f45d44b143d511f211eba2d2/hopper/flash_fwd_kernel_sm90.h#L81-L82
    alias num_producer_regs = 56 if num_consumer == 1 else (
        (24 if USE_TMA else 56) if num_consumer == 2 else 32
    )
    alias num_consumer_regs = 256 if num_consumer == 1 else (
        (240 if USE_TMA else 224) if num_consumer == 2 else 160
    )
    # alias num_producer_regs = 56
    # alias num_consumer_regs = 224

    # constructing calls barrier() if static
    var tile_summary = MHATileSummary(
        batch_size,
        ceildiv(max_seq_len.as_uint32(), BM) * partition.num_partitions(),
        valid_length,
        max_seq_len.as_uint32(),
    )
    var state: MHATileState = scheduler.initial_state(
        block_idx_ptr, tile_summary
    )

    # returns `true` if we are done
    @parameter
    @always_inline
    fn advance[
        producer: Bool,
        sync: MHASchedulerSynchronization = MHASchedulerSynchronization.DEFAULT,
    ](pipeline_idx: UInt32) -> OptionalReg[SeqInfo]:
        return scheduler.advance[ragged, producer, sync](
            tile_summary, state, pipeline_idx
        )

    # The persistent kernels limit the grid size.
    # initial_seq_info = scheduler.unsafe_get_current_work_info(tile_summary, state)

    initial_seq_info = scheduler.unsafe_seq_info[ragged](tile_summary, state)

    @parameter
    if not decoding:
        if not initial_seq_info.is_valid():

            @parameter
            if persistent:
                seq_info = advance[True, MHASchedulerSynchronization.ALL](1)
                if seq_info:
                    initial_seq_info = seq_info.value()
                else:
                    return
            else:
                return

    if tid == 0:

        @parameter
        for i in range(pipeline_stages):
            produced_mbar_kv[i].init(1)
            consumed_mbar_kv[i].init(num_consumer_threads)

        @parameter
        if persistent:

            @parameter
            for i in range(2):
                produced_mbar_q[i].init(1)
                consumed_mbar_q[i].init(num_consumer_threads)

    alias PositionType = MHAPosition[BM, BN, depth, num_heads, group, decoding]

    @parameter
    @always_inline
    fn k_tile(
        idx: UInt32,
        out k_smem: LayoutTensor[
            kv_type,
            k_smem_layout,
            MutableAnyOrigin,
            address_space = AddressSpace.SHARED,
            layout_int_type = DType.int32,
            linear_idx_type = DType.int32,
            alignment=128,
        ],
    ):
        alias sz = BN * depth
        k_smem = __type_of(k_smem)(kv_smem + sz * idx)

    @parameter
    @always_inline
    fn v_tile(
        idx: UInt32,
        out v_smem: LayoutTensor[
            kv_type,
            v_smem_layout,
            MutableAnyOrigin,
            address_space = AddressSpace.SHARED,
            layout_int_type = DType.int32,
            linear_idx_type = DType.int32,
            alignment=128,
        ],
    ):
        alias sz = BN * depth
        v_smem = __type_of(v_smem)(kv_smem + sz * idx)

    @parameter
    @always_inline
    fn get_position(seq_info: SeqInfo) -> PositionType:
        return _get_position[
            BM, BN, depth, num_heads, group, ragged, _is_cache_length_accurate
        ](
            seq_info,
            kv_lut,
            max_seq_len,
            num_keys_arg,
            kv_input_row_offsets,
        )

    var position: PositionType = get_position(initial_seq_info)

    q_pipeline_state = PipelineState[2]()

    barrier()
    # For intra-warp overlap, we initiate wgmmas as
    # Q @ K_0, Q @ K_1, P_0 @ V_0, Q @ K_2, P_1 @ V_1, ...
    # ..., Q @ K_{N-1}, P_{N-2} @ V_{N-2}, P_{N-1} @ V_{N-1}
    #
    # Due to this, we can overlap wgmmas and softmax calculations.
    if warp_group_idx == 0:
        # producer
        warpgroup_reg_dealloc[num_producer_regs]()
        if thread_idx.x == 0:
            produce[
                pipeline_stages=pipeline_stages,
                ragged=ragged,
                _is_cache_length_accurate=_is_cache_length_accurate,
            ](
                q_tma_op,
                k_tma_op,
                v_tma_op,
                q_smem,
                kv_smem,
                produced_mbar_kv,
                consumed_mbar_kv,
                produced_mbar_q,
                consumed_mbar_q,
                kv_lut,
                position,
                partition,
                scheduler,
                mask,
                tile_summary,
                state,
                max_seq_len,
                num_keys_arg,
                kv_input_row_offsets,
            )

    else:
        warpgroup_reg_alloc[num_consumer_regs]()

        # arrive to unblock the producers
        # TODO: skip this by not waiting on the first set
        @parameter
        for i in range(pipeline_stages):
            _ = consumed_mbar_kv[i].arrive()

        @parameter
        if persistent:
            _ = consumed_mbar_q[0].arrive()
        var local_warp_group_idx: UInt32 = warp_group_idx - 1

        @parameter
        @always_inline("nodebug")
        fn q_consumer(
            q_idx: UInt32,
        ) -> LayoutTensor[
            kv_type,
            q_smem_layout_consumer,
            MutableAnyOrigin,
            address_space = AddressSpace.SHARED,
            alignment=128,
        ]:
            return {q_smem + q_size * q_idx}

        # layout is
        # shape  = (2, num_m_mmas) x (2, num_n_mmas)
        # stride = (2, 4*num_n_mmas) x (1, 4)
        alias s_reg_tile_layout = Layout.row_major(
            num_m_mmas * num_n_mmas, p_frag_size
        )
        alias o_reg_tile_layout = Layout.row_major(
            num_m_mmas * num_n_mmas, o_frag_size
        )
        p_reg_tile = LayoutTensor[
            accum_type,
            s_reg_tile_layout,
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ].stack_allocation()
        output_reg_tile = (
            LayoutTensor[
                accum_type,
                o_reg_tile_layout,
                MutableAnyOrigin,
                address_space = AddressSpace.LOCAL,
            ]
            .stack_allocation()
            .fill(0)
        )
        alias p_reg_tile_layout = Layout.row_major(
            num_m_mmas * num_n_mmas * frag_ratio, a_frag_size
        )
        p_frag = LayoutTensor[
            kv_type,
            p_reg_tile_layout,
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ].stack_allocation()

        @parameter
        @always_inline
        fn vectorize_p_reg_tile(
            out result: LayoutTensor[
                accum_type,
                p_vec_output_layout,
                MutableAnyOrigin,
                address_space = AddressSpace.LOCAL,
                element_layout=element_layout,
            ],
        ):
            result = __type_of(result)(p_reg_tile.ptr)

        @parameter
        @always_inline
        fn vectorize_o_reg_tile(
            out result: LayoutTensor[
                accum_type,
                o_vec_output_layout,
                MutableAnyOrigin,
                address_space = AddressSpace.LOCAL,
                element_layout=element_layout,
            ],
        ):
            result = __type_of(result)(output_reg_tile.ptr)

        rowmax = LayoutTensor[
            accum_type,
            Layout.row_major(num_rows_per_warp),
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ].stack_allocation()
        rowsum = LayoutTensor[
            accum_type,
            Layout.row_major(num_rows_per_warp),
            MutableAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ].stack_allocation()

        # Mask global memory iterator.
        mask_warp_row = warp_y * WM
        var scale_log2e: Scalar[accum_type] = (
            scale.cast[accum_type]() if use_score_mod
            or MaskType.apply_log2e_after_mask else scale.cast[accum_type]()
            * log2e
        )

        @parameter
        @always_inline
        fn q_mul_k(read_idx: UInt32, read_phase: UInt32, q_idx: UInt32):
            k_smem_sub = k_tile(read_idx)
            var q_smem_sub = q_consumer(q_idx)
            produced_mbar_kv[read_idx].wait(read_phase)
            warpgroup_fence(p_reg_tile)
            wgmma_0.arrive()
            wgmma_0.wgmma[num_consumer, scale_c=0](
                q_smem_sub,
                k_smem_sub,
                p_reg_tile,
                Int(local_warp_group_idx),
            )
            wgmma_0.commit_group()
            warpgroup_fence(p_reg_tile)

        @parameter
        @always_inline
        fn p_mul_v(read_idx: UInt32, read_phase: UInt32):
            v_smem_sub = v_tile(read_idx)
            produced_mbar_kv[read_idx].wait(read_phase)
            warpgroup_fence(output_reg_tile)
            wgmma_1.arrive()
            wgmma_1.wgmma(
                p_frag,
                v_smem_sub,
                output_reg_tile,
            )
            wgmma_1.commit_group()
            warpgroup_fence(output_reg_tile)

        @parameter
        @always_inline
        fn wait_for_q_mul_k[wgmma_left_in_flight: Int](read_idx: UInt32):
            wgmma_0.wait_group[wgmma_left_in_flight]()  # P is available
            _ = consumed_mbar_kv[read_idx].arrive()

        @parameter
        @always_inline
        fn wait_for_p_mul_v(read_idx: UInt32):
            wgmma_1.wait_group[0]()  # output is available
            _ = consumed_mbar_kv[read_idx].arrive()

        @parameter
        @always_inline
        fn apply_mask(
            position: PositionType,
            mask_status: TileMaskStatus,
            kv_tile_start_row: UInt32,
        ):
            var max_len: UInt32 = (
                num_keys_arg if decoding else max_seq_len.as_uint32()
            )
            _apply_mask[WM, MMA_N0, num_m_mmas, num_n_mmas, use_score_mod](
                mask_warp_row,
                position,
                lane,
                max_len,
                scale_log2e,
                kv_tile_start_row,
                mask,
                mask_status,
                score_mod,
                vectorize_p_reg_tile(),
            )

        @parameter
        @always_inline
        fn scale_output(correction: __type_of(rowmax)):
            # we are now able to read/modify `output_reg_tile` and modify `p_frag`
            vout = vectorize_o_reg_tile()

            # Correct output
            # We could avoid this on the first iter
            # if we specialize and unswitch on `first_iter`
            # otherwise, the branch requires synchronization
            @parameter
            for row in range(num_rows_per_warp):
                c = SIMD[accum_type, element_layout.size()](
                    rebind[Scalar[accum_type]](correction[row])
                )

                @parameter
                for col in range(num_cols_output):
                    vout[row, col] = vout[row, col] * c

        @always_inline
        fn elementwise_reciprocal(
            old_rowsum: __type_of(rowsum), new_rowsum: __type_of(rowsum)
        ):
            # new_rowsum, old_rowsum = 1/old_rowsum, new_rowsum
            @parameter
            for row in range(num_rows_per_warp):
                old = old_rowsum[row]
                new = new_rowsum[row]
                new_rowsum[row] = recip(old)[0]
                old_rowsum[row] = new

        @parameter
        @always_inline
        fn write_output(
            position: PositionType,
            q_idx: UInt32,
            rowsum_inv: __type_of(rowsum),
        ):
            vout = vectorize_o_reg_tile()

            # Apply softmax denumerator.
            @parameter
            for row in range(num_rows_per_warp):
                rs_inv = vout.element_type(rowsum_inv[row][0])

                @parameter
                for col in range(num_cols_output):
                    vout[row, col] = vout[row, col] * rs_inv

            var output_ptr: UnsafePointer[Scalar[output_type]] = o_ptr_arg

            @parameter
            if decoding and PartitionType.do_partition:
                output_ptr = output_ptr.offset(
                    depth * num_heads * batch_size * position.prompt_offset
                )
            output_gmem_tile = position.q_out_gmem_tensor(output_ptr)

            alias swizzle = make_swizzle[
                num_rows = MMA_M // 2, row_size=BN, access_size=8
            ]()
            # Reuse a_smem for c tile in smem
            alias q_tile_size: UInt32 = q_smem_size // 2

            # ensure all threads have finished reading `q_smem`
            named_barrier[num_consumer_threads]()
            accum_smem_tile = output_reg_to_smem[
                BM,
                BN,
                WM,
                depth,
                kv_type,
                output_type,
                accum_type,
                o_reg_tile_layout,
                o_frag_size,
                num_consumer_threads,
                simd_size,
                swizzle,
                num_m_mmas,
                num_consumer,
                mma_thread_layout,
            ](
                tid,
                local_warp_group_idx,
                warp_x,
                warp_y,
                q_smem + q_idx * q_tile_size,
                output_reg_tile,
            )
            # Guard writing to shared memory.
            named_barrier[num_consumer_threads]()
            # Vectorized copy from shared to global memory, during which every 2 FP32
            # are cast to 2 BF16 so that 2 4xFP32 vectors are merged into 1 8xBF16
            # vector and stored using 16B store instruction.
            copy_sram_to_dram[
                thread_layout = Layout.row_major(
                    num_consumer_threads * simd_size // depth,
                    depth // simd_size,
                ),
                swizzle=swizzle,
            ](
                output_gmem_tile.vectorize[1, simd_size](),
                accum_smem_tile.vectorize[1, simd_size](),
            )

        startend = position.get_start_and_end_for_partitions[BN=BN](partition)
        var kv_tile_start_row: UInt32 = startend[0]
        var end: UInt32 = startend[1]

        @parameter
        if decoding and PartitionType.do_partition:
            if kv_tile_start_row >= end:
                if thread_idx.x % 4 == 0 and thread_idx.x < 4 * group + 128:
                    exp_sum_ptr, qk_max_ptr = position.exp_sum_qk_max_ptr(
                        partition, batch_size
                    )
                    var q_head_idx = position.head_idx * group + lane // 4
                    exp_sum_ptr[q_head_idx] = Scalar[PartitionType.accum_dtype](
                        0
                    )
                    qk_max_ptr[q_head_idx] = min_or_neg_inf[
                        PartitionType.accum_dtype
                    ]()

                write_output(position, q_pipeline_state.index(), rowsum)
                return

        var mask_status: TileMaskStatus
        while True:
            mask_status = position.mask_status(mask, kv_tile_start_row)
            if mask_status != TileMaskStatus.FULL_MASK:
                break
            kv_tile_start_row += BN

        read_pipeline_states = PipelineState[pipeline_stages]()

        # q_mul_k must wait on fetching q and k
        # therefore, we find `kv_tile_start_row` first.
        var read_idx_q: UInt32 = read_pipeline_states.index()
        q_mul_k(
            read_idx_q,
            read_pipeline_states.phase(),
            q_pipeline_state.index(),
        )
        read_pipeline_states.step()
        wait_for_q_mul_k[0](read_idx_q)

        apply_mask(position, mask_status, kv_tile_start_row)

        # Compute initial rowmax
        var attention_rowmax = _rowmax_online_softmax[
            # threads layout by warp
            1,
            mma_thread_layout,
            use_exp2=True,
        ](vectorize_p_reg_tile(), rowmax, init_rowmax=True)

        # Include sink_weights in rowmax computation if present
        @parameter
        if sink:
            var head_idx = position.head_idx
            var sink_weight = sink_weights_ptr[head_idx]

            @parameter
            for i in range(num_rows_per_warp):
                attention_rowmax[i] = max(
                    attention_rowmax[i], sink_weight.cast[accum_type]()
                )

        rowmax.copy_from(attention_rowmax)

        # Compute rowsum
        var attention_rowsum = _rowsum[mma_thread_layout](
            vectorize_p_reg_tile()
        )

        # Add sink weight contribution to rowsum
        @parameter
        if sink:
            var head_idx = position.head_idx
            var sink_weight = sink_weights_ptr[head_idx].cast[accum_type]()

            @parameter
            for i in range(num_rows_per_warp):
                # Compute exp2((sink_weight - rowmax[i]) * log2e)
                var sink_contribution = exp2((sink_weight - rowmax[i]) * log2e)
                attention_rowsum[i] = attention_rowsum[i] + sink_contribution[0]

        rowsum.copy_from(attention_rowsum)

        var position_prev: PositionType = position
        var q_idx_old: UInt32 = q_pipeline_state.index()
        var q_phase_old: UInt32 = q_pipeline_state.phase()

        # Consumption order:
        # Preheader: Q0, K0
        # Body: Q1, K1, V0, Q2, K2, V1, ..., Q{-1}, K{-1}, V{-2}
        # Exit: V{-1}
        @parameter
        if persistent:
            kv_tile_start_row += BN
        while True:
            while True:

                @parameter
                if not persistent:
                    kv_tile_start_row += BN
                if kv_tile_start_row >= end:
                    break

                # this loops over num_keys
                mask_status = position.mask_status(mask, kv_tile_start_row)
                if mask_status == TileMaskStatus.FULL_MASK:

                    @parameter
                    if persistent:
                        kv_tile_start_row += BN
                    continue
                p_frag.vectorize[
                    1, a_frag_size
                ]().copy_from(  # copy new pfrag, used by `p_mul_v` on next iter
                    p_reg_tile.reshape[
                        Layout.row_major(
                            num_m_mmas * num_n_mmas * frag_ratio,
                            a_frag_size,
                        )
                    ]().vectorize[1, a_frag_size](),
                )

                # new pipeline states
                var read_idx_q: UInt32 = read_pipeline_states.index()
                # start wgmmas
                q_mul_k(
                    read_idx_q,
                    read_pipeline_states.phase(),
                    q_pipeline_state.index(),
                )  # can't rw `p_reg_tile`
                read_pipeline_states.step()
                var read_idx_v: UInt32 = read_pipeline_states.index()
                p_mul_v(
                    read_idx_v, read_pipeline_states.phase()
                )  # can't rw output or pfrag
                read_pipeline_states.step()
                wait_for_q_mul_k[1](read_idx_q)  # can rw `p_reg_tile`

                apply_mask(position, mask_status, kv_tile_start_row)
                new_q = persistent and q_idx_old != q_pipeline_state.index()
                # Compute rowmax for current scores
                var current_rowmax = _rowmax_online_softmax[
                    # threads layout by warp
                    1,
                    mma_thread_layout,
                    use_exp2=True,
                ](vectorize_p_reg_tile(), rowmax, new_q)

                # Include sink_weights in rowmax if present
                @parameter
                if sink:
                    var head_idx = position.head_idx
                    var sink_weight = sink_weights_ptr[head_idx]

                    @parameter
                    for i in range(num_rows_per_warp):
                        current_rowmax[i] = max(
                            current_rowmax[i], sink_weight.cast[accum_type]()
                        )

                score_frag_rowmax = current_rowmax
                if new_q:

                    @parameter
                    if decoding and PartitionType.do_partition:
                        if (
                            thread_idx.x % 4 == 0
                            and thread_idx.x < 4 * group + 128
                        ):
                            exp_sum_ptr, qk_max_ptr = (
                                position_prev.exp_sum_qk_max_ptr(
                                    partition, batch_size
                                )
                            )
                            var q_head_idx = (
                                position_prev.head_idx * group + lane // 4
                            )
                            exp_sum_ptr[q_head_idx] = rebind[
                                Scalar[PartitionType.accum_dtype]
                            ](rowsum[0])
                            qk_max_ptr[q_head_idx] = rebind[
                                Scalar[PartitionType.accum_dtype]
                            ](rowmax[0])
                    score_frag_rowsum = rebind[__type_of(rowsum)](
                        _rowsum[mma_thread_layout](vectorize_p_reg_tile())
                    )
                    rowmax.copy_from(score_frag_rowmax)
                    elementwise_reciprocal(rowsum, score_frag_rowsum)
                    wait_for_p_mul_v(read_idx_v)  # can rw output and pfrag
                    # we `^ 1` to access the previous
                    # Two separate issues:
                    # 0. Which q do we use for `accum_smem`?
                    # 1. Which qs, if any, do we `arrive` at?
                    #
                    # If the next q_idx != the current q_idx (i.e. q_idx_n != q_idx)
                    # then we can use the current q for writing smem.
                    # If `q_idx_n == q_idx`, then we use the old q_idx (i.e. q_idx_o).
                    # This means we were not allowed to `arrive` at `q_idx_o`.
                    #
                    # Letting `0` indicate inequality, and `1` equality,
                    # let x = q_idx == q_idx_n
                    # let y = q_idx_n == q_idx_n_n
                    # We thus have 4 states `xy`:
                    # 0. 00: We use q_idx and arrive
                    # 1. 01: We use q_idx, but do not arrive on q_idx
                    # 2. 10: We use q_idx_o, do not arrive on q_idx
                    # 3. 11: We use q_idx_o, do not arrive on q_idx
                    #
                    # Only in `00` do we get to arrive on `q_idx` early.
                    # Given `BN < num_keys`, it won't often be the case
                    # that we can arrive at Q early; we need a series
                    # of q_tile_idx and head_idx that have a lot of
                    # `FULL_MASK`s, which our iteration scheme is supposed
                    # to make unlikely.
                    # Thus, we're going to simplify the problem by assuming
                    # scenario `0.` is unlikely unless `BN >= num_keys`,
                    # in which case it is guaranteed.
                    # var q_idx: UInt32 = q_pipeline_state.index() if few_keys else q_idx_old
                    write_output(position_prev, q_idx_old, score_frag_rowsum)
                    var q_idx_new: UInt32 = q_pipeline_state.index()

                    _ = consumed_mbar_q[q_idx_new].arrive()
                    _ = output_reg_tile.vectorize[accum_simd_width]().fill(0)
                    position_prev = position
                    q_idx_old = q_idx_new
                    q_phase_old = q_pipeline_state.phase()
                else:
                    score_frag_rowsum = rebind[__type_of(rowsum)](
                        _rowsum[mma_thread_layout](vectorize_p_reg_tile())
                    )

                    # Add sink weight contribution to score_frag_rowsum
                    @parameter
                    if sink:
                        var head_idx = position.head_idx
                        var sink_weight = sink_weights_ptr[head_idx].cast[
                            accum_type
                        ]()

                        @parameter
                        for i in range(num_rows_per_warp):
                            # Compute exp2((sink_weight - rowmax[i]) * log2e)
                            var sink_contribution = exp2(
                                (sink_weight - rowmax[i]) * log2e
                            )
                            score_frag_rowsum[i] = (
                                score_frag_rowsum[i] + sink_contribution
                            )

                    _online_softmax_correction[use_exp2=True](
                        rowmax, score_frag_rowmax
                    )
                    # rowmax now holds score_frag_rowmax
                    # score_frag_rowmax now holds the correction

                    @parameter
                    for i in range(num_rows_per_warp):
                        rowsum[i] = (
                            rowsum[i] * score_frag_rowmax[i]
                            + score_frag_rowsum[i]
                        )

                    wait_for_p_mul_v(read_idx_v)  # can rw output and pfrag
                    scale_output(score_frag_rowmax)  # scale output

                @parameter
                if persistent:
                    kv_tile_start_row += BN

            @parameter
            if persistent:
                var q_idx_old: UInt32 = q_pipeline_state.index()
                var q_phase_old: UInt32 = q_pipeline_state.phase()
                q_pipeline_state.step()
                produced_mbar_q[q_idx_old].wait(q_phase_old)
                docontinue = advance[False](q_idx_old)
                if not docontinue:
                    break
                position = get_position(docontinue.value())
                start, new_end = position.get_start_and_end_for_partitions[
                    BN=BN
                ](partition)
                kv_tile_start_row = start
                end = new_end
            else:
                break

        p_frag.vectorize[1, a_frag_size]().copy_from(
            p_reg_tile.reshape[
                Layout.row_major(
                    num_m_mmas * num_n_mmas * frag_ratio, a_frag_size
                )
            ]().vectorize[1, a_frag_size](),
        )
        p_mul_v(
            read_pipeline_states.index(),
            read_pipeline_states.phase(),
        )

        @parameter
        if decoding and PartitionType.do_partition:
            if thread_idx.x % 4 == 0 and thread_idx.x < 4 * group + 128:
                exp_sum_ptr, qk_max_ptr = position.exp_sum_qk_max_ptr(
                    partition, batch_size
                )
                var q_head_idx = position.head_idx * group + lane // 4
                exp_sum_ptr[q_head_idx] = rebind[
                    Scalar[PartitionType.accum_dtype]
                ](rowsum[0])
                qk_max_ptr[q_head_idx] = rebind[
                    Scalar[PartitionType.accum_dtype]
                ](rowmax[0])

        @parameter
        for row in range(num_rows_per_warp):
            rowsum[row] = recip(rowsum[row])[0]
        wgmma_1.wait_group()
        write_output(position, q_pipeline_state.index(), rowsum)
        # don't arrive
