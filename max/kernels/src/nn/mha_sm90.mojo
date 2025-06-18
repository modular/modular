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
from math import ceildiv, recip
from math.constants import log2e
from sys import alignof, env_get_int, simdwidthof, sizeof

import gpu.warp as warp
from algorithm.functional import unswitch
from buffer import NDBuffer
from gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    barrier,
    block_dim,
    global_idx,
    lane_id,
    thread_idx,
)
from gpu.host import DeviceContext, FuncAttribute
from gpu.host._nvidia_cuda import TensorMapSwizzle
from gpu.host.info import H100
from gpu.intrinsics import warpgroup_reg_alloc, warpgroup_reg_dealloc
from gpu.memory import AddressSpace, external_memory
from gpu.sync import async_copy_arrive, named_barrier
from layout.int_tuple import IntTuple
from layout.layout import Layout
from layout.layout_tensor import (
    LayoutTensor,
    LayoutTensorIter,
    copy,
    copy_dram_to_sram_async,
    copy_local_to_dram,
    copy_sram_to_dram,
    cp_async_k_major,
    cp_async_mn_major,
)
from layout.runtime_layout import RuntimeLayout, RuntimeTuple
from layout.swizzle import make_swizzle
from layout.tensor_core import get_fragment_size
from layout.tensor_core_async import (
    TensorCoreAsync,
    tile_layout_k_major,
    tile_layout_mn_major,
)
from layout.tma_async import PipelineState, SharedMemBarrier
from nn.mha_mask import MHAMask, TileMaskStatus
from nn.mha_operand import MHAOperand
from nn.mha_score_mod import ScoreModTrait
from nn.mha_tile_scheduler import (
    MHASchedule,
    MHASchedulerSynchronization,
    MHATileScheduler,
    MHATileState,
    MHATileSummary,
    QueuedTileScheduler,
    SeqInfo,
    TileScheduler,
    TransientScheduler,
    WorkInfo,
)
from nn.mha_utils import (
    DynamicInt,
    FlashAttentionAlgorithm,
    MHAConfig,
    MHAPartitionScheme,
    NoPartition,
    OptionallyStaticInt,
    SplitKPartition,
    StaticInt,
    _copy_frag_to_smem,
    _is_decoding,
    _kernel_mask,
    get_start_and_end_for_partitions,
)
from nn.softmax import (
    _online_softmax_correction,
    _rowmax_online_softmax,
    _rowsum,
)
from tensor_internal import ManagedTensorSlice

from utils.index import Index, IndexList
from utils.numerics import get_accum_type, min_or_neg_inf
from utils.static_tuple import StaticTuple


@always_inline
fn mha_sm90_dispatch[
    k_t: MHAOperand,
    v_t: MHAOperand,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    type: DType,
    output_type: DType,
    max_prompt_len_t: OptionallyStaticInt,
    partition_t: MHAPartitionScheme, //,
    config: MHAConfig,
    group: Int,
    use_score_mod: Bool,
    ragged: Bool,
    _is_cache_length_accurate: Bool,
](
    output: UnsafePointer[Scalar[output_type]],
    q: UnsafePointer[Scalar[type]],
    k: k_t,
    v: v_t,
    mask_functor: mask_t,
    score_mod_functor: score_mod_t,
    valid_length: ManagedTensorSlice[dtype = DType.uint32, rank=1],
    max_prompt_len_arg: max_prompt_len_t,
    max_cache_valid_length_arg: Int,
    scale: Float32,
    kv_input_row_offsets: OptionalReg[
        NDBuffer[DType.uint32, 1, MutableAnyOrigin]
    ],
    batch_size_arg: Int,
    partition: partition_t,
    ctx: DeviceContext,
) raises:
    alias decoding: Bool = max_prompt_len_t.static_value.or_else(0) == 1
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
    alias persistent = 0 if partition_t.do_partition else env_get_int[
        "USE_EXPERIMENTAL_KERNELS", 0
    ]()
    constrained[new_config.algorithm == FlashAttentionAlgorithm(3)]()

    var max_cache_valid_length: UInt32 = UInt32(max_cache_valid_length_arg)
    var batch_size: UInt32 = UInt32(batch_size_arg)
    var max_prompt_len: UInt32 = max_prompt_len_arg.as_uint32()
    var max_num_prompt_tiles: UInt32 = ceildiv(max_prompt_len, BM)
    var block_x: UInt32 = max_num_prompt_tiles * partition.num_partitions()

    alias num_scheduler_heads = config.num_heads // group if decoding else config.num_heads
    # if decoding,
    alias scheduler_tile_shape = 1 if decoding else BM

    @parameter
    if persistent == 0:
        alias scheduler_t = TransientScheduler[
            scheduler_tile_shape, num_scheduler_heads
        ]
        alias kernel_sm90 = _mha_sm90[
            new_config.type,
            k_t,
            v_t,
            output_type,
            mask_t,
            score_mod_t,
            scheduler_t,
            new_config,
            group=group,
            use_score_mod=use_score_mod,
            ragged=ragged,
            _is_cache_length_accurate=_is_cache_length_accurate,
            max_seq_len_t=max_prompt_len_t,
            partition_t=partition_t,
        ]
        var scheduler: scheduler_t = scheduler_t()
        gd = scheduler_t.grid_dim(batch_size, block_x)

        @parameter
        if max_prompt_len_t.static_value:

            @parameter
            if partition_t.do_partition:
                ctx.enqueue_function[kernel_sm90](
                    q,
                    k,
                    v,
                    output,
                    scale,
                    batch_size,
                    max_cache_valid_length,
                    valid_length_managed_tensor_slice_to_ndbuffer(valid_length),
                    kv_input_row_offsets,
                    partition,
                    mask_functor,
                    score_mod_functor,
                    grid_dim=scheduler_t.grid_dim(batch_size, block_x),
                    block_dim=(Int(num_threads), 1, 1),
                    shared_mem_bytes=Int(smem_use),
                    func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                        smem_use
                    ),
                )
            else:
                ctx.enqueue_function[kernel_sm90](
                    q,
                    k,
                    v,
                    output,
                    scale,
                    batch_size,
                    max_cache_valid_length,
                    valid_length_managed_tensor_slice_to_ndbuffer(valid_length),
                    kv_input_row_offsets,
                    mask_functor,
                    score_mod_functor,
                    grid_dim=scheduler_t.grid_dim(batch_size, block_x),
                    block_dim=(Int(num_threads), 1, 1),
                    shared_mem_bytes=Int(smem_use),
                    func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                        smem_use
                    ),
                )

        else:

            @parameter
            if partition_t.do_partition:
                ctx.enqueue_function[kernel_sm90](
                    q,
                    k,
                    v,
                    output,
                    scale,
                    batch_size,
                    max_prompt_len,
                    max_cache_valid_length,
                    valid_length_managed_tensor_slice_to_ndbuffer(valid_length),
                    kv_input_row_offsets,
                    partition,
                    mask_functor,
                    score_mod_functor,
                    grid_dim=scheduler_t.grid_dim(batch_size, block_x),
                    block_dim=(Int(num_threads), 1, 1),
                    shared_mem_bytes=Int(smem_use),
                    func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                        smem_use
                    ),
                )
            else:
                ctx.enqueue_function[kernel_sm90](
                    q,
                    k,
                    v,
                    output,
                    scale,
                    batch_size,
                    max_prompt_len,
                    max_cache_valid_length,
                    valid_length_managed_tensor_slice_to_ndbuffer(valid_length),
                    kv_input_row_offsets,
                    mask_functor,
                    score_mod_functor,
                    grid_dim=scheduler_t.grid_dim(batch_size, block_x),
                    block_dim=(Int(num_threads), 1, 1),
                    shared_mem_bytes=Int(smem_use),
                    func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                        smem_use
                    ),
                )
    elif persistent == 2:
        alias scheduler_t = TileScheduler[
            scheduler_tile_shape, num_scheduler_heads
        ]
        alias kernel_sm90 = _mha_sm90[
            new_config.type,
            k_t,
            v_t,
            output_type,
            mask_t,
            score_mod_t,
            scheduler_t,
            new_config,
            group=group,
            use_score_mod=use_score_mod,
            ragged=ragged,
            _is_cache_length_accurate=_is_cache_length_accurate,
            max_seq_len_t=max_prompt_len_t,
            partition_t=partition_t,
        ]
        var scheduler: scheduler_t = scheduler_t()

        @parameter
        if max_prompt_len_t.static_value:

            @parameter
            if partition_t.do_partition:
                ctx.enqueue_function[kernel_sm90](
                    q,
                    k,
                    v,
                    output,
                    scale,
                    batch_size,
                    max_cache_valid_length,
                    valid_length_managed_tensor_slice_to_ndbuffer(valid_length),
                    kv_input_row_offsets,
                    partition,
                    mask_functor,
                    score_mod_functor,
                    scheduler,
                    grid_dim=scheduler_t.grid_dim(batch_size, block_x),
                    block_dim=(Int(num_threads), 1, 1),
                    shared_mem_bytes=Int(smem_use),
                    func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                        smem_use
                    ),
                )
            else:
                ctx.enqueue_function[kernel_sm90](
                    q,
                    k,
                    v,
                    output,
                    scale,
                    batch_size,
                    max_cache_valid_length,
                    valid_length_managed_tensor_slice_to_ndbuffer(valid_length),
                    kv_input_row_offsets,
                    mask_functor,
                    score_mod_functor,
                    scheduler,
                    grid_dim=scheduler_t.grid_dim(batch_size, block_x),
                    block_dim=(Int(num_threads), 1, 1),
                    shared_mem_bytes=Int(smem_use),
                    func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                        smem_use
                    ),
                )
        else:

            @parameter
            if partition_t.do_partition:
                ctx.enqueue_function[kernel_sm90](
                    q,
                    k,
                    v,
                    output,
                    scale,
                    batch_size,
                    max_prompt_len,
                    max_cache_valid_length,
                    valid_length_managed_tensor_slice_to_ndbuffer(valid_length),
                    kv_input_row_offsets,
                    partition,
                    mask_functor,
                    score_mod_functor,
                    scheduler,
                    grid_dim=scheduler_t.grid_dim(batch_size, block_x),
                    block_dim=(Int(num_threads), 1, 1),
                    shared_mem_bytes=Int(smem_use),
                    func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                        smem_use
                    ),
                )
            else:
                ctx.enqueue_function[kernel_sm90](
                    q,
                    k,
                    v,
                    output,
                    scale,
                    batch_size,
                    max_prompt_len,
                    max_cache_valid_length,
                    valid_length_managed_tensor_slice_to_ndbuffer(valid_length),
                    kv_input_row_offsets,
                    mask_functor,
                    score_mod_functor,
                    scheduler,
                    grid_dim=scheduler_t.grid_dim(batch_size, block_x),
                    block_dim=(Int(num_threads), 1, 1),
                    shared_mem_bytes=Int(smem_use),
                    func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                        smem_use
                    ),
                )
    else:
        alias scheduler_t = QueuedTileScheduler[
            scheduler_tile_shape, num_scheduler_heads, decoding=decoding
        ]
        alias kernel_sm90 = _mha_sm90[
            new_config.type,
            k_t,
            v_t,
            output_type,
            mask_t,
            score_mod_t,
            scheduler_t,
            new_config,
            group=group,
            use_score_mod=use_score_mod,
            ragged=ragged,
            _is_cache_length_accurate=_is_cache_length_accurate,
            max_seq_len_t=max_prompt_len_t,
            partition_t=partition_t,
        ]
        var schedule = ctx.enqueue_create_buffer[DType.uint32](1).enqueue_fill(
            UInt32(H100.sm_count)
        )
        ctx.synchronize()
        var scheduler: scheduler_t = scheduler_t(schedule.unsafe_ptr())

        # these nested branches are to reduce risk of memory corruption
        # when passing 0-sized arguments, which is currently not handled
        # correctly in Mojo
        # TODO: Remove and simplify when KERN-1753 is fixed
        @parameter
        if max_prompt_len_t.static_value:

            @parameter
            if partition_t.do_partition:
                ctx.enqueue_function[kernel_sm90](
                    rebind[scheduler_t](scheduler),
                    q,
                    k,
                    v,
                    output,
                    rebind[Float32](scale),
                    batch_size,
                    max_cache_valid_length,
                    valid_length_managed_tensor_slice_to_ndbuffer(valid_length),
                    rebind[
                        OptionalReg[NDBuffer[DType.uint32, 1, MutableAnyOrigin]]
                    ](kv_input_row_offsets),
                    partition,
                    rebind[mask_t](mask_functor),
                    rebind[score_mod_t](score_mod_functor),
                    grid_dim=scheduler_t.grid_dim(batch_size, block_x),
                    block_dim=(Int(num_threads), 1, 1),
                    shared_mem_bytes=Int(smem_use),
                    func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                        smem_use
                    ),
                )
            else:
                ctx.enqueue_function[kernel_sm90](
                    rebind[scheduler_t](scheduler),
                    q,
                    k,
                    v,
                    output,
                    rebind[Float32](scale),
                    batch_size,
                    max_cache_valid_length,
                    valid_length_managed_tensor_slice_to_ndbuffer(valid_length),
                    rebind[
                        OptionalReg[NDBuffer[DType.uint32, 1, MutableAnyOrigin]]
                    ](kv_input_row_offsets),
                    rebind[mask_t](mask_functor),
                    rebind[score_mod_t](score_mod_functor),
                    grid_dim=scheduler_t.grid_dim(batch_size, block_x),
                    block_dim=(Int(num_threads), 1, 1),
                    shared_mem_bytes=Int(smem_use),
                    func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                        smem_use
                    ),
                )
        else:

            @parameter
            if partition_t.do_partition:
                ctx.enqueue_function[kernel_sm90](
                    rebind[scheduler_t](scheduler),
                    q,
                    k,
                    v,
                    output,
                    rebind[Float32](scale),
                    batch_size,
                    max_prompt_len,
                    max_cache_valid_length,
                    valid_length_managed_tensor_slice_to_ndbuffer(valid_length),
                    rebind[
                        OptionalReg[NDBuffer[DType.uint32, 1, MutableAnyOrigin]]
                    ](kv_input_row_offsets),
                    partition,
                    rebind[mask_t](mask_functor),
                    rebind[score_mod_t](score_mod_functor),
                    grid_dim=scheduler_t.grid_dim(batch_size, block_x),
                    block_dim=(Int(num_threads), 1, 1),
                    shared_mem_bytes=Int(smem_use),
                    func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                        smem_use
                    ),
                )
            else:
                ctx.enqueue_function[kernel_sm90](
                    rebind[scheduler_t](scheduler),
                    q,
                    k,
                    v,
                    output,
                    rebind[Float32](scale),
                    batch_size,
                    max_prompt_len,
                    max_cache_valid_length,
                    valid_length_managed_tensor_slice_to_ndbuffer(valid_length),
                    rebind[
                        OptionalReg[NDBuffer[DType.uint32, 1, MutableAnyOrigin]]
                    ](kv_input_row_offsets),
                    rebind[mask_t](mask_functor),
                    rebind[score_mod_t](score_mod_functor),
                    grid_dim=scheduler_t.grid_dim(batch_size, block_x),
                    block_dim=(Int(num_threads), 1, 1),
                    shared_mem_bytes=Int(smem_use),
                    func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                        smem_use
                    ),
                )
        _ = schedule


@register_passable("trivial")
struct MHAPosition[
    BM: Int, BN: Int, depth: Int, num_heads: Int, group: Int, decoding: Bool
](Copyable, Movable):
    """
    Position of the MHA-kernel.
    When `decoding=False`, `q_head_stride == num_heads`.
    When `decoding=True`, `q_head_stride == 1`.
    """

    var q_out_offset: Int
    var num_keys: UInt32
    var start_pos: UInt32
    var seq_len: UInt32
    var head_idx: UInt32  # when decoding, kv_head_idx
    var prompt_offset: UInt32  # when decoding, this is the position_idx
    var prompt_idx: UInt32

    alias q_stride: Int = Self.depth if decoding else Self.depth * Self.num_heads
    alias q_output_gmem_layout = Layout(
        IntTuple(Self.BM, Self.depth), IntTuple(Self.q_stride, 1)
    )

    @always_inline
    fn __init__(
        out self,
        q_out_offset: Int,
        num_keys: UInt32,
        start_pos: UInt32,
        seq_info: SeqInfo,
    ):
        self.q_out_offset = q_out_offset
        self.num_keys = num_keys
        self.start_pos = start_pos
        self.seq_len = seq_info.seq_len
        self.head_idx = seq_info.head_idx
        self.prompt_offset = seq_info.prompt_offset
        self.prompt_idx = seq_info.prompt_idx  # batch idx

    @always_inline
    fn q_head_idx(self) -> UInt32:
        @parameter
        if Self.decoding:
            return self.head_idx * Self.group
        else:
            return self.head_idx

    @always_inline
    fn kv_head_idx(self) -> UInt32:
        @parameter
        if Self.decoding:
            return self.head_idx
        else:
            return self.head_idx // Self.group

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        writer.write(
            "(",
            self.q_out_offset,
            ", ",
            self.seq_len,
            ", ",
            self.num_keys,
            ", ",
            self.start_pos,
            ", ",
            self.prompt_offset,
            ", ",
            self.head_idx,
            ", ",
            self.prompt_idx,
            ")",
        )

    @always_inline
    fn q_tile_num_rows(self) -> UInt32:
        @parameter
        if decoding:
            return Self.group
        else:
            return min(BM, self.seq_len - self.prompt_offset)

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        return self.q_out_offset == other.q_out_offset

    @always_inline
    fn __ne__(self, other: Self) -> Bool:
        return self.q_out_offset != other.q_out_offset

    @always_inline
    fn q_out_gmem_tensor[
        dtype: DType
    ](
        self,
        ptr: UnsafePointer[Scalar[dtype]],
        out gmem_block: LayoutTensor[
            dtype,
            Self.q_output_gmem_layout,
            MutableAnyOrigin,
            layout_int_type = DType.int32,
            linear_idx_type = DType.int32,
            masked=True,
        ],
    ):
        gmem_block = __type_of(gmem_block)(
            ptr + self.q_out_offset,
            __type_of(gmem_block.runtime_layout)(
                __type_of(gmem_block.runtime_layout.shape)(
                    Int(self.q_tile_num_rows()), depth
                ),
                __type_of(gmem_block.runtime_layout.stride)(Self.q_stride, 1),
            ),
        )

    @always_inline
    fn mask_status[
        mask_t: MHAMask
    ](self, mask: mask_t, kv_tile_start_row: UInt32) -> TileMaskStatus:
        @parameter
        if decoding:
            return TileMaskStatus.PARTIAL_MASK
        else:
            return mask.status(
                Index[dtype = DType.int32](
                    Int(self.prompt_offset + self.start_pos),
                    Int(kv_tile_start_row),
                ),
                Index[dtype = DType.int32](Int(Self.BM), Int(Self.BN)),
            )

    @always_inline
    fn exp_sum_qk_max_ptr[
        partition_t: MHAPartitionScheme
    ](
        self,
        partition: partition_t,
        batch_size: UInt32,
    ) -> Tuple[
        UnsafePointer[Scalar[partition_t.accum_dtype]],
        UnsafePointer[Scalar[partition_t.accum_dtype]],
    ]:
        exp_sum_offset = Self.num_heads * (
            self.prompt_idx + batch_size * self.prompt_offset
        )
        exp_sum_ptr = partition.get_exp_sum_qk_max_pointer().offset(
            exp_sum_offset
        )
        qk_max_ptr = exp_sum_ptr.offset(
            Self.num_heads * batch_size * partition.num_partitions()
        )
        return (exp_sum_ptr, qk_max_ptr)

    @always_inline
    fn get_start_and_end_for_partitions[
        partition_t: MHAPartitionScheme, //, BN: Int
    ](self, partition: partition_t) -> Tuple[UInt32, UInt32]:
        @parameter
        if partition_t.do_partition:
            start, end = get_start_and_end_for_partitions[BN](
                Int(self.num_keys),
                Int(partition.num_partitions()),
                Int(self.prompt_offset),
            )
            return (UInt32(start), UInt32(end))
        else:
            return (UInt32(0), self.num_keys)


@always_inline
fn _get_position[
    k_t: MHAOperand,
    max_seq_len_t: OptionallyStaticInt, //,
    config: MHAConfig,
    group: Int,
    ragged: Bool,
    _is_cache_length_accurate: Bool,
](
    out ret: MHAPosition[
        config.block_m(),
        config.block_n(),
        config.depth,
        config.num_heads,
        group,
        _is_decoding[max_seq_len_t](),
    ],
    seq_info: SeqInfo,
    k: k_t,
    max_seq_len: max_seq_len_t,
    num_keys_arg: UInt32,
    valid_length: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
    kv_input_row_offsets: OptionalReg[
        NDBuffer[DType.uint32, 1, MutableAnyOrigin]
    ],
):
    alias depth = config.depth
    alias num_heads = config.num_heads

    var batch_idx: UInt32 = seq_info.prompt_idx
    # mha inputs
    var seq_len: UInt32 = seq_info.seq_len
    var num_keys: UInt32
    var start_pos: UInt32
    var q_offset: Int

    @parameter
    if ragged:
        cache_len = k.cache_length(Int(batch_idx))

        @parameter
        if not _is_cache_length_accurate:
            start_pos = cache_len
        else:
            start_pos = 0

        # this is used for cross attention where we get the num_keys
        # from kv_input_row_offsets. This is when num_keys != seq_len
        if kv_input_row_offsets:
            var kv_row_offsets = kv_input_row_offsets.value()
            kv_seq_start = Int(kv_row_offsets[Int(batch_idx)])
            kv_seq_end = Int(kv_row_offsets[Int(batch_idx) + 1])
            cur_kv_len = kv_seq_end - kv_seq_start
            num_keys = cur_kv_len + cache_len
        else:
            num_keys = seq_len + cache_len
        q_offset = Int(seq_info.start_of_seq) * Int(num_heads)

    # NDBuffer inputs, homogeneous batching.
    else:
        num_keys = num_keys_arg

        # When cache length (num_keys) is greater, we assume it has
        # prefix preceding the input seq_len.
        start_pos = num_keys - seq_len
        q_offset = Int(num_heads * batch_idx) * Int(max_seq_len)

    var kv_head_idx: UInt32

    @parameter
    if _is_decoding[max_seq_len_t]():
        q_offset += Int(seq_info.head_idx) * group
    else:  # head_idx is for q_heads
        q_offset += Int(seq_info.head_idx) + Int(seq_info.prompt_offset) * Int(
            num_heads
        )
    ret = __type_of(ret)(
        Int(depth) * q_offset,
        num_keys,
        start_pos,
        seq_info,
    )


@always_inline("nodebug")
fn _produce[
    smem_layout: Layout,
    kv_t: MHAOperand,
    BM: Int,
    BN: Int,
    depth: Int,
    num_heads: Int,
    group: Int,
    decoding: Bool, //,
    kv_num_heads: Int,
    *,
    axis: Int,
    wait: Bool,
](
    write_idx: UInt32,
    write_phase: UInt32,
    kv_tile_start_row: UInt32,
    position: MHAPosition[BM, BN, depth, num_heads, group, decoding],
    consumed_mbar: UnsafePointer[
        SharedMemBarrier, address_space = AddressSpace.SHARED
    ],
    produced_mbar: UnsafePointer[
        SharedMemBarrier, address_space = AddressSpace.SHARED
    ],
    kv: kv_t,
    smem_tile: LayoutTensor[
        kv_t.type,
        smem_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        layout_int_type = DType.int32,
        linear_idx_type = DType.int32,
    ],
):
    kv_tile_num_rows = min(
        Int(BN), Int(position.num_keys) - Int(kv_tile_start_row)
    )
    alias kv_gmem_layout = Layout(
        IntTuple(Int(BN), Int(depth)),
        IntTuple(Int(kv_num_heads * depth), 1),
    )
    alias kv_runtime_layout_t = RuntimeLayout[
        kv_gmem_layout,
        element_type = DType.int32,
        linear_idx_type = DType.int32,
    ]

    kv_runtime_layout = kv_runtime_layout_t(
        RuntimeTuple[
            kv_gmem_layout.shape,
            element_type = kv_runtime_layout_t.element_type,
        ](kv_tile_num_rows, depth),
        RuntimeTuple[
            kv_gmem_layout.stride,
            element_type = kv_runtime_layout_t.linear_idx_type,
        ](kv_num_heads * depth, 1),
    )

    gmem_ptr = kv.block_paged_ptr[BN](
        position.prompt_idx,
        Int(kv_tile_start_row),
        Int(position.kv_head_idx()),
        0,
    )

    @parameter
    if wait:
        consumed_mbar[write_idx].wait(write_phase)

    @parameter
    @always_inline("nodebug")
    fn copy_gmem_to_smem[masked: Bool]():
        gmem_block = LayoutTensor[
            kv_t.type,
            kv_gmem_layout,
            MutableAnyOrigin,
            layout_int_type = DType.int32,
            linear_idx_type = DType.int32,
            masked=masked,
        ](gmem_ptr, kv_runtime_layout)

        @parameter
        if axis == 0:
            cp_async_mn_major(smem_tile, gmem_block)
        else:
            cp_async_k_major(smem_tile, gmem_block)

    # if kv_tile_num_rows >= BN:
    #     copy_gmem_to_smem[False]()
    # else:
    #     copy_gmem_to_smem[True]()
    unswitch[copy_gmem_to_smem](kv_tile_num_rows < BN)

    p_mbar = produced_mbar + write_idx
    async_copy_arrive(p_mbar)
    _ = p_mbar[].arrive()


@always_inline
fn _apply_mask[
    BM: Int,
    BN: Int,
    depth: Int,
    num_heads: Int,
    group: Int,
    decoding: Bool,
    accum_type: DType,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    reg_tile_layout: Layout,
    element_layout: Layout, //,
    # last_iter: Bool,
    MMA_M: Int,
    MMA_N: Int,
    num_m_mmas: Int,
    num_n_mmas: Int,
    use_score_mod: Bool,
](
    mask_warp_row_arg: UInt32,
    mask_warp_col_arg: UInt32,
    position: MHAPosition[BM, BN, depth, num_heads, group, decoding],
    lane: UInt32,
    max_seq_len: UInt32,
    scale_log2e: Scalar[accum_type],
    kv_tile_start_row: UInt32,
    mask: mask_t,
    mask_status: TileMaskStatus,
    score_mod: score_mod_t,
    p_reg_tile: LayoutTensor[
        accum_type,
        reg_tile_layout,
        MutableAnyOrigin,
        address_space = AddressSpace.LOCAL,
        element_layout=element_layout,
    ],
):
    alias num_groups_per_thread = min(2, ceildiv(group, 8)) if decoding else 2
    var batch_cache_valid_length: UInt32

    @parameter
    if decoding:
        if warp.broadcast((thread_idx.x - 128) // 32) > ((group - 1) // 16):
            return
        if lane >= 4 * group:
            return
        batch_cache_valid_length = position.num_keys - 1
    else:
        batch_cache_valid_length = 0

    alias p_frag_simdwidth = element_layout.size()
    # Vectorize by 2.
    var fragment_row: UInt32 = lane // 4
    var fragment_col: UInt32 = (lane * p_frag_simdwidth % MMA_N) % 8
    # Offset to current thread's fragment
    var mask_warp_row: UInt32 = mask_warp_row_arg + fragment_row
    var mask_warp_col: UInt32 = (
        mask_warp_col_arg + kv_tile_start_row + fragment_col
    )

    @parameter
    @always_inline
    fn _apply_mask_capture[masked: Bool]():
        @parameter
        for m_mma in range(num_m_mmas):

            @parameter
            for n_mma in range(num_n_mmas):
                # Coordinates in mask for current mma tile.
                mask_frag_row = mask_warp_row + m_mma * MMA_M
                mask_frag_col = mask_warp_col + n_mma * MMA_N

                @parameter
                for i in range(num_groups_per_thread):
                    var q_head_idx: UInt32 = position.head_idx

                    @parameter
                    if decoding:
                        group_idx = i * 8 + fragment_row
                        q_head_idx = group * q_head_idx + group_idx
                    # The row in score matrix of shape seq_len x num_keys.
                    # Mask col is score col since we don't partition in col.
                    var score_row: UInt32
                    var score_row_with_start_pos: UInt32

                    @parameter
                    if decoding:
                        score_row = batch_cache_valid_length
                        score_row_with_start_pos = score_row
                    else:
                        score_row = (
                            position.prompt_offset
                            + mask_frag_row
                            + i * MMA_M // 2
                        )
                        score_row_with_start_pos = (
                            score_row + position.start_pos
                        )

                    @parameter
                    for j in range(MMA_N // 8):
                        score_col = mask_frag_col + j * 8
                        alias idx = IntTuple(
                            IntTuple(i, m_mma), IntTuple(j, n_mma)
                        )
                        p = p_reg_tile._get[idx=idx]()

                        @parameter
                        if masked:
                            p = mask.mask(
                                IndexList[4, element_type = DType.uint32](
                                    Int(position.prompt_idx),
                                    Int(q_head_idx),
                                    Int(score_row_with_start_pos),
                                    Int(score_col),
                                ),
                                p * scale_log2e,
                            )
                        else:
                            p *= scale_log2e

                        @parameter
                        if use_score_mod:
                            p = (
                                score_mod.score_mod(
                                    IndexList[4, element_type = DType.uint32](
                                        Int(position.prompt_idx),
                                        Int(q_head_idx),
                                        Int(score_row_with_start_pos),
                                        Int(score_col),
                                    ),
                                    p,
                                    Int(max_seq_len),
                                )
                                * log2e
                            )
                        elif mask_t.apply_log2e_after_mask:
                            p *= log2e

                        var bound: IndexList[2, element_type = DType.uint32]

                        @parameter
                        if decoding:
                            bound = IndexList[2, element_type = DType.uint32](
                                Int(position.num_keys),
                                Int(
                                    min(
                                        BN + kv_tile_start_row,
                                        position.num_keys,
                                    )
                                ),
                            )
                            p = _kernel_mask(
                                IndexList[2, element_type = DType.uint32](
                                    Int(score_row), Int(score_col)
                                ),
                                bound,
                                p,
                            )
                        elif masked:
                            bound = IndexList[2, element_type = DType.uint32](
                                Int(position.seq_len),
                                Int(position.num_keys),
                            )
                            p = _kernel_mask(
                                IndexList[2, element_type = DType.uint32](
                                    Int(score_row), Int(score_col)
                                ),
                                bound,
                                p,
                            )
                        p_reg_tile._set[idx=idx](p)

    @parameter
    if decoding:
        _apply_mask_capture[True]()
    else:
        unswitch[_apply_mask_capture](
            (mask_status == TileMaskStatus.PARTIAL_MASK)
            # NOTE: mask_status should be either PARTIAL_MASK or NO_MASK at
            # this point.
            # In the NO_MASK case, we still need to mask out the scores for the
            # last tile, which goes beyond num_keys (for num_keys % 128 != 0).
            or (not decoding and (BN + kv_tile_start_row > position.num_keys))
        )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        config.num_threads[True]()
    )
)
fn _mha_sm90[
    q_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    output_type: DType,
    mask_t: MHAMask,
    score_mod_t: ScoreModTrait,
    scheduler_t: MHATileScheduler,
    config: MHAConfig,
    group: Int,
    use_score_mod: Bool,
    ragged: Bool,
    _is_cache_length_accurate: Bool,
    max_seq_len_t: OptionallyStaticInt,
    partition_t: MHAPartitionScheme,
](
    scheduler: scheduler_t,
    q_ptr_arg: UnsafePointer[Scalar[q_type]],
    k: k_t,
    v: v_t,
    output_ptr_arg: UnsafePointer[Scalar[output_type]],
    scale: Float32,
    batch_size: UInt32,
    max_seq_len: max_seq_len_t,  # sequence length after padding.
    num_keys_arg: UInt32,
    valid_length: NDBuffer[DType.uint32, 1, MutableAnyOrigin],
    kv_input_row_offsets: OptionalReg[
        NDBuffer[DType.uint32, 1, MutableAnyOrigin]
    ],
    partition: partition_t,
    mask: mask_t,
    score_mod: score_mod_t,
):
    """MHA for token gen where seqlen = 1 and num_keys >= 1.

    The general data layout and steps conform to flash attention. Two exceptions:

    1 Partition across B, H, and num_keys (TODO).  The last one is split-K and
      will need a separate reduction kernel at the end.

    2 First bmm becomes gemv and second bmm becomes gevm.
      TODO: use more optimized kernels for them

    """
    alias k_type = k_t.type
    alias v_type = v_t.type
    constrained[q_type == k_type and k_type == v_type]()
    alias decoding: Bool = _is_decoding[max_seq_len_t]()

    alias simd_size = simdwidthof[q_type]()

    alias num_warps_m = config.num_warps_m()
    alias num_consumer_threads = config.num_consumer_threads()
    alias BM = config.block_m()
    alias BN = config.block_n()
    alias num_heads = config.num_heads
    alias depth = config.depth
    # num_consumer_threads ignores the producers
    # actual number of threads is num_consumer_threads + 128
    alias num_consumer = num_consumer_threads // 128
    alias pipeline_stages = Int(config.num_pipeline_stages)
    var tid: UInt32 = thread_idx.x
    var warp_group_idx: UInt32 = warp.broadcast(tid // 128)

    constrained[
        num_warps_m == (num_consumer_threads // WARP_SIZE),
        "Number of warps doesn't match warp tile sizes.",
    ]()

    var warp_id: UInt32 = warp.broadcast((tid - 128) // WARP_SIZE)
    var lane: UInt32 = lane_id()

    # Coordinates of the current warp.
    var warp_y: UInt32 = warp_id  # // num_warps_n
    alias warp_x: UInt32 = 0  # warp_id % num_warps_n

    alias q_smem_layout = tile_layout_k_major[
        DType.bfloat16, BM, depth, swizzle_mode = TensorMapSwizzle.SWIZZLE_128B
    ]()
    alias k_smem_layout = tile_layout_k_major[
        DType.bfloat16, BN, depth, swizzle_mode = TensorMapSwizzle.SWIZZLE_128B
    ]()
    alias v_smem_layout = tile_layout_mn_major[
        DType.bfloat16, depth, BN, swizzle_mode = TensorMapSwizzle.SWIZZLE_128B
    ]()
    # for wgmma_0, we mutliply BM x depth @ depth x BN -> BM x BN
    # for wgmma_1, we multiply BM x BN @ BN x depth -> BM x depth
    # For wgmma_0, we iterate over (depth//BK) tiles of size BKxBN
    # For wgmma_1, we iterate over (BN//BK) tiles of size BKxdepth
    alias persistent = scheduler_t.may_advance

    # The entire query block (BM x depth) is tiled in shared memory.
    alias q_smem_size = config.q_smem_size(True, persistent)
    q_smem = external_memory[
        Scalar[q_type],
        address_space = AddressSpace.SHARED,
        alignment = alignof[SIMD[q_type, simd_size]](),
        name="mha_dynamic_shared_memory",
    ]()
    # We have depth//BK * (1 if persistent else 1) instances
    q_smem_iter = LayoutTensorIter[
        q_type,
        q_smem_layout,
        # Layout.row_major(BM, BK),
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment = q_smem.alignment,
    ](
        rebind[
            __type_of(
                LayoutTensorIter[
                    q_type,
                    q_smem_layout,
                    # Layout.row_major(BM, BK),
                    MutableAnyOrigin,
                    address_space = AddressSpace.SHARED,
                    alignment = q_smem.alignment,
                ]().ptr
            )
        ](q_smem),
        q_smem_size,
    )
    # We have `num_pipeline_stages` instances of each
    alias kv_smem_size = config.kv_smem_size(True)
    kv_smem = (q_smem + q_smem_size).bitcast[Scalar[k_type]]()

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

    alias accum_type = get_accum_type[q_type]()
    alias p_frag_size = MMA_M * MMA_N0 // WARP_SIZE
    alias o_frag_size = MMA_M * MMA_N1 // WARP_SIZE
    alias frag_simdwidth = 2

    alias a_frag_size = MMA_M * MMA_K // WARP_SIZE
    alias frag_ratio = p_frag_size // a_frag_size

    alias q_swizzle = TensorMapSwizzle.SWIZZLE_128B
    alias k_swizzle = TensorMapSwizzle.SWIZZLE_128B
    alias v_swizzle = TensorMapSwizzle.SWIZZLE_128B
    # the first mma is BMxdepth @ depthxBN
    alias wgmma_0 = TensorCoreAsync[
        accum_type,
        q_type,
        k_type,
        Index(4 * MMA_M, MMA_N0, 16),
        a_swizzle=q_swizzle,
        b_swizzle=k_swizzle,
        transpose_b=True,
    ]()
    # the second mma is BMxBN @ BNxdepth
    alias wgmma_1 = TensorCoreAsync[
        accum_type,
        v_type,
        v_type,
        Index(4 * MMA_M, MMA_N1, 16),
        a_swizzle = TensorMapSwizzle.SWIZZLE_NONE,
        b_swizzle=v_swizzle,
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
    alias accum_simd_width = simdwidthof[accum_type]()
    alias row_alignment = alignof[SIMD[accum_type, accum_simd_width]]()
    # Account for group query.
    alias kv_num_heads = num_heads // group

    # var lane_predicate = elect_one_sync() # not needed with async_copy

    alias mma_thread_layout = Layout.row_major(8, 4)

    produced_mbar_kv = (kv_smem + kv_smem_size).bitcast[SharedMemBarrier]()
    consumed_mbar_kv = produced_mbar_kv + pipeline_stages
    produced_mbar_q = consumed_mbar_kv + pipeline_stages
    consumed_mbar_q = produced_mbar_q + 2
    block_idx_ptr = (consumed_mbar_q + 2).bitcast[UInt32]()

    alias USE_TMA = False
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
            # until we can use TMA, we need 128 producers working on async copies
            produced_mbar_kv[i].init(128)
            consumed_mbar_kv[i].init(num_consumer_threads)

        @parameter
        if persistent:

            @parameter
            for i in range(2):
                produced_mbar_q[i].init(128)
                consumed_mbar_q[i].init(num_consumer_threads)

    alias position_t = MHAPosition[BM, BN, depth, num_heads, group, decoding]

    @parameter
    @always_inline
    fn k_tile(
        idx: UInt32,
        out k_smem: LayoutTensor[
            k_t.type,
            k_smem_layout,
            MutableAnyOrigin,
            address_space = AddressSpace.SHARED,
            layout_int_type = DType.int32,
            linear_idx_type = DType.int32,
        ],
    ):
        alias sz = BN * depth
        k_smem = __type_of(k_smem)(kv_smem + sz * idx)

    @parameter
    @always_inline
    fn v_tile(
        idx: UInt32,
        out v_smem: LayoutTensor[
            v_t.type,
            v_smem_layout,
            MutableAnyOrigin,
            address_space = AddressSpace.SHARED,
            layout_int_type = DType.int32,
            linear_idx_type = DType.int32,
        ],
    ):
        alias sz = BN * depth
        v_smem = __type_of(v_smem)(
            rebind[
                UnsafePointer[
                    Scalar[v_type], address_space = AddressSpace.SHARED
                ]
            ](kv_smem)
            + sz * idx
        )

    @parameter
    @always_inline
    fn get_position(seq_info: SeqInfo) -> position_t:
        return _get_position[config, group, ragged, _is_cache_length_accurate](
            seq_info,
            k,
            max_seq_len,
            num_keys_arg,
            valid_length,
            kv_input_row_offsets,
        )

    var position: position_t = get_position(initial_seq_info)

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
        write_pipeline_states = PipelineState[pipeline_stages]()

        @parameter
        if partition_t.do_partition:
            start, end = position.get_start_and_end_for_partitions[BN=BN](
                partition
            )
            if start >= end:
                return

        # note that Q does not wait or arrive...
        # it assumes you can use K's
        @parameter
        @always_inline("nodebug")
        fn produce_q(
            position: position_t,
            pipeline_idx: UInt32,
        ):
            # Query global memory iterator
            q_gmem_block = position.q_out_gmem_tensor(q_ptr_arg)
            q_smem_sub = q_smem_iter.next_unsafe(Int(pipeline_idx))[]

            cp_async_k_major(q_smem_sub, q_gmem_block)

        @parameter
        @always_inline("nodebug")
        fn produce_k[
            wait: Bool
        ](
            mut state: PipelineState[pipeline_stages],
            kv_tile_start_row: UInt32,
            position: position_t,
        ):
            var write_idx: UInt32 = state.index()
            var write_phase: UInt32 = state.phase()
            _produce[kv_num_heads, axis=1, wait=wait](
                write_idx,
                write_phase,
                kv_tile_start_row,
                position,
                consumed_mbar_kv,
                produced_mbar_kv,
                k,
                k_tile(write_idx),
            )
            state.step()

        @parameter
        @always_inline("nodebug")
        fn produce_v[
            wait: Bool
        ](
            mut state: PipelineState[pipeline_stages],
            kv_tile_start_row: UInt32,
            position: position_t,
        ):
            var write_idx: UInt32 = state.index()
            var write_phase: UInt32 = state.phase()
            _produce[kv_num_heads, axis=0, wait=wait](
                write_idx,
                write_phase,
                kv_tile_start_row,
                position,
                consumed_mbar_kv,
                produced_mbar_kv,
                v,
                v_tile(write_idx),
            )
            state.step()

        produce_q(position, q_pipeline_state.index())

        start, end = position.get_start_and_end_for_partitions[BN=BN](partition)
        var kv_tile_start_row: UInt32 = start

        while (
            position.mask_status(mask, kv_tile_start_row)
            == TileMaskStatus.FULL_MASK
        ):
            kv_tile_start_row += BN

        var kv_tile_start_row_prev: UInt32 = kv_tile_start_row
        var position_prev: position_t = position

        produce_k[False](write_pipeline_states, kv_tile_start_row, position)

        # wait to flip phase, but only bother after producing
        # there isn't any memory we can throttle
        # the order of the consumer's arrivals determines the
        # order of the producer's waits.
        # few_keys = num_keys <= BN

        # Process work with the tile size until there's not enough remaining work
        # to fit in a tile.
        # Production order:
        # Preheader: Q0, K0
        # Body: Q1, K1, V0, Q2, K2, V1, ..., Q{-1}, K{-1}, V{-2}
        # Exit: V{-1}
        while True:
            # this loops over num_keys
            kv_tile_start_row += BN
            if kv_tile_start_row >= end:

                @parameter
                if persistent:
                    kv_tile_start_row = 0
                    var q_idx_old: UInt32 = q_pipeline_state.index()
                    var q_phase_old: UInt32 = q_pipeline_state.phase()
                    q_pipeline_state.step()
                    consumed_mbar_q[q_idx_old].wait(q_phase_old)
                    # we must wait before advancing, as this mbar
                    # is for both `q_smem` and `sidx_ptr`
                    var q_idx: UInt32 = q_pipeline_state.index()
                    docontinue = advance[True](q_idx_old)
                    # commit writes to `q_smem` and to `sidx_ptr`
                    async_copy_arrive(produced_mbar_q + q_idx_old)
                    _ = produced_mbar_q[q_idx_old].arrive()
                    if not docontinue:
                        break
                    position = get_position(docontinue.value())
                    # merge with following produce_k
                    produce_q(position, q_idx)
                    start, new_end = position.get_start_and_end_for_partitions[
                        BN=BN
                    ](partition)
                    kv_tile_start_row = start
                    end = new_end
                else:
                    break

            if (
                position.mask_status(mask, kv_tile_start_row)
                == TileMaskStatus.FULL_MASK
            ):
                continue
            # new pipeline states
            produce_k[True](
                write_pipeline_states,
                kv_tile_start_row,
                position,
            )
            produce_v[True](
                write_pipeline_states,
                kv_tile_start_row_prev,
                position_prev,
            )
            kv_tile_start_row_prev = kv_tile_start_row
            position_prev = position

        produce_v[True](
            write_pipeline_states,
            kv_tile_start_row_prev,
            position_prev,
        )

    else:
        warpgroup_reg_alloc[num_consumer_regs]()

        # TODO: skip this by not waiting on the first set
        @parameter
        for i in range(pipeline_stages):
            _ = consumed_mbar_kv[i].arrive()

        @parameter
        if persistent:
            _ = consumed_mbar_q[0].arrive()
        var local_warp_group_idx: UInt32 = warp_group_idx - 1

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
            v_type,
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
        mask_warp_col_base = warp_x * BN
        var scale_log2e: Scalar[accum_type] = (
            scale.cast[accum_type]() if use_score_mod
            or mask_t.apply_log2e_after_mask else scale.cast[accum_type]()
            * log2e
        )
        read_pipeline_states = PipelineState[pipeline_stages]()

        @parameter
        @always_inline
        fn q_mul_k(read_idx: UInt32, read_phase: UInt32, q_idx: UInt32):
            k_smem_sub = k_tile(read_idx)
            q_smem_sub = q_smem_iter.next_unsafe(Int(q_idx))[]
            produced_mbar_kv[read_idx].wait(read_phase)
            wgmma_0.arrive()

            wgmma_0.wgmma[num_consumer, scale_c=0](
                q_smem_sub,
                k_smem_sub,
                p_reg_tile,
                Int(local_warp_group_idx),
            )
            wgmma_0.commit_group()

        @parameter
        @always_inline
        fn p_mul_v(read_idx: UInt32, read_phase: UInt32):
            v_smem_sub = v_tile(read_idx)
            produced_mbar_kv[read_idx].wait(read_phase)
            wgmma_1.arrive()

            wgmma_1.wgmma(
                p_frag,
                v_smem_sub,
                output_reg_tile,
            )
            wgmma_1.commit_group()

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
            position: position_t,
            mask_status: TileMaskStatus,
            mask_warp_col: UInt32,
            kv_tile_start_row: UInt32,
        ):
            var max_len: UInt32 = (
                num_keys_arg if decoding else max_seq_len.as_uint32()
            )
            _apply_mask[MMA_M, MMA_N0, num_m_mmas, num_n_mmas, use_score_mod,](
                mask_warp_row,
                mask_warp_col,
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
                c = correction._get[row, size = element_layout.size()]()

                @parameter
                for col in range(num_cols_output):
                    vout._set[row, col](vout._get[row, col]() * c)

        @always_inline
        fn elementwise_reciprocal(
            old_rowsum: __type_of(rowsum), new_rowsum: __type_of(rowsum)
        ):
            # new_rowsum, old_rowsum = 1/old_rowsum, new_rowsum
            @parameter
            for row in range(num_rows_per_warp):
                old = old_rowsum._get[row]()
                new = new_rowsum._get[row]()
                new_rowsum._set[row](recip(old)[0])
                old_rowsum._set[row](new)

        @parameter
        @always_inline
        fn write_output(
            position: position_t,
            q_idx: UInt32,
            rowsum_inv: __type_of(rowsum),
        ):
            # Apply softmax denumerator.
            vout = vectorize_o_reg_tile()

            @parameter
            for row in range(num_rows_per_warp):
                rs_inv = vout.element_type(rowsum_inv._get[row]()[0])

                @parameter
                for col in range(num_cols_output):
                    vout._set[row, col](vout._get[row, col]() * rs_inv)

            var output_ptr: UnsafePointer[Scalar[output_type]] = output_ptr_arg

            @parameter
            if decoding and partition_t.do_partition:
                output_ptr = output_ptr.offset(
                    depth * num_heads * batch_size * position.prompt_offset
                )
            output_gmem_tile = position.q_out_gmem_tensor(output_ptr)

            # Write to global memory.
            constrained[
                output_type.is_half_float(), "we don't support Float32 output"
            ]()
            constrained[sizeof[q_type]() == sizeof[output_type]()]()
            alias swizzle = make_swizzle[
                num_rows = MMA_M // 2, row_size=BN, access_size=8
            ]()
            # Reuse a_smem for c tile in smem
            alias q_tile_size: UInt32 = q_smem_size // 2
            accum_smem_tile = LayoutTensor[
                output_type,
                Layout.row_major(BM, depth),
                address_space = AddressSpace.SHARED,
            ]((q_smem + q_idx * q_tile_size).bitcast[Scalar[output_type]]())
            accum_smem_warp_tile = accum_smem_tile.tile[WM, BN](
                Int(warp_y), Int(warp_x)
            )

            # ensure all threads have finished reading `q_smem`
            named_barrier[num_consumer_threads]()
            copy[thread_layout=mma_thread_layout, swizzle=swizzle](
                accum_smem_warp_tile.vectorize[1, 2](),
                output_reg_tile.vectorize[1, 2]().transpose(),
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

        start, end = position.get_start_and_end_for_partitions[BN=BN](partition)
        var execute: Bool = Bool(
            start < end
        ) if partition_t.do_partition else True
        if execute:
            var kv_tile_start_row: UInt32 = start
            var mask_status: TileMaskStatus
            while True:
                mask_status = position.mask_status(mask, kv_tile_start_row)
                if mask_status != TileMaskStatus.FULL_MASK:
                    break
                kv_tile_start_row += BN

            # q_mul_k must wait on fetching q and k
            # therefore, we find `kv_tile_start_row` first.
            var read_idx: UInt32 = read_pipeline_states.index()
            q_mul_k(
                read_idx, read_pipeline_states.phase(), q_pipeline_state.index()
            )
            read_pipeline_states.step()
            wait_for_q_mul_k[0](read_idx)

            apply_mask(
                position, mask_status, mask_warp_col_base, kv_tile_start_row
            )
            rowmax.copy_from(
                _rowmax_online_softmax[
                    # threads layout by warp
                    1,
                    mma_thread_layout,
                    use_exp2=True,
                ](vectorize_p_reg_tile(), rowmax, init_rowmax=True)
            )
            rowsum.copy_from(_rowsum[mma_thread_layout](vectorize_p_reg_tile()))

            var position_prev: position_t = position
            var q_idx_old: UInt32 = q_pipeline_state.index()
            var q_phase_old: UInt32 = q_pipeline_state.phase()
            # Consumption order:
            # Preheader: Q0, K0
            # Body: Q1, K1, V0, Q2, K2, V1, ..., Q{-1}, K{-1}, V{-2}
            # Exit: V{-1}
            kv_tile_start_row += BN
            while True:
                while kv_tile_start_row < end:
                    # this loops over num_keys
                    mask_status = position.mask_status(mask, kv_tile_start_row)
                    if mask_status == TileMaskStatus.FULL_MASK:
                        kv_tile_start_row += BN
                        continue
                    # new pipeline states
                    var read_idx_q: UInt32 = read_pipeline_states.index()
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

                    apply_mask(
                        position,
                        mask_status,
                        mask_warp_col_base,
                        kv_tile_start_row,
                    )
                    new_q = persistent and q_idx_old != q_pipeline_state.index()
                    score_frag_rowmax = _rowmax_online_softmax[
                        # threads layout by warp
                        1,
                        mma_thread_layout,
                        use_exp2=True,
                    ](vectorize_p_reg_tile(), rowmax, new_q)
                    if new_q:

                        @parameter
                        if decoding and partition_t.do_partition:
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
                                    Scalar[partition_t.accum_dtype]
                                ](rowsum[0])
                                qk_max_ptr[q_head_idx] = rebind[
                                    Scalar[partition_t.accum_dtype]
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
                        write_output(
                            position_prev, q_idx_old, score_frag_rowsum
                        )
                        var q_idx_new: UInt32 = q_pipeline_state.index()

                        _ = consumed_mbar_q[q_idx_new].arrive()
                        _ = output_reg_tile.vectorize[accum_simd_width]().fill(
                            0
                        )
                        position_prev = position
                        q_idx_old = q_idx_new
                        q_phase_old = q_pipeline_state.phase()
                    else:
                        score_frag_rowsum = rebind[__type_of(rowsum)](
                            _rowsum[mma_thread_layout](vectorize_p_reg_tile())
                        )
                        _online_softmax_correction[use_exp2=True](
                            rowmax, score_frag_rowmax
                        )
                        # rowmax now holds score_frag_rowmax
                        # score_frag_rowmax now holds the correction

                        @parameter
                        for i in range(num_rows_per_warp):
                            rowsum._set[i](
                                rowsum._get[i]() * score_frag_rowmax._get[i]()
                                + rebind[Scalar[accum_type]](
                                    score_frag_rowsum._get[i]()
                                )
                            )

                        wait_for_p_mul_v(read_idx_v)  # can rw output and pfrag
                        scale_output(score_frag_rowmax)  # scale output
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
            p_mul_v(read_pipeline_states.index(), read_pipeline_states.phase())

            @parameter
            if decoding and partition_t.do_partition:
                if thread_idx.x % 4 == 0 and thread_idx.x < 4 * group + 128:
                    exp_sum_ptr, qk_max_ptr = position.exp_sum_qk_max_ptr(
                        partition, batch_size
                    )
                    var q_head_idx = position.head_idx * group + lane // 4
                    exp_sum_ptr[q_head_idx] = rebind[
                        Scalar[partition_t.accum_dtype]
                    ](rowsum[0])
                    qk_max_ptr[q_head_idx] = rebind[
                        Scalar[partition_t.accum_dtype]
                    ](rowmax[0])

            @parameter
            for row in range(num_rows_per_warp):
                rowsum._set[row](recip(rowsum._get[row]())[0])
            wgmma_1.wait_group()
        else:

            @parameter
            if decoding and partition_t.do_partition:
                if thread_idx.x % 4 == 0 and thread_idx.x < 4 * group + 128:
                    exp_sum_ptr, qk_max_ptr = position.exp_sum_qk_max_ptr(
                        partition, batch_size
                    )
                    var q_head_idx = position.head_idx * group + lane // 4
                    exp_sum_ptr[q_head_idx] = Scalar[partition_t.accum_dtype](0)
                    qk_max_ptr[q_head_idx] = min_or_neg_inf[
                        partition_t.accum_dtype
                    ]()

        write_output(position, q_pipeline_state.index(), rowsum)
        # don't arrive


# TODO: Remove this when we're no longer using NDBuffers.
@always_inline
fn valid_length_managed_tensor_slice_to_ndbuffer(
    tensor: ManagedTensorSlice[dtype = DType.uint32, rank=1]
) -> NDBuffer[DType.uint32, 1, MutableAnyOrigin]:
    var ptr = tensor._ptr.address_space_cast[AddressSpace.GENERIC]()
    return NDBuffer[DType.uint32, 1, MutableAnyOrigin](
        ptr, tensor.shape(), tensor._runtime_strides
    )
