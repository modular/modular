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

from std.collections import OptionalReg

from std.math import align_up, ceildiv
from std.math.uutils import umod
from std.memory import stack_allocation

from std.atomic import Atomic
from std.sys.info import simd_width_of

import std.gpu.primitives.warp as warp
from std.bit import pop_count, log2_floor
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    barrier,
    block_idx,
    warp_id,
    lane_id,
    thread_idx,
)
from std.gpu.primitives.grid_controls import (
    PDL,
    PDLLevel,
    pdl_launch_attributes,
)
from std.gpu.host.info import is_gpu
from layout import (
    Coord,
    Idx,
    TensorLayout,
    TileTensor,
    row_major,
    stack_allocation as tensor_alloc,
)
from std.runtime.asyncrt import DeviceContextPtr
from std.runtime.tracing import Trace, TraceLevel

from std.utils.index import IndexList, StaticTuple
from std.builtin.dtype import _uint_type_of_width

from nn.topk import TopK_2


@always_inline
def calculate_warp_offset[
    MaskType: DType
](state: Bool) -> Tuple[UInt64, UInt64]:
    # sets bits to 1 for all threads that voted true
    var mask = UInt64(warp.vote[MaskType](state))

    # counts the number of bits that are set to 1
    var writes = pop_count(mask)

    # masks out all bits that are set to 1 for higher thread IDs
    var preceding_mask = mask & ((UInt64(1) << UInt64(thread_idx.x)) - 1)

    # counts the number of bits that are set to 1 in the preceding mask
    var offset = pop_count(preceding_mask)

    return writes, offset


struct _BucketGroupParams[num_threads: Int, input_type: DType]:
    comptime MaskType = _uint_type_of_width[Self.num_threads]()
    comptime width = simd_width_of[Self.input_type]()

    var expert: Int
    var reads_per_iteration: Int
    var topk_ids_length: Int
    var topk_ids_length_rounded: Int
    var start_idx: Int
    var remainder_start_idx: Int

    def __init__(out self, top_k_length: Int):
        self.expert = block_idx.x
        self.reads_per_iteration = Self.num_threads * Self.width
        self.topk_ids_length = top_k_length
        self.topk_ids_length_rounded = align_up(
            self.topk_ids_length, self.reads_per_iteration
        )
        self.start_idx = thread_idx.x * Self.width
        self.remainder_start_idx = (
            self.topk_ids_length // Self.width
        ) * Self.width + thread_idx.x


@always_inline
def _count_expert_tokens[
    num_threads: Int,
    input_type: DType,
    //,
    expected_count: Int,
](
    topk_ids: TileTensor[input_type, ...],
    smem: TileTensor[mut=True, DType.uint32, ...],
    bg_params: _BucketGroupParams[num_threads, input_type],
) -> UInt64:
    comptime assert topk_ids.flat_rank == 2
    comptime assert smem.flat_rank == 2
    comptime assert topk_ids.flat_rank >= 2

    comptime width = bg_params.width
    comptime MaskType = bg_params.MaskType

    var total_writes: UInt64 = 0

    # Vectorized scan of expert IDs from global memory
    # Each thread loads 'width' expert IDs and checks which match this block's expert
    for idx in range(
        bg_params.start_idx,
        bg_params.topk_ids_length_rounded,
        bg_params.reads_per_iteration,
    ):
        var g_vector: SIMD[input_type, width]

        if idx + width <= bg_params.topk_ids_length:
            g_vector = topk_ids.load[width=width](Coord(Idx(0), Idx(idx)))
        else:
            g_vector = SIMD[input_type, width](bg_params.expert + 1)

        # Use warp-level voting to efficiently count matching tokens
        # All threads in the warp vote, and we count how many threads
        # before us also voted true to determine our write offset
        comptime for i in range(width):
            var expert_id = g_vector[i]
            var state = expert_id == Scalar[input_type](bg_params.expert)

            var offset = total_writes

            # if state is true this thread will write to smem
            # but we need to know how many threads will write to smem before us
            # to get the correct offset. So all threads vote and we tally the votes
            # before us

            var warp_writes, preceding_thread_writes = calculate_warp_offset[
                MaskType
            ](state)
            total_writes += warp_writes
            offset += preceding_thread_writes

            # If this token matches, store its index in shared memory
            if state and offset < UInt64(expected_count):
                smem[0, offset] = UInt32(idx + i)

    var expert_id = (
        topk_ids[
            0, bg_params.remainder_start_idx
        ] if bg_params.remainder_start_idx
        < bg_params.topk_ids_length else Scalar[input_type](bg_params.expert)
        + 1
    )
    var state = expert_id == Scalar[input_type](bg_params.expert)

    # Use same warp voting technique for remainder elements
    var warp_writes, preceding_thread_writes = calculate_warp_offset[MaskType](
        state
    )
    var offset = total_writes + preceding_thread_writes
    total_writes += warp_writes

    if state and offset < UInt64(expected_count):
        smem[0, offset] = UInt32(bg_params.remainder_start_idx)

    return total_writes


@always_inline
def _get_index_and_offset(
    lock: TileTensor[mut=True, DType.uint64, ...],
    total_writes: UInt32,
    aligned_total_writes: UInt32,
) -> Tuple[UInt32, UInt32, UInt32]:
    var expert_idx_and_offsets: UInt64 = 0

    # in order to write back to gmem we need to know the current available offset
    # so we use atomics to get the next available offset

    if thread_idx.x == 0:
        # Pack expert index (12 bits), current total writes (26 bits), and
        # aligned total writes (26 bits) into single atomic update
        # Upper 12 bits: expert counter (which expert slot to use)
        # Middle 26 bits: current total writes
        # Lower 26 bits: aligned total writes
        expert_idx_and_offsets = Atomic.fetch_add(
            lock.ptr,
            (UInt64(1) << 52)
            | (UInt64(total_writes) << 26)
            | UInt64(aligned_total_writes),
        )

    # Broadcast the atomic result to all threads in the warp
    expert_idx_and_offsets = warp.broadcast(expert_idx_and_offsets)
    var expert_idx = UInt32(expert_idx_and_offsets >> 52)
    var base_g_offset = UInt32(expert_idx_and_offsets >> 26) & 0x03FFFFFF
    var aligned_g_offset = UInt32(expert_idx_and_offsets) & 0x03FFFFFF

    return expert_idx, base_g_offset, aligned_g_offset


@always_inline
def _copy_tokens_smem_to_gmem[
    num_threads: Int,
    input_type: DType,
    //,
    expected_count: Int,
](
    token_expert_order: TileTensor[mut=True, DType.uint32, ...],
    restore_token_order: TileTensor[mut=True, DType.uint32, ...],
    smem: TileTensor[DType.uint32, ...],
    g_offset: UInt32,
    total_writes: UInt64,
    bg_params: _BucketGroupParams[num_threads, input_type],
):
    comptime assert smem.flat_rank == 2
    comptime assert token_expert_order.flat_rank == 1
    comptime assert restore_token_order.flat_rank == 1
    comptime assert smem.flat_rank >= 2
    comptime assert token_expert_order.flat_rank >= 1

    var g_offset_copy = g_offset
    comptime width = bg_params.width

    var total_reads_rounded = align_up(
        Int(total_writes), bg_params.reads_per_iteration
    )

    var total_smem_reads = align_up(
        expected_count, bg_params.reads_per_iteration
    )
    var rounded_smem_reads = min(total_smem_reads, total_reads_rounded)
    var smem_writes = min(UInt64(expected_count), total_writes)

    for smem_idx in range(
        bg_params.start_idx, rounded_smem_reads, bg_params.reads_per_iteration
    ):
        if smem_idx + width <= Int(smem_writes):
            var source_vector = smem.load[width=width](
                Coord(Idx(0), Idx(smem_idx))
            )

            comptime for i in range(width):
                token_expert_order[
                    g_offset_copy + UInt32(smem_idx) + UInt32(i)
                ] = source_vector[i]
                restore_token_order[Int(source_vector[i])] = (
                    g_offset_copy + UInt32(smem_idx) + UInt32(i)
                )

    var start_idx: UInt64 = (smem_writes // UInt64(width)) * UInt64(width)

    g_offset_copy += UInt32(start_idx)

    if UInt64(thread_idx.x) < smem_writes - start_idx:
        var smem_val = smem[0, start_idx + UInt64(thread_idx.x)]
        token_expert_order.store(
            Coord(Idx(Int(g_offset_copy + UInt32(thread_idx.x)))),
            smem_val,
        )

        restore_token_order[Int(smem_val)] = g_offset_copy + UInt32(
            thread_idx.x
        )


@always_inline
def _copy_tokens_to_gmem[
    num_threads: Int,
    input_type: DType,
    //,
    expected_count: Int,
](
    topk_ids: TileTensor[input_type, ...],
    smem: TileTensor[DType.uint32, ...],
    token_expert_order: TileTensor[mut=True, DType.uint32, ...],
    restore_token_order: TileTensor[mut=True, DType.uint32, ...],
    total_writes: UInt64,
    g_offset: UInt32,
    bg_params: _BucketGroupParams[num_threads, input_type],
):
    comptime assert topk_ids.flat_rank == 2
    comptime assert token_expert_order.flat_rank == 1
    comptime assert restore_token_order.flat_rank == 1
    comptime assert topk_ids.flat_rank >= 2

    comptime width = bg_params.width
    comptime MaskType = bg_params.MaskType

    var g_offset_copy = g_offset

    # keep track of how many tokens we have come across
    var tokens_seen: UInt64 = 0

    # load all tokens in vectorized manner from global memory into registers
    for idx in range(
        bg_params.start_idx,
        bg_params.topk_ids_length_rounded,
        bg_params.reads_per_iteration,
    ):
        var g_vector: SIMD[input_type, width]

        if idx + width <= bg_params.topk_ids_length:
            g_vector = topk_ids.load[width=width](Coord(Idx(0), Idx(idx)))
        else:
            g_vector = SIMD[input_type, width](bg_params.expert + 1)

        comptime for i in range(width):
            var expert_id = g_vector[i]
            var state = expert_id == Scalar[input_type](bg_params.expert)

            var warp_writes, preceding_thread_writes = calculate_warp_offset[
                MaskType
            ](state)
            var thr_tokens_seen = (
                tokens_seen
                + preceding_thread_writes
                + UInt64(1 if state else 0)
            )

            # we have already writeen expected_count tokens to global memory since they were in shared memory.
            # so we only need to write the remaining tokens to global memory.
            if thr_tokens_seen >= UInt64(expected_count) and state:
                token_expert_order[
                    g_offset_copy + UInt32(preceding_thread_writes)
                ] = UInt32(idx + i)
                restore_token_order[idx + i] = g_offset_copy + UInt32(
                    preceding_thread_writes
                )

            tokens_seen += warp_writes
            g_offset_copy += UInt32(warp_writes)

    # Handle remainder elements that couldn't be vectorized
    var expert_id = (
        topk_ids[
            0, bg_params.remainder_start_idx
        ] if bg_params.remainder_start_idx
        < bg_params.topk_ids_length else Scalar[input_type](bg_params.expert)
        + 1
    )
    var state = expert_id == Scalar[input_type](bg_params.expert)

    # Use same warp voting technique for remainder elements
    var _, preceding_thread_writes = calculate_warp_offset[MaskType](state)
    var temp_current_writes = (
        tokens_seen + preceding_thread_writes + UInt64(1 if state else 0)
    )

    if temp_current_writes >= UInt64(expected_count) and state:
        token_expert_order[
            g_offset_copy + UInt32(preceding_thread_writes)
        ] = UInt32(bg_params.remainder_start_idx)
        restore_token_order[
            bg_params.remainder_start_idx
        ] = g_offset_copy + UInt32(preceding_thread_writes)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
@__name(
    t"moe_create_indices_bucket_group_{input_type}_t{num_threads}", mangle=True
)
def moe_create_indices_bucket_group_kernel[
    input_type: DType,
    TokenExpertOrderLayoutType: TensorLayout,
    LockLayoutType: TensorLayout,
    ExpertStartIndicesLayoutType: TensorLayout,
    RestoreTokenOrderLayoutType: TensorLayout,
    ExpertIdsLayoutType: TensorLayout,
    ExpertUsageStatsLayoutType: TensorLayout,
    TopkIdsLayoutType: TensorLayout,
    num_threads: Int = WARP_SIZE,
    expected_count: Int = 8192,
    _scale_alignment: UInt32 = 128,
](
    token_expert_order: TileTensor[
        mut=True, DType.uint32, TokenExpertOrderLayoutType, MutAnyOrigin
    ],
    lock: TileTensor[DType.uint64, LockLayoutType, MutAnyOrigin],
    expert_start_indices: TileTensor[
        mut=True, DType.uint32, ExpertStartIndicesLayoutType, MutAnyOrigin
    ],
    restore_token_order: TileTensor[
        mut=True, DType.uint32, RestoreTokenOrderLayoutType, MutAnyOrigin
    ],
    expert_ids: TileTensor[
        mut=True, DType.int32, ExpertIdsLayoutType, MutAnyOrigin
    ],
    expert_usage_stats: TileTensor[
        mut=True, DType.uint32, ExpertUsageStatsLayoutType, MutAnyOrigin
    ],
    topk_ids: TileTensor[input_type, TopkIdsLayoutType, MutAnyOrigin],
    scales_offset_p: Optional[UnsafePointer[UInt32, MutAnyOrigin]],
):
    """Create indices for MoE routing using bucket sort algorithm.

    The main goal of this kernel is to group tokens that use the same expert together.
    This allows for efficient batching when used by other kernels such as grouped matmul.

    This is a GPU-optimized bucket sort implementation that uses:
    - Warp-level voting to count matching tokens
    - Shared memory for temporary storage
    - Atomic operations for thread-safe global memory updates

    topk_ids: a 1D tensor of expert ids, the index of each expert_id corresponds to a token.
    For example if topk_ids is [1, 0, 1, 3, 4, 2], then the corresponding tokens are [0, 1, 2, 3, 4, 5]

    token_expert_order: a 1D tensor of tokens grouped together by expert id.
    Using the previous topk_ids, the token expert order could be [0, 2, 1, 3, 4, 5]

    expert_ids: a 1D tensor of all the experts that are being used. Using the previous topk_ids the
    our expert_ids would be [1, 0, 3, 4, 2]

    expert_start_indices: tells us where each expert starts and end in the token_expert_order. Based on the
    order of our expert_ids our expert_start_indices would be [0, 2, 3, 4, 5, 6]. So if you wanted to see where
    expert 1 starts and ends you would get the index 'i' of expert 1 in expert_ids and would query expert_start_indices[i]
    and query expert_start_indices[i + 1] which is 0 and 2 respectively.

    lock: a 1D tensor that holds a single scalar value, this single integer will be used to atomically
    synchronize the writes back to global memory. It will do this by storing how many blocks have finished
    writing and the current global memory offset.

    expert_usage_stats: contains two values, the maximum number of tokens assigned to any expert and the
    number of active experts. For our example the stats would be [2, 5]

    restore_token_order: a 1D tensor where each index represents a corresponding token and holds the new index of the token
    in the token_expert_order tensor. For our example the restore_token_order would be [0, 2, 1, 3, 4, 5]
    """

    comptime assert token_expert_order.flat_rank == 1
    comptime assert lock.flat_rank == 1
    comptime assert expert_start_indices.flat_rank == 1
    comptime assert restore_token_order.flat_rank == 1
    comptime assert expert_ids.flat_rank == 1
    comptime assert expert_usage_stats.flat_rank == 1
    comptime assert topk_ids.flat_rank == 2

    comptime assert num_threads in (
        32,
        64,
    ), "Only support 32 or 64 threads per warp"

    comptime BucketParamsType = _BucketGroupParams[num_threads, input_type]

    # Allocate shared memory for temporary storage of matching token indices
    # alignment=128,
    var smem = tensor_alloc[DType.uint32, address_space=AddressSpace.SHARED](
        row_major[1, expected_count]()
    )

    comptime assert (
        expected_count % BucketParamsType.width == 0
    ), "Expected count must be a multiple of the simd width"

    var bucket_group_params = BucketParamsType(Int(topk_ids.dim(1)))

    # count tokens per expert and store as many we can in shared memory
    var total_writes = _count_expert_tokens[expected_count](
        topk_ids, smem, bucket_group_params
    )

    var aligned_total_writes = align_up(UInt32(total_writes), _scale_alignment)

    var expert_idx, g_offset, aligned_g_offset = _get_index_and_offset(
        lock, UInt32(total_writes), aligned_total_writes
    )

    if scales_offset_p:
        var _ptr = scales_offset_p.value()
        _ptr[expert_idx] = (
            aligned_g_offset // _scale_alignment - g_offset // _scale_alignment
        )

    # Record which expert is active at this index
    # this signals this expert is being used
    expert_ids[expert_idx] = Int32(bucket_group_params.expert)

    # Store the ending index for this expert (start of next expert)
    # NOTE: expert_start_indices must be zero-initialized for this to work correctly
    expert_start_indices[expert_idx + 1] = g_offset + UInt32(total_writes)

    # First expert always starts at index 0
    if expert_idx == 0:
        expert_start_indices[expert_idx] = 0

    if total_writes > 0:
        # Copy all tokens in shared memory back to global memory
        _copy_tokens_smem_to_gmem[expected_count](
            token_expert_order,
            restore_token_order,
            smem,
            g_offset,
            total_writes,
            bucket_group_params,
        )

        # write the rest of the tokens not in shared memory into global memory
        if total_writes > UInt64(expected_count):
            _copy_tokens_to_gmem[expected_count](
                topk_ids,
                smem,
                token_expert_order,
                restore_token_order,
                total_writes,
                g_offset,
                bucket_group_params,
            )

    # update expert_usage_stats.
    if thread_idx.x == 0:
        _ = Atomic.fetch_add(expert_usage_stats.ptr + 1, 1)

        # NOTE: must be zero initialized otherwise atomic max will not work
        _ = Atomic.max(expert_usage_stats.ptr, UInt32(total_writes))


@always_inline
def moe_create_indices[
    input_type: DType,
    //,
    target: StaticString,
    expected_count: Int = 8192,
](
    token_expert_order: TileTensor[mut=True, DType.uint32, ...],
    expert_start_indices: TileTensor[mut=True, DType.uint32, ...],
    restore_token_order: TileTensor[mut=True, DType.uint32, ...],
    expert_ids: TileTensor[mut=True, DType.int32, ...],
    expert_usage_stats: TileTensor[mut=True, DType.uint32, ...],
    topk_ids: TileTensor[input_type, ...],
    context: DeviceContextPtr,
    scales_offset_p: Optional[UnsafePointer[UInt32, MutAnyOrigin]] = None,
) raises:
    comptime assert is_gpu[
        target
    ](), "Creating MoE indices is only supported on GPU"

    var cuda_ctx = context.get_device_context()

    with Trace[TraceLevel.OP, target=target](
        "mo.moe.create_indices", task_id=Int(context.get_device_context().id())
    ):
        var lock_buffer = cuda_ctx.enqueue_create_buffer[DType.uint64](1)

        def fill_zero_kernel(
            lock_ptr: UnsafePointer[UInt64, MutAnyOrigin],
            expert_usage_stats_ptr: UnsafePointer[UInt32, MutAnyOrigin],
        ):
            lock_ptr.store(0)
            expert_usage_stats_ptr.store(0)
            expert_usage_stats_ptr.store(1, 0)

        cuda_ctx.enqueue_function[fill_zero_kernel](
            lock_buffer,
            expert_usage_stats.ptr,
            grid_dim=(1,),
            block_dim=(1,),
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )

        var lock = TileTensor(lock_buffer, row_major[1]())

        var topk_2D = TileTensor(
            topk_ids.ptr,
            row_major(Coord(Idx(1), Idx(Int(topk_ids.dim(0))))),
        )

        var num_experts = expert_ids.dim(0)

        comptime kernel = moe_create_indices_bucket_group_kernel[
            input_type,
            token_expert_order.LayoutType,
            lock.LayoutType,
            expert_start_indices.LayoutType,
            restore_token_order.LayoutType,
            expert_ids.LayoutType,
            expert_usage_stats.LayoutType,
            topk_2D.LayoutType,
            expected_count=expected_count,
        ]

        cuda_ctx.enqueue_function[kernel](
            token_expert_order,
            lock,
            expert_start_indices,
            restore_token_order,
            expert_ids,
            expert_usage_stats,
            topk_2D,
            scales_offset_p,
            grid_dim=(num_experts),
            block_dim=(WARP_SIZE),
        )


# Function to perform warp-level sorting
@always_inline
@parameter
def _warp_bitonic_sort[
    T: DType,
    num_lanes: Int = WARP_SIZE,
    descending: Bool = True,
](_val: TopK_2[T]) -> TopK_2[T]:
    """
    Performs warp-level bitonic sort to sort TopK_2 elements.

    Parameters:
        T: DType - Data type of the values being compared.
        num_lanes: Int - Number of lanes that participate in the reduction.
        descending: Bool - Whether to sort in descending order.

    Arguments:
        _val: TopK_2[T] - TopK_2 value from each thread to be sorted.

    Returns:
        TopK_2[T] - Sorted TopK_2 value across the warp.
    """

    comptime assert num_lanes.is_power_of_two(), "num_lanes must be power of 2"

    @always_inline
    def bitonic_sort_step(
        v: TopK_2[T],
        step: UInt32,
        stage: UInt32,
        i: UInt32,
    ) -> TopK_2[T]:
        var partner = TopK_2[T](
            u=warp.shuffle_xor(v.u, step),  # u is the value
            p=Int(warp.shuffle_xor(Int32(v.p), step)),  # p is the index
        )

        var cmp_val = (v.u < partner.u) ^ descending
        if v.u == partner.u:
            cmp_val = v.p > partner.p

        var merge_direction = pop_count(i & (stage | step)) == 1

        if cmp_val == merge_direction:
            return partner
        else:
            return v

    var val = _val
    # Use modulo so merge direction is consistent across all lane groups
    var i = UInt32(umod(lane_id(), num_lanes))

    comptime for stage_i in range(1, log2_floor(num_lanes) + 1):
        var stage = 1 << stage_i

        comptime for step_i in reversed(range(stage_i)):
            var step = 1 << step_i
            val = bitonic_sort_step(val, UInt32(step), UInt32(stage), i)

    return val


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
@__name(
    t"group_limited_router_{scores_type}_{bias_type}_t{num_threads}",
    mangle=True,
)
def group_limited_router_kernel[
    scores_type: DType,
    bias_type: DType,
    ExpertIndicesLayoutType: TensorLayout,
    ExpertWeightsLayoutType: TensorLayout,
    ExpertScoresLayoutType: TensorLayout,
    ExpertBiasLayoutType: TensorLayout,
    n_routed_experts: Int,
    n_experts_per_tok: Int,
    n_groups: Int,
    topk_group: Int,
    norm_weights: Bool,
    num_threads: Int,
    scores_input_fn: OptionalReg[
        def[width: Int](IndexList[2]) capturing -> SIMD[scores_type, width]
    ] = None,
](
    expert_indices: TileTensor[
        mut=True, DType.int32, ExpertIndicesLayoutType, MutAnyOrigin
    ],
    expert_weights: TileTensor[
        mut=True, scores_type, ExpertWeightsLayoutType, MutAnyOrigin
    ],
    expert_scores: TileTensor[
        scores_type, ExpertScoresLayoutType, ImmutAnyOrigin
    ],
    expert_bias: TileTensor[bias_type, ExpertBiasLayoutType, ImmutAnyOrigin],
    routed_scaling_factor: Float32,
):
    """
    A manually fused MoE router with the group-limited strategy. It divides all
    the experts into `n_groups` groups and then finds the top `topk_group`
    groups with the highest scores. The final experts for each token are
    selected from the experts in the selected groups. The bias will be applied
    to the scores during the selection process, but the final weights will not
    include the bias.
    """
    comptime assert expert_indices.flat_rank == 2
    comptime assert expert_weights.flat_rank == 2
    comptime assert expert_scores.flat_rank == 2
    comptime assert expert_bias.flat_rank == 1
    comptime assert expert_bias.flat_rank >= 1
    comptime assert expert_scores.flat_rank >= 2
    comptime assert expert_indices.flat_rank >= 2

    comptime assert (
        expert_scores.static_shape[1] == n_routed_experts
    ), "expert_scores.static_shape[1] must be equal to n_routed_experts"

    comptime assert (
        expert_indices.static_shape[1] == n_experts_per_tok
    ), "expert_indices.static_shape[1] must be equal to n_experts_per_tok"
    comptime assert (
        expert_weights.static_shape[1] == n_experts_per_tok
    ), "expert_weights.static_shape[1] must be equal to n_experts_per_tok"

    comptime group_size = n_routed_experts // n_groups
    comptime assert (
        WARP_SIZE % group_size == 0
    ), "WARP_SIZE must be divisible by group_size"
    comptime n_groups_per_warp = WARP_SIZE // group_size
    comptime assert (
        topk_group * n_experts_per_tok <= WARP_SIZE
    ), "topk_group * n_experts_per_tok must be less than or equal to WARP_SIZE"

    comptime assert (
        num_threads == n_routed_experts
    ), "num_threads must be equal to n_routed_experts"

    var token_idx = block_idx.x
    var tid = thread_idx.x
    var warp_id = tid // WARP_SIZE

    var num_tokens = expert_scores.dim(0)

    var shared_mem = stack_allocation[
        topk_group * n_experts_per_tok,
        TopK_2[scores_type],
        address_space=AddressSpace.SHARED,
    ]()
    var selected_group = stack_allocation[
        topk_group, DType.int32, address_space=AddressSpace.SHARED
    ]()
    var thread_group_id, tid_in_group = divmod(tid, group_size)

    var thread_expert_bias = expert_bias.load[width=1](Coord(Idx(tid))).cast[
        scores_type
    ]()

    with PDL():
        var thread_expert_score: Scalar[scores_type]

        comptime if scores_input_fn:
            comptime scores_fn = scores_input_fn.value()
            thread_expert_score = scores_fn[width=1]((token_idx, tid))
        else:
            thread_expert_score = expert_scores.load[width=1](
                (Idx(token_idx), Idx(tid))
            )

        thread_expert_score += thread_expert_bias
        var thd_topk2 = TopK_2(u=thread_expert_score, p=tid)
        var sorted_group = _warp_bitonic_sort[num_lanes=group_size](thd_topk2)

        # In each group, the sum of the first two highest scores is the
        # score for the group. Store the two scores in shared memory.

        if tid_in_group == 0 or tid_in_group == 1:
            shared_mem[2 * thread_group_id + tid_in_group] = TopK_2(
                u=sorted_group.u, p=thread_group_id
            )
        barrier()

        # The first warp finds the `topk_group` groups with the highest scores.
        if warp_id == 0:
            if tid < n_groups:
                var group_scores = (
                    shared_mem[2 * tid].u + shared_mem[2 * tid + 1].u
                )
                thd_topk2 = TopK_2(u=group_scores, p=tid)
            else:
                thd_topk2 = TopK_2[scores_type]()

            var sorted_group_id = _warp_bitonic_sort[num_lanes=n_groups](
                thd_topk2
            )

            if tid < topk_group:
                selected_group[tid] = Int32(sorted_group_id.p)

        # Check if this group is selected
        barrier()
        var selected_group_smem_offset: Int32 = -1

        comptime for i in range(topk_group):
            if selected_group[i] == Int32(thread_group_id):
                selected_group_smem_offset = Int32(i * n_experts_per_tok)

        if selected_group_smem_offset >= 0:
            # Store the selected group's top `n_experts_per_tok` experts in
            # shared memory.
            if tid_in_group < n_experts_per_tok:
                shared_mem[
                    selected_group_smem_offset + Int32(tid_in_group)
                ] = sorted_group

        # Now, we use the first warp to find the global top `n_experts_per_tok` experts.
        barrier()
        if warp_id == 0:
            if tid < topk_group * n_experts_per_tok:
                thd_topk2 = shared_mem[tid]
            else:
                thd_topk2 = TopK_2[scores_type]()

            var global_topk_result = _warp_bitonic_sort[
                num_lanes=topk_group * n_experts_per_tok
            ](thd_topk2)

            var weights_sum: Scalar[scores_type] = 0
            var original_weight: Scalar[scores_type] = 0

            if tid < n_experts_per_tok:
                # We need to subtract the expert bias from the weight to get the original score.
                # This global load shouldn't be a problem since the expert bias is likely to be cached in L1.
                original_weight = (
                    global_topk_result.u
                    - expert_bias.load[width=1](
                        Coord(Idx(global_topk_result.p))
                    ).cast[scores_type]()
                )

            weights_sum = warp.lane_group_sum[num_lanes=n_experts_per_tok](
                original_weight
            )

            comptime if norm_weights:
                original_weight /= weights_sum

            original_weight *= Scalar[scores_type](routed_scaling_factor)

            if tid < n_experts_per_tok:
                expert_indices.store(
                    (Idx(token_idx), Idx(tid)), Int32(global_topk_result.p)
                )
                expert_weights[token_idx, tid] = original_weight


@always_inline
def router_group_limited[
    scores_type: DType,
    bias_type: DType,
    //,
    n_routed_experts: Int,
    n_experts_per_tok: Int,
    n_groups: Int,
    topk_group: Int,
    norm_weights: Bool,
    target: StaticString,
    scores_input_fn: OptionalReg[
        def[width: Int](IndexList[2]) capturing -> SIMD[scores_type, width]
    ] = None,
](
    expert_indices: TileTensor[mut=True, DType.int32, ...],
    expert_weights: TileTensor[mut=True, scores_type, ...],
    expert_scores: TileTensor[scores_type, ...],
    expert_bias: TileTensor[bias_type, ...],
    routed_scaling_factor: Float32,
    context: DeviceContextPtr,
) raises:
    """
    A manually fused MoE router with the group-limited strategy.

    Reference: https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/model.py#L566.

    Parameters:
        scores_type: The data type of the scores and the output weights.
        bias_type: The data type of the expert bias.
        n_routed_experts: The number of experts to route to.
        n_experts_per_tok: The number of experts to be selected per token.
        n_groups: The number of expert groups.
        topk_group: The number of expert groups to be selected per token.
        norm_weights: Whether to normalize the selected weights.
        target: The target device to run the kernel on.
        scores_input_fn: Input lambda function to load the scores.

    Inputs:
        expert_indices: The indices of the routed experts for each token.
            Shape: [num_tokens, num_experts_per_tok].
        expert_weights: The weights of the routed experts for each token.
            Shape: [num_tokens, num_experts_per_tok].
        expert_scores: The scores for each expert for each token. Shape:
            [num_tokens, n_routed_experts].
        expert_bias: The bias for each expert. Shape: [n_routed_experts].
        routed_scaling_factor: The scaling factor for the routed expert weights.
        context: DeviceContextPtr.
    """
    comptime assert is_gpu[
        target
    ](), "Group limited MoE router is only supported on GPU"

    if expert_scores.dim(0) == 0:
        return

    var gpu_ctx = context.get_device_context()

    with Trace[TraceLevel.OP, target=target](
        "mo.moe.router_group_limited", task_id=Int(gpu_ctx.id())
    ):
        comptime num_threads = n_routed_experts
        comptime hw_info = gpu_ctx.default_device_info
        comptime blocks_per_sm = hw_info.threads_per_multiprocessor // num_threads

        comptime num_sms = hw_info.sm_count

        comptime kernel = group_limited_router_kernel[
            scores_type,
            bias_type,
            expert_indices.LayoutType,
            expert_weights.LayoutType,
            expert_scores.LayoutType,
            expert_bias.LayoutType,
            n_routed_experts,
            n_experts_per_tok,
            n_groups,
            topk_group,
            norm_weights,
            num_threads,
            scores_input_fn=scores_input_fn,
        ]

        gpu_ctx.enqueue_function[kernel](
            expert_indices,
            expert_weights,
            expert_scores,
            expert_bias,
            routed_scaling_factor,
            grid_dim=expert_scores.dim(0),
            block_dim=num_threads,
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
@__name(
    t"single_group_router_{scores_type}_{bias_type}_t{num_threads}", mangle=True
)
def single_group_router_kernel[
    scores_type: DType,
    bias_type: DType,
    ExpertIndicesLayoutType: TensorLayout,
    ExpertWeightsLayoutType: TensorLayout,
    ExpertScoresLayoutType: TensorLayout,
    ExpertBiasLayoutType: TensorLayout,
    n_routed_experts: Int,
    n_experts_per_tok: Int,
    norm_weights: Bool,
    num_threads: Int,
    scores_input_fn: OptionalReg[
        def[width: Int](IndexList[2]) capturing -> SIMD[scores_type, width]
    ] = None,
](
    expert_indices: TileTensor[
        mut=True, DType.int32, ExpertIndicesLayoutType, MutAnyOrigin
    ],
    expert_weights: TileTensor[
        mut=True, scores_type, ExpertWeightsLayoutType, MutAnyOrigin
    ],
    expert_scores: TileTensor[
        scores_type, ExpertScoresLayoutType, ImmutAnyOrigin
    ],
    expert_bias: TileTensor[bias_type, ExpertBiasLayoutType, ImmutAnyOrigin],
    routed_scaling_factor: Float32,
):
    """Single-group MoE router kernel. One block per token, one thread per expert.

    Fuses: corrected = scores + bias → top-k selection → weight = corrected - bias
    → optional normalize → scale.
    Uses warp-bitonic sort across 2 or 3 phases
    depending on WARP_SIZE. NVIDIA (WARP_SIZE=32): 3-phase. AMD (WARP_SIZE=64):
    2-phase (phase 2 eliminated at compile time when phase1_candidates fits in
    one wavefront).
    """

    comptime assert expert_indices.flat_rank == 2
    comptime assert expert_weights.flat_rank == 2
    comptime assert expert_scores.flat_rank == 2
    comptime assert expert_bias.flat_rank == 1

    comptime assert (
        expert_scores.static_shape[1] == n_routed_experts
    ), "expert_scores.static_shape[1] must be equal to n_routed_experts"

    comptime assert (
        expert_weights.static_shape[1] == n_experts_per_tok
    ), "expert_weights.static_shape[1] must be equal to n_experts_per_tok"

    comptime assert (
        expert_indices.static_shape[1] == n_experts_per_tok
    ), "expert_indices.static_shape[1] must be equal to n_experts_per_tok"

    comptime assert (
        num_threads == n_routed_experts
    ), "num_threads must be equal to n_routed_experts"

    comptime assert (
        num_threads % WARP_SIZE == 0
    ), "WARP_SIZE must be divisible by num_threads"

    # k = n_experts_per_tok must be a power of 2 because _warp_bitonic_sort[num_lanes=k] uses a bitonic sort algorithm
    comptime assert (
        n_experts_per_tok.is_power_of_two()
    ), "n_experts_per_tok must be a power of two"

    # Phase 1 produces num_warps × n_experts_per_tok survivors:
    comptime num_warps = num_threads // WARP_SIZE
    comptime phase1_candidates = num_warps * n_experts_per_tok

    # Phase 2 assign selected candidates to ceil(ph1/32) warps, each sorting 32:
    comptime num_phase2_warps = ceildiv(phase1_candidates, WARP_SIZE)
    comptime phase2_candidates = num_phase2_warps * n_experts_per_tok

    # AMD has 64 warps per SM, so we can skip phase 2 for KIMIk2.5 with 384 MOE
    comptime skip_phase2 = (num_phase2_warps == 1)
    # Phase 3 takes ph2_candidates padded up to WARP_SIZE.
    comptime assert (
        phase2_candidates <= WARP_SIZE
    ), "phase2_candidates must be less than or equal to WARP_SIZE"

    comptime if skip_phase2:
        # When skipping phase 2, warp 0 reads directly from smem_phase1.
        # Requires phase1_candidates to fit within one warp.
        comptime assert (
            phase1_candidates <= WARP_SIZE
        ), "phase1_candidates exceeds WARP_SIZE — cannot skip phase 2"
    # comptime ph3_padding = WARP_SIZE - phase2_candidates

    comptime total_smem = phase1_candidates if skip_phase2 else (
        phase1_candidates + phase2_candidates
    )

    var token_idx = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var warp_id = warp_id()
    var lane_id = lane_id()

    var shared_mem = stack_allocation[
        total_smem,
        TopK_2[scores_type],
        address_space=AddressSpace.SHARED,
    ]()

    var shared_mem_phase1 = shared_mem
    var shared_mem_phase2 = shared_mem + phase1_candidates

    with PDL():
        var thread_expert_bias = expert_bias.load[width=1](
            Coord(Idx(tid))
        ).cast[scores_type]()

        var thread_expert_score: Scalar[scores_type]
        comptime if scores_input_fn:
            comptime scores_fn = scores_input_fn.value()
            thread_expert_score = scores_fn[width=1]((token_idx, tid))
        else:
            thread_expert_score = expert_scores.load[width=1](
                (Idx(token_idx), Idx(tid))
            )
        var biased_score = thread_expert_score + thread_expert_bias

        var val = TopK_2(u=biased_score, p=tid)
        var sorted_val = _warp_bitonic_sort[num_lanes=WARP_SIZE](val)

        if lane_id < n_experts_per_tok:
            shared_mem_phase1[
                warp_id * n_experts_per_tok + lane_id
            ] = sorted_val

        barrier()

        comptime if not skip_phase2:
            var val2: TopK_2[scores_type]
            if warp_id < num_phase2_warps:
                # Sequential read: warp W reads smem_phase1[W*32 .. W*32+31].
                # No bank conflicts — each warp reads a contiguous slice.
                val2 = shared_mem_phase1[tid]
            else:
                # Inactive warps: dead-value cannot corrupt the sort because
                # _warp_bitonic_sort is fully intra-warp.
                val2 = TopK_2[scores_type]()

            var sorted_val2 = _warp_bitonic_sort[num_lanes=WARP_SIZE](val2)

            if warp_id < num_phase2_warps and lane_id < n_experts_per_tok:
                shared_mem_phase2[
                    warp_id * n_experts_per_tok + lane_id
                ] = sorted_val2

            barrier()

        # WARP 0 ONLY gives top n_experts_per_tok
        if warp_id == 0:
            var val3: TopK_2[scores_type]

            comptime if skip_phase2:
                # AMD path: read phase1_candidates (48) entries directly.
                if lane_id < phase1_candidates:
                    val3 = shared_mem_phase1[lane_id]
                else:
                    val3 = TopK_2[scores_type]()  # padding: -inf
            else:
                # NVIDIA path: read phase2_candidates (24) entries.
                if lane_id < phase2_candidates:
                    val3 = shared_mem_phase2[lane_id]
                else:
                    val3 = TopK_2[scores_type]()  # padding: -inf

            var sorted_val3 = _warp_bitonic_sort[num_lanes=WARP_SIZE](val3)

            # get the original weights and normalize them
            var original_weight: Scalar[scores_type] = 0
            if lane_id < n_experts_per_tok:
                comptime if scores_input_fn:
                    comptime d_fn = scores_input_fn.value()
                    original_weight = d_fn[width=1]((token_idx, sorted_val3.p))
                else:
                    original_weight = expert_scores.load[width=1](
                        (Idx(token_idx), Idx(sorted_val3.p))
                    )

            var weights_sum = warp.lane_group_sum[num_lanes=n_experts_per_tok](
                original_weight
            )

            comptime if norm_weights:
                original_weight /= weights_sum

            original_weight *= Scalar[scores_type](routed_scaling_factor)

            # Write expert index and weight for this token.
            if lane_id < n_experts_per_tok:
                expert_indices.store(
                    (Idx(token_idx), Idx(lane_id)), Int32(sorted_val3.p)
                )
                expert_weights[token_idx, lane_id] = original_weight


@always_inline
def single_group_router[
    scores_type: DType,
    bias_type: DType,
    //,
    n_routed_experts: Int,
    n_experts_per_tok: Int,
    norm_weights: Bool,
    target: StaticString,
    scores_input_fn: OptionalReg[
        def[width: Int](IndexList[2]) capturing -> SIMD[scores_type, width]
    ] = None,
](
    expert_indices: TileTensor[mut=True, DType.int32, ...],
    expert_weight: TileTensor[mut=True, scores_type, ...],
    expert_scores: TileTensor[scores_type, ...],
    expert_bias: TileTensor[bias_type, ...],
    routed_scaling_factor: Float32,
    context: DeviceContextPtr,
) raises:
    """Launch the single-group MoE router on GPU.

    One block per token, one thread per expert. Selects top n_experts_per_tok
    experts using warp-bitonic sort with 2 or 3 reduction phases depending on
    hardware warp size (AMD skips phase 2 at compile time).

    Parameters:
        scores_type: DType of routing scores and output weights.
        bias_type: DType of the expert correction bias.
        n_routed_experts: Total number of experts (e.g. 384 for Kimi K2.5).
        n_experts_per_tok: Experts selected per token — must be a power of 2
            (e.g. 8 for Kimi K2.5).
        norm_weights: If True, normalize selected weights to sum to 1 before
            applying routed_scaling_factor.
        target: The target device to run the kernel on.
        scores_input_fn: Optional fused input lambda to load scores. If None,
            scores are loaded directly from expert_scores.

    Inputs:
        expert_indices: Output expert indices. Shape: [num_tokens, n_experts_per_tok].
        expert_weights: Output expert weights. Shape: [num_tokens, n_experts_per_tok].
        expert_scores: Input routing scores. Shape: [num_tokens, n_routed_experts].
        expert_bias: Per-expert correction bias used for selection only.
        routed_scaling_factor: Scalar multiplied into every output weight.
        context: DeviceContextPtr.
    """
    comptime assert is_gpu[
        target
    ](), "Single group router is only supported on GPU"

    if expert_scores.dim(0) == 0:
        return

    var gpu_ctx = context.get_device_context()

    with Trace[TraceLevel.OP, target=target](
        "mo.moe.router_single_group", task_id=Int(gpu_ctx.id())
    ):
        # comptime num_tokens = Int(expert_scores.dim(0))
        comptime num_threads = n_routed_experts
        comptime hw_info = gpu_ctx.default_device_info
        comptime blocks_per_sm = hw_info.threads_per_multiprocessor // num_threads

        comptime num_sms = hw_info.sm_count

        comptime kernel = single_group_router_kernel[
            scores_type,
            bias_type,
            expert_indices.LayoutType,
            expert_weight.LayoutType,
            expert_scores.LayoutType,
            expert_bias.LayoutType,
            n_routed_experts,
            n_experts_per_tok,
            norm_weights,
            num_threads,
            scores_input_fn=scores_input_fn,
        ]

        # launch the kernle using gpu_ctx
        gpu_ctx.enqueue_function[kernel](
            expert_indices,
            expert_weight,
            expert_scores,
            expert_bias,
            routed_scaling_factor,
            grid_dim=expert_scores.dim(0),
            block_dim=num_threads,
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )
