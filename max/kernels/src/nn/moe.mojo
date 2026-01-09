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


from math import align_up, ceildiv
from memory import LegacyUnsafePointer, stack_allocation

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from os.atomic import Atomic
from sys.info import simd_width_of

import gpu.warp as warp
from bit import next_power_of_two, pop_count
from gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    barrier,
    block_idx,
    grid_dim,
    thread_idx,
)
from gpu.grid_controls import PDL, pdl_launch_attributes
from gpu.host.info import is_gpu
from layout import UNKNOWN_VALUE, Layout, LayoutTensor, RuntimeLayout
from runtime.asyncrt import DeviceContextPtr
from runtime.tracing import Trace, TraceLevel

from utils.index import IndexList, StaticTuple
from builtin.dtype import _uint_type_of_width

from nn.topk import _warp_reduce_topk, TopK_2


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](num_threads)
)
fn moe_create_indices_kernel[
    input_type: DType,
    num_threads: Int,
    token_expert_order_layout: Layout,
    expert_start_indices_layout: Layout,
    restore_token_order_layout: Layout,
    expert_ids_layout: Layout,
    expert_usage_stats_layout: Layout,
    indices_padded_layout: Layout,
    padded_input_layout: Layout,
    topk_ids_layout: Layout,
](
    token_expert_order: LayoutTensor[
        mut=True, DType.uint32, token_expert_order_layout, MutAnyOrigin
    ],
    expert_start_indices: LayoutTensor[
        mut=True, DType.uint32, expert_start_indices_layout, MutAnyOrigin
    ],
    restore_token_order: LayoutTensor[
        mut=True, DType.uint32, restore_token_order_layout, MutAnyOrigin
    ],
    expert_ids: LayoutTensor[
        mut=True, DType.int32, expert_ids_layout, MutAnyOrigin
    ],
    expert_usage_stats: LayoutTensor[
        mut=True, DType.uint32, expert_usage_stats_layout, MutAnyOrigin
    ],
    indices_padded: LayoutTensor[
        mut=True, DType.uint32, indices_padded_layout, MutAnyOrigin
    ],
    topk_ids_padded: LayoutTensor[
        mut=True, input_type, padded_input_layout, MutAnyOrigin
    ],
    topk_ids: LayoutTensor[input_type, topk_ids_layout, MutAnyOrigin],
):
    comptime indices_type = DType.uint32
    var num_tokens: Int = Int(topk_ids.runtime_layout.shape[0])
    var num_tokens_padded: Int = Int(indices_padded.runtime_layout.shape[0])
    var num_tokens_per_thread = ceildiv(num_tokens_padded, num_threads)
    var thd_tok_idx = thread_idx.x * UInt(num_tokens_per_thread)

    # first copy topk_ids to topk_ids_padded and fill indices_padded
    for tok_id in range(num_tokens_per_thread):
        var i = thd_tok_idx + UInt(tok_id)
        if i < UInt(num_tokens):
            indices_padded[i] = i
            topk_ids_padded[i] = rebind[Scalar[input_type]](topk_ids[i])
        elif i < UInt(num_tokens_padded):
            indices_padded[i] = Scalar[indices_type].MAX_FINITE
            topk_ids_padded[i] = Scalar[input_type].MAX_FINITE
        else:
            pass

    # Use Bitonic sort algorithm to sort expert IDs and their corresponding token indices.
    @always_inline
    fn bitonic_sort_step[
        indices_layout: Layout, input_layout: Layout
    ](
        indices: LayoutTensor[
            mut=True, DType.uint32, indices_layout, MutAnyOrigin
        ],
        input: LayoutTensor[mut=True, input_type, input_layout, MutAnyOrigin],
        n: Int,
        step: Int,
        stage: Int,
        i: Int,
    ) -> None:
        """Perform one step of bitonic sort.

        Bitonic sort works by comparing elements at distance 'step' apart and
        swapping them based on the direction of the current stage.

        Parameters:
            indices_layout: Layout of the indices tensor.
            input_layout: Layout of the input tensor.

        Args:
            indices: Token indices to be sorted alongside input values.
            input: Expert IDs to sort.
            n: Total number of elements.
            step: Distance between elements to compare.
            stage: Current stage size (power of 2), determines sort direction.
            i: Index of the current element.
        """
        if i >= n:
            return

        # Calculate partner index using XOR - determines the element to compare with
        var partner = i ^ step

        # Compare if partner is greater than current index to avoid redundant comparisons
        if partner > i and partner < n:
            var cmp_val = input[i] > input[partner]

            # Determine sort direction for this part of the bitonic sequence
            # (i & stage) == 0 should be in ascending order
            # (i & stage) != 0 should be in descending order
            var bitonic_merge_direction = (i & stage) == 0

            # Swap if elements are in wrong order for current direction
            if cmp_val == bitonic_merge_direction:
                swap(input[i], input[partner])
                swap(indices[i], indices[partner])

    # Synchronize all threads before starting sort
    barrier()

    # Bitonic sort main loop: build bitonic sequences of increasing sizes
    # Starting from stage=2 (pairs), double the stage size each iteration
    var stage = 2
    while stage <= num_tokens_padded:
        # For each stage, perform multiple merge steps
        # Start with step = stage/2 and halve it each iteration
        var step = stage // 2
        while step > 0:
            for tok_id in range(num_tokens_per_thread):
                var i = thd_tok_idx + UInt(tok_id)
                bitonic_sort_step(
                    indices_padded,
                    topk_ids_padded,
                    num_tokens_padded,
                    step,
                    stage,
                    Int(i),
                )
            barrier()
            step //= 2
        stage *= 2

    # fill the expert_offsets array with sentinel value
    var num_experts = Int(expert_start_indices.runtime_layout.shape[0])
    var num_experts_per_thread = ceildiv(num_experts, num_threads)
    for i in range(num_experts_per_thread):
        var expert_id = thread_idx.x * UInt(num_experts_per_thread) + UInt(i)
        if expert_id < UInt(num_experts):
            expert_start_indices[expert_id] = Scalar[indices_type].MAX_FINITE
    barrier()

    # check if this is the start of a new expert
    for tok_id in range(num_tokens_per_thread):
        var i = thd_tok_idx + UInt(tok_id)
        if i < UInt(num_tokens):
            # copy results back to token_expert_order
            token_expert_order[i] = indices_padded[i]

            # also, fill the restore_token_order array
            restore_token_order[Int(indices_padded[i])] = i

            # check if this is the start of a new expert
            if i != 0:
                if topk_ids_padded[i] != topk_ids_padded[i - 1]:
                    expert_start_indices[Int(topk_ids_padded[i])] = i
            else:
                expert_start_indices[Int(topk_ids_padded[i])] = 0
    barrier()

    if thread_idx.x == 0:
        # squeeze the expert_start_indices array to remove all the sentinel values
        var num_experts_used = 0
        var max_M: UInt32 = 0
        for i in range(num_experts):
            # check if this is an active expert
            if expert_start_indices[i] != Scalar[indices_type].MAX_FINITE:
                # fill the expert_start_indices array with the active expert's start index
                expert_start_indices[num_experts_used] = expert_start_indices[i]
                if num_experts_used > 0:
                    max_M = max(
                        max_M,
                        rebind[Scalar[indices_type]](
                            expert_start_indices[num_experts_used]
                            - expert_start_indices[num_experts_used - 1]
                        ),
                    )

                # fill the expert_ids array with the active expert ids
                expert_ids[num_experts_used] = i

                num_experts_used += 1

        # this is the token length for the last expert
        expert_start_indices[num_experts_used] = num_tokens
        var last_expert_token_length = (
            num_tokens - expert_start_indices[num_experts_used - 1]
        )
        max_M = max(
            max_M, rebind[Scalar[indices_type]](last_expert_token_length)
        )

        expert_usage_stats[0] = max_M
        expert_usage_stats[1] = num_experts_used


@always_inline
fn calculate_warp_offset[MaskType: DType](state: Bool) -> Tuple[UInt64, UInt64]:
    # sets bits to 1 for all threads that voted true
    var mask = UInt64(warp.vote[MaskType](state))

    # counts the number of bits that are set to 1
    var writes = pop_count(mask)

    # masks out all bits that are set to 1 for higher thread IDs
    var preceding_mask = mask & ((UInt64(1) << thread_idx.x) - 1)

    # counts the number of bits that are set to 1 in the preceding mask
    var offset = pop_count(preceding_mask)

    return writes, offset


struct _BucketGroupParams[num_threads: Int, input_type: DType]:
    comptime MaskType = _uint_type_of_width[Self.num_threads]()
    comptime width = simd_width_of[Self.input_type]()

    var expert: UInt
    var reads_per_iteration: Int
    var topk_ids_length: Int
    var topk_ids_length_rounded: Int
    var start_idx: Int
    var remainder_start_idx: Int

    fn __init__(out self, top_k_length: Int):
        self.expert = block_idx.x
        self.reads_per_iteration = Self.num_threads * Self.width
        self.topk_ids_length = top_k_length
        self.topk_ids_length_rounded = align_up(
            self.topk_ids_length, self.reads_per_iteration
        )
        self.start_idx = Int(thread_idx.x) * Self.width
        self.remainder_start_idx = (
            self.topk_ids_length // Self.width
        ) * Self.width + Int(thread_idx.x)


@always_inline
fn _count_expert_tokens[
    num_threads: Int,
    input_type: DType,
    //,
    expected_count: Int,
](
    topk_ids: LayoutTensor[input_type, ...],
    smem: LayoutTensor[mut=True, DType.uint32, ...],
    bg_params: _BucketGroupParams[num_threads, input_type],
) -> UInt64:
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
            g_vector = topk_ids.aligned_load[width=width](0, idx)
        else:
            g_vector = SIMD[input_type, width](bg_params.expert + 1)

        # Use warp-level voting to efficiently count matching tokens
        # All threads in the warp vote, and we count how many threads
        # before us also voted true to determine our write offset
        @parameter
        for i in range(width):
            var expert_id = g_vector[i]
            var state = expert_id == bg_params.expert

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
            if state and offset < expected_count:
                smem[0, offset] = idx + i

    var expert_id = (
        topk_ids[
            0, bg_params.remainder_start_idx
        ] if bg_params.remainder_start_idx
        < bg_params.topk_ids_length else bg_params.expert + 1
    )
    var state = expert_id == bg_params.expert

    # Use same warp voting technique for remainder elements
    var warp_writes, preceding_thread_writes = calculate_warp_offset[MaskType](
        state
    )
    var offset = total_writes + preceding_thread_writes
    total_writes += warp_writes

    if state and offset < expected_count:
        smem[0, offset] = bg_params.remainder_start_idx

    return total_writes


@always_inline
fn _get_index_and_offset(
    lock: LayoutTensor[DType.uint32, Layout.row_major(1), MutAnyOrigin],
    total_writes: UInt64,
) -> Tuple[UInt32, UInt32]:
    var expert_idx_and_offsets: UInt32 = 0

    # in order to write back to gmem we need to know the current available offset
    # so we use atomics to get the next available offset

    if thread_idx.x == 0:
        # Pack expert index (8 bits) and offset (24 bits) into single atomic update
        # Upper 8 bits: expert counter (which expert slot to use)
        # Lower 24 bits: offset in token_expert_order array
        expert_idx_and_offsets = Atomic.fetch_add(
            lock.ptr, UInt32(total_writes) | 0x01000000
        )

    # Broadcast the atomic result to all threads in the warp
    expert_idx_and_offsets = warp.broadcast(expert_idx_and_offsets)
    var expert_idx = expert_idx_and_offsets >> 24
    var base_g_offset = expert_idx_and_offsets & 0x00FFFFFF

    return expert_idx, base_g_offset


@always_inline
fn _copy_tokens_smem_to_gmem[
    num_threads: Int,
    input_type: DType,
    //,
    expected_count: Int,
](
    token_expert_order: LayoutTensor[mut=True, DType.uint32, ...],
    restore_token_order: LayoutTensor[mut=True, DType.uint32, ...],
    smem: LayoutTensor[DType.uint32, ...],
    g_offset: UInt32,
    total_writes: UInt64,
    bg_params: _BucketGroupParams[num_threads, input_type],
):
    var g_offset_copy = g_offset
    comptime width = bg_params.width

    var total_reads_rounded = align_up(
        Int(total_writes), bg_params.reads_per_iteration
    )

    var total_smem_reads = align_up(
        expected_count, bg_params.reads_per_iteration
    )
    var rounded_smem_reads = min(total_smem_reads, total_reads_rounded)
    var smem_writes = min(expected_count, total_writes)

    for smem_idx in range(
        bg_params.start_idx, rounded_smem_reads, bg_params.reads_per_iteration
    ):
        if smem_idx + width <= Int(smem_writes):
            var source_vector = smem.aligned_load[width=width](0, smem_idx)

            @parameter
            for i in range(width):
                token_expert_order[
                    g_offset_copy + smem_idx + i
                ] = source_vector[i]
                restore_token_order[source_vector[i]] = (
                    g_offset_copy + smem_idx + i
                )

    var start_idx = UInt((smem_writes // width) * width)

    g_offset_copy += start_idx

    if thread_idx.x < UInt(smem_writes - start_idx):
        token_expert_order[Int(g_offset_copy + thread_idx.x)] = rebind[
            SIMD[DType.uint32, token_expert_order.element_size]
        ](smem[0, start_idx + thread_idx.x])

        restore_token_order[smem[0, start_idx + thread_idx.x]] = (
            g_offset_copy + thread_idx.x
        )


@always_inline
fn _copy_tokens_to_gmem[
    num_threads: Int,
    input_type: DType,
    //,
    expected_count: Int,
](
    topk_ids: LayoutTensor[input_type, ...],
    smem: LayoutTensor[DType.uint32, ...],
    token_expert_order: LayoutTensor[mut=True, DType.uint32, ...],
    restore_token_order: LayoutTensor[mut=True, DType.uint32, ...],
    total_writes: UInt64,
    g_offset: UInt32,
    bg_params: _BucketGroupParams[num_threads, input_type],
):
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
            g_vector = topk_ids.aligned_load[width=width](0, idx)
        else:
            g_vector = SIMD[input_type, width](bg_params.expert + 1)

        @parameter
        for i in range(width):
            var expert_id = g_vector[i]
            var state = expert_id == bg_params.expert

            var warp_writes, preceding_thread_writes = calculate_warp_offset[
                MaskType
            ](state)
            var thr_tokens_seen = (
                tokens_seen + preceding_thread_writes + (1 if state else 0)
            )

            # we have already writeen expected_count tokens to global memory since they were in shared memory.
            # so we only need to write the remaining tokens to global memory.
            if thr_tokens_seen >= expected_count and state:
                token_expert_order[
                    g_offset_copy + UInt(preceding_thread_writes)
                ] = (idx + i)
                restore_token_order[idx + i] = g_offset_copy + UInt(
                    preceding_thread_writes
                )

            tokens_seen += warp_writes
            g_offset_copy += UInt(warp_writes)

    # Handle remainder elements that couldn't be vectorized
    var expert_id = (
        topk_ids[
            0, bg_params.remainder_start_idx
        ] if bg_params.remainder_start_idx
        < bg_params.topk_ids_length else bg_params.expert + 1
    )
    var state = expert_id == bg_params.expert

    # Use same warp voting technique for remainder elements
    var _, preceding_thread_writes = calculate_warp_offset[MaskType](state)
    var temp_current_writes = (
        tokens_seen + preceding_thread_writes + (1 if state else 0)
    )

    if temp_current_writes >= expected_count and state:
        token_expert_order[
            g_offset_copy + UInt(preceding_thread_writes)
        ] = bg_params.remainder_start_idx
        restore_token_order[
            bg_params.remainder_start_idx
        ] = g_offset_copy + UInt(preceding_thread_writes)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](num_threads)
)
fn moe_create_indices_bucket_group_kernel[
    input_type: DType,
    token_expert_order_layout: Layout,
    expert_start_indices_layout: Layout,
    restore_token_order_layout: Layout,
    expert_ids_layout: Layout,
    expert_usage_stats_layout: Layout,
    topk_ids_layout: Layout,
    num_threads: Int = WARP_SIZE,
    expected_count: Int = 8192,
](
    token_expert_order: LayoutTensor[
        mut=True, DType.uint32, token_expert_order_layout, MutAnyOrigin
    ],
    lock: LayoutTensor[DType.uint32, Layout.row_major(1), MutAnyOrigin],
    expert_start_indices: LayoutTensor[
        mut=True, DType.uint32, expert_start_indices_layout, MutAnyOrigin
    ],
    restore_token_order: LayoutTensor[
        mut=True, DType.uint32, restore_token_order_layout, MutAnyOrigin
    ],
    expert_ids: LayoutTensor[
        mut=True, DType.int32, expert_ids_layout, MutAnyOrigin
    ],
    expert_usage_stats: LayoutTensor[
        mut=True, DType.uint32, expert_usage_stats_layout, MutAnyOrigin
    ],
    topk_ids: LayoutTensor[input_type, topk_ids_layout, MutAnyOrigin],
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

    restore_token_order: a 1D tensor where each index represents a cooresponding token and holds the new index of the token
    in the token_expert_order tensor. For our example the restore_token_order would be [0, 2, 1, 3, 4, 5]
    """

    __comptime_assert num_threads in (
        32,
        64,
    ), "Only support 32 or 64 threads per warp"

    comptime BucketParamsType = _BucketGroupParams[num_threads, input_type]
    comptime SmemVectorType = LayoutTensor[
        DType.uint32,
        Layout.row_major(1, expected_count),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]

    # Allocate shared memory for temporary storage of matching token indices
    var smem = SmemVectorType.stack_allocation()

    __comptime_assert (
        expected_count % BucketParamsType.width == 0
    ), "Expected count must be a multiple of the simd width"

    var bucket_group_params = BucketParamsType(topk_ids.dim(1))

    # count tokens per expert and store as many we can in shared memory
    var total_writes = _count_expert_tokens[expected_count](
        topk_ids, smem, bucket_group_params
    )

    if total_writes > 0:
        # Copy all tokens in shared memory back to global memory
        var expert_idx, g_offset = _get_index_and_offset(lock, total_writes)

        # Record which expert is active at this index
        # this signals this expert is being used
        expert_ids[expert_idx] = bucket_group_params.expert

        # Store the ending index for this expert (start of next expert)
        # NOTE: expert_start_indices must be zero-initialized for this to work correctly
        expert_start_indices[expert_idx + 1] = g_offset + UInt32(total_writes)

        # First expert always starts at index 0
        if expert_idx == 0:
            expert_start_indices[expert_idx] = 0

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
        if total_writes > expected_count:
            _copy_tokens_to_gmem[expected_count](
                topk_ids,
                smem,
                token_expert_order,
                restore_token_order,
                total_writes,
                g_offset,
                bucket_group_params,
            )

        # update expert_usage_stats
        if thread_idx.x == 0:
            _ = Atomic.fetch_add(expert_usage_stats.ptr + 1, 1)

            # NOTE: must be zero initialized otherwise atomic max will not work
            _ = Atomic.max(expert_usage_stats.ptr, UInt32(total_writes))


@always_inline
fn moe_create_indices[
    input_type: DType,
    //,
    target: StaticString,
    expected_count: Int = 8192,
](
    token_expert_order: LayoutTensor[mut=True, DType.uint32, ...],
    expert_start_indices: LayoutTensor[mut=True, DType.uint32, ...],
    restore_token_order: LayoutTensor[mut=True, DType.uint32, ...],
    expert_ids: LayoutTensor[mut=True, DType.int32, ...],
    expert_usage_stats: LayoutTensor[mut=True, DType.uint32, ...],
    topk_ids: LayoutTensor[input_type, ...],
    context: DeviceContextPtr,
) raises:
    __comptime_assert is_gpu[
        target
    ](), "Creating MoE indices is only supported on GPU"

    var cuda_ctx = context.get_device_context()

    with Trace[TraceLevel.OP, target=target](
        "mo.moe.create_indices", task_id=Int(context.get_device_context().id())
    ):
        var lock_buffer = cuda_ctx.enqueue_create_buffer[DType.uint32](1)
        lock_buffer.enqueue_fill(0)
        var lock = LayoutTensor[
            DType.uint32, Layout.row_major(1), MutAnyOrigin
        ](lock_buffer.unsafe_ptr())

        comptime topk_layout = Layout.row_major(1, UNKNOWN_VALUE)

        var topk_2D = LayoutTensor[input_type, topk_layout, MutAnyOrigin](
            rebind[UnsafePointer[Scalar[input_type]]](topk_ids.ptr),
            RuntimeLayout[topk_layout].row_major(
                IndexList[2](1, topk_ids.dim(0))
            ),
        )

        var num_experts = expert_ids.dim(0)

        var expert_usage_stats_host = cuda_ctx.enqueue_create_host_buffer[
            DType.uint32
        ](2)
        expert_usage_stats_host.enqueue_fill(0)
        cuda_ctx.enqueue_copy[DType.uint32](
            rebind[UnsafePointer[UInt32]](expert_usage_stats.ptr),
            expert_usage_stats_host,
        )

        comptime kernel = moe_create_indices_bucket_group_kernel[
            input_type,
            token_expert_order.layout,
            expert_start_indices.layout,
            restore_token_order.layout,
            expert_ids.layout,
            expert_usage_stats.layout,
            topk_layout,
            expected_count=expected_count,
        ]

        cuda_ctx.enqueue_function[kernel, kernel](
            token_expert_order,
            lock,
            expert_start_indices,
            restore_token_order,
            expert_ids,
            expert_usage_stats,
            topk_2D,
            grid_dim=(num_experts),
            block_dim=(WARP_SIZE),
        )

        _ = lock_buffer^
        _ = expert_usage_stats_host^


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](num_threads)
)
fn group_limited_router_kernel[
    scores_type: DType,
    bias_type: DType,
    expert_indices_layout: Layout,
    expert_weights_layout: Layout,
    expert_scores_layout: Layout,
    expert_bias_layout: Layout,
    n_routed_experts: Int,
    n_experts_per_tok: Int,
    n_groups: Int,
    topk_group: Int,
    norm_weights: Bool,
    num_threads: Int,
](
    expert_indices: LayoutTensor[
        DType.int32, expert_indices_layout, MutAnyOrigin
    ],
    expert_weights: LayoutTensor[
        scores_type, expert_weights_layout, MutAnyOrigin
    ],
    expert_scores: LayoutTensor[
        scores_type, expert_scores_layout, ImmutAnyOrigin
    ],
    expert_bias: LayoutTensor[bias_type, expert_bias_layout, ImmutAnyOrigin],
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

    __comptime_assert (
        expert_scores.shape[1]() == n_routed_experts
    ), "expert_scores.shape[1] must be equal to n_routed_experts"

    __comptime_assert (
        expert_indices.shape[1]() == n_experts_per_tok
    ), "expert_indices.shape[1] must be equal to n_experts_per_tok"
    __comptime_assert (
        expert_weights.shape[1]() == n_experts_per_tok
    ), "expert_weights.shape[1] must be equal to n_experts_per_tok"

    comptime group_size = n_routed_experts // n_groups
    __comptime_assert (
        WARP_SIZE % group_size == 0
    ), "WARP_SIZE must be divisible by group_size"
    comptime n_groups_per_warp = WARP_SIZE // group_size
    __comptime_assert (
        topk_group * n_experts_per_tok <= WARP_SIZE
    ), "topk_group * n_experts_per_tok must be less than or equal to WARP_SIZE"

    __comptime_assert (
        num_threads == n_routed_experts
    ), "num_threads must be equal to n_routed_experts"

    var grid_size = grid_dim.x
    var bid = block_idx.x
    var tid = Int(thread_idx.x)
    var warp_id = tid // WARP_SIZE

    var num_tokens = expert_scores.dim(0)

    var shared_mem = stack_allocation[
        topk_group * n_experts_per_tok,
        TopK_2[scores_type],
        address_space = AddressSpace.SHARED,
    ]()
    var selected_group = stack_allocation[
        topk_group, DType.int32, address_space = AddressSpace.SHARED
    ]()
    var thread_group_id = tid // group_size

    var thread_expert_bias = expert_bias.load[width=1](IndexList[1](tid)).cast[
        scores_type
    ]()

    with PDL():
        for token_idx in range(bid, num_tokens, grid_size):
            var thread_expert_score = (
                expert_scores.load[width=1](token_idx, tid) + thread_expert_bias
            )
            var thread_topk_candidate = TopK_2(u=thread_expert_score, p=tid)

            var group_scores: Scalar[scores_type] = 0

            # First, In each group, find the first two experts with the highest scores.
            # The sum of the two experts' scores is the score for the group.
            @parameter
            for _ in range(2):
                var warp_topk_result = _warp_reduce_topk[
                    num_lanes=group_size, broadcast=True
                ](thread_topk_candidate)

                group_scores += warp_topk_result.u
                if thread_topk_candidate.p == warp_topk_result.p:
                    thread_topk_candidate = TopK_2[scores_type]()
            barrier()

            # Store the group score in shared memory. Only the first thread in each group
            # does this.
            if tid % group_size == 0:
                shared_mem[thread_group_id] = TopK_2(
                    u=group_scores, p=thread_group_id
                )
            barrier()

            # The first warp finds the `topk_group` groups with the highest scores.
            if warp_id == 0:
                if tid < n_groups:
                    thread_topk_candidate = shared_mem[tid]
                else:
                    thread_topk_candidate = TopK_2[scores_type]()

                # repeat Top‑1 K times (greedy top‑K)
                @parameter
                for i in range(topk_group):
                    var warp_topk_result = _warp_reduce_topk[
                        num_lanes=n_groups, broadcast=True
                    ](thread_topk_candidate)

                    # Skip threads that hold dead values
                    if tid < n_groups:
                        if thread_topk_candidate.p == warp_topk_result.p:
                            selected_group[i] = tid
                            thread_topk_candidate = TopK_2[scores_type]()
            barrier()

            var selected_group_smem_offset: Int32 = -1

            @parameter
            for i in range(topk_group):
                if selected_group[i] == thread_group_id:
                    selected_group_smem_offset = i * n_experts_per_tok

            # Find the top `n_experts_per_tok` experts with the highest scores in each group.
            # However, only the selected group's threads will store the results in shared memory.
            thread_topk_candidate = TopK_2(u=thread_expert_score, p=tid)

            # repeat Top‑1 K times (greedy top‑K)
            @parameter
            for i in range(n_experts_per_tok):
                var warp_topk_result = _warp_reduce_topk[
                    num_lanes=group_size, broadcast=True
                ](thread_topk_candidate)

                if thread_topk_candidate.p == warp_topk_result.p:
                    thread_topk_candidate = TopK_2[scores_type]()
                    if selected_group_smem_offset >= 0:
                        shared_mem[
                            selected_group_smem_offset + i
                        ] = warp_topk_result
            barrier()

            # For all the selected groups, their top `n_experts_per_tok` experts are stored in shared memory.
            # Now, we use the first warp to find the global top `n_experts_per_tok` experts.
            if warp_id == 0:
                if tid < topk_group * n_experts_per_tok:
                    thread_topk_candidate = shared_mem[tid]
                else:
                    thread_topk_candidate = TopK_2[scores_type]()

                var weights_sum: Scalar[scores_type] = 0
                var topk_idx: Int = -1
                var selected_weight: Scalar[scores_type] = 0
                var selected_index: Int32 = -1

                @parameter
                for i in range(n_experts_per_tok):
                    var warp_topk_result = _warp_reduce_topk[
                        num_lanes = topk_group * n_experts_per_tok,
                        broadcast=True,
                    ](thread_topk_candidate)

                    # We need to subtract the expert bias from the weight to get the original score.
                    # This global load shouldn't be a problem since the expert bias is likely to be cached in L1.
                    var original_weight = (
                        warp_topk_result.u
                        - expert_bias.load[width=1](
                            IndexList[1](warp_topk_result.p)
                        ).cast[scores_type]()
                    )
                    weights_sum += original_weight
                    # Skip threads that hold dead values
                    if tid < topk_group * n_experts_per_tok:
                        if thread_topk_candidate.p == warp_topk_result.p:
                            selected_weight = original_weight
                            selected_index = warp_topk_result.p
                            thread_topk_candidate = TopK_2[scores_type]()
                            topk_idx = i

                @parameter
                if norm_weights:
                    selected_weight /= weights_sum

                selected_weight *= Scalar[scores_type](routed_scaling_factor)

                if topk_idx != -1:
                    expert_indices.store(token_idx, topk_idx, selected_index)
                    expert_weights.store(token_idx, topk_idx, selected_weight)


@always_inline
fn router_group_limited[
    scores_type: DType,
    bias_type: DType,
    //,
    n_routed_experts: Int,
    n_experts_per_tok: Int,
    n_groups: Int,
    topk_group: Int,
    norm_weights: Bool,
    target: StaticString,
](
    expert_indices: LayoutTensor[mut=True, DType.int32, ...],
    expert_weights: LayoutTensor[mut=True, scores_type, ...],
    expert_scores: LayoutTensor[scores_type, ...],
    expert_bias: LayoutTensor[bias_type, ...],
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
    __comptime_assert is_gpu[
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
        var num_blocks = min(num_sms * blocks_per_sm, expert_scores.dim(0))

        comptime kernel = group_limited_router_kernel[
            scores_type,
            bias_type,
            expert_indices.layout,
            expert_weights.layout,
            expert_scores.layout,
            expert_bias.layout,
            n_routed_experts,
            n_experts_per_tok,
            n_groups,
            topk_group,
            norm_weights,
            num_threads,
        ]

        gpu_ctx.enqueue_function[kernel, kernel](
            expert_indices,
            expert_weights,
            expert_scores,
            expert_bias,
            routed_scaling_factor,
            grid_dim=num_blocks,
            block_dim=num_threads,
            attributes=pdl_launch_attributes(),
        )
