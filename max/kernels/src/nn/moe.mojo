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
from os.atomic import Atomic
from sys.info import simd_width_of

import gpu.warp as warp
from bit import next_power_of_two, pop_count
from gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    barrier,
    block_idx,
    thread_idx,
)
from gpu.host.info import is_gpu
from gpu.memory import AddressSpace
from layout import UNKNOWN_VALUE, Layout, LayoutTensor, RuntimeLayout
from runtime.asyncrt import DeviceContextPtr
from runtime.tracing import Trace, TraceLevel

from utils.index import IndexList, StaticTuple
from builtin.dtype import _uint_type_of_width


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
        mut=True, DType.uint32, token_expert_order_layout, MutableAnyOrigin
    ],
    expert_start_indices: LayoutTensor[
        mut=True, DType.uint32, expert_start_indices_layout, MutableAnyOrigin
    ],
    restore_token_order: LayoutTensor[
        mut=True, DType.uint32, restore_token_order_layout, MutableAnyOrigin
    ],
    expert_ids: LayoutTensor[
        mut=True, DType.int32, expert_ids_layout, MutableAnyOrigin
    ],
    expert_usage_stats: LayoutTensor[
        mut=True, DType.uint32, expert_usage_stats_layout, MutableAnyOrigin
    ],
    indices_padded: LayoutTensor[
        mut=True, DType.uint32, indices_padded_layout, MutableAnyOrigin
    ],
    topk_ids_padded: LayoutTensor[
        mut=True, input_type, padded_input_layout, MutableAnyOrigin
    ],
    topk_ids: LayoutTensor[input_type, topk_ids_layout, MutableAnyOrigin],
):
    alias indices_type = DType.uint32
    var num_tokens: Int = Int(topk_ids.runtime_layout.shape[0])
    var num_tokens_padded: Int = Int(indices_padded.runtime_layout.shape[0])
    var num_tokens_per_thread = ceildiv(num_tokens_padded, num_threads)
    var thd_tok_idx = thread_idx.x * num_tokens_per_thread

    # first copy topk_ids to topk_ids_padded and fill indices_padded
    for tok_id in range(num_tokens_per_thread):
        var i = thd_tok_idx + tok_id
        if i < num_tokens:
            indices_padded[i] = i
            topk_ids_padded[i] = rebind[Scalar[input_type]](topk_ids[i])
        elif i < num_tokens_padded:
            indices_padded[i] = Scalar[indices_type].MAX_FINITE
            topk_ids_padded[i] = Scalar[input_type].MAX_FINITE
        else:
            pass

    # use Bitonic sort algorithm
    @always_inline
    fn bitonic_sort_step[
        indices_layout: Layout, input_layout: Layout
    ](
        indices: LayoutTensor[
            mut=True, DType.uint32, indices_layout, MutableAnyOrigin
        ],
        input: LayoutTensor[
            mut=True, input_type, input_layout, MutableAnyOrigin
        ],
        n: Int,
        step: Int,
        stage: Int,
        i: Int,
    ) -> None:
        if i >= n:
            return

        var partner = i ^ step

        if partner > i and partner < n:
            var cmp_val = input[i] > input[partner]

            # Determine if we are in ascending or descending part of bitonic merge.
            var bitonic_merge_direction = (i & stage) == 0

            if cmp_val == bitonic_merge_direction:
                swap(input[i], input[partner])
                swap(indices[i], indices[partner])

    barrier()
    var stage = 2
    # Iterate through increasing sequence lengths
    while stage <= num_tokens_padded:
        var step = stage // 2
        while step > 0:
            for tok_id in range(num_tokens_per_thread):
                var i = thd_tok_idx + tok_id
                bitonic_sort_step(
                    indices_padded,
                    topk_ids_padded,
                    num_tokens_padded,
                    step,
                    stage,
                    i,
                )
            barrier()
            step //= 2
        stage *= 2

    # fill the expert_offsets array with sentinel value
    var num_experts = Int(expert_start_indices.runtime_layout.shape[0])
    var num_experts_per_thread = ceildiv(num_experts, num_threads)
    for i in range(num_experts_per_thread):
        var expert_id = thread_idx.x * num_experts_per_thread + i
        if expert_id < num_experts:
            expert_start_indices[expert_id] = Scalar[indices_type].MAX_FINITE
    barrier()

    # check if this is the start of a new expert
    for tok_id in range(num_tokens_per_thread):
        var i = thd_tok_idx + tok_id
        if i < num_tokens:
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


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](num_threads)
)
fn moe_create_indices_bucket_sort_kernel[
    input_type: DType,
    token_expert_order_layout: Layout,
    expert_start_indices_layout: Layout,
    restore_token_order_layout: Layout,
    expert_ids_layout: Layout,
    expert_usage_stats_layout: Layout,
    topk_ids_layout: Layout,
    num_threads: Int = WARP_SIZE,
    expected_count: Int = 8192,  # the max topk_ids size
](
    token_expert_order: LayoutTensor[
        mut=True, DType.uint32, token_expert_order_layout, MutableAnyOrigin
    ],
    lock: LayoutTensor[DType.uint32, Layout.row_major(1), MutableAnyOrigin],
    expert_start_indices: LayoutTensor[
        mut=True, DType.uint32, expert_start_indices_layout, MutableAnyOrigin
    ],
    restore_token_order: LayoutTensor[
        mut=True, DType.uint32, restore_token_order_layout, MutableAnyOrigin
    ],
    expert_ids: LayoutTensor[
        mut=True, DType.int32, expert_ids_layout, MutableAnyOrigin
    ],
    expert_usage_stats: LayoutTensor[
        mut=True, DType.uint32, expert_usage_stats_layout, MutableAnyOrigin
    ],
    topk_ids: LayoutTensor[input_type, topk_ids_layout, MutableAnyOrigin],
):
    """
    The main goal of this kernel is to group tokens that use the same expert together.
    This allows for efficent batching when used by other kernels such as grouped matmul.

    topk_ids: a 1D tensor of expert ids, the index of each expert_id cooresponds to a token.
    For example if topk_ids is [1, 0, 1, 3, 4, 2], then the cooresponding tokens are [0, 1, 2, 3, 4, 5]

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

    constrained[
        num_threads in (32, 64),
        "Only support 32 or 64 threads per warp",
    ]()
    alias mask_type = _uint_type_of_width[num_threads]()

    alias SmemVectorType = LayoutTensor[
        DType.uint32,
        Layout.row_major(1, expected_count),
        MutableAnyOrigin,
        address_space = AddressSpace.SHARED,
        alignment=128,
    ]

    var smem = SmemVectorType.stack_allocation()

    alias token_expert_order_length = token_expert_order.layout.size()
    alias width = simd_width_of[input_type]()

    # each block is responsible for one expert
    var expert = block_idx.x

    var reads_per_iteration = num_threads * width
    var topk_ids_length = topk_ids.dim(1)
    var topk_ids_length_rounded = align_up(topk_ids_length, reads_per_iteration)

    var total_writes: UInt64 = 0

    var start_idx = thread_idx.x * width

    # vectorized loads from gmem
    for idx in range(start_idx, topk_ids_length_rounded, reads_per_iteration):
        var g_vector: SIMD[input_type, width]

        if idx + width <= topk_ids_length:
            g_vector = topk_ids.aligned_load[width=width](0, idx)
        else:
            g_vector = SIMD[input_type, width](expert + 1)

        @parameter
        for i in range(width):
            var expert_id = g_vector[i]
            var state = expert_id == expert

            var offset = total_writes

            # if state is true this thread will write to smem
            # but we need to know how many threads will write to smem before us
            # to get the correct offset. So all threads vote and we tally the votes
            # before us
            var mask = UInt64(warp.vote[mask_type](state))

            var writes = pop_count(mask)
            total_writes += writes

            var preceding_mask = mask & ((UInt64(1) << thread_idx.x) - 1)
            offset += pop_count(preceding_mask)

            if state:
                smem[0, offset] = idx + i

    # the rest that can't be vectorized are loaded normally from gmem
    start_idx = (topk_ids_length // width) * width + thread_idx.x

    var expert_id = (
        topk_ids[0, start_idx] if start_idx < topk_ids_length else expert + 1
    )
    var state = expert_id == expert

    var offset = total_writes

    var mask = UInt64(warp.vote[mask_type](state))
    var writes = pop_count(mask)
    total_writes += writes

    var preceding_mask = mask & ((UInt64(1) << thread_idx.x) - 1)
    offset += pop_count(preceding_mask)

    if state:
        smem[0, offset] = start_idx

    # copy back to token_expert_order
    if total_writes > 0:
        var expert_idx_and_offsets: UInt32 = 0

        # in order to write back to gmem we need to know the current avilable offset
        # so we use atomics to get the next available offset

        if thread_idx.x == 0:
            # we atomically update the offset and expert index with one update by adding to the offset using the last 24 bits
            # and the expert index using the upper 8 bits
            expert_idx_and_offsets = Atomic.fetch_add(
                lock.ptr, UInt32(total_writes) | 0x01000000
            )

        expert_idx_and_offsets = warp.broadcast(expert_idx_and_offsets)
        var expert_idx = expert_idx_and_offsets >> 24
        var g_offset = expert_idx_and_offsets & 0x00FFFFFF

        # add expert_id to expert_ids, this signals this exper is being used
        expert_ids[expert_idx] = expert

        # NOTE: expert_start_indices must be zero initialized otherwise the first offset will not be zero
        # add starting index for the next expert, to expert_start_indices
        expert_start_indices[expert_idx + 1] = g_offset + UInt32(total_writes)

        if expert_idx == 0:
            expert_start_indices[expert_idx] = 0

        var total_writes_rounded = align_up(
            Int(total_writes), reads_per_iteration
        )

        start_idx = thread_idx.x * width

        for smem_idx in range(
            start_idx, total_writes_rounded, reads_per_iteration
        ):
            if smem_idx + width <= Int(total_writes):
                var source_vector = smem.aligned_load[width=width](0, smem_idx)

                @parameter
                for i in range(width):
                    token_expert_order[g_offset + smem_idx + i] = source_vector[
                        i
                    ]
                    restore_token_order[source_vector[i]] = (
                        g_offset + smem_idx + i
                    )

        start_idx = UInt((total_writes // width) * width)

        g_offset += start_idx

        if thread_idx.x < Int(total_writes - start_idx):
            token_expert_order[Int(g_offset + thread_idx.x)] = smem[
                0, start_idx + thread_idx.x
            ]
            restore_token_order[smem[0, start_idx + thread_idx.x]] = (
                g_offset + thread_idx.x
            )

        # update expert_usage_stats
        if thread_idx.x == 0:
            _ = Atomic.fetch_add(expert_usage_stats.ptr + 1, 1)

            # NOTE: must be zero initialized otherwise atomic max will not work
            _ = Atomic.max(expert_usage_stats.ptr, UInt32(total_writes))


@always_inline
fn moe_create_indices[
    input_type: DType, //,
    target: StaticString,
](
    token_expert_order: LayoutTensor[mut=True, DType.uint32, **_],
    expert_start_indices: LayoutTensor[mut=True, DType.uint32, **_],
    restore_token_order: LayoutTensor[mut=True, DType.uint32, **_],
    expert_ids: LayoutTensor[mut=True, DType.int32, **_],
    expert_usage_stats: LayoutTensor[mut=True, DType.uint32, **_],
    topk_ids: LayoutTensor[input_type, **_],
    context: DeviceContextPtr,
) raises:
    constrained[
        is_gpu[target](), "Creating MoE indices is only supported on GPU"
    ]()

    var cuda_ctx = context.get_device_context()

    with Trace[TraceLevel.OP, target=target](
        "mo.moe.create_indices", task_id=Int(context.get_device_context().id())
    ):
        var lock_buffer = cuda_ctx.enqueue_create_buffer[DType.uint32](
            1
        ).enqueue_fill(0)
        var lock = LayoutTensor[
            DType.uint32, Layout.row_major(1), MutableAnyOrigin
        ](lock_buffer.unsafe_ptr())

        alias topk_layout = Layout.row_major(1, UNKNOWN_VALUE)

        var topk_2D = LayoutTensor[input_type, topk_layout, MutableAnyOrigin](
            rebind[UnsafePointer[Scalar[input_type]]](topk_ids.ptr),
            RuntimeLayout[topk_layout].row_major(
                IndexList[2](1, topk_ids.dim(0))
            ),
        )

        var num_experts = expert_ids.dim(0)

        var expert_usage_stats_host = cuda_ctx.enqueue_create_host_buffer[
            DType.uint32
        ](2).enqueue_fill(0)
        cuda_ctx.enqueue_copy[DType.uint32](
            rebind[UnsafePointer[Scalar[DType.uint32]]](expert_usage_stats.ptr),
            expert_usage_stats_host,
        )

        alias kernel = moe_create_indices_bucket_sort_kernel[
            input_type,
            token_expert_order.layout,
            expert_start_indices.layout,
            restore_token_order.layout,
            expert_ids.layout,
            expert_usage_stats.layout,
            topk_layout,
        ]

        cuda_ctx.enqueue_function_checked[kernel, kernel](
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
