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


from math import ceildiv

from bit import next_power_of_two
from gpu import MAX_THREADS_PER_BLOCK_METADATA, barrier, thread_idx
from gpu.host import DeviceContext
from gpu.host.info import is_gpu
from layout import UNKNOWN_VALUE, Layout, LayoutTensor, RuntimeLayout
from runtime.asyncrt import DeviceContextPtr
from runtime.tracing import Trace, TraceLevel

from utils.index import IndexList, StaticTuple


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
        mut=True, DType.uint32, expert_ids_layout, MutableAnyOrigin
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
        var last_expert_token_length = num_tokens - expert_start_indices[
            num_experts_used - 1
        ]
        max_M = max(
            max_M, rebind[Scalar[indices_type]](last_expert_token_length)
        )

        expert_usage_stats[0] = max_M
        expert_usage_stats[1] = num_experts_used


@always_inline
fn moe_create_indices[
    input_type: DType, //,
    target: StaticString,
](
    token_expert_order: LayoutTensor[mut=True, DType.uint32, **_],
    expert_start_indices: LayoutTensor[mut=True, DType.uint32, **_],
    restore_token_order: LayoutTensor[mut=True, DType.uint32, **_],
    expert_ids: LayoutTensor[mut=True, DType.uint32, **_],
    expert_usage_stats: LayoutTensor[mut=True, DType.uint32, **_],
    topk_ids: LayoutTensor[input_type, **_],
    context: DeviceContextPtr,
) raises:
    constrained[
        is_gpu[target](), "Creating MoE indices is only supported on GPU"
    ]()

    var cuda_ctx = context.get_device_context()

    with Trace[TraceLevel.OP, target=target]("mo.moe.create_indices"):
        var n = topk_ids.size()
        var pow_2_length = next_power_of_two(n)
        var padded_input_buffer = cuda_ctx.enqueue_create_buffer[input_type](
            pow_2_length
        )
        alias unknown_layout = Layout.row_major(UNKNOWN_VALUE)
        var padded_input = LayoutTensor[
            mut=True, input_type, unknown_layout, token_expert_order.origin
        ](
            padded_input_buffer.unsafe_ptr(),
            RuntimeLayout[unknown_layout].row_major(IndexList[1](pow_2_length)),
        )

        var padded_indices_buffer = cuda_ctx.enqueue_create_buffer[
            DType.uint32
        ](pow_2_length)
        var padded_indices = LayoutTensor[
            mut=True, DType.uint32, unknown_layout, token_expert_order.origin
        ](
            padded_indices_buffer.unsafe_ptr(),
            RuntimeLayout[unknown_layout].row_major(IndexList[1](pow_2_length)),
        )

        alias hw_info = cuda_ctx.device_info
        alias registers_per_thread = 255
        alias registers_per_block = hw_info.max_registers_per_block
        alias block_size_unrounded = registers_per_block // registers_per_thread
        alias block_size: UInt = block_size_unrounded - (
            block_size_unrounded % 2
        )

        alias kernel = moe_create_indices_kernel[
            input_type,
            block_size,
            token_expert_order.layout,
            expert_start_indices.layout,
            restore_token_order.layout,
            expert_ids.layout,
            expert_usage_stats.layout,
            padded_indices.layout,
            padded_input.layout,
            topk_ids.layout,
        ]

        cuda_ctx.enqueue_function[kernel](
            token_expert_order,
            expert_start_indices,
            restore_token_order,
            expert_ids,
            expert_usage_stats,
            padded_indices,
            padded_input,
            topk_ids,
            grid_dim=(1, 1, 1),
            block_dim=(block_size, 1, 1),
        )

        # Free the temporary input buffer
        _ = padded_input_buffer^
        _ = padded_indices_buffer^
