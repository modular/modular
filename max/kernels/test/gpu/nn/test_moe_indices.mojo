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


from gpu.host import DeviceContext, HostBuffer
from layout import UNKNOWN_VALUE, Layout, LayoutTensor, RuntimeLayout
from layout._fillers import random
from nn.moe import moe_create_indices
from testing import assert_equal

from utils import IndexList


fn get_expert_dictionary(
    topk_ids: HostBuffer[DType.uint32], num_tokens: Int
) -> Dict[UInt32, UInt32]:
    var expert_dictionary = Dict[UInt32, UInt32]()

    for i in range(num_tokens):
        var expert_id = topk_ids[i]
        var current_value = expert_dictionary.get(expert_id, 0)
        current_value += 1
        expert_dictionary[expert_id] = current_value

    return expert_dictionary^


fn check_token_expert_order(
    token_expert_order: HostBuffer[DType.uint32],
    topk_ids: HostBuffer[DType.uint32],
    num_tokens: Int,
) raises:
    """
    This function asserts all tokens of the same expert are together in the token_expert_order.
    """

    var expert_dictionary = get_expert_dictionary(topk_ids, num_tokens)
    var current_expert_id = topk_ids[Int(token_expert_order[0])]
    var token_count = expert_dictionary.get(current_expert_id, 0) - 1

    for i in range(1, num_tokens):
        var expert_id = topk_ids[Int(token_expert_order[i])]

        if expert_id != current_expert_id:
            assert_equal(token_count, 0, "tokens are grouped incorrectly")
            expert_dictionary[current_expert_id] = 0
            current_expert_id = expert_id
            token_count = expert_dictionary.get(current_expert_id, 0) - 1
        else:
            token_count -= 1

    assert_equal(token_count, 0, "tokens are grouped incorrectly")
    expert_dictionary[current_expert_id] = 0

    for k_v in expert_dictionary.take_items():
        assert_equal(k_v.value, 0, "tokens are grouped incorrectly")


fn check_expert_stats(
    expert_usage_stats: HostBuffer[DType.uint32],
    topk_ids: HostBuffer[DType.uint32],
    num_tokens: Int,
) raises:
    """
    Checks if the most frequent expert is accurate, and if the number of experts is correct.
    """

    var expert_dictionary = get_expert_dictionary(topk_ids, num_tokens)
    var total_experts = len(expert_dictionary)

    var mx_value: UInt32 = 0

    for k_v in expert_dictionary.take_items():
        if k_v.value > mx_value:
            mx_value = k_v.value

    assert_equal(
        expert_usage_stats[0], mx_value, "most frequent expert is incorrect"
    )
    assert_equal(
        expert_usage_stats[1], total_experts, "number of experts is incorrect"
    )


fn check_expert_indices(
    expert_start_indices: HostBuffer[DType.uint32],
    expert_ids: HostBuffer[DType.int32],
    token_expert_order: HostBuffer[DType.uint32],
    expert_usage_stats: HostBuffer[DType.uint32],
    topk_ids: HostBuffer[DType.uint32],
    num_tokens: Int,
) raises:
    """
    Checks if the provided start indices are correct.
    """

    for i in range(expert_usage_stats[1]):
        var start_idx = expert_start_indices[Int(i)]
        var end_idx = expert_start_indices[Int(i + 1)]
        var expert_id = expert_ids[Int(i)]

        for j in range(start_idx, end_idx):
            var current_expert_id = topk_ids[Int(token_expert_order[Int(j)])]
            assert_equal(
                expert_id,
                Int32(current_expert_id),
                "expert range in start indices is incorrect",
            )


fn check_restore_token_order(
    restore_token_order: HostBuffer[DType.uint32],
    token_expert_order: HostBuffer[DType.uint32],
    num_tokens: Int,
) raises:
    """
    Checks if original export order can be restored.
    """

    for i in range(num_tokens):
        assert_equal(
            i,
            Int(token_expert_order[Int(restore_token_order[Int(i)])]),
            "restore token order is incorrect",
        )


fn test_moe_create_indices(
    token_expert_order_length: Int,
    ctx: DeviceContext,
) raises:
    alias num_experts = 32

    var token_expert_order_buffer_host = ctx.enqueue_create_host_buffer[
        DType.uint32
    ](token_expert_order_length)
    var top_k_buffer_host = ctx.enqueue_create_host_buffer[DType.uint32](
        token_expert_order_length
    )
    var restore_token_order_buffer_host = ctx.enqueue_create_host_buffer[
        DType.uint32
    ](token_expert_order_length)
    var expert_usage_stats_buffer_host = ctx.enqueue_create_host_buffer[
        DType.uint32
    ](2)
    var expert_start_indices_buffer_host = ctx.enqueue_create_host_buffer[
        DType.uint32
    ](num_experts + 1)
    var expert_ids_buffer_host = ctx.enqueue_create_host_buffer[DType.int32](
        num_experts
    )

    var token_expert_order_buffer_device = ctx.enqueue_create_buffer[
        DType.uint32
    ](token_expert_order_length)
    var expert_start_indices_buffer = ctx.enqueue_create_buffer[DType.uint32](
        num_experts + 1
    )
    var restore_token_order_buffer = ctx.enqueue_create_buffer[DType.uint32](
        token_expert_order_length
    )
    var expert_ids_buffer = ctx.enqueue_create_buffer[DType.int32](num_experts)
    var expert_usage_stats_buffer = ctx.enqueue_create_buffer[DType.uint32](2)
    var top_k_buffer_device = ctx.enqueue_create_buffer[DType.uint32](
        token_expert_order_length
    )

    alias layout = Layout.row_major(UNKNOWN_VALUE)

    var token_expert_order = LayoutTensor[
        DType.uint32, layout, MutableAnyOrigin
    ](
        token_expert_order_buffer_device,
        RuntimeLayout[layout].row_major(
            IndexList[1](token_expert_order_length)
        ),
    )

    var expert_start_indices = LayoutTensor[
        DType.uint32, layout, MutableAnyOrigin
    ](
        expert_start_indices_buffer.unsafe_ptr(),
        RuntimeLayout[layout].row_major(IndexList[1](num_experts + 1)),
    )

    var restore_token_order = LayoutTensor[
        DType.uint32, layout, MutableAnyOrigin
    ](
        restore_token_order_buffer.unsafe_ptr(),
        RuntimeLayout[layout].row_major(
            IndexList[1](token_expert_order_length)
        ),
    )

    var expert_ids = LayoutTensor[DType.int32, layout, MutableAnyOrigin](
        expert_ids_buffer.unsafe_ptr(),
        RuntimeLayout[layout].row_major(IndexList[1](num_experts)),
    )

    var expert_usage_stats = LayoutTensor[
        DType.uint32, layout, MutableAnyOrigin
    ](
        expert_usage_stats_buffer.unsafe_ptr(),
        RuntimeLayout[layout].row_major(IndexList[1](2)),
    )

    var top_k = LayoutTensor[DType.uint32, layout, MutableAnyOrigin](
        top_k_buffer_device.unsafe_ptr(),
        RuntimeLayout[layout].row_major(
            IndexList[1](token_expert_order_length)
        ),
    )

    var top_k_host = LayoutTensor[DType.uint32, layout, MutableAnyOrigin](
        top_k_buffer_host.unsafe_ptr(),
        RuntimeLayout[layout].row_major(
            IndexList[1](token_expert_order_length)
        ),
    )

    ctx.synchronize()

    random(top_k_host, min=0, max=num_experts)
    ctx.enqueue_copy(top_k_buffer_device, top_k_buffer_host)

    moe_create_indices["gpu"](
        token_expert_order,
        expert_start_indices,
        restore_token_order,
        expert_ids,
        expert_usage_stats,
        top_k,
        ctx,
    )

    ctx.enqueue_copy(
        token_expert_order_buffer_host, token_expert_order_buffer_device
    )
    ctx.enqueue_copy(
        restore_token_order_buffer_host, restore_token_order_buffer
    )
    ctx.enqueue_copy(expert_usage_stats_buffer_host, expert_usage_stats_buffer)
    ctx.enqueue_copy(expert_ids_buffer_host, expert_ids_buffer)
    ctx.enqueue_copy(
        expert_start_indices_buffer_host, expert_start_indices_buffer
    )
    ctx.synchronize()

    check_token_expert_order(
        token_expert_order_buffer_host,
        top_k_buffer_host,
        token_expert_order_length,
    )
    check_expert_stats(
        expert_usage_stats_buffer_host,
        top_k_buffer_host,
        token_expert_order_length,
    )
    check_expert_indices(
        expert_start_indices_buffer_host,
        expert_ids_buffer_host,
        token_expert_order_buffer_host,
        expert_usage_stats_buffer_host,
        top_k_buffer_host,
        token_expert_order_length,
    )

    check_restore_token_order(
        restore_token_order_buffer_host,
        token_expert_order_buffer_host,
        token_expert_order_length,
    )


def main():
    with DeviceContext() as ctx:
        test_moe_create_indices(
            197,
            ctx,
        )

        test_moe_create_indices(
            2500,
            ctx,
        )

        test_moe_create_indices(
            11,
            ctx,
        )

        test_moe_create_indices(
            1,
            ctx,
        )

        test_moe_create_indices(
            20660,
            ctx,
        )
