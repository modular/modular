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

"""
Expert Parallelism (EP) Communication Kernels.

This file contains the kernels for Expert Parallelism (EP) communication.
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import BufferValue, DeviceRef, Dim, TensorType, TensorValue, ops

from .ep_config import NUM_GROUPS, EPConfig


def call_ep_init(
    atomic_counter_group_0: BufferValue,
    atomic_counter_group_1: BufferValue,
    config: EPConfig,
) -> TensorValue:
    """Initialize Expert Parallelism communication infrastructure by creating
    a custom operation that initializes SHMEM context and allocates symmetric
    memory buffers for EP communication.

    This operation only initializes the vendor library and allocates the
    symmetric memory buffers for current GPU. To prevent deadlocks, it needs to
    be called for each GPU separately through different threads.

    Args:
        atomic_counter_group_0: Atomic counters for buffer group 0.
        atomic_counter_group_1: Atomic counters for buffer group 1.
        config: EP configuration.

    Returns:
        TensorValue containing device pointers to allocated SHMEM buffers. The
        tensor has shape [NUM_GROUPS, 3] where each group contains pointers to:
        [send_buffer, recv_buffer, recv_count_buffer].
    """

    parameters: dict[str, bool | int | str | DType] = {
        "dispatch_dtype": config.dispatch_dtype,
        "combine_dtype": config.combine_dtype,
        "hidden_size": config.hidden_size,
        "top_k": config.top_k,
        "n_experts": config.n_experts,
        "max_token_per_rank": config.max_tokens_per_rank,
        "n_gpus_per_node": config.n_gpus_per_node,
    }
    return ops.inplace_custom(
        "ep.init",
        device=atomic_counter_group_0.device,
        values=[atomic_counter_group_0, atomic_counter_group_1],
        out_types=[
            TensorType(DType.uint64, [NUM_GROUPS, 3], device=DeviceRef.CPU())
        ],
        parameters=parameters,
    )[0].tensor


def call_ep_dispatch(
    input_tokens: TensorValue,
    topk_ids: TensorValue,
    atomic_counter: BufferValue,
    send_buf_ptrs: TensorValue,
    recv_buf_ptrs: TensorValue,
    recv_count_ptrs: TensorValue,
    config: EPConfig,
) -> None:
    """Initiate Expert Parallelism token dispatch phase.

    This function launches the EP dispatch kernel that distributes input tokens
    to expert devices based on top-k routing decisions. The kernel uses
    non-blocking SHMEM communication and returns immediately after initiating
    transfers.

    Args:
        input_tokens: Input tokens to be dispatched to experts.
            Shape: (num_tokens, hidden_size)
        topk_ids: Expert IDs selected for each token by the router.
            Shape: (num_tokens, top_k)
            Values: Expert indices in range [0, n_experts)
        atomic_counter: Buffer for synchronization between thread blocks.
        send_buf_ptrs: Device pointers to SHMEM send buffers for each GPU.
            Shape: (n_gpus_per_node,) each points to a buffer of shape
            (max_tokens_per_rank, msg_bytes)
        recv_buf_ptrs: Device pointers to SHMEM receive buffers for each GPU.
            Shape: (n_gpus_per_node,) each points to a buffer of shape
            (n_local_experts, n_ranks, max_tokens_per_rank, msg_bytes)
        recv_count_ptrs: Device pointers to SHMEM receive count buffers for
            each GPU.
            Shape: (n_gpus_per_node,) each points to a buffer of shape
            (n_local_experts, n_ranks)
        config: EP configuration.

    Note:
        This is a non-blocking operation. Call call_ep_dispatch_cb() to wait for
        completion and collect the dispatched tokens.
    """

    parameters: dict[str, bool | int | str | DType] = {
        "dispatch_dtype": config.dispatch_dtype,
        "hidden_size": config.hidden_size,
        "top_k": config.top_k,
        "n_experts": config.n_experts,
        "max_token_per_rank": config.max_tokens_per_rank,
        "n_gpus_per_node": config.n_gpus_per_node,
        "n_nodes": config.n_nodes,
    }

    ops.inplace_custom(
        "ep.dispatch",
        device=input_tokens.device,
        values=[
            atomic_counter,
            input_tokens,
            topk_ids,
            send_buf_ptrs,
            recv_buf_ptrs,
            recv_count_ptrs,
        ],
        out_types=[],
        parameters=parameters,
    )


def call_ep_dispatch_cb(
    atomic_counter: BufferValue,
    recv_buf_ptrs: TensorValue,
    recv_count_ptrs: TensorValue,
    config: EPConfig,
) -> tuple[TensorValue, TensorValue, TensorValue, TensorValue, TensorValue]:
    """Complete Expert Parallelism token dispatch and prepare for expert
    computation.

    This function launches the EP dispatch callback kernel that waits for all
    local token transfers to complete, then organizes the received tokens into
    a format suitable for grouped matmul computation.

    Args:
        atomic_counter: Buffer for synchronization between thread blocks.
        recv_buf_ptrs: Device pointers to SHMEM receive buffers for each GPU.
            Shape: (n_gpus_per_node,) each points to a buffer of shape
            (n_local_experts, n_ranks, max_tokens_per_rank, msg_bytes)
        recv_count_ptrs: Device pointers to SHMEM receive count buffers for
            each GPU.
            Shape: (n_gpus_per_node,) each points to a buffer of shape
            (n_local_experts, n_ranks)
        config: EP configuration.

    Returns:
        A tuple containing:
        - output_tokens: Aggregated tokens ready for grouped matmul computation.
            Shape: (max_recv_tokens, hidden_size)
        - expert_start_indices: Row offsets for grouped matmul operation.
            Shape: (n_local_experts + 1,)
        - expert_ids: Local expert IDs for the grouped operation.
            Shape: (n_local_experts,)
            Maps position in row_offsets to actual expert ID
        - expert_usage_stats: Statistics for the grouped matmul kernel.
            Shape: (2,) on CPU
            [max_tokens_per_expert, n_active_experts]
        - src_info: Source routing information for combine phase.
            Shape: (max_recv_tokens, 2)
            [original_token_index, topk_index] for each received token

    Note:
        This function blocks until all expected tokens have been received from
        remote devices.
    """

    parameters: dict[str, bool | int | str | DType] = {
        "dispatch_dtype": config.dispatch_dtype,
        "hidden_size": config.hidden_size,
        "top_k": config.top_k,
        "n_experts": config.n_experts,
        "max_token_per_rank": config.max_tokens_per_rank,
        "n_gpus_per_node": config.n_gpus_per_node,
        "n_nodes": config.n_nodes,
    }

    max_recv_tokens = config.max_tokens_per_rank * config.n_experts
    n_ranks = config.n_gpus_per_node * config.n_nodes
    n_local_experts = config.n_experts // n_ranks

    device_ref = atomic_counter.device

    results = ops.inplace_custom(
        "ep.dispatch_cb",
        device=device_ref,
        values=[atomic_counter, recv_buf_ptrs, recv_count_ptrs],
        out_types=[
            TensorType(
                dtype=config.dispatch_dtype,
                shape=[max_recv_tokens, config.hidden_size],
                device=device_ref,
            ),  # output_tokens
            TensorType(
                dtype=DType.uint32,
                shape=[n_local_experts + 1],
                device=device_ref,
            ),  # expert_start_indices
            TensorType(
                dtype=DType.int32,
                shape=[n_local_experts],
                device=device_ref,
            ),  # expert_ids
            TensorType(
                dtype=DType.uint32, shape=[2], device=DeviceRef.CPU()
            ),  # expert_usage_stats
            TensorType(
                dtype=DType.int32,
                shape=[max_recv_tokens, 2],
                device=device_ref,
            ),  # src_info
        ],
        parameters=parameters,
    )

    return (
        results[0].tensor,
        results[1].tensor,
        results[2].tensor,
        results[3].tensor,
        results[4].tensor,
    )


def call_ep_combine(
    input_tokens: TensorValue,
    src_info: TensorValue,
    atomic_counter: BufferValue,
    send_buf_ptrs: TensorValue,
    recv_buf_ptrs: TensorValue,
    recv_count_ptrs: TensorValue,
    config: EPConfig,
) -> None:
    """Initiate Expert Parallelism token combine phase.

    This function launches the EP combine kernel that sends expert outputs back
    to their original devices based on source routing information. The kernel
    uses non-blocking SHMEM communication and returns immediately after
    initiating transfers.

    Args:
        input_tokens: Expert output tokens to send back to original devices.
            Shape: (max_tokens_per_rank, hidden_size)
            Results from expert computation that need to be routed back
        src_info: Source routing information from dispatch phase.
            Shape: (max_tokens_per_rank, 2)
            [original_token_index, topk_index] for each token
        atomic_counter: Buffer for synchronization between thread blocks.
        send_buf_ptrs: Device pointers to SHMEM send buffers for each GPU.
            Shape: (n_gpus_per_node,) each points to a buffer of shape
            (n_local_experts * n_ranks * max_tokens_per_rank, msg_bytes).
        recv_buf_ptrs: Device pointers to SHMEM receive buffers for each GPU.
            Shape: (n_gpus_per_node,) each points to a buffer of shape
            (max_tokens_per_rank, top_k, msg_bytes).
        recv_count_ptrs: Device pointers to SHMEM receive count buffers for
            each GPU.
            Shape: (n_gpus_per_node,) each points to a buffer of shape
            (n_experts,)
        config: EP configuration.

    Note:
        This is a non-blocking operation. Call call_ep_combine_cb() to wait for
        completion and collect the final outputs.
    """

    parameters: dict[str, bool | int | str | DType] = {
        "combine_dtype": config.combine_dtype,
        "hidden_size": config.hidden_size,
        "top_k": config.top_k,
        "n_experts": config.n_experts,
        "max_token_per_rank": config.max_tokens_per_rank,
        "n_gpus_per_node": config.n_gpus_per_node,
        "n_nodes": config.n_nodes,
    }

    ops.inplace_custom(
        "ep.combine",
        device=input_tokens.device,
        values=[
            atomic_counter,
            input_tokens,
            src_info,
            send_buf_ptrs,
            recv_buf_ptrs,
            recv_count_ptrs,
        ],
        out_types=[],
        parameters=parameters,
    )


def call_ep_combine_cb(
    atomic_counter: BufferValue,
    recv_buf_ptrs: TensorValue,
    recv_count_ptrs: TensorValue,
    config: EPConfig,
    num_tokens: Dim,
) -> TensorValue:
    """Complete Expert Parallelism token combine and return final outputs.

    This function launches the EP combine callback kernel that waits for all
    expert output transfers to complete, then organizes the received tokens
    back into their original format and positions.

    Args:
        atomic_counter: Buffer for synchronization between thread blocks.
        recv_buf_ptrs: Device pointers to SHMEM receive buffers for each GPU.
            Shape: (n_gpus_per_node,) each points to a buffer of shape
            (max_tokens_per_rank, top_k, msg_bytes)
        recv_count_ptrs: Device pointers to SHMEM receive count buffers for
            each GPU.
            Shape: (n_gpus_per_node,) each points to a buffer of shape
            (n_experts,)
        config: EP configuration.
        num_tokens: Number of original input tokens before expert processing.

    Returns:
        output_tokens: Final output tensor with expert results.
            Shape: (num_tokens, top_k, hidden_size)
            Expert outputs arranged back in original token order.

    Note:
        This function blocks until all expected expert outputs have been
        received from remote devices.
    """

    parameters: dict[str, bool | int | str | DType] = {
        "combine_dtype": config.combine_dtype,
        "hidden_size": config.hidden_size,
        "top_k": config.top_k,
        "n_experts": config.n_experts,
        "max_token_per_rank": config.max_tokens_per_rank,
        "n_gpus_per_node": config.n_gpus_per_node,
        "n_nodes": config.n_nodes,
    }

    device_ref = atomic_counter.device

    result = ops.inplace_custom(
        "ep.combine_cb",
        device=device_ref,
        values=[atomic_counter, recv_buf_ptrs, recv_count_ptrs],
        out_types=[
            TensorType(
                dtype=config.combine_dtype,
                shape=[num_tokens, config.top_k, config.hidden_size],
                device=device_ref,
            ),  # output_tokens
        ],
        parameters=parameters,
    )

    return result[0].tensor
