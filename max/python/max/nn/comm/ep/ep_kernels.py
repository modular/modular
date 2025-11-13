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
from max.graph import (
    BufferValue,
    DeviceRef,
    Dim,
    StaticDim,
    TensorType,
    TensorValue,
    ops,
)

from ...float8_config import Float8Config
from .ep_config import NUM_GROUPS, EPConfig


def call_ep_init(
    atomic_counter_group_0: BufferValue,
    atomic_counter_group_1: BufferValue,
    config: EPConfig,
) -> tuple[TensorValue, TensorValue]:
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
        A tuple containing:
        - device_ptrs: TensorValue containing device pointers to allocated SHMEM buffers.
            The tensor has shape [NUM_GROUPS, 3] where each group contains pointers to:
            [send_buffer, recv_buffer, recv_count_buffer].
        - my_rank: TensorValue containing the rank of the current GPU. The
            tensor has shape [1,].
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
    if config.dispatch_dtype.is_float8():
        assert config.dispatch_fp8_config is not None
        parameters["dispatch_scale_granularity"] = str(
            config.dispatch_fp8_config.granularity
        )
        parameters["dispatch_scale_dtype"] = config.dispatch_fp8_config.dtype
    else:
        # fill in dummy values for non-float8 cases
        parameters["dispatch_scale_granularity"] = "none"
        parameters["dispatch_scale_dtype"] = DType.float32

    results = ops.inplace_custom(
        "ep.init",
        device=atomic_counter_group_0.device,
        values=[atomic_counter_group_0, atomic_counter_group_1],
        out_types=[
            TensorType(DType.uint64, [NUM_GROUPS, 3], device=DeviceRef.CPU()),
            TensorType(DType.int32, [1], DeviceRef.CPU()),
        ],
        parameters=parameters,
    )

    return results[0].tensor, results[1].tensor


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
        "hidden_size": config.hidden_size,
        "top_k": config.top_k,
        "n_experts": config.n_experts,
        "max_token_per_rank": config.max_tokens_per_rank,
        "n_gpus_per_node": config.n_gpus_per_node,
        "n_nodes": config.n_nodes,
    }
    op_name = "ep.dispatch"
    if config.dispatch_dtype.is_float8():
        assert config.dispatch_fp8_config is not None
        op_name += ".fp8"
        parameters["dispatch_dtype"] = config.dispatch_dtype
        parameters["dispatch_scale_granularity"] = str(
            config.dispatch_fp8_config.granularity
        )
        parameters["dispatch_scale_dtype"] = config.dispatch_fp8_config.dtype

    ops.inplace_custom(
        op_name,
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
) -> tuple[TensorValue, ...]:
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

    return tuple([v.tensor for v in results])


def call_ep_dispatch_cb_fp8(
    atomic_counter: BufferValue,
    recv_buf_ptrs: TensorValue,
    recv_count_ptrs: TensorValue,
    config: EPConfig,
) -> tuple[TensorValue, ...]:
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
        - output_scales: Aggregated scales ready for grouped matmul computation.
            Shape: (hidden_size // block_size, max_recv_tokens)
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

    assert config.dispatch_fp8_config is not None
    assert (
        config.dispatch_fp8_config.block_size is not None
        and config.dispatch_fp8_config.block_size[1] == 128
    ), "Only support block_size=[1, 128] for input activations."

    parameters: dict[str, bool | int | str | DType] = {
        "dispatch_dtype": config.dispatch_dtype,
        "hidden_size": config.hidden_size,
        "top_k": config.top_k,
        "n_experts": config.n_experts,
        "max_token_per_rank": config.max_tokens_per_rank,
        "n_gpus_per_node": config.n_gpus_per_node,
        "n_nodes": config.n_nodes,
        "dispatch_scale_granularity": str(
            config.dispatch_fp8_config.granularity
        ),
    }

    max_recv_tokens = config.max_tokens_per_rank * config.n_experts
    n_ranks = config.n_gpus_per_node * config.n_nodes
    n_local_experts = config.n_experts // n_ranks

    device_ref = atomic_counter.device

    results = ops.inplace_custom(
        "ep.dispatch_cb.fp8",
        device=device_ref,
        values=[atomic_counter, recv_buf_ptrs, recv_count_ptrs],
        out_types=[
            TensorType(
                dtype=config.dispatch_dtype,
                shape=[max_recv_tokens, config.hidden_size],
                device=device_ref,
            ),  # output_tokens
            TensorType(
                dtype=config.dispatch_fp8_config.dtype,
                shape=[
                    config.hidden_size
                    // config.dispatch_fp8_config.block_size[1],
                    max_recv_tokens,
                ],
                device=device_ref,
            ),  # output_scales
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

    return tuple([v.tensor for v in results])


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


# ===-----------------------------------------------------------------------===#
# Expert Parallelism Utils
# ===-----------------------------------------------------------------------===#


def fused_silu(
    input: TensorValue,
    row_offsets: TensorValue,
) -> TensorValue:
    """Perform fused SILU operation for all the MLPs in the EP MoE module.

    We need to manually implement the custom operation here is because after
    the EP dispatch phase, the actual number of received tokens is not known to
    the host. This kernel will read the row offsets to determine the actual
    number of received tokens in the input tensor, and then only perform the
    SILU operation on the received tokens.

    Args:
        input_tokens: Input tokens to perform the SILU operation.
            Shape: (max_recv_tokens, hidden_size)
        row_offsets: Row offsets to determine the actual number of received
            tokens in the input tensor.
            Shape: (n_local_experts + 1,)

    Returns:
        output_tokens: Output tokens after the SILU operation.
            Shape: (max_recv_tokens, hidden_size)
    """

    if input.rank != 2:
        raise ValueError("input must be rank 2 tensor")

    if not isinstance(input.shape[1], StaticDim):
        raise ValueError(
            f"input.shape[1] must be a statically known dimension. Input shape received: {input.shape}"
        )

    hidden_size = input.shape[1] // 2

    return ops.custom(
        "ep.fused_silu",
        device=input.device,
        values=[input, row_offsets],
        out_types=[
            TensorType(
                dtype=input.dtype,
                shape=[input.shape[0], hidden_size],
                device=input.device,
            ),
        ],
    )[0].tensor


def fused_silu_fp8(
    input: TensorValue,
    row_offsets: TensorValue,
    fp8_config: Float8Config,
    out_type: DType,
) -> tuple[TensorValue, TensorValue]:
    """Perform fused SILU operation for all the MLPs in the EP MoE module.

    We need to manually implement the custom operation here is because after
    the EP dispatch phase, the actual number of received tokens is not known to
    the host. This kernel will read the row offsets to determine the actual
    number of received tokens in the input tensor, and then only perform the
    SILU operation on the received tokens. Once the SILU operation is performed,
    the output will be quantized to the FP8 format. The scales will be stored
    in a transposed way.

    Args:
        input: Input tokens to perform the SILU operation.
            Shape: (max_recv_tokens, hidden_size)
        row_offsets: Row offsets to determine the actual number of received
            tokens in the input tensor.
            Shape: (n_local_experts + 1,)
        fp8_config: FP8 configuration.

    Returns:
        A tuple containing:
        - output_tokens: Output tokens after the SILU operation.
            Shape: (max_recv_tokens, hidden_size)
        - output_scales: Output scales after the SILU operation.
            Shape: (hidden_size // block_size, max_recv_tokens)
    """

    if input.rank != 2:
        raise ValueError("input_tokens must be rank 2 tensor")

    if out_type not in (DType.float8_e4m3fn, DType.float8_e4m3fnuz):
        raise ValueError("out_type must be float8_e4m3fn or float8_e4m3fnuz")

    if not isinstance(input.shape[1], StaticDim):
        raise ValueError(
            f"input.shape[1] must be a statically known dimension. Input shape received: {input.shape}"
        )

    if (
        fp8_config.input_scale.block_size is None
        or fp8_config.input_scale.block_size[1] != 128
    ):
        raise ValueError(
            "Only support block_size=[1, 128] for input activations."
        )

    hidden_size = input.shape[1] // 2
    block_size = fp8_config.input_scale.block_size[1]
    num_blocks = hidden_size // block_size
    scales_type = fp8_config.input_scale.dtype

    # For blockwise scaling pad the a_scales to 16 Bytes. This is required by NVIDIA SM90+ TMA instructions
    padding_size = 16 // scales_type.size_in_bytes
    a_scales_dim1 = (
        (input.shape[0] + padding_size - 1) // padding_size
    ) * padding_size

    result = ops.custom(
        "ep.fused_silu_fp8",
        device=input.device,
        values=[input, row_offsets],
        out_types=[
            TensorType(
                dtype=out_type,
                shape=[input.shape[0], hidden_size],
                device=input.device,
            ),
            TensorType(
                dtype=scales_type,
                shape=[num_blocks, a_scales_dim1],
                device=input.device,
            ),
        ],
    )

    return result[0].tensor, result[1].tensor
