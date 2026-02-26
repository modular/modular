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

"""Expert Parallelism (EP) Communication Kernels.

This file contains the kernels for Expert Parallelism (EP) communication.
"""

from __future__ import annotations

from typing import Any

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    Dim,
    Shape,
    StaticDim,
    TensorType,
    TensorValue,
    Value,
    ops,
)

from ...float8_config import Float8Config
from .ep_config import NUM_GROUPS, EPConfig


def _ep_dispatch_output_types(
    max_recv_tokens: int,
    token_last_dim: int,
    n_local_experts: int,
    config: EPConfig,
    device_ref: DeviceRef,
) -> list[TensorType]:
    """Returns the output types for the EP dispatch kernel."""
    output_tokens_type = TensorType(
        dtype=config.dispatch_dtype,
        shape=[max_recv_tokens, token_last_dim],
        device=device_ref,
    )
    expert_start_indices_type = TensorType(
        dtype=DType.uint32,
        shape=[n_local_experts + 1],
        device=device_ref,
    )
    expert_ids_type = TensorType(
        dtype=DType.int32,
        shape=[n_local_experts],
        device=device_ref,
    )
    src_info_type = TensorType(
        dtype=DType.int32,
        shape=[max_recv_tokens, 2],
        device=device_ref,
    )

    if config.dispatch_fp8_config is not None:
        float8_config = config.dispatch_fp8_config

        out_scales_type = float8_config.quantized_scales_type(
            Shape([max_recv_tokens, config.hidden_size]), device_ref
        )

        if config.dispatch_dtype.is_float8():
            return [
                output_tokens_type,
                out_scales_type,
                expert_start_indices_type,
                expert_ids_type,
                src_info_type,
            ]
        elif float8_config.is_nvfp4:
            # NVFP4 format will produce an extra tensor for offsets of the
            # padded scales.
            scales_offsets_type = TensorType(
                dtype=DType.uint32,
                shape=[n_local_experts],
                device=device_ref,
            )
            return [
                output_tokens_type,
                out_scales_type,
                expert_start_indices_type,
                scales_offsets_type,
                expert_ids_type,
                src_info_type,
            ]
        else:
            raise ValueError(
                f"Unsupported dispatch dtype: {config.dispatch_dtype}"
            )

    return [
        output_tokens_type,
        expert_start_indices_type,
        expert_ids_type,
        src_info_type,
    ]


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
        "n_nodes": config.n_nodes,
    }
    if config.dispatch_fp8_config is not None:
        if config.dispatch_fp8_config.is_nvfp4:
            parameters["dispatch_fmt_str"] = "NVFP4"
            parameters["dispatch_scale_dtype"] = DType.float8_e4m3fn
        elif config.dispatch_dtype.is_float8():
            parameters["dispatch_fmt_str"] = "BlockwiseFP8"
            parameters["dispatch_scale_dtype"] = DType.float32
        else:
            raise ValueError(
                f"Unsupported dispatch dtype: {config.dispatch_dtype}"
            )
    else:
        # fill in dummy values for non-quantized cases
        parameters["dispatch_fmt_str"] = "BF16"
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


def call_ep_dispatch_async(
    input_tokens: TensorValue,
    topk_ids: TensorValue,
    atomic_counter: BufferValue,
    send_buf_ptrs: TensorValue,
    recv_buf_ptrs: TensorValue,
    recv_count_ptrs: TensorValue,
    config: EPConfig,
    input_scales: TensorValue | None = None,
) -> None:
    """Initiate Expert Parallelism token dispatch phase (async).

    This function launches the EP async dispatch kernel that distributes input
    tokens to expert devices based on top-k routing decisions. The kernel uses
    non-blocking SHMEM communication in multi-node scenarios and returns
    immediately after initiating transfers.

    Args:
        input_tokens: Input tokens to be dispatched to experts.
            Shape: (num_tokens, hidden_size)
        topk_ids: Expert IDs selected for each token by the router.
            Shape: (num_tokens, top_k)
            Values: Expert indices in range [0, n_experts)
        atomic_counter: Buffer for synchronization between thread blocks.
        send_buf_ptrs: Device pointers to the send buffers for each GPU.
            Shape: (n_gpus_per_node,) each points to a buffer of shape
            (max_tokens_per_rank, msg_bytes)
        recv_buf_ptrs: Device pointers to the receive buffers for each GPU.
            Shape: (n_gpus_per_node,) each points to a buffer of shape
            (n_local_experts, n_ranks, max_tokens_per_rank, msg_bytes)
        recv_count_ptrs: Device pointers to the receive count buffers for
            each GPU.
            Shape: (n_gpus_per_node,) each points to a buffer of shape
            (n_local_experts, n_ranks)
        config: EP configuration.
        input_scales: Optional input scales tensor. Required for NVFP4
            dispatch. Shape: (1,) or (n_experts,).

    Note:
        This is a non-blocking operation. Call call_ep_dispatch_wait() to wait
        for completion and collect the dispatched tokens.
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
    op_name = "ep.dispatch_async"
    input_vals: list[Value[Any]] = [
        atomic_counter,
        input_tokens,
        topk_ids,
        send_buf_ptrs,
        recv_buf_ptrs,
        recv_count_ptrs,
    ]

    if config.dispatch_fp8_config is not None:
        float8_config = config.dispatch_fp8_config
        if float8_config.is_nvfp4:
            if input_scales is None:
                raise ValueError(
                    "input_scales must be provided when using NVFP4 dispatch"
                )
            op_name += ".nvfp4"
            input_vals.append(1.0 / input_scales.to(input_tokens.device))
        elif config.dispatch_dtype.is_float8():
            parameters["dispatch_fmt_str"] = "BlockwiseFP8"
        else:
            raise ValueError(
                f"Unsupported dispatch dtype: {config.dispatch_dtype}"
            )

    elif config.dispatch_dtype == DType.bfloat16:
        parameters["dispatch_fmt_str"] = "BF16"
    else:
        raise ValueError(f"Unsupported dispatch dtype: {config.dispatch_dtype}")

    ops.inplace_custom(
        op_name,
        device=input_tokens.device,
        values=input_vals,
        out_types=[],
        parameters=parameters,
    )


def call_ep_dispatch_wait(
    atomic_counter: BufferValue,
    recv_buf_ptrs: TensorValue,
    recv_count_ptrs: TensorValue,
    config: EPConfig,
    input_tokens: TensorValue | None = None,
) -> tuple[TensorValue, ...]:
    """Wait for Expert Parallelism token dispatch and prepare for expert
    computation.

    This function launches the EP dispatch wait kernel that waits for all
    inter-device communication to complete, then organizes the received tokens
    into a format suitable for grouped matmul computation.

    Args:
        atomic_counter: Buffer for synchronization between thread blocks.
        recv_buf_ptrs: Device pointers to the receive buffers for each GPU.
            Shape: (n_gpus_per_node,) each points to a buffer of shape
            (n_local_experts, n_ranks, max_tokens_per_rank, msg_bytes)
        recv_count_ptrs: Device pointers to the receive count buffers for
            each GPU.
            Shape: (n_gpus_per_node,) each points to a buffer of shape
            (n_local_experts, n_ranks)
        config: EP configuration.
        input_tokens: Input tokens for the shared expert. If shared expert
            fusion is enabled, this will be bundled with the inputs of the
            routed experts, and passed to the grouped matmul kernel.

    Returns:
        A tuple containing:
        - output_tokens: Aggregated tokens ready for grouped matmul computation.
            Shape: (max_recv_tokens, hidden_size)
        - expert_start_indices: Row offsets for grouped matmul operation.
            Shape: (n_local_experts + 1,)
        - expert_ids: Local expert IDs for the grouped operation.
            Shape: (n_local_experts,)
            Maps position in row_offsets to actual expert ID
        - src_info: Source routing information for combine phase.
            Shape: (max_recv_tokens, 2)
            [original_token_index, topk_index] for each received token

    Note:
        This function blocks until all expected tokens have been received from
        remote devices. For Quantized dispatch format, the output will also
        include the aggregated scales as the second element of the tuple.
    """
    parameters: dict[str, bool | int | str | DType] = {
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

    op_name = "ep.dispatch_wait"
    input_vals: list[Value[Any]] = [
        atomic_counter,
        recv_buf_ptrs,
        recv_count_ptrs,
    ]

    if input_tokens is not None:
        assert config.fused_shared_expert, (
            "Shared experts fusion must be enabled when input_tokens is provided"
        )
        op_name += ".fused_shared_expert"
        input_vals.append(input_tokens)
        max_recv_tokens += config.max_tokens_per_rank
        n_local_experts += 1

    output_last_dim = config.hidden_size
    if (
        config.dispatch_fp8_config is not None
        and config.dispatch_fp8_config.is_nvfp4
    ):
        output_last_dim //= 2

    output_vals = _ep_dispatch_output_types(
        max_recv_tokens,
        output_last_dim,
        n_local_experts,
        config,
        device_ref,
    )

    if config.dispatch_fp8_config is not None:
        float8_config = config.dispatch_fp8_config

        if config.dispatch_dtype.is_float8():
            op_name += ".fp8"
            parameters["dispatch_scale_granularity"] = str(
                float8_config.input_scale.granularity
            )
        elif float8_config.is_nvfp4:
            op_name += ".nvfp4"
            if config.fused_shared_expert:
                raise ValueError(
                    "NVFP4 dispatch with fused shared expert is not supported"
                )
        else:
            raise ValueError(
                f"Unsupported dispatch dtype: {config.dispatch_dtype}"
            )

    results = ops.inplace_custom(
        op_name,
        device=device_ref,
        values=input_vals,
        out_types=output_vals,
        parameters=parameters,
    )

    return tuple([v.tensor for v in results])


def call_ep_combine_async(
    input_tokens: TensorValue,
    src_info: TensorValue,
    atomic_counter: BufferValue,
    send_buf_ptrs: TensorValue,
    recv_buf_ptrs: TensorValue,
    recv_count_ptrs: TensorValue,
    config: EPConfig,
    num_output_tokens: Dim | None = None,
) -> TensorValue | None:
    """Initiate Expert Parallelism token combine phase (async).

    This function launches the EP async combine kernel that sends expert outputs
    back to their original devices based on source routing information. The
    kernel uses non-blocking SHMEM communication in multi-node scenarios and
    returns immediately after initiating transfers.

    Args:
        input_tokens: Expert output tokens to send back to original devices.
            Shape: (max_tokens_per_rank, hidden_size)
            Results from expert computation that need to be routed back
        src_info: Source routing information from dispatch phase.
            Shape: (max_tokens_per_rank, 2)
            [original_token_index, topk_index] for each token
        atomic_counter: Buffer for synchronization between thread blocks.
        send_buf_ptrs: Device pointers to the send buffers for each GPU.
            Shape: (n_gpus_per_node,) each points to a buffer of shape
            (n_local_experts * n_ranks * max_tokens_per_rank, msg_bytes).
        recv_buf_ptrs: Device pointers to the receive buffers for each GPU.
            Shape: (n_gpus_per_node,) each points to a buffer of shape
            (max_tokens_per_rank, top_k, msg_bytes).
        recv_count_ptrs: Device pointers to the receive count buffers for
            each GPU.
            Shape: (n_gpus_per_node,) each points to a buffer of shape
            (n_experts,)
        config: EP configuration.
        num_output_tokens: Number of output tokens. If provided, the shared
            expert outputs will be filtered out and stored in a separate tensor.

    Returns:
        shared_expert_output: Output tokens from the shared expert. Only
        returned when fused_shared_expert is enabled. Shape:
        (num_output_tokens, hidden_size).

    Note:
        This is a non-blocking operation. Call call_ep_combine_wait() to wait
        for completion and collect the final outputs.
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

    op_name = "ep.combine_async"
    out_types: list[TensorType] = []

    if config.fused_shared_expert:
        op_name += ".fused_shared_expert"

        assert num_output_tokens is not None, (
            "num_output_tokens must be provided when fused_shared_expert is enabled"
        )
        out_types.append(
            TensorType(
                dtype=config.combine_dtype,
                shape=[num_output_tokens, config.hidden_size],
                device=atomic_counter.device,
            )
        )

    result = ops.inplace_custom(
        op_name,
        device=input_tokens.device,
        values=[
            atomic_counter,
            input_tokens,
            src_info,
            send_buf_ptrs,
            recv_buf_ptrs,
            recv_count_ptrs,
        ],
        out_types=out_types,
        parameters=parameters,
    )

    if config.fused_shared_expert:
        return result[0].tensor
    else:
        return None


def call_ep_combine_wait(
    atomic_counter: BufferValue,
    recv_buf_ptrs: TensorValue,
    recv_count_ptrs: TensorValue,
    config: EPConfig,
    num_tokens: Dim,
    router_weights: TensorValue,
) -> TensorValue:
    """Wait for Expert Parallelism token combine and return final outputs.

    This function launches the EP combine wait kernel, which waits for all
    inter-device communication to complete, then computes the weighted sum of
    routed expert outputs for each token.

    Args:
        atomic_counter: Buffer for synchronization between thread blocks.
        recv_buf_ptrs: Device pointers to the receive buffers for each GPU.
            Shape: (n_gpus_per_node,) each points to a buffer of shape
            (max_tokens_per_rank, top_k, msg_bytes)
        recv_count_ptrs: Device pointers to the receive count buffers for
            each GPU.
            Shape: (n_gpus_per_node,) each points to a buffer of shape
            (n_experts,)
        config: EP configuration.
        num_tokens: Number of original input tokens before expert processing.
        router_weights: Router weights for the current device. Once all tokens
            are received, all routed experts' outputs for each token will be
            weighted and summed to produce the final output for the token.
            Shape: (num_tokens, top_k)

    Returns:
        output_tokens: Final output tensor with expert results.
            Shape: (num_tokens, hidden_size)
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
        "ep.combine_wait",
        device=device_ref,
        values=[atomic_counter, recv_buf_ptrs, recv_count_ptrs, router_weights],
        out_types=[
            TensorType(
                dtype=config.combine_dtype,
                shape=[num_tokens, config.hidden_size],
                device=device_ref,
            ),  # output_tokens
        ],
        parameters=parameters,
    )

    return result[0].tensor


# ===-----------------------------------------------------------------------===#
# Expert Parallelism Fused Kernels
# ===-----------------------------------------------------------------------===#


def call_ep_dispatch(
    input_tokens: TensorValue,
    topk_ids: TensorValue,
    atomic_counter: BufferValue,
    send_buf_ptrs: TensorValue,
    recv_buf_ptrs: TensorValue,
    recv_count_ptrs: TensorValue,
    config: EPConfig,
    input_scales: TensorValue | None = None,
) -> tuple[TensorValue, ...]:
    """Execute fused Expert Parallelism token dispatch (async + wait).

    This function launches the fused EP dispatch kernel that combines both
    dispatch_async and dispatch_wait functionality in a single kernel launch.
    It distributes input tokens to expert devices based on top-k routing
    decisions, waits for all tokens to arrive, and organizes received tokens
    into a format suitable for grouped matmul computation.

    Args:
        input_tokens: Input tokens to be dispatched to experts.
            Shape: (num_tokens, hidden_size)
        topk_ids: Expert IDs selected for each token by the router.
            Shape: (num_tokens, top_k)
            Values: Expert indices in range [0, n_experts)
        atomic_counter: Buffer for synchronization between thread blocks.
        send_buf_ptrs: Device pointers to the send buffers for each GPU.
            Shape: (n_gpus_per_node,) each points to a buffer of shape
            (max_tokens_per_rank, msg_bytes)
        recv_buf_ptrs: Device pointers to the receive buffers for each GPU.
            Shape: (n_gpus_per_node,) each points to a buffer of shape
            (n_local_experts, n_ranks, max_tokens_per_rank, msg_bytes)
        recv_count_ptrs: Device pointers to the receive count buffers for
            each GPU.
            Shape: (n_gpus_per_node,) each points to a buffer of shape
            (n_local_experts, n_ranks)
        config: EP configuration.
        input_scales: Optional input scales tensor. Needed for NVFP4 dispatch.
            Shape: (1,)

    Returns:
        A tuple containing:
        - output_tokens: Aggregated tokens ready for grouped matmul computation.
            Shape: (max_recv_tokens, hidden_size)
        - expert_start_indices: Row offsets for grouped matmul operation.
            Shape: (n_local_experts + 1,)
        - expert_ids: Local expert IDs for the grouped operation.
            Shape: (n_local_experts,)
        - src_info: Source routing information for combine phase.
            Shape: (max_recv_tokens, 2)
            [original_token_index, topk_index] for each received token

    Note:
        For Quantized dispatch format, the output will also include the
        aggregated scales as the second element of the tuple.
    """
    parameters: dict[str, bool | int | str | DType] = {
        "hidden_size": config.hidden_size,
        "top_k": config.top_k,
        "n_experts": config.n_experts,
        "max_token_per_rank": config.max_tokens_per_rank,
        "n_gpus_per_node": config.n_gpus_per_node,
        "n_nodes": config.n_nodes,
        "fused_shared_expert": config.fused_shared_expert,
    }

    device_ref = atomic_counter.device
    op_name = "ep.dispatch"
    max_recv_tokens = config.max_tokens_per_rank * config.n_experts
    n_ranks = config.n_gpus_per_node * config.n_nodes
    n_local_experts = config.n_experts // n_ranks

    if config.fused_shared_expert:
        max_recv_tokens += config.max_tokens_per_rank
        n_local_experts += 1

    input_vals: list[Value[Any]] = [
        atomic_counter,
        input_tokens,
        topk_ids,
        send_buf_ptrs,
        recv_buf_ptrs,
        recv_count_ptrs,
    ]

    token_last_dim = config.hidden_size
    if (
        config.dispatch_fp8_config is not None
        and config.dispatch_fp8_config.is_nvfp4
    ):
        token_last_dim //= 2

    output_vals = _ep_dispatch_output_types(
        max_recv_tokens,
        token_last_dim,
        n_local_experts,
        config,
        device_ref,
    )

    if config.dispatch_fp8_config is not None:
        float8_config = config.dispatch_fp8_config

        if config.dispatch_dtype.is_float8():
            op_name += ".fp8"
            parameters["dispatch_scale_granularity"] = str(
                float8_config.input_scale.granularity
            )
        elif float8_config.is_nvfp4:
            if input_scales is None:
                raise ValueError(
                    "input_scales must be provided when using NVFP4 dispatch"
                )
            op_name += ".nvfp4"
            input_vals.append(1.0 / input_scales.to(device_ref))
        else:
            raise ValueError(
                f"Unsupported dispatch dtype: {config.dispatch_dtype}"
            )

    results = ops.inplace_custom(
        op_name,
        device=device_ref,
        values=input_vals,
        out_types=output_vals,
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
    num_tokens: Dim,
    router_weights: TensorValue,
) -> TensorValue:
    """Execute fused Expert Parallelism token combine (async + wait).

    This function launches the fused EP combine kernel that combines both
    combine_async and combine_wait functionality in a single kernel launch.
    It sends expert outputs back to their original devices, waits for all
    transfers to complete, and computes the weighted sum of routed expert
    outputs for each token.

    Args:
        input_tokens: Expert output tokens to send back to original devices.
            Shape: (max_tokens_per_rank, hidden_size)
            Results from expert computation that need to be routed back.
            For fused_shared_expert mode, this also contains shared expert
            outputs at the start.
        src_info: Source routing information from dispatch phase.
            Shape: (max_tokens_per_rank, 2)
            [original_token_index, topk_index] for each token
        atomic_counter: Buffer for synchronization between thread blocks.
        send_buf_ptrs: Device pointers to the send buffers for each GPU.
            Shape: (n_gpus_per_node,) each points to a buffer of shape
            (n_local_experts * n_ranks * max_tokens_per_rank, msg_bytes).
        recv_buf_ptrs: Device pointers to the receive buffers for each GPU.
            Shape: (n_gpus_per_node,) each points to a buffer of shape
            (max_tokens_per_rank, top_k, msg_bytes).
        recv_count_ptrs: Device pointers to the receive count buffers for
            each GPU.
            Shape: (n_gpus_per_node,) each points to a buffer of shape
            (n_experts,)
        config: EP configuration.
        num_tokens: Number of original input tokens before expert processing.
        router_weights: Router weights for the current device. All routed
            experts' outputs for a token will be weighted and summed.
            Shape: (num_tokens, top_k)
        fused_shared_expert: Whether to add shared expert outputs to the
            combined result. When True, the shared expert outputs are read
            from input_tokens and added to the reduced routed expert outputs.

    Returns:
        output_tokens: Final output tensor with expert results.
            Shape: (num_tokens, hidden_size)
            Expert outputs arranged back in original token order.
    """
    parameters: dict[str, bool | int | str | DType] = {
        "hidden_size": config.hidden_size,
        "top_k": config.top_k,
        "n_experts": config.n_experts,
        "max_token_per_rank": config.max_tokens_per_rank,
        "n_gpus_per_node": config.n_gpus_per_node,
        "n_nodes": config.n_nodes,
        "fused_shared_expert": config.fused_shared_expert,
    }

    device_ref = atomic_counter.device

    result = ops.inplace_custom(
        "ep.combine",
        device=device_ref,
        values=[
            atomic_counter,
            input_tokens,
            src_info,
            send_buf_ptrs,
            recv_buf_ptrs,
            recv_count_ptrs,
            router_weights,
        ],
        out_types=[
            TensorType(
                dtype=config.combine_dtype,
                shape=[num_tokens, config.hidden_size],
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


def fused_silu_quantized(
    input: TensorValue,
    row_offsets: TensorValue,
    fp8_config: Float8Config,
    out_type: DType,
    input_scales: TensorValue | None = None,
    scales_offsets: TensorValue | None = None,
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
        out_type: Output dtype.
        input_scales: Optional input scales tensor. Needed by NVFP4.
        scales_offsets: Optional scales offsets tensor. Needed by NVFP4.

    Returns:
        A tuple containing:
        - output_tokens: Output tokens after the SILU operation.
            Shape: (max_recv_tokens, hidden_size)
        - output_scales: Output scales after the SILU operation. Shape depends
            on the quantization format.
    """
    if input.rank != 2:
        raise ValueError("input_tokens must be rank 2 tensor")

    if not isinstance(input.shape[1], StaticDim):
        raise ValueError(
            f"input.shape[1] must be a statically known dimension. Input shape received: {input.shape}"
        )

    hidden_size = input.shape[1] // 2
    op_name = "ep.fused_silu"
    input_vals: list[Value[Any]] = [input, row_offsets]

    if fp8_config.is_nvfp4:
        op_name += ".nvfp4"
        hidden_size //= 2  # Two FP4 elements are packed into one uint8 element
        assert scales_offsets is not None and input_scales is not None, (
            "scales_offsets and input_scales must be provided when using NVFP4"
        )
        input_vals.append(scales_offsets)
        input_vals.append(1.0 / input_scales.to(input.device))
    elif out_type.is_float8():
        op_name += ".fp8"
    else:
        raise ValueError(
            f"Unsupported quantization format: {fp8_config.quant_method}"
        )

    out_scales_type = fp8_config.quantized_scales_type(
        Shape([input.shape[0], input.shape[1] // 2]), input.device
    )

    result = ops.custom(
        op_name,
        device=input.device,
        values=input_vals,
        out_types=[
            TensorType(
                dtype=out_type,
                shape=[input.shape[0], hidden_size],
                device=input.device,
            ),
            out_scales_type,
        ],
    )

    return result[0].tensor, result[1].tensor
