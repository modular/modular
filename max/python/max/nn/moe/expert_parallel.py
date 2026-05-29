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
"""Expert-parallel forward pass for MoE layers.

Provides :func:`forward_moe_sharded_layers`, the single entry point for
running EP-sharded layers (EP MoE *or* DP replicated MLP) in the
forward pass.  Internally it checks whether the shards are EP-enabled
MoE and dispatches to the EP-specific logic; otherwise it falls back to
:func:`forward_sharded_layers`.

The caller is responsible for calling
:meth:`EPBatchManager.fetch_buffers` before invoking this function.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

from max.dtype import DType
from max.graph import (
    TensorValue,
    ops,
)

from ..transformer.distributed_transformer import forward_sharded_layers
from .moe import MoE


def _ep_forward(
    moe_shards: list[MoE],
    xs: list[TensorValue],
) -> list[TensorValue]:
    """Runs the EP MoE forward pass using multi-device dispatch and combine.

    Uses single multi-device graph ops for both dispatch and combine,
    with per-shard gate and local expert compute in between:
    gate -> multi-device dispatch -> local compute -> multi-device combine.
    """

    all_topk_ids: list[TensorValue] = []
    all_router_weights: list[TensorValue] = []
    all_input_scales: list[TensorValue | None] = []
    device_ids: list[int] = []

    for shard, x in zip(moe_shards, xs, strict=True):
        router_idx, router_weight = shard.gate(x)
        all_topk_ids.append(ops.cast(router_idx, DType.int32))
        all_router_weights.append(router_weight)
        all_input_scales.append(shard._ep_dispatch_input_scales())
        device_ids.append(shard.devices[0].id)

    # Collect non-None scales into a list (all-or-nothing for NVFP4).
    scales: list[TensorValue] | None = None
    if all_input_scales[0] is not None:
        scales = [s for s in all_input_scales if s is not None]

    batch_mgr = moe_shards[0].ep_batch_manager

    # When the model has an unfused shared expert and non-allreduce EP, split
    # the per-device dispatch and combine into async launch + wait and run the
    # shared-expert subgraph in the gap. It reads only ``x`` and has no data
    # dependency on dispatch, so the graph compiler can schedule it
    # concurrently with the EP comms on each device's stream.
    has_unfused_shared = (
        moe_shards[0].has_shared_experts
        and not batch_mgr.config.fused_shared_expert
    )
    overlap_shared_expert = (
        has_unfused_shared and not batch_mgr.config.use_allreduce
    )
    shared_outs: list[TensorValue | None] | None = None

    if batch_mgr.config.use_allreduce:
        # launch per-device dispatch since they don't need to do cross-device
        # communication.
        all_dispatch_results = []
        for i, (shard, x) in enumerate(zip(moe_shards, xs, strict=True)):
            shard_mgr = shard.ep_batch_manager
            dispatch_result = shard_mgr.ep_dispatch(
                x,
                all_topk_ids[i],
                device_ids[i],
                input_scales=scales[i] if scales is not None else None,
            )
            all_dispatch_results.append(dispatch_result)
        if has_unfused_shared:
            # All devices hold the same ``x`` (TP attention replicates the
            # input). The caller AllReduces the per-device combine outputs in
            # ``_post_mlp``, so adding ``shared_experts(x)`` on every device
            # would multiply the shared contribution by ``n_devices`` after
            # the reduction. Add it on device 0 only so ``AllReduce.sum``
            # recovers a single copy.
            shared_outs = [
                moe_shards[0].shared_experts(xs[0]) if i == 0 else None
                for i in range(len(moe_shards))
            ]
    elif overlap_shared_expert:
        # Per-device async dispatch so we can interleave the shared-expert
        # subgraph between launch and wait.
        for i, (shard, x) in enumerate(zip(moe_shards, xs, strict=True)):
            shard.ep_batch_manager.ep_dispatch_async(
                x,
                all_topk_ids[i],
                device_ids[i],
                input_scales=scales[i] if scales is not None else None,
            )
        shared_outs = [
            shard.shared_experts(x)
            for shard, x in zip(moe_shards, xs, strict=True)
        ]
        all_dispatch_results = [
            shard.ep_batch_manager.ep_dispatch_wait(device_ids[i])
            for i, shard in enumerate(moe_shards)
        ]
    else:
        # Multi-device dispatch (single op).
        all_dispatch_results = batch_mgr.ep_dispatch_all(
            xs, all_topk_ids, device_ids, input_scales=scales
        )
        if has_unfused_shared:
            shared_outs = [
                shard.shared_experts(x)
                for shard, x in zip(moe_shards, xs, strict=True)
            ]

    # Estimated total token-expert pairs across all devices.
    total_tokens = ops.shape_to_tensor(xs[0].shape)[0]
    for x in xs[1:]:
        total_tokens = total_tokens + ops.shape_to_tensor(x.shape)[0]
    estimated_total_m = (
        total_tokens
        * moe_shards[0].num_experts_per_token
        // batch_mgr.config.n_gpus_per_node
    ).cast(DType.uint32)

    # Per-shard local expert compute.
    all_down_projs: list[TensorValue] = []
    for i, (shard, x) in enumerate(zip(moe_shards, xs, strict=True)):
        expert_inputs = all_dispatch_results[i]
        down = shard._local_ep_compute(expert_inputs, x, estimated_total_m)
        all_down_projs.append(down)

    if batch_mgr.config.use_allreduce:
        # launch per-device combine since they don't need to do cross-device
        # communication.
        combine_results: list[TensorValue] = []
        for i, shard in enumerate(moe_shards):
            shard_mgr = shard.ep_batch_manager
            combine_result = shard_mgr.ep_combine(
                all_down_projs[i],
                all_router_weights[i],
                device_ids[i],
                all_topk_ids[i],
            )
            combine_results.append(combine_result)
    elif overlap_shared_expert:
        # Per-device async combine. The shared-expert subgraph was issued
        # earlier between dispatch_async and dispatch_wait; combine_async +
        # combine_wait gives the scheduler a second window to absorb any
        # remaining shared-expert work.
        for i, shard in enumerate(moe_shards):
            shard.ep_batch_manager.ep_combine_async(
                all_down_projs[i], device_ids[i]
            )
        combine_results = [
            shard.ep_batch_manager.ep_combine_wait(
                all_router_weights[i], device_ids[i]
            )
            for i, shard in enumerate(moe_shards)
        ]
    else:
        # Multi-device combine (single op).
        combine_results = batch_mgr.ep_combine_all(
            all_down_projs, all_router_weights, device_ids
        )

    outputs: list[TensorValue] = []
    for i, x in enumerate(xs):
        out = combine_results[i]
        shared_out = shared_outs[i] if shared_outs is not None else None
        if shared_out is not None:
            out += shared_out
        outputs.append(out.cast(x.dtype))
    return outputs


def forward_moe_sharded_layers(
    shards: Sequence[Callable[[TensorValue], TensorValue]],
    xs: list[TensorValue],
) -> list[TensorValue]:
    """Forward pass through DP-sharded layers (EP MoE or replicated MLP/MoE).

    For EP-enabled MoE shards this runs the full expert-parallel
    communication path (dispatch -> local compute -> combine).
    For everything else (replicated MLP, non-EP MoE) it falls back to
    :func:`forward_sharded_layers`.

    Args:
        shards: Per-device shard callables (MoE, MLP, etc.).
        xs: Input tensors, one per shard.

    Returns:
        Output tensors, one per shard.
    """
    first = shards[0]
    if (
        hasattr(first, "_ep_batch_manager")
        and first._ep_batch_manager is not None
    ):
        return _ep_forward(cast(list[MoE], list(shards)), xs)
    return forward_sharded_layers(shards, xs)
