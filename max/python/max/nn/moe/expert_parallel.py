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
running DP-sharded layers (EP MoE *or* replicated MLP/MoE) in the
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
    """Runs the EP MoE forward pass with centralized communication.

    For each device shard, this function orchestrates:
    gate -> ep_dispatch -> local expert compute -> ep_combine -> shared experts.
    """
    outputs: list[TensorValue] = []
    for shard, x in zip(moe_shards, xs, strict=True):
        router_idx, router_weight = shard.gate(x)
        device_id = shard.devices[0].id

        input_scales = shard._ep_dispatch_input_scales()
        expert_inputs = shard.ep_batch_manager.ep_dispatch(
            x, ops.cast(router_idx, DType.int32), device_id, input_scales
        )

        down = shard._local_ep_compute(expert_inputs, x)

        out = shard.ep_batch_manager.ep_combine(down, router_weight, device_id)

        if (
            shard.has_shared_experts
            and not shard.ep_batch_manager.config.fused_shared_expert
        ):
            out += shard.shared_experts(x)

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
