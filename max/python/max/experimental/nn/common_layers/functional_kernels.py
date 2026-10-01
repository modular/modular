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

"""Functional wrappers for MAX kernel operations used in attention layers."""

from collections.abc import Sequence
from typing import Any

from max.experimental import functional as F
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    Sharded,
)
from max.experimental.sharding.action import ActionSet, AxisAssignment
from max.experimental.sharding.cost import (
    P,
    R,
    build_action_set,
    force_replicated_action_set,
)
from max.experimental.sharding.types import TensorLayout
from max.experimental.tensor import Tensor
from max.graph import TensorValue, ops
from max.nn.comm.ep.ep_kernels import (
    fused_silu as _fused_silu,
)
from max.nn.comm.ep.ep_kernels import (
    fused_silu_quantized as _fused_silu_quantized,
)
from max.nn.kernels import (
    flare_mla_prefill_plan as _flare_mla_prefill_plan,
)
from max.nn.kernels import (
    flash_attention_gpu as _flash_attention_gpu,
)
from max.nn.kernels import (
    flash_attention_ragged as _flash_attention_ragged,
)
from max.nn.kernels import (
    flash_attention_ragged_gpu as _flash_attention_ragged_gpu,
)
from max.nn.kernels import (
    grouped_matmul_ragged as _grouped_matmul_ragged,
)
from max.nn.kernels import (
    hyper_connection_gates as _hyper_connection_gates,
)
from max.nn.kernels import (
    mla_decode_graph as _mla_decode_graph,
)
from max.nn.kernels import (
    mla_prefill_decode_graph as _mla_prefill_decode_graph,
)
from max.nn.kernels import (
    mla_prefill_graph as _mla_prefill_graph,
)
from max.nn.kernels import (
    moe_create_indices as _moe_create_indices,
)
from max.nn.kernels import (
    moe_router_group_limited as _moe_router_group_limited,
)
from max.nn.kernels import (
    rms_norm_key_cache as _rms_norm_key_cache,
)
from max.nn.kernels import (
    rope_split_store_ragged as _rope_split_store_ragged,
)


def grouped_matmul_ragged_rule(
    hidden_states: TensorLayout,
    weight: TensorLayout,
    expert_start_indices: TensorLayout,
    expert_ids: TensorLayout,
    expert_usage_stats: TensorLayout,
) -> ActionSet:
    """Strategies for the MoE grouped matmul ``hidden_states @ weight.T``.

    ``weight`` is ``[num_experts, N, K]`` (Linear convention) and
    ``hidden_states`` is ``[tokens, K]``, producing ``[tokens, N]``. Mirrors
    the bf16 dense-matmul strategies: column-parallel (weight's ``N`` axis
    sharded, matching output axis) and row-parallel (weight's contraction
    ``K`` axis sharded, together with ``hidden_states``' matching axis,
    producing a partial sum). ``expert_start_indices`` / ``expert_ids`` /
    ``expert_usage_stats`` are small per-call metadata, always ``Replicated``.
    """
    layouts = (
        hidden_states,
        weight,
        expert_start_indices,
        expert_ids,
        expert_usage_stats,
    )
    rows = [
        AxisAssignment((R, R, R, R, R), R),
        # Column-parallel: weight's N (out) axis sharded -> output's N axis.
        AxisAssignment((R, Sharded(1), R, R, R), Sharded(1)),
        # Row-parallel: weight's K (contraction) axis sharded, matched by
        # hidden_states' K axis -> partial sum.
        AxisAssignment((Sharded(1), Sharded(2), R, R, R), P),
    ]
    return build_action_set(rows, layouts=layouts)


grouped_matmul_ragged = F.functional(
    _grouped_matmul_ragged, rule=grouped_matmul_ragged_rule
)


def _moe_create_indices_rule(lhs: TensorLayout, *args: Any) -> ActionSet:
    return force_replicated_action_set(lhs)


moe_create_indices = F.functional(
    _moe_create_indices, rule=_moe_create_indices_rule
)

inplace_custom = F.functional(ops.inplace_custom)
shard_and_stack = F.functional(ops.shard_and_stack)


# ─── Operations that should be dispatched per-device on distributed inputs ────


flash_attention_gpu = F.functional(_flash_attention_gpu)
flash_attention_ragged = F.functional(_flash_attention_ragged)
flash_attention_ragged_gpu = F.functional(_flash_attention_ragged_gpu)
rope_split_store_ragged = F.functional(_rope_split_store_ragged)
rms_norm_key_cache = F.functional(_rms_norm_key_cache)
flare_mla_prefill_plan = F.functional(_flare_mla_prefill_plan)
mla_prefill_graph = F.functional(_mla_prefill_graph)
mla_decode_graph = F.functional(_mla_decode_graph)
mla_prefill_decode_graph = F.functional(_mla_prefill_decode_graph)
hyper_connection_gates = F.functional(_hyper_connection_gates)


def fused_silu_rule(x: TensorLayout, row_offsets: TensorLayout) -> ActionSet:
    """Strategies for ``fused_silu``: preserves every input axis (nonlinear).

    ``row_offsets`` is the small per-call expert boundary tensor; it is
    always ``Replicated``. No ``Partial`` row: SiLU is nonlinear.
    """
    rows = [AxisAssignment((R, R), R)]
    rows += [AxisAssignment((Sharded(d), R), Sharded(d)) for d in range(x.rank)]
    return build_action_set(rows, layouts=(x, row_offsets))


fused_silu = F.functional(_fused_silu, rule=fused_silu_rule)
# Fused SiLU+FP8-quantize. The EP grouped_silu routes through this so the
# down-projection reads an already-quantized activation instead of a separate
# quantize pass; the per-128-block scale is shard-invariant.
fused_silu_quantized = F.functional(_fused_silu_quantized)

# Routing decisions must match the placement of the (replicated) router
# scores so every device agrees on expert assignment under TP/EP.
moe_router_group_limited = F.functional(_moe_router_group_limited)


def stack_device_shards(
    shards: Sequence[Tensor], axis: int, mesh: DeviceMesh
) -> Tensor:
    """Reassembles a per-device weight-shard bundle into one ``Sharded`` tensor."""
    if len(shards) == 1:
        return shards[0]
    mapping = DeviceMapping(mesh, (Sharded(axis=axis),))
    return Tensor.from_shard_values([TensorValue(s) for s in shards], mapping)


__all__ = [
    "flash_attention_gpu",
    "flash_attention_ragged",
    "flash_attention_ragged_gpu",
    "fused_silu",
    "grouped_matmul_ragged",
    "hyper_connection_gates",
    "moe_create_indices",
    "moe_router_group_limited",
    "rms_norm_key_cache",
    "rope_split_store_ragged",
    "stack_device_shards",
]
