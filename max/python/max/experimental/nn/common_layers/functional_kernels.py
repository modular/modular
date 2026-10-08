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

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.sharding import (
    DeviceMapping,
    DeviceMesh,
    Sharded,
    Unknown,
)
from max.experimental.sharding.action import AxisAssignment
from max.experimental.sharding.cost import P, R
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
    _moe_sigmoid_gemv_router,
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
    mla_fp8_index_top_k as _mla_fp8_index_top_k,
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
    moe_finalize as _moe_finalize,
)
from max.nn.kernels import (
    moe_router_group_limited as _moe_router_group_limited,
)
from max.nn.kernels import (
    mtp_eh_norm as _mtp_eh_norm,
)
from max.nn.kernels import (
    quantize_dynamic_scaled_float8 as _quantize_dynamic_scaled_float8,
)
from max.nn.kernels import (
    rms_norm_key_cache as _rms_norm_key_cache,
)
from max.nn.kernels import (
    rope_ragged as _rope_ragged,
)
from max.nn.kernels import (
    rope_split_store_ragged as _rope_split_store_ragged,
)
from max.nn.kernels import (
    store_k_cache_ragged as _store_k_cache_ragged,
)
from max.nn.kernels import (
    store_k_scale_cache_ragged as _store_k_scale_cache_ragged,
)


def grouped_matmul_ragged_rule(
    hidden_states: TensorLayout,
    weight: TensorLayout,
    expert_start_indices: TensorLayout,
    expert_ids: TensorLayout,
    expert_usage_stats: TensorLayout,
) -> list[AxisAssignment]:
    """Strategies for the MoE grouped matmul ``hidden_states @ weight.T``.

    ``weight`` is ``[num_experts, N, K]`` (Linear convention) and
    ``hidden_states`` is ``[tokens, K]``, producing ``[tokens, N]``. Mirrors
    the bf16 dense-matmul strategies: column-parallel (weight's ``N`` axis
    sharded, matching output axis) and row-parallel (weight's contraction
    ``K`` axis sharded, together with ``hidden_states``' matching axis,
    producing a partial sum). ``expert_start_indices`` / ``expert_ids`` /
    ``expert_usage_stats`` are small per-call metadata, always ``Replicated``.
    """
    return [
        AxisAssignment((R, R, R, R, R), (R,)),
        # Column-parallel: weight's N (out) axis sharded -> output's N axis.
        AxisAssignment((R, Sharded(1), R, R, R), (Sharded(1),)),
        # Row-parallel: weight's K (contraction) axis sharded, matched by
        # hidden_states' K axis -> partial sum.
        AxisAssignment((Sharded(1), Sharded(2), R, R, R), (P,)),
    ]


grouped_matmul_ragged = F.functional(
    _grouped_matmul_ragged, rule=grouped_matmul_ragged_rule
)


def _moe_create_indices_rule(
    lhs: TensorLayout,
    num_local_experts: int,
    *,
    needs_scales_offset: bool = False,
) -> list[AxisAssignment]:
    """Returns the rows for ``moe_create_indices``.

    Replicated ids give whole-batch positions; sharded ids give local ones.
    """
    # Over a shard of the batch the outputs mix per-token and per-expert
    # extents that no one global tensor spans, so the row forgets instead.
    return [
        AxisAssignment((R,), (R,) * (6 if needs_scales_offset else 5)),
        AxisAssignment(
            (Sharded(0),), (Unknown(),) * (6 if needs_scales_offset else 5)
        ),
    ]


moe_create_indices = F.functional(
    _moe_create_indices, rule=_moe_create_indices_rule
)


def moe_finalize_rule(
    down_projs: TensorLayout,
    restore_token_order: TensorLayout,
    router_weight: TensorLayout,
    out_type: DType,
) -> list[AxisAssignment]:
    """Strategies for ``moe_finalize``: linear in ``down_projs``.

    The weighted row sum keeps ``down_projs``' hidden-axis sharding and
    passes a partial sum through. ``restore_token_order`` and
    ``router_weight`` are per-token routing state, always ``Replicated``.
    """
    return [
        AxisAssignment((R, R, R), (R,)),
        AxisAssignment((Sharded(1), R, R), (Sharded(1),)),
        AxisAssignment((P, R, R), (P,)),
    ]


moe_finalize = F.functional(_moe_finalize, rule=moe_finalize_rule)

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
mla_fp8_index_top_k = F.functional(_mla_fp8_index_top_k)
rope_ragged = F.functional(_rope_ragged)
quantize_dynamic_scaled_float8 = F.functional(_quantize_dynamic_scaled_float8)
store_k_cache_ragged = F.functional(_store_k_cache_ragged)
store_k_scale_cache_ragged = F.functional(_store_k_scale_cache_ragged)


def fused_silu_rule(
    x: TensorLayout, row_offsets: TensorLayout
) -> list[AxisAssignment]:
    """Strategies for ``fused_silu``: preserves every input axis (nonlinear).

    ``row_offsets`` is the small per-call expert boundary tensor; it is
    always ``Replicated``. No ``Partial`` row: SiLU is nonlinear.
    """
    rows = [AxisAssignment((R, R), (R,))]
    rows += [
        AxisAssignment((Sharded(d), R), (Sharded(d),)) for d in range(x.rank)
    ]
    return rows


fused_silu = F.functional(_fused_silu, rule=fused_silu_rule)
# Fused SiLU+quantize for the FP4 and MX formats. The EP grouped_silu routes
# through this so the down-projection reads an already-quantized activation
# instead of a separate quantize pass; the block scale is shard-invariant.
fused_silu_quantized = F.functional(_fused_silu_quantized)

# Routing decisions must match the placement of the (replicated) router
# scores so every device agrees on expert assignment under TP/EP.
moe_router_group_limited = F.functional(_moe_router_group_limited)
mtp_eh_norm = F.functional(_mtp_eh_norm)


def _moe_sigmoid_gemv_router_rule(
    hidden_states: TensorLayout,
    gate_weight: TensorLayout,
    expert_bias: TensorLayout,
    n_experts_per_tok: int,
    norm_weights: bool,
    routed_scaling_factor: float,
) -> list[AxisAssignment]:
    # Sigmoid and top-k need the full gate dot product on every device, so
    # this op runs on replicated inputs rather than on contraction shards.
    return [AxisAssignment((R, R, R), (R, R))]


moe_sigmoid_gemv_router = F.functional(
    _moe_sigmoid_gemv_router, rule=_moe_sigmoid_gemv_router_rule
)


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
    "mla_fp8_index_top_k",
    "moe_create_indices",
    "moe_finalize",
    "moe_router_group_limited",
    "moe_sigmoid_gemv_router",
    "mtp_eh_norm",
    "quantize_dynamic_scaled_float8",
    "rms_norm_key_cache",
    "rope_ragged",
    "rope_split_store_ragged",
    "stack_device_shards",
    "store_k_cache_ragged",
    "store_k_scale_cache_ragged",
]
