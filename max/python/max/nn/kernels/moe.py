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

"""Mixture-of-experts kernels as ``Tensor`` ops, with their sharding rules."""

from max import functional as F
from max.dtype import DType
from max.nn._graph_ops import _moe_sigmoid_gemv_router
from max.nn._graph_ops import grouped_matmul_ragged as _grouped_matmul_ragged
from max.nn._graph_ops import moe_create_indices as _moe_create_indices
from max.nn._graph_ops import (
    moe_create_indices_packed as _moe_create_indices_packed,
)
from max.nn._graph_ops import moe_finalize as _moe_finalize
from max.nn._graph_ops import (
    moe_router_group_limited as _moe_router_group_limited,
)
from max.nn._graph_ops import (
    moe_single_group_router_packed as _moe_single_group_router_packed,
)
from max.nn._graph_ops.ep import fused_silu as _fused_silu
from max.nn._graph_ops.ep import fused_silu_quantized as _fused_silu_quantized
from max.nn.activation import GatedActivation
from max.tensor.sharding import Sharded, Unknown
from max.tensor.sharding.action import AxisAssignment
from max.tensor.sharding.cost import P, R
from max.tensor.sharding.types import TensorLayout


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
    ``expert_usage_stats`` are small per-call metadata, always ``Replicated``
    under tensor parallelism.

    Under expert parallelism, each device already holds its own experts
    (``weight``'s leading axis) and the tokens dispatched to them, so
    ``hidden_states`` and ``weight`` share a device-opaque split instead of a
    global one; ``expert_start_indices`` / ``expert_ids`` are each device's
    own (``Unknown``), while ``expert_usage_stats`` stays the same value on
    every device.
    """
    split = Sharded(0)
    return [
        AxisAssignment((R, R, R, R, R), (R,)),
        # Column-parallel: weight's N (out) axis sharded -> output's N axis.
        AxisAssignment((R, Sharded(1), R, R, R), (Sharded(1),)),
        # Row-parallel: weight's K (contraction) axis sharded, matched by
        # hidden_states' K axis -> partial sum.
        AxisAssignment((Sharded(1), Sharded(2), R, R, R), (P,)),
        # Expert-parallel: each device's own tokens against its own experts.
        AxisAssignment((split, split, Unknown(), Unknown(), R), (split,)),
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


def fused_silu_rule(
    x: TensorLayout,
    row_offsets: TensorLayout,
    activation: GatedActivation | None = None,
    interleaved: bool = False,
) -> list[AxisAssignment]:
    """Strategies for ``fused_silu``: preserves every input axis (nonlinear).

    ``row_offsets`` is the small per-call expert boundary tensor; it is
    always ``Replicated``. No ``Partial`` row: the gated activation is
    nonlinear, whichever one ``activation`` selects.
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
moe_single_group_router_packed = F.functional(_moe_single_group_router_packed)
moe_create_indices_packed = F.functional(_moe_create_indices_packed)


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


__all__ = [
    "fused_silu",
    "fused_silu_quantized",
    "grouped_matmul_ragged",
    "moe_create_indices",
    "moe_create_indices_packed",
    "moe_finalize",
    "moe_router_group_limited",
    "moe_sigmoid_gemv_router",
    "moe_single_group_router_packed",
]
