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
"""DeepSeek-V4's custom kernels as eager-tensor ops, with sharding rules.

Each is the graph-API binding in ``max.nn._graph_ops`` under
``F.functional``, which hands a paged cache over as its graph values (blocks
as buffers). The rules are the V2 tensor-parallel split: attention and
indexer heads on the ``tp`` axis, everything else (caches, offsets, the
routing bookkeeping) replicated, since every device holds the whole
single-head latent cache.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any

from max import functional as F
from max import tree
from max.nn import _graph_ops as kernels
from max.tensor.sharding import (
    AxisAssignment,
    Partial,
    Placement,
    Replicated,
    Sharded,
    TensorLayout,
    replicated_rows,
)

__all__ = [
    "compress_windows_ragged",
    "indexer_score_ragged",
    "latent_sparse_attention_ragged",
    "moe_create_indices_with_slots",
    "moe_router_group_limited",
    "quantize_dynamic_scaled_float8_grouped_mx",
    "replicated_rule",
    "rope_ragged",
    "top_k_per_row",
]

R = Replicated()
HEADS = Sharded(1)
"""The head axis of a ``[T, heads, head_dim]`` query or output."""


def replicated_rule(output_count: int = 1) -> Callable[..., Any]:
    """A rule for an op that only runs on whole tensors."""

    def rule(*args: Any, **kwargs: Any) -> list[AxisAssignment]:
        return replicated_rows(args, kwargs, output_count=output_count)

    return rule


def _named_rule(
    graph_op: Callable[..., Any],
    split: Mapping[str, Placement],
    outputs: tuple[Placement, ...],
) -> Callable[..., Any]:
    """The replicated row, plus one placing the ``split`` parameters' tensors.

    Every tensor of a parameter not in ``split`` (a cache's included) is
    ``Replicated`` in the split row too.
    """
    signature = inspect.signature(graph_op)

    def rule(*args: Any, **kwargs: Any) -> list[AxisAssignment]:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        needed: list[Placement] = []
        for name, value in bound.arguments.items():
            count = len(tree.leaves(value, leaf=TensorLayout))
            needed += [split.get(name, R)] * count
        whole = AxisAssignment((R,) * len(needed), (R,) * len(outputs))
        return [whole, AxisAssignment(tuple(needed), outputs)]

    return rule


latent_sparse_attention_ragged = F.functional(
    kernels.latent_sparse_attention_ragged,
    rule=_named_rule(
        kernels.latent_sparse_attention_ragged,
        {"q": HEADS, "attn_sink": Sharded(0)},
        (HEADS,),
    ),
)
# Linear in the heads: a device's share of them sums to its share of a score.
indexer_score_ragged = F.functional(
    kernels.indexer_score_ragged,
    rule=_named_rule(
        kernels.indexer_score_ragged,
        {"q": HEADS, "weights": HEADS},
        (Partial(),),
    ),
)
rope_ragged = F.functional(
    kernels.rope_ragged,
    rule=_named_rule(kernels.rope_ragged, {"input": HEADS}, (HEADS,)),
)
compress_windows_ragged = F.functional(
    kernels.compress_windows_ragged, rule=replicated_rule()
)
top_k_per_row = F.functional(kernels.top_k_per_row, rule=replicated_rule(2))
moe_create_indices_with_slots = F.functional(
    kernels.moe_create_indices_with_slots, rule=replicated_rule(9)
)
# Inside the routed body, on each device's own slots.
quantize_dynamic_scaled_float8_grouped_mx = F.functional(
    kernels.quantize_dynamic_scaled_float8_grouped_mx, rule=replicated_rule(2)
)
# Every device routes the same replicated scores to the same experts.
moe_router_group_limited = F.functional(
    kernels.moe_router_group_limited, rule=replicated_rule(2)
)
