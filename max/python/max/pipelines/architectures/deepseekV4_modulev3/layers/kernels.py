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
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from max import functional as F
from max import tree
from max.driver import Accelerator
from max.graph import DeviceRef, TensorValue
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
    "block_diagonal_block_scaled_matmul",
    "block_scales_interleave",
    "compress_windows_ragged",
    "device_architectures",
    "indexer_score_ragged",
    "latent_sparse_attention_ragged",
    "moe_create_indices_with_slots",
    "moe_router_group_limited",
    "quantize_dynamic_scaled_float8_grouped_mx",
    "replicated_rule",
    "rope_quantize_dynamic_block_scaled",
    "rope_ragged",
    "top_k_per_row",
    "wo_a_mxfp8_supported",
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


# A weight's rows split across devices in whole 128-row scale tiles.
block_scales_interleave = F.functional(
    kernels.block_scales_interleave,
    rule=_named_rule(
        kernels.block_scales_interleave, {"scales": Sharded(0)}, (Sharded(0),)
    ),
)
# Each device's whole heads quantize to its columns and their scale tiles.
rope_quantize_dynamic_block_scaled = F.functional(
    kernels.rope_quantize_dynamic_block_scaled,
    rule=_named_rule(
        kernels.rope_quantize_dynamic_block_scaled,
        {"x": Sharded(1)},
        (Sharded(1), Sharded(1)),
    ),
)


def _block_diagonal_matmul(
    a: TensorValue,
    b: TensorValue,
    a_scales: TensorValue,
    b_scales: TensorValue,
    group_rows: int,
) -> TensorValue:
    """The block-diagonal matmul over a shard's groups of ``group_rows`` rows."""
    return kernels.block_diagonal_block_scaled_matmul(
        a, b, a_scales, b_scales, int(b.shape[0]) // group_rows
    )


# A device's columns of ``a`` are its groups: its rows of ``b`` and output
# columns.
block_diagonal_block_scaled_matmul = F.functional(
    _block_diagonal_matmul,
    rule=_named_rule(
        _block_diagonal_matmul,
        {
            "a": Sharded(1),
            "b": Sharded(0),
            "a_scales": Sharded(1),
            "b_scales": Sharded(0),
        },
        (Sharded(1),),
    ),
)


def device_architectures(devices: Iterable[object]) -> list[str]:
    """Each device's architecture name, ``""`` for one that is not a GPU."""
    names = []
    for device in devices:
        if isinstance(device, DeviceRef):
            device = device.to_device() if device.is_gpu() else None
        names.append(
            device.architecture_name if isinstance(device, Accelerator) else ""
        )
    return names


def wo_a_mxfp8_supported(
    architectures: Sequence[str],
    groups: int,
    o_lora_rank: int,
    group_width: int,
) -> bool:
    """Whether ``wo_a`` runs as one block-diagonal MXFP8 matmul on every device.

    The matmul is SM100-only and needs each of a device's ``groups`` blocks to
    be whole 128 x 128 scale tiles. A device holding a single group is left
    out: the matmul's fallback configs for a large token count are built only
    for two groups or more.
    """
    return (
        bool(architectures)
        and all(name.startswith("sm_10") for name in architectures)
        and groups >= 2
        and o_lora_rank % 128 == 0
        and group_width % 128 == 0
    )
