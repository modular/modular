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

"""Elementwise distributed op dispatch (binary and unary).

Each op handles both sharded and non-sharded inputs. For non-sharded
inputs, the call falls through to the graph op directly.

Partial placement resolution is handled upstream by the
:func:`~max.experimental.distributed_functional._context_provider.functional` wrapper
(via its ``linear`` parameter) before these dispatch functions run.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from max.dtype import DType
from max.experimental import tensor
from max.experimental.sharding import (
    Placement,
    PlacementMapping,
    Replicated,
    Sharded,
)
from max.graph import TensorValue, ops
from max.graph.value import TensorValueLike

from ._context_provider import functional
from .collectives import _has_distributed, make_distributed, to_shard_tvs

# ─── Placement resolution for binary elementwise ops ──────────────────


def _broadcast_adjust_placement(p: Placement, rank_delta: int) -> Placement:
    """Adjusts a Sharded axis index for NumPy-style broadcasting.

    When a lower-rank operand (rank r) is broadcast against a higher-rank
    operand (rank R), leading dimensions are prepended.  A ``Sharded(d)``
    on the lower-rank tensor corresponds to ``Sharded(d + R - r)`` in the
    output shape.  Non-Sharded placements are returned unchanged.
    """
    if isinstance(p, Sharded) and rank_delta > 0:
        return Sharded(p.axis + rank_delta)
    return p


def _resolve_binary_placements(
    lhs: tensor.Tensor, rhs: tensor.Tensor, op_name: str
) -> tuple[
    tensor.DeviceMesh, tuple[Placement, ...], tuple[Placement, ...], int
]:
    """Resolves placements for binary elementwise ops.

    Returns (mesh, lhs_placements, rhs_placements, n_shards).
    Both operands must be distributed on the same mesh (enforced by
    ``_validate_distributed_args`` in the ``@functional`` wrapper).

    When the two operands have different tensor ranks (broadcast add of a
    bias, for example), Sharded axis indices on the lower-rank operand are
    adjusted by the rank difference before comparison so that placements
    referring to the same physical dimension are recognized as compatible.
    """
    mesh = lhs.mesh
    lhs_p = lhs.placements
    rhs_p = rhs.placements

    # Adjust Sharded axis indices for broadcast rank mismatch.
    lhs_rank = len(lhs.shape)
    rhs_rank = len(rhs.shape)
    if lhs_rank > rhs_rank:
        rhs_p = tuple(
            _broadcast_adjust_placement(p, lhs_rank - rhs_rank) for p in rhs_p
        )
    elif rhs_rank > lhs_rank:
        lhs_p = tuple(
            _broadcast_adjust_placement(p, rhs_rank - lhs_rank) for p in lhs_p
        )

    # Element-wise: placements must match (or one side is Replicated)
    for ax, (pl, pr) in enumerate(zip(lhs_p, rhs_p, strict=False)):
        if (
            pl != pr
            and not isinstance(pl, Replicated)
            and not isinstance(pr, Replicated)
        ):
            raise ValueError(
                f"{op_name}: incompatible placements on axis {ax}: "
                f"{pl} vs {pr}. Redistribute first."
            )
    return mesh, lhs_p, rhs_p, mesh.num_devices


def _binary_out_placements(
    lhs_p: tuple[Placement, ...], rhs_p: tuple[Placement, ...]
) -> tuple[Placement, ...]:
    """Output placements for element-wise: prefer the non-Replicated side."""
    return tuple(
        pl if not isinstance(pl, Replicated) else pr
        for pl, pr in zip(lhs_p, rhs_p, strict=False)
    )


# ─── Generic unary/binary dispatch ─────────────────────────────────────


def _unary_elementwise(
    graph_op: Callable[..., TensorValueLike],
    x: TensorValueLike,
    *,
    op_name: str = "",
    **kwargs: Any,
) -> TensorValueLike:
    """Generic dispatch for unary elementwise distributed ops."""
    if not _has_distributed(x):
        return graph_op(x, **kwargs)
    assert isinstance(x, tensor.Tensor)
    n = x.mesh.num_devices
    shards = to_shard_tvs(x, n)
    results = [TensorValue(graph_op(s, **kwargs)) for s in shards]
    return make_distributed(results, PlacementMapping(x.mesh, x.placements))


def _binary_elementwise(
    graph_op: Callable[..., TensorValueLike],
    lhs: TensorValueLike,
    rhs: TensorValueLike,
    *,
    op_name: str = "",
    **kwargs: Any,
) -> TensorValueLike:
    """Generic dispatch for binary elementwise distributed ops."""
    if not _has_distributed(lhs, rhs):
        return graph_op(lhs, rhs, **kwargs)

    # Identify the distributed operand; the other may be a scalar (non-Tensor).
    # Mixed distributed/non-distributed Tensors are caught by the wrapper.
    dist = lhs if isinstance(lhs, tensor.Tensor) and lhs.is_distributed else rhs
    assert isinstance(dist, tensor.Tensor)
    mesh = dist.mesh
    n = mesh.num_devices

    if isinstance(lhs, tensor.Tensor) and isinstance(rhs, tensor.Tensor):
        # Both are distributed Tensors (enforced by @functional wrapper).
        mesh, lhs_adj_p, rhs_adj_p, n = _resolve_binary_placements(
            lhs, rhs, op_name
        )
        lhs_s = to_shard_tvs(lhs, n)
        rhs_s = to_shard_tvs(rhs, n)
        out_p = _binary_out_placements(lhs_adj_p, rhs_adj_p)
    elif isinstance(lhs, tensor.Tensor):
        # Distributed Tensor op scalar — create constant on each shard's device.
        lhs_s = to_shard_tvs(lhs, n)
        rhs_s = [ops.constant(rhs, lhs.dtype, s.device) for s in lhs_s]  # type: ignore[arg-type]
        out_p = dist.placements
    else:
        # Scalar op distributed Tensor.
        assert isinstance(rhs, tensor.Tensor)
        rhs_s = to_shard_tvs(rhs, n)
        lhs_s = [ops.constant(lhs, rhs.dtype, s.device) for s in rhs_s]  # type: ignore[arg-type]
        out_p = dist.placements

    results = [
        TensorValue(graph_op(l, r, **kwargs))
        for l, r in zip(lhs_s, rhs_s, strict=False)
    ]
    return make_distributed(results, PlacementMapping(mesh, out_p))


# ─── Factory helpers ─────────────────────────────────────────────────────


def _unary(
    graph_op: Callable[..., TensorValueLike],
) -> Callable[..., TensorValueLike]:
    """Create a distributed unary elementwise op from a graph op."""

    @functools.wraps(graph_op)
    def wrapper(x: TensorValueLike, **kwargs: Any) -> TensorValueLike:
        return _unary_elementwise(
            graph_op, x, op_name=graph_op.__name__, **kwargs
        )

    return wrapper


def _binary(
    graph_op: Callable[..., TensorValueLike],
) -> Callable[..., TensorValueLike]:
    """Create a distributed binary elementwise op from a graph op."""

    @functools.wraps(graph_op)
    def wrapper(
        lhs: TensorValueLike, rhs: TensorValueLike, **kwargs: Any
    ) -> TensorValueLike:
        return _binary_elementwise(
            graph_op, lhs, rhs, op_name=graph_op.__name__, **kwargs
        )

    return wrapper


# ─── Binary ops ────────────────────────────────────────────────────────

add = functional(linear=True)(_binary(ops.add))
sub = functional(linear=True)(_binary(ops.sub))
mul = functional(_binary(ops.mul))
div = functional(_binary(ops.div))
pow = functional(_binary(ops.pow))
mod = functional(_binary(ops.mod))

# ─── Unary ops ─────────────────────────────────────────────────────────

negate = functional(linear=True)(_unary(ops.negate))


@functional
def cast(x: TensorValueLike, dtype: DType) -> TensorValueLike:
    """Distributed cast."""
    return _unary_elementwise(lambda t: ops.cast(t, dtype), x, op_name="cast")


relu = functional(_unary(ops.relu))
abs = functional(_unary(ops.abs))
exp = functional(_unary(ops.exp))
log = functional(_unary(ops.log))
sqrt = functional(_unary(ops.sqrt))
rsqrt = functional(_unary(ops.rsqrt))
sigmoid = functional(_unary(ops.sigmoid))
silu = functional(_unary(ops.silu))
gelu = functional(_unary(ops.gelu))
tanh = functional(_unary(ops.tanh))
cos = functional(_unary(ops.cos))
sin = functional(_unary(ops.sin))
erf = functional(_unary(ops.erf))
