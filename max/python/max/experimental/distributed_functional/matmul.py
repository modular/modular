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

"""Distributed matmul dispatch with placement-aware output rules.

Rules are applied **per mesh axis independently**, so they work for
any mesh dimensionality (1D, 2D, ...) and any tensor rank (2D, 3D
batched matmul, etc.).  The ``axis`` numbers in ``Sharded(axis)``
refer to tensor axes, not mesh axes.

For ``[..., M, K] @ [..., K, N]``, axis roles are:

* **Batch dims**: axes ``0`` to ``max(lhs_rank, rhs_rank) - 3``.
* **Contracting**: ``lhs[-1]`` (K) and ``rhs[-2]`` (K).
* **Non-contracting**: ``lhs[-2]`` (M) and ``rhs[-1]`` (N).

Per-mesh-axis placement rules:

============  ============  ===========  ================================
lhs           rhs           output       pattern
============  ============  ===========  ================================
R             R             R            trivial
S(batch)      S(batch)      S(batch)     batch parallel
S(batch)      R             S(batch)     batch sharded
R             S(batch)      S(batch)     batch sharded
S(M)          R             S(M)         data parallel
R             S(N)          S(N)         column tensor parallel
S(any)        S(K_rhs)      Partial      row tensor parallel
P             R             P            matmul is bilinear in lhs
R             P             P            matmul is bilinear in rhs
============  ============  ===========  ================================

The bilinear rules (PxR, RxP) follow from ``(A1+A2)xB = A1xB + A2xB``.

All other Partial combinations (PxP, SxP, PxS) are auto-reduced
when ``auto_reduce_partial`` is enabled, or raise otherwise.
"""

from __future__ import annotations

from collections.abc import Callable

from max.experimental import tensor
from max.experimental.sharding import (
    DeviceMesh,
    Partial,
    Placement,
    PlacementMapping,
    Replicated,
    Sharded,
)
from max.graph import ops
from max.graph.value import TensorValueLike

from ._context_provider import functional
from .collectives import _has_distributed, make_distributed, to_shard_tvs

# Sentinel: rule function returns this when a Partial input cannot pass
# through and needs an explicit reduce.
NEEDS_REDUCE = object()


def resolve_per_axis_placements(
    lhs_p: tuple[Placement, ...],
    rhs_p: tuple[Placement, ...],
    mesh: DeviceMesh,
    rule: Callable[[Placement, Placement], Placement | object],
    op_name: str,
) -> tuple[Placement, ...]:
    """Applies a per-mesh-axis placement rule to produce output placements.

    For each mesh axis, ``rule(pl, pr)`` is called with the lhs/rhs
    placements.  It must return:

    * A :class:`Placement` — used as the output placement for that axis.
    * ``NEEDS_REDUCE`` — signals that a Partial input cannot pass through.
    * Raise ``NotImplementedError`` for unsupported combinations.
    """
    out_p: list[Placement] = []
    needs_reduce_sides: set[str] = set()

    for ax in range(mesh.ndim):
        pl = lhs_p[ax] if len(lhs_p) > ax else Replicated()
        pr = rhs_p[ax] if len(rhs_p) > ax else Replicated()

        result = rule(pl, pr)

        if result is NEEDS_REDUCE:
            if isinstance(pl, Partial):
                needs_reduce_sides.add("lhs")
            if isinstance(pr, Partial):
                needs_reduce_sides.add("rhs")
        else:
            assert isinstance(result, Placement)
            out_p.append(result)

    if needs_reduce_sides:
        raise ValueError(
            f"{op_name}: unsupported Partial placement on "
            f"{', '.join(sorted(needs_reduce_sides))}. "
            f"Use F.all_reduce_sum() or F.resolve_partials() to reduce "
            f"Partial inputs before {op_name}. "
            f"(Supported: PxR -> P, RxP -> P)"
        )

    return tuple(out_p)


def _matmul_rule(
    batch_dims: set[int],
    lhs_non_contract: int | None,
    rhs_non_contract: int,
    rhs_contract: int,
) -> Callable[[Placement, Placement], Placement | object]:
    """Returns a per-axis placement rule for matmul.

    The returned callable maps ``(pl, pr) -> Placement | NEEDS_REDUCE``
    for use with :func:`resolve_per_axis_placements`.
    """

    def rule(pl: Placement, pr: Placement) -> Placement | object:
        # R x R -> R
        if isinstance(pl, Replicated) and isinstance(pr, Replicated):
            return Replicated()
        # S(batch) x S(batch) -> S(batch)
        if (
            isinstance(pl, Sharded)
            and isinstance(pr, Sharded)
            and pl.axis in batch_dims
            and pl.axis == pr.axis
        ):
            return Sharded(pl.axis)
        # S(batch) x R -> S(batch)
        if (
            isinstance(pl, Sharded)
            and pl.axis in batch_dims
            and isinstance(pr, Replicated)
        ):
            return Sharded(pl.axis)
        # R x S(batch) -> S(batch)
        if (
            isinstance(pl, Replicated)
            and isinstance(pr, Sharded)
            and pr.axis in batch_dims
        ):
            return Sharded(pr.axis)
        # S(M) x R -> S(M)
        if (
            isinstance(pl, Sharded)
            and lhs_non_contract is not None
            and pl.axis == lhs_non_contract
            and isinstance(pr, Replicated)
        ):
            return Sharded(pl.axis)
        # R x S(N) -> S(N)
        if (
            isinstance(pl, Replicated)
            and isinstance(pr, Sharded)
            and pr.axis == rhs_non_contract
        ):
            return Sharded(pr.axis)
        # S(*) x S(K_rhs) -> Partial
        if (
            isinstance(pl, Sharded)
            and isinstance(pr, Sharded)
            and pr.axis == rhs_contract
        ):
            return Partial()
        # P x R -> P (bilinear)
        if isinstance(pl, Partial) and isinstance(pr, Replicated):
            return Partial(pl.reduce_op)
        # R x P -> P (bilinear)
        if isinstance(pl, Replicated) and isinstance(pr, Partial):
            return Partial(pr.reduce_op)
        # Other Partial combos need reduce
        if isinstance(pl, Partial) or isinstance(pr, Partial):
            return NEEDS_REDUCE
        raise NotImplementedError(
            f"matmul: unsupported placements: {pl} x {pr}"
        )

    return rule


@functional(linear=None)
def matmul(lhs: TensorValueLike, rhs: TensorValueLike) -> TensorValueLike:
    """Distributed matmul with placement-aware output rules.

    Rules use semantic axis roles (batch / non-contracting / contracting)
    derived from tensor ranks, so they work for 2-D, 3-D batched, and
    higher-rank matmuls.  See module docstring for the full rule table.
    """
    if not _has_distributed(lhs, rhs):
        return ops.matmul(lhs, rhs)
    # Both must be distributed on the same mesh (enforced by @functional).
    assert isinstance(lhs, tensor.Tensor) and isinstance(rhs, tensor.Tensor)
    mesh = lhs.mesh
    n = mesh.num_devices

    lhs_p = lhs.placements
    rhs_p = rhs.placements

    # Semantic axis roles for [..., M, K] @ [..., K, N].
    lhs_rank = len(lhs.shape)
    rhs_rank = len(rhs.shape)
    rhs_contract = max(0, rhs_rank - 2)
    lhs_non_contract = lhs_rank - 2 if lhs_rank >= 2 else None
    rhs_non_contract = rhs_rank - 1
    batch_dims = set(range(max(0, max(lhs_rank, rhs_rank) - 2)))

    out_p = resolve_per_axis_placements(
        lhs_p,
        rhs_p,
        mesh,
        _matmul_rule(
            batch_dims, lhs_non_contract, rhs_non_contract, rhs_contract
        ),
        "matmul",
    )

    lhs_shards = to_shard_tvs(lhs, n)
    rhs_shards = to_shard_tvs(rhs, n)

    results = [
        ops.matmul(l, r) for l, r in zip(lhs_shards, rhs_shards, strict=False)
    ]
    return make_distributed(results, PlacementMapping(mesh, tuple(out_p)))
