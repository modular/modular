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

"""TensorLayout-based rules for matmul-family ops (matmul, qmatmul, layer_norm).

Per-mesh-axis placement rules for ``[..., M, K] @ [..., K, N]``::

    lhs         rhs         output      pattern
    R           R           R           trivial
    S(batch)    S(batch)    S(batch)    batch parallel
    S(batch)    R           S(batch)    batch sharded
    R           S(batch)    S(batch)    batch sharded
    S(M)        R           S(M)        data parallel
    R           S(N)        S(N)        column tensor parallel
    S(any)      S(K_rhs)    Partial     row tensor parallel
    P           R           P           bilinear in lhs (linear keeps P)
    R           P           P           bilinear in rhs (linear keeps P)

Partial inputs that cannot pass through (e.g. P x S) are suggested for
resolution to Replicated, triggering all_reduce in the dispatch engine.
"""

from __future__ import annotations

import builtins
from collections.abc import Callable

from max.experimental.sharding.mappings import PlacementMapping
from max.experimental.sharding.placements import (
    Partial,
    Placement,
    Replicated,
    Sharded,
)
from max.experimental.sharding.types import TensorLayout

from ._common import RuleSignature


def _resolve_per_axis_placements(
    lhs_p: tuple[Placement, ...],
    rhs_p: tuple[Placement, ...],
    mesh_ndim: int,
    rule: Callable[
        [Placement, Placement], tuple[Placement, Placement, Placement]
    ],
    op_name: str,
) -> tuple[tuple[Placement, ...], tuple[Placement, ...], tuple[Placement, ...]]:
    """Applies a per-mesh-axis rule to produce suggested inputs + output placements."""
    suggested_lhs: list[Placement] = []
    suggested_rhs: list[Placement] = []
    out_p: list[Placement] = []

    for ax in range(mesh_ndim):
        pl = lhs_p[ax] if len(lhs_p) > ax else Replicated()
        pr = rhs_p[ax] if len(rhs_p) > ax else Replicated()
        sl, sr, o = rule(pl, pr)
        suggested_lhs.append(sl)
        suggested_rhs.append(sr)
        out_p.append(o)

    return tuple(suggested_lhs), tuple(suggested_rhs), tuple(out_p)


def _is_matrix_dim(axis: int, non_contract: int | None, contract: int) -> bool:
    return axis in (contract, non_contract)


def _matmul_axis_rule(
    lhs_non_contract: int | None,
    lhs_contract: int,
    rhs_non_contract: int,
    rhs_contract: int,
    out_non_contract_lhs: int | None,
    out_non_contract_rhs: int,
) -> Callable[[Placement, Placement], tuple[Placement, Placement, Placement]]:
    """Returns a per-axis rule that produces (suggested_lhs, suggested_rhs, out)."""

    def rule(
        pl: Placement, pr: Placement
    ) -> tuple[Placement, Placement, Placement]:
        if isinstance(pl, Replicated) and isinstance(pr, Replicated):
            return (Replicated(), Replicated(), Replicated())

        if isinstance(pl, Sharded) and not _is_matrix_dim(
            pl.axis, lhs_non_contract, lhs_contract
        ):
            if isinstance(pr, Sharded) and pl.axis == pr.axis:
                return (pl, pr, Sharded(pl.axis))
            if isinstance(pr, Replicated):
                return (pl, pr, Sharded(pl.axis))
        if isinstance(pr, Sharded) and not _is_matrix_dim(
            pr.axis, rhs_non_contract, rhs_contract
        ):
            if isinstance(pl, Replicated):
                return (pl, pr, Sharded(pr.axis))

        if (
            isinstance(pl, Sharded)
            and lhs_non_contract is not None
            and pl.axis == lhs_non_contract
            and isinstance(pr, Replicated)
        ):
            out_axis = (
                out_non_contract_lhs
                if out_non_contract_lhs is not None
                else pl.axis
            )
            return (pl, pr, Sharded(out_axis))

        if (
            isinstance(pl, Replicated)
            and isinstance(pr, Sharded)
            and pr.axis == rhs_non_contract
        ):
            return (pl, pr, Sharded(out_non_contract_rhs))

        if (
            isinstance(pl, Sharded)
            and isinstance(pr, Sharded)
            and pr.axis == rhs_contract
        ):
            return (pl, pr, Partial())

        if isinstance(pl, Partial) and isinstance(pr, Replicated):
            return (pl, pr, Partial(pl.reduce_op))
        if isinstance(pl, Replicated) and isinstance(pr, Partial):
            return (pl, pr, Partial(pr.reduce_op))

        if isinstance(pl, Partial) and isinstance(pr, Partial):
            raise ValueError(
                "matmul: Partial x Partial is not supported. "
                "Use resolve_partials() on at least one input first."
            )

        if isinstance(pl, Partial) and not isinstance(pr, Replicated):
            return (Replicated(), pr, _output_for_clean(Replicated(), pr))
        if isinstance(pr, Partial) and not isinstance(pl, Replicated):
            return (pl, Replicated(), _output_for_clean(pl, Replicated()))

        raise NotImplementedError(
            f"matmul: unsupported placements: {pl} x {pr}"
        )

    def _output_for_clean(pl: Placement, pr: Placement) -> Placement:
        if isinstance(pl, Replicated) and isinstance(pr, Replicated):
            return Replicated()
        if isinstance(pl, Replicated) and isinstance(pr, Sharded):
            if pr.axis == rhs_non_contract:
                return Sharded(out_non_contract_rhs)
            return pr
        if isinstance(pl, Sharded) and isinstance(pr, Replicated):
            if lhs_non_contract is not None and pl.axis == lhs_non_contract:
                out_axis = (
                    out_non_contract_lhs
                    if out_non_contract_lhs is not None
                    else pl.axis
                )
                return Sharded(out_axis)
            return pl
        if isinstance(pl, Sharded) and isinstance(pr, Sharded):
            if pr.axis == rhs_contract:
                return Partial()
        return Replicated()

    return rule


def matmul_rule(lhs: TensorLayout, rhs: TensorLayout) -> RuleSignature:
    """Placement rule for matmul-family ops."""
    lhs_p = lhs.mapping.to_placements()
    rhs_p = rhs.mapping.to_placements()
    mesh = lhs.mapping.mesh
    mesh_ndim = mesh.ndim

    lhs_rank = lhs.rank
    rhs_rank = rhs.rank
    lhs_contract = lhs_rank - 1
    lhs_non_contract = lhs_rank - 2 if lhs_rank >= 2 else None
    rhs_contract = builtins.max(0, rhs_rank - 2)
    rhs_non_contract = rhs_rank - 1

    out_rank = builtins.max(lhs_rank, rhs_rank)
    out_non_contract_rhs = out_rank - 1
    out_non_contract_lhs = out_rank - 2 if out_rank >= 2 else None

    suggested_lhs_p, suggested_rhs_p, out_p = _resolve_per_axis_placements(
        lhs_p,
        rhs_p,
        mesh_ndim,
        _matmul_axis_rule(
            lhs_non_contract,
            lhs_contract,
            rhs_non_contract,
            rhs_contract,
            out_non_contract_lhs,
            out_non_contract_rhs,
        ),
        "matmul",
    )

    return (
        (
            PlacementMapping(mesh, suggested_lhs_p),
            PlacementMapping(mesh, suggested_rhs_p),
        ),
        (PlacementMapping(mesh, out_p),),
    )


def layer_norm_rule(
    x: TensorLayout,
    gamma: TensorLayout,
    beta: TensorLayout,
    epsilon: float = 1e-5,
) -> RuleSignature:
    """Placement rule for layer_norm: rejects sharded norm dimensions."""
    x_p = x.mapping.to_placements()
    mesh = x.mapping.mesh
    w_rank = gamma.rank
    norm_start = x.rank - w_rank

    suggested_x_p = tuple(
        Replicated() if isinstance(p, Partial) else p for p in x_p
    )

    for p in suggested_x_p:
        if isinstance(p, Sharded) and p.axis >= norm_start:
            raise ValueError(
                f"layer_norm: cannot normalize along sharded axis "
                f"{p.axis}. Gather first or shard a different axis."
            )

    suggested_x_mapping = PlacementMapping(mesh, suggested_x_p)
    out_mapping = PlacementMapping(mesh, suggested_x_p)

    return (
        (suggested_x_mapping, gamma.mapping, beta.mapping, epsilon),
        (out_mapping,),
    )
