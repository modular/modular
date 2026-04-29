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

"""Placement rules for elementwise ops.

- ``unary_rule`` / ``linear_unary_rule`` — nonlinear / linear unary
- ``binary_rule`` / ``linear_binary_rule`` — nonlinear / linear binary
- ``ternary_rule`` — always nonlinear (where)
"""

from __future__ import annotations

from max.experimental.sharding.mappings import PlacementMapping
from max.experimental.sharding.placements import (
    Partial,
    Placement,
    Replicated,
    Sharded,
)
from max.experimental.sharding.types import TensorLayout

from ._common import RuleSignature

# ─── Helpers ──────────────────────────────────────────────────────────


def _broadcast_adjust(p: Placement, delta: int) -> Placement:
    return (
        Sharded(p.axis + delta) if isinstance(p, Sharded) and delta > 0 else p
    )


def _resolve_partial(p: Placement) -> Placement:
    return Replicated() if isinstance(p, Partial) else p


def _maybe_resolve(
    p: tuple[Placement, ...], resolve: bool
) -> tuple[Placement, ...]:
    return tuple(_resolve_partial(x) for x in p) if resolve else p


def _prefer_non_replicated(
    *all_p: tuple[Placement, ...],
) -> tuple[Placement, ...]:
    result: list[Placement] = []
    for axis_ps in zip(*all_p, strict=True):
        chosen = Replicated()
        for p in axis_ps:
            if not isinstance(p, Replicated):
                chosen = p
                break
        result.append(chosen)
    return tuple(result)


def _align_to_output(
    input_p: tuple[Placement, ...],
    output_p: tuple[Placement, ...],
    input_rank: int,
    output_rank: int,
) -> tuple[Placement, ...]:
    delta = output_rank - input_rank
    result = list(input_p)
    for ax in range(len(result)):
        out_p_ax = output_p[ax]
        if isinstance(result[ax], Replicated) and isinstance(out_p_ax, Sharded):
            in_axis = out_p_ax.axis - delta
            if in_axis >= 0:
                result[ax] = Sharded(in_axis)
    return tuple(result)


# ─── Unary rules ─────────────────────────────────────────────────────


def _unary_impl(x: TensorLayout, *extra: object, linear: bool) -> RuleSignature:
    mesh = x.mapping.mesh
    suggested_p = _maybe_resolve(x.mapping.to_placements(), resolve=not linear)
    m = PlacementMapping(mesh, suggested_p)
    return (m, *extra), (m,)


def unary_rule(x: TensorLayout, *extra: object) -> RuleSignature:
    """Nonlinear unary: resolves Partials."""
    return _unary_impl(x, *extra, linear=False)


def linear_unary_rule(x: TensorLayout, *extra: object) -> RuleSignature:
    """Linear unary: Partials pass through."""
    return _unary_impl(x, *extra, linear=True)


# ─── Binary rules ────────────────────────────────────────────────────


def _binary_impl(
    lhs: TensorLayout,
    rhs: TensorLayout,
    linear: bool,
) -> RuleSignature:
    mesh = lhs.mapping.mesh
    lhs_p, rhs_p = lhs.mapping.to_placements(), rhs.mapping.to_placements()
    lhs_rank, rhs_rank = lhs.rank, rhs.rank

    if lhs_rank > rhs_rank:
        rhs_p = tuple(_broadcast_adjust(p, lhs_rank - rhs_rank) for p in rhs_p)
    elif rhs_rank > lhs_rank:
        lhs_p = tuple(_broadcast_adjust(p, rhs_rank - lhs_rank) for p in lhs_p)

    lhs_has_p = any(isinstance(p, Partial) for p in lhs_p)
    rhs_has_p = any(isinstance(p, Partial) for p in rhs_p)
    if not (linear and lhs_has_p and rhs_has_p):
        lhs_p = _maybe_resolve(lhs_p, resolve=lhs_has_p)
        rhs_p = _maybe_resolve(rhs_p, resolve=rhs_has_p)

    for ax, (pl, pr) in enumerate(zip(lhs_p, rhs_p, strict=True)):
        if (
            pl != pr
            and not isinstance(pl, Replicated)
            and not isinstance(pr, Replicated)
        ):
            raise ValueError(
                f"binary elementwise: incompatible placements on axis {ax}: "
                f"{pl} vs {pr}. Redistribute first."
            )

    out_p = _prefer_non_replicated(lhs_p, rhs_p)
    out_rank = max(lhs_rank, rhs_rank)
    lhs_m = PlacementMapping(
        mesh, _align_to_output(lhs_p, out_p, lhs.rank, out_rank)
    )
    rhs_m = PlacementMapping(
        mesh, _align_to_output(rhs_p, out_p, rhs.rank, out_rank)
    )
    return (lhs_m, rhs_m), (PlacementMapping(mesh, out_p),)


def binary_rule(lhs: TensorLayout, rhs: TensorLayout) -> RuleSignature:
    """Nonlinear binary: resolves Partials."""
    return _binary_impl(lhs, rhs, linear=False)


def linear_binary_rule(lhs: TensorLayout, rhs: TensorLayout) -> RuleSignature:
    """Linear binary: Partials pass through when both sides are Partial."""
    return _binary_impl(lhs, rhs, linear=True)


# ─── Ternary rule ────────────────────────────────────────────────────


def ternary_rule(
    condition: TensorLayout,
    x: TensorLayout,
    y: TensorLayout,
) -> RuleSignature:
    """Ternary (where): always nonlinear, resolves all Partials."""
    layouts = [condition, x, y]
    max_rank = max(l.rank for l in layouts)
    mesh = condition.mapping.mesh

    all_p = [
        _maybe_resolve(
            tuple(
                _broadcast_adjust(p, max_rank - l.rank)
                for p in l.mapping.to_placements()
            ),
            resolve=True,
        )
        for l in layouts
    ]

    for ax in range(len(all_p[0])):
        non_rep = [p[ax] for p in all_p if not isinstance(p[ax], Replicated)]
        if len(set(str(p) for p in non_rep)) > 1:
            raise ValueError(
                f"ternary elementwise: incompatible on mesh axis {ax}."
            )

    out_p = _prefer_non_replicated(*all_p)
    aligned = [
        _align_to_output(sp, out_p, l.rank, max_rank)
        for sp, l in zip(all_p, layouts, strict=True)
    ]
    mappings = tuple(PlacementMapping(mesh, ap) for ap in aligned)
    return mappings, (PlacementMapping(mesh, out_p),)


# ─── Backward compatibility aliases ─────────────────────────────────

unary_passthrough = unary_rule
binary_elementwise = binary_rule
ternary_elementwise = ternary_rule
