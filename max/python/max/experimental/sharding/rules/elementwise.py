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

"""Placement rules for unary, binary, and ternary elementwise ops."""

from __future__ import annotations

from typing import Any

from max.dtype import DType
from max.experimental.sharding.placements import Sharded
from max.experimental.sharding.types import TensorLayout
from max.graph.dim import Dim, StaticDim

from ..action import AxisAssignment
from ..cost import P, R


def _is_size_one(dim: Dim) -> bool:
    """``True`` for a static size-1 dim (a broadcast axis)."""
    return isinstance(dim, StaticDim) and dim.dim == 1


def _aligned_axis(layout: TensorLayout, out_axis: int, out_rank: int) -> int:
    """Input tensor axis that trailing-aligns to ``out_axis``, or ``-1`` if absent."""
    return out_axis - (out_rank - layout.rank)


def _elementwise_rows(
    layouts: tuple[TensorLayout, ...], *, linear: bool
) -> list[AxisAssignment]:
    """Returns elementwise rows, aligned by trailing axis.

    For each output axis, offers one row sharding every input that carries the
    axis at full extent. Inputs that broadcast the axis (absent or size 1)
    stay Replicated, since a broadcast operand is whole on every device
    already. Prepends the ``(R, ...) -> R`` fallback; ``linear`` ops add a
    ``(P, ...) -> P`` row.
    """
    n_in = len(layouts)
    out_rank = max(layout.rank for layout in layouts)
    rows: list[AxisAssignment] = [AxisAssignment((R,) * n_in, (R,))]
    for out_axis in range(out_rank):
        aligned = [
            _aligned_axis(layout, out_axis, out_rank) for layout in layouts
        ]
        shardable = [
            i
            for i, (k, layout) in enumerate(zip(aligned, layouts, strict=True))
            if k >= 0 and not _is_size_one(layout.shape[k])
        ]
        if not shardable:
            continue
        # Every input carrying the axis at full extent shards together. A row
        # that sharded only one of them would leave the others whole along an
        # axis they do not broadcast, and the shapes would not line up.
        needed = tuple(
            Sharded(aligned[i]) if i in shardable else R for i in range(n_in)
        )
        rows.append(AxisAssignment(needed, (Sharded(out_axis),)))
    if linear:
        rows.append(AxisAssignment((P,) * n_in, (P,)))
    return rows


def unary_rule(
    x: TensorLayout, *extra: Any, **kwargs: Any
) -> list[AxisAssignment]:
    """Strategies for nonlinear unary ops (relu, exp, ...): Replicated or Sharded."""
    return [
        AxisAssignment((R,), (R,)),
        *(AxisAssignment((Sharded(d),), (Sharded(d),)) for d in range(x.rank)),
    ]


def linear_unary_rule(
    x: TensorLayout, *extra: Any, **kwargs: Any
) -> list[AxisAssignment]:
    """Strategies for linear unary ops (negate, ...): adds Partial passthrough."""
    return [
        AxisAssignment((R,), (R,)),
        *(AxisAssignment((Sharded(d),), (Sharded(d),)) for d in range(x.rank)),
        AxisAssignment((P,), (P,)),
    ]


def cast_rule(x: TensorLayout, dtype: DType) -> list[AxisAssignment]:
    """Preserves partial sums only when the cast is an identity."""
    return linear_unary_rule(x) if x.dtype == dtype else unary_rule(x)


def binary_rule(lhs: TensorLayout, rhs: TensorLayout) -> list[AxisAssignment]:
    """Strategies for elementwise binary ops (mul, div, ...): no Partial passthrough."""
    return _elementwise_rows((lhs, rhs), linear=False)


def mul_rule(lhs: TensorLayout, rhs: TensorLayout) -> list[AxisAssignment]:
    """Strategies for ``mul``: a partial sum times a copy is a partial sum."""
    return [
        *binary_rule(lhs, rhs),
        AxisAssignment((P, R), (P,)),
        AxisAssignment((R, P), (P,)),
    ]


def div_rule(lhs: TensorLayout, rhs: TensorLayout) -> list[AxisAssignment]:
    """Strategies for ``div``: a partial sum over a copy is a partial sum."""
    return [*binary_rule(lhs, rhs), AxisAssignment((P, R), (P,))]


def linear_binary_rule(
    lhs: TensorLayout, rhs: TensorLayout
) -> list[AxisAssignment]:
    """Strategies for linear binary ops (add, sub): Partial passthrough when both Partial."""
    return _elementwise_rows((lhs, rhs), linear=True)


def ternary_rule(
    condition: TensorLayout, x: TensorLayout, y: TensorLayout
) -> list[AxisAssignment]:
    """Strategies for elementwise ternary (``where``): trailing-aligned Sharded."""
    return _elementwise_rows((condition, x, y), linear=False)
