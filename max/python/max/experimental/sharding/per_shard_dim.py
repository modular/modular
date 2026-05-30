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

"""Per-shard :class:`~max.graph.Dim` wrapper carrying one value per device.

A :class:`PerShardDim` bundles one :class:`~max.graph.Dim` per mesh shard
(device index along a :class:`Sharded` mesh axis). Arithmetic distributes
element-wise. :func:`make_per_shard_dim` collapses to a plain :class:`Dim`
when every cell is equal. Wrappers must never reach MLIR; they are projected
per shard by shape rules before any :meth:`Dim.to_mlir` call.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import NoReturn, TypeGuard

from max.graph.dim import Dim, SymbolicDim
from max.graph.shape import Shape

__all__ = [
    "PerShardDim",
    "cell_at",
    "global_dim",
    "global_shape",
    "is_one",
    "is_per_shard_dim",
    "make_per_shard_dim",
    "shape_at",
]


def _all_equal(items: Sequence[object]) -> bool:
    if not items:
        return True
    first = items[0]
    return all(x == first for x in items[1:])


class PerShardDim(Dim):
    """A :class:`~max.graph.Dim` whose ``per_shard`` tuple lists one cell per mesh shard.

    Used on :class:`Sharded` axes when shards hold different per-device
    sizes (uneven static splits, dynamic axes minted per-shard). The
    wrapper must be projected per shard via :func:`cell_at` before
    reaching MLIR.
    """

    __slots__ = ("per_shard",)

    per_shard: tuple[Dim, ...]

    def __init__(self, per_shard: Iterable[Dim] | PerShardDim) -> None:
        if isinstance(per_shard, PerShardDim):
            if per_shard is self:
                return
            per_shard = per_shard.per_shard
        object.__setattr__(self, "per_shard", tuple(per_shard))

    def to_mlir(self) -> NoReturn:
        """Raises; wrappers must be projected per shard before reaching MLIR."""
        raise TypeError(
            f"PerShardDim cannot be lowered to MLIR: {self!r}. "
            "Project per shard via the shape rule before reaching graph IR."
        )

    @property
    def parameters(self) -> Iterable[SymbolicDim]:
        """Distinct symbolic-dim parameters referenced across all cells."""
        seen: set[str] = set()
        for d in self.per_shard:
            for p in d.parameters:
                if p.name not in seen:
                    seen.add(p.name)
                    yield p

    def __hash__(self) -> int:
        return hash(self.per_shard)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PerShardDim):
            return self.per_shard == other.per_shard
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        eq = self.__eq__(other)
        return NotImplemented if eq is NotImplemented else not eq

    def __str__(self) -> str:
        return "(" + " | ".join(str(d) for d in self.per_shard) + ")"

    def __repr__(self) -> str:
        return f"PerShardDim({self.per_shard!r})"

    def __add__(self, rhs: object) -> Dim:
        return _binop(self, rhs, lambda a, b: a + b)

    def __radd__(self, lhs: object) -> Dim:
        return _binop(lhs, self, lambda a, b: a + b)

    def __mul__(self, rhs: object) -> Dim:
        return _binop(self, rhs, lambda a, b: a * b)

    def __rmul__(self, lhs: object) -> Dim:
        return _binop(lhs, self, lambda a, b: a * b)

    def __floordiv__(self, rhs: object) -> Dim:
        return _binop(self, rhs, lambda a, b: a // b)

    def __rfloordiv__(self, lhs: object) -> Dim:
        return _binop(lhs, self, lambda a, b: a // b)

    def __neg__(self) -> Dim:
        return make_per_shard_dim(tuple(-d for d in self.per_shard))

    def __sub__(self, rhs: object) -> Dim:
        return _binop(self, rhs, lambda a, b: a - b)

    def __rsub__(self, lhs: object) -> Dim:
        return _binop(lhs, self, lambda a, b: a - b)

    def __int__(self) -> int:
        if len(self.per_shard) == 1:
            return int(self.per_shard[0])
        raise TypeError(
            f"int(PerShardDim) is undefined for per-shard cells "
            f"{self.per_shard!r}; use symbolic Dim arithmetic instead."
        )

    def __index__(self) -> int:
        return self.__int__()


def _cells_of(x: object) -> tuple[Dim, ...] | None:
    """Returns ``x.per_shard`` if ``x`` is a :class:`PerShardDim`, else ``None``."""
    if isinstance(x, PerShardDim):
        return x.per_shard
    return None


def _binop(
    lhs: object,
    rhs: object,
    op: Callable[[Dim, Dim], Dim],
) -> Dim:
    lhs_cells = _cells_of(lhs)
    rhs_cells = _cells_of(rhs)
    if lhs_cells is not None and rhs_cells is not None:
        if len(lhs_cells) != len(rhs_cells):
            raise ValueError(
                f"PerShardDim arity mismatch: "
                f"{len(lhs_cells)} vs {len(rhs_cells)}."
            )
        cells = tuple(
            op(a, b) for a, b in zip(lhs_cells, rhs_cells, strict=False)
        )
    elif lhs_cells is not None:
        assert isinstance(rhs, (int, str, Dim))
        rhs_dim = Dim(rhs)
        cells = tuple(op(a, rhs_dim) for a in lhs_cells)
    elif rhs_cells is not None:
        assert isinstance(lhs, (int, str, Dim))
        lhs_dim = Dim(lhs)
        cells = tuple(op(lhs_dim, b) for b in rhs_cells)
    else:
        raise TypeError(
            f"_binop called without per-shard operand: {lhs!r}, {rhs!r}"
        )
    return make_per_shard_dim(cells, force_wrap=True)


def make_per_shard_dim(
    per_shard: Sequence[Dim], *, force_wrap: bool = False
) -> Dim:
    """Wraps ``per_shard`` in a :class:`PerShardDim`, or collapses to a plain :class:`Dim`.

    Collapses to ``per_shard[0]`` when every entry is equal. Pass
    ``force_wrap=True`` to keep the wrapper even when entries happen to
    be equal (used on Sharded axes so :func:`global_dim` recovers the
    global extent by summing).
    """
    if not per_shard:
        raise ValueError("make_per_shard_dim: empty per_shard tuple.")
    cells = tuple(Dim(d) for d in per_shard)
    if not force_wrap and _all_equal(cells):
        return cells[0]
    return PerShardDim(cells)


def global_dim(d: Dim) -> Dim:
    """Folds a :class:`PerShardDim` into a single global :class:`Dim` by summing.

    Convenience for the common single-mesh-axis :class:`Sharded` case.
    Multi-mesh-axis sharding requires the placement's
    :meth:`Placement.global_dim`.
    """
    if not is_per_shard_dim(d):
        return d
    total: Dim = d.per_shard[0]
    for x in d.per_shard[1:]:
        total = total + x
    return total


def is_per_shard_dim(d: object) -> TypeGuard[PerShardDim]:
    """``True`` if ``d`` is a :class:`PerShardDim`."""
    return isinstance(d, PerShardDim)


def is_one(d: Dim) -> bool:
    """``True`` if every cell of ``d`` is equal to 1.

    For a :class:`PerShardDim`, checks every cell recursively. For any
    other :class:`~max.graph.Dim`, checks ``d == 1`` via integer
    comparison; non-static dims return ``False``.
    """
    from max.graph.dim import StaticDim

    if is_per_shard_dim(d):
        return all(is_one(c) for c in d.per_shard)
    return isinstance(d, StaticDim) and d.dim == 1


def cell_at(d: Dim, shard: int) -> Dim:
    """Returns the dimension value at ``shard``.

    If ``d`` is a regular (non-distributed) :class:`Dim`, then ``d`` is
    returned unchanged.
    """
    if is_per_shard_dim(d):
        return d.per_shard[shard]
    return d


def shape_at(shape: Sequence[Dim], shard: int) -> Shape:
    """Projects every axis of ``shape`` onto a single ``shard`` index."""
    return Shape([cell_at(Dim(d), shard) for d in shape])


def global_shape(shape: Sequence[Dim]) -> Shape:
    """Returns the global shape by summing every :class:`PerShardDim` axis."""
    return Shape([global_dim(Dim(d)) for d in shape])
