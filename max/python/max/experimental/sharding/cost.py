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

"""SPMD redistribution cost model and row feasibility."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from max.graph.dim import StaticDim

from .action import AxisAssignment, input_placements_at_axis
from .mesh import DeviceMesh
from .per_shard_dim import global_dim
from .placements import (
    Collective,
    Partial,
    Placement,
    Replicated,
    ShardingError,
    Unknown,
)
from .types import TensorLayout

R: Placement = Replicated()
"""The :class:`Replicated` placement singleton."""

P: Placement = Partial()
"""The :class:`Partial` placement singleton."""


# ─── Per-collective ring formulas ────────────────────────────────────


# An arbitrary cost discount for local memory slicing. With it, a slice costs
# more than moving nothing and less than a collective.
_LOCAL_MEMORY_DISCOUNT = 1.0 / 64


def _ring_allgather(
    message_bytes: float, mesh: DeviceMesh, axis_index: int
) -> float:
    """Ring-allgather cost: ``bytes * (N-1)/N`` on ``axis_index``."""
    n = mesh.mesh_shape[axis_index]
    factor = (n - 1) / n if n > 1 else 0.0
    return message_bytes * factor


def _ring_allreduce(
    message_bytes: float, mesh: DeviceMesh, axis_index: int
) -> float:
    """Ring-allreduce cost: 2x the allgather term (reduce-scatter + allgather)."""
    n = mesh.mesh_shape[axis_index]
    factor = (n - 1) / n if n > 1 else 0.0
    return 2.0 * message_bytes * factor


def _ring_reduce_scatter(
    message_bytes: float, mesh: DeviceMesh, axis_index: int
) -> float:
    """Ring reduce-scatter cost: same volume class as allgather."""
    n = mesh.mesh_shape[axis_index]
    factor = (n - 1) / n if n > 1 else 0.0
    return message_bytes * factor


def transition_cost(
    source: Placement,
    dest: Placement,
    *,
    message_bytes: float,
    mesh: DeviceMesh,
    axis_index: int,
) -> float:
    """Returns the cost of redistributing ``source`` to ``dest`` on ``axis_index``.

    Dispatches on :meth:`Placement.transition_to`, so custom
    :class:`Placement` subclasses participate as long as they return a
    :class:`Collective` member. Anything else is reported as infeasible
    (``+inf``) so the picker rejects it.
    """
    if source == dest:
        return 0.0
    name: Collective = source.transition_to(dest)
    if name is Collective.NOOP:
        return 0.0
    if name in (Collective.LOCAL_SLICE, Collective.KEEP_ONE_COPY):
        n = mesh.mesh_shape[axis_index]
        return message_bytes / n * _LOCAL_MEMORY_DISCOUNT
    if name in (Collective.ALLGATHER, Collective.ALL_TO_ALL):
        return _ring_allgather(message_bytes, mesh, axis_index)
    if name == Collective.ALLREDUCE:
        return _ring_allreduce(message_bytes, mesh, axis_index)
    if name == Collective.REDUCE_SCATTER:
        return _ring_reduce_scatter(message_bytes, mesh, axis_index)
    return float("inf")


# ─── Feasibility ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class _FeasibilityContext:
    """The state one feasibility check needs about *one* mesh axis.

    Internal to :mod:`cost`. Feasibility is decided per mesh axis: given
    the actual per-input placements along that axis and the candidate
    :class:`AxisAssignment`, does the candidate leave every input it moves
    with non-empty shards? This context carries everything the helpers in
    :func:`action_is_feasible` consult. The per-axis size is derived from
    ``mesh`` and ``mesh_axis_idx`` via :attr:`group_size`.
    """

    layouts: tuple[TensorLayout, ...]
    """Per-tensor input layouts. Indexed positionally; one entry per
    tensor argument the rule received."""
    mesh: DeviceMesh
    """The device mesh; only the per-axis size is read."""
    mesh_axis_idx: int
    """The mesh axis under evaluation. Indexes ``mesh.mesh_shape`` and the
    per-axis tuple of every input's placement."""
    input_placements: Sequence[tuple[Placement, ...]] | None = None
    """Current per-input placements (one tuple per tensor input). ``None``
    falls back to reading ``layouts[i].mapping``."""

    @property
    def group_size(self) -> int:
        """The size of the mesh axis under evaluation."""
        return self.mesh.mesh_shape[self.mesh_axis_idx]


def action_is_feasible(
    axs: AxisAssignment,
    actuals: tuple[Placement, ...],
    ctx: _FeasibilityContext,
) -> bool:
    """Returns whether a row can run on this mesh axis.

    A row is infeasible when it would empty a shard or ask for a placement
    no collective can reach.
    """
    for i, p in enumerate(axs.needed_inputs):
        actual = actuals[i]
        if ctx.layouts[i].mapping.mesh.num_devices == 1 and p != R:
            return False
        # A row may only ask for a placement the operand can reach, which
        # also keeps rows off an Unknown operand.
        if actual.transition_to(p) is Collective.INFEASIBLE:
            return False
    for i, p in enumerate(axs.needed_inputs):
        if not _input_passes_empty_shard(i, p, ctx):
            return False
    return True


def _input_passes_empty_shard(
    i: int,
    p: Placement,
    ctx: _FeasibilityContext,
) -> bool:
    """True if input ``i`` under placement ``p`` would not produce an empty shard."""
    p_axis = p.localized_axis()
    if p_axis is None:
        return True
    original = ctx.layouts[i].placements
    current = (
        ctx.input_placements[i]
        if ctx.input_placements is not None
        else original
    )
    if tuple(current) == original and original[ctx.mesh_axis_idx] == p:
        return True
    shape = ctx.layouts[i].shape
    if not 0 <= p_axis < len(shape):
        return False
    dim = global_dim(shape[p_axis])
    effective = _cumulative_axis_group_size(i, p_axis, ctx)
    return not (isinstance(dim, StaticDim) and dim.dim < effective)


def _cumulative_axis_group_size(
    i: int, p_axis: int, ctx: _FeasibilityContext
) -> int:
    """Product of mesh-axis sizes already sharding tensor axis ``p_axis``."""
    placements = (
        ctx.input_placements[i]
        if ctx.input_placements is not None
        else ctx.layouts[i].mapping.placements
    )
    return ctx.group_size * math.prod(
        ctx.mesh.mesh_shape[j]
        for j, other in enumerate(placements)
        if j != ctx.mesh_axis_idx and other.localized_axis() == p_axis
    )


# ─── Byte counts ──────────────────────────────────────────────────────


def _global_axis_size(layout: TensorLayout, tensor_axis: int) -> int:
    """Best-effort global static extent of tensor axis ``tensor_axis``."""
    dim = global_dim(layout.shape[tensor_axis])
    if isinstance(dim, StaticDim):
        return dim.dim
    return 1


def tensor_byte_count(layout: TensorLayout) -> float:
    """Returns the global byte count for ``layout``.

    Symbolic axes fall back to ``1``.
    """
    total = float(layout.dtype.size_in_bytes)
    for tensor_axis in range(len(layout.shape)):
        total *= _global_axis_size(layout, tensor_axis)
    return total


def feasible_rows_at_axis(
    rows: Sequence[AxisAssignment],
    layouts: tuple[TensorLayout, ...],
    mesh: DeviceMesh,
    mesh_axis: int,
    per_input_placements: Sequence[tuple[Placement, ...]],
) -> tuple[AxisAssignment, ...]:
    """Rows feasible at ``mesh_axis``."""
    actuals = input_placements_at_axis(per_input_placements, mesh_axis)
    if any(isinstance(placement, Unknown) for placement in actuals):
        if any(isinstance(placement, Partial) for placement in actuals):
            raise ShardingError(
                "Resolve Partial inputs before mixing them with Unknown inputs, "
                "or rebind them as Unknown."
            )
        return (AxisAssignment(actuals, (Unknown(),) * len(rows[0].outputs)),)
    ctx = _FeasibilityContext(
        layouts=layouts,
        mesh=mesh,
        mesh_axis_idx=mesh_axis,
        input_placements=per_input_placements,
    )
    return tuple(row for row in rows if action_is_feasible(row, actuals, ctx))
